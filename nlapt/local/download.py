"""Resumable HTTPS file downloader for catalog GGUF files (stdlib only).

Streams into ``<dest>.part`` and atomically renames on completion, so an
interrupted download never leaves a half-written model file at the final
path and resumes from the partial file via a ``Range`` request. When the
caller supplies ``expected_sha256`` the completed file is digest-verified
before the rename (an existing ``dest`` that already matches
``expected_bytes`` is trusted without re-hashing — integrity is checked
once, at download time). Progress and cancellation are injectable
callables so the GUI can report and abort from another thread without
this module knowing about Qt.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nlapt.core.errors import DownloadCancelledError, DownloadError, ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

CHUNK_SIZE = 256 * 1024
PART_SUFFIX = ".part"
DOWNLOAD_TIMEOUT_SECONDS = 30.0
USER_AGENT = "NLapt-local-downloader"
# HTTPS only: every catalog URL is https and TLS is the transport integrity
# layer; plain http would silently disable it.
ALLOWED_SCHEMES = ("https://",)

# done_bytes (including any resumed prefix), total_bytes (None when unknown).
ProgressFn = Callable[[int, int | None], None]


@runtime_checkable
class DownloadResponse(Protocol):
    """Minimal urlopen-response surface the downloader relies on."""

    headers: Any  # Mapping-like with .get() (email.message.Message for urllib)

    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...


# (request, timeout_seconds) -> open response.
OpenerFn = Callable[[urllib.request.Request, float], DownloadResponse]


class _StaleRangeError(Exception):
    """Internal: the server rejected our resume Range (HTTP 416)."""


def _default_opener(
    request: urllib.request.Request, timeout: float
) -> DownloadResponse:
    """Open via urllib (scheme already validated against ALLOWED_SCHEMES)."""
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


def part_path(dest: Path) -> Path:
    """The temporary partial-download path used for ``dest``."""
    return dest.with_name(dest.name + PART_SUFFIX)


def _content_length(response: DownloadResponse) -> int | None:
    """Parsed Content-Length header, or None when absent / invalid."""
    raw = response.headers.get("Content-Length") if response.headers else None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _sha256_of(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Streaming SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _close_quietly(response: DownloadResponse) -> None:
    """Close a response; close-time errors are logged, never raised."""
    close = getattr(response, "close", None)
    if not callable(close):
        return
    try:
        close()
    except OSError:
        _LOGGER.debug("response close failed", exc_info=True)


def _open_with_resume(
    url: str, resume_from: int, *, opener: OpenerFn, timeout: float
) -> tuple[DownloadResponse, int]:
    """Open ``url`` (with a Range header when resuming); return (response, status).

    Raises :class:`_StaleRangeError` on HTTP 416 during a resume so the
    caller can restart cleanly, and :class:`DownloadError` on any other
    HTTP / network failure (a non-200/206 response is closed here).
    """
    headers = {"User-Agent": USER_AGENT}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    try:
        response = opener(request, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and resume_from > 0:
            raise _StaleRangeError() from exc
        raise DownloadError(f"下载失败 (HTTP {exc.code}): {url}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise DownloadError(f"下载失败: {exc}") from exc
    status = int(getattr(response, "status", 200) or 200)
    if status not in (200, 206):
        _close_quietly(response)
        raise DownloadError(f"下载失败 (HTTP {status}): {url}")
    return response, status


def _stream_to_part(
    response: DownloadResponse,
    part: Path,
    *,
    url: str,
    resume_from: int,
    cap: int | None,
    total: int | None,
    progress: ProgressFn | None,
    cancel: threading.Event | None,
    chunk_size: int,
) -> None:
    """Stream the response body into ``part`` (append mode when resuming).

    Raises :class:`DownloadCancelledError` when ``cancel`` fires (the part
    file is kept for resuming) and :class:`DownloadError` on read failures
    or when the stream exceeds ``cap`` (the oversized part is discarded —
    a misbehaving endpoint must not fill the disk).
    """
    done = resume_from
    overrun = False
    mode = "ab" if resume_from > 0 else "wb"
    with open(part, mode) as handle:
        if progress is not None:
            progress(done, total)
        while True:
            if cancel is not None and cancel.is_set():
                _LOGGER.info("download cancelled at %d bytes: %s", done, url)
                raise DownloadCancelledError("下载已取消(已下载部分保留,可续传)")
            try:
                chunk = response.read(chunk_size)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                raise DownloadError(f"下载中断: {exc}") from exc
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if cap is not None and done > cap:
                overrun = True
                break
            if progress is not None:
                progress(done, total)
        handle.flush()
        os.fsync(handle.fileno())
    if overrun:
        part.unlink(missing_ok=True)
        raise DownloadError(f"下载数据超出预期大小(> {cap} 字节),已丢弃")


def _verify_and_finalize(
    part: Path,
    dest: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
    chunk_size: int,
) -> Path:
    """Size + SHA256 checks on ``part``, then the atomic rename to ``dest``."""
    final_size = part.stat().st_size
    if expected_bytes is not None and final_size != expected_bytes:
        if final_size > expected_bytes:
            # Oversized partials cannot be resumed into a valid file.
            part.unlink(missing_ok=True)
            raise DownloadError(
                f"下载文件大小异常({final_size} > 预期 {expected_bytes}),已丢弃"
            )
        raise DownloadError(
            f"下载不完整({final_size} / {expected_bytes} 字节),重试可续传"
        )
    if expected_sha256:
        digest = _sha256_of(part, chunk_size)
        if digest != expected_sha256.lower():
            # A wrong digest cannot be repaired by resuming — discard it.
            part.unlink(missing_ok=True)
            raise DownloadError(
                "文件校验失败(SHA256 与官方指纹不符),已丢弃,请重新下载"
            )
    try:
        os.replace(part, dest)
    except OSError as exc:
        raise DownloadError(f"无法保存下载文件 {dest}: {exc}") from exc
    _LOGGER.info("downloaded %s (%d bytes)", dest, final_size)
    return dest


def download_file(
    url: str,
    dest: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    opener: OpenerFn | None = None,
    chunk_size: int = CHUNK_SIZE,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Download ``url`` to ``dest`` (resumable). Returns ``dest``.

    Raises :class:`ValidationError` for bad input,
    :class:`DownloadCancelledError` when ``cancel`` fires (the ``.part``
    file is kept for resuming) and :class:`DownloadError` on network /
    size / digest failures.
    """
    if not isinstance(url, str) or not url.lower().startswith(ALLOWED_SCHEMES):
        raise ValidationError(f"下载地址必须是 https URL,得到: {url!r}")
    if chunk_size <= 0:
        raise ValidationError(f"chunk_size must be positive, got {chunk_size}")

    if dest.exists() and expected_bytes is not None:
        if dest.stat().st_size == expected_bytes:
            _LOGGER.info("already downloaded: %s", dest)
            if progress is not None:
                progress(expected_bytes, expected_bytes)
            return dest
        _LOGGER.warning("size mismatch on existing %s; re-downloading", dest)

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadError(f"无法创建下载目录 {dest.parent}: {exc}") from exc

    part = part_path(dest)
    resume_from = 0
    if part.exists():
        try:
            resume_from = part.stat().st_size
        except OSError:
            resume_from = 0

    open_fn = opener if opener is not None else _default_opener
    # Bounded retry: a stale-range 416 restarts once from byte zero (a
    # range-less request cannot 416 again on a compliant server).
    for _attempt in range(2):
        try:
            response, status = _open_with_resume(
                url, resume_from, opener=open_fn, timeout=timeout
            )
            break
        except _StaleRangeError:
            _LOGGER.warning("server rejected resume range for %s; restarting", url)
            part.unlink(missing_ok=True)
            resume_from = 0
    else:  # pragma: no cover - needs a non-compliant server 416ing range-less requests
        raise DownloadError(f"下载失败(服务器持续拒绝范围请求): {url}")

    try:
        if resume_from > 0 and status == 200:
            # Server ignored the Range header — restart from byte zero.
            _LOGGER.info("server does not support resume for %s; restarting", url)
            resume_from = 0
        remaining = _content_length(response)
        total = resume_from + remaining if remaining is not None else expected_bytes
        cap = expected_bytes if expected_bytes is not None else total
        _stream_to_part(
            response,
            part,
            url=url,
            resume_from=resume_from,
            cap=cap,
            total=total,
            progress=progress,
            cancel=cancel,
            chunk_size=chunk_size,
        )
    finally:
        _close_quietly(response)

    return _verify_and_finalize(
        part,
        dest,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        chunk_size=chunk_size,
    )
