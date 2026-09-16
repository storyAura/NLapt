"""Process-wide download slot shared by caption models and Hy-MT2.

A transfer outlives the dialog that started it. Progress/finish go through
one hub so a reopened 设置 window can re-attach. One download at a time.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.core.errors import DownloadCancelledError
from nlapt.diagnostics import get_logger
from nlapt.local.download import download_file
from nlapt.local.runtime import RuntimeAsset, ensure_runtime

from nlapt_gui.resources import app_data_dir
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

PROGRESS_EMIT_STEP_BYTES = 8 * 1024 * 1024
RUNTIME_DIR_NAME = "runtime"
DOWNLOAD_OK = "ok"
DOWNLOAD_CANCELLED = "cancelled"
DOWNLOAD_ERROR = "error"


class DownloadHub(QObject):
    """Fan-out point for download signals; outlives every dialog/bridge."""

    progress = Signal(str, str, object, object)  # family_id, quant_label, done, total
    finished = Signal(str, str, str, str)  # family_id, quant_label, status, message


@dataclass(frozen=True)
class _ActiveDownload:
    """Snapshot of the single in-flight download (replaced, never mutated)."""

    family_id: str
    quant_label: str
    cancel: threading.Event
    done: int = 0
    total: int | None = None


_HUB: DownloadHub | None = None
_ACTIVE_TASK: _ActiveDownload | None = None
_ACTIVE_LOCK = threading.Lock()
_DOWNLOAD_POOL: QThreadPool | None = None
DOWNLOAD_POOL_MAX_THREADS = 1


def get_download_pool() -> QThreadPool:
    """Single-thread pool so model downloads never queue behind translation."""
    global _DOWNLOAD_POOL
    if _DOWNLOAD_POOL is None:
        pool = QThreadPool()
        pool.setMaxThreadCount(DOWNLOAD_POOL_MAX_THREADS)
        _DOWNLOAD_POOL = pool
    return _DOWNLOAD_POOL


def get_download_hub() -> DownloadHub:
    """Process-wide download signal hub (created lazily on the GUI thread)."""
    global _HUB
    if _HUB is None:
        _HUB = DownloadHub()
    return _HUB


def active_download() -> tuple[str, str, int, int | None] | None:
    """(family_id, quant_label, done_bytes, total_bytes) or None."""
    with _ACTIVE_LOCK:
        task = _ACTIVE_TASK
        if task is None:
            return None
        return (task.family_id, task.quant_label, task.done, task.total)


def launch_download_jobs(
    family_id: str,
    quant_label: str,
    jobs: list[tuple[str, Path, int, str]],
    runtime_asset: RuntimeAsset | None,
    pool: QThreadPool | None = None,
    headers: Mapping[str, str] | None = None,
) -> bool:
    """Run download jobs on ``pool`` (or the dedicated download pool).

    ``expected == 0`` in a job means "do not verify size" (provisional
    catalog files). Those bytes are still counted toward progress using
    the on-disk size after each file finishes.
    """
    known = sum(expected for _url, _dest, expected, _sha in jobs if expected > 0)
    total_bytes: int | None = known if known else None
    if runtime_asset is not None:
        total_bytes = (total_bytes or 0) + runtime_asset.size_bytes
    cancel = threading.Event()
    global _ACTIVE_TASK
    with _ACTIVE_LOCK:
        if _ACTIVE_TASK is not None:
            return False
        _ACTIVE_TASK = _ActiveDownload(
            family_id=family_id,
            quant_label=quant_label,
            cancel=cancel,
            done=0,
            total=total_bytes,
        )
    hub = get_download_hub()

    def work() -> str:
        finished_prefix = 0
        last_emitted = -PROGRESS_EMIT_STEP_BYTES

        def report(done: int, _total: int | None, *, base: int) -> None:
            nonlocal last_emitted
            global _ACTIVE_TASK
            overall = base + done
            with _ACTIVE_LOCK:
                if _ACTIVE_TASK is not None:
                    _ACTIVE_TASK = _dc_replace(_ACTIVE_TASK, done=overall)
            if (
                overall - last_emitted < PROGRESS_EMIT_STEP_BYTES
                and overall != total_bytes
            ):
                return
            last_emitted = overall
            hub.progress.emit(family_id, quant_label, overall, total_bytes)

        try:
            if runtime_asset is not None:
                ensure_runtime(
                    app_data_dir() / RUNTIME_DIR_NAME,
                    progress=lambda done, _t: report(done, None, base=0),
                    cancel=cancel,
                )
                finished_prefix = runtime_asset.size_bytes
            for url, dest, expected, sha256 in jobs:
                download_file(
                    url,
                    dest,
                    expected_bytes=expected if expected > 0 else None,
                    expected_sha256=sha256 or None,
                    progress=lambda done, _t, *, b=finished_prefix: report(
                        done, None, base=b
                    ),
                    cancel=cancel,
                    headers=headers,
                )
                if expected > 0:
                    finished_prefix += expected
                elif dest.is_file():
                    finished_prefix += dest.stat().st_size
        except DownloadCancelledError:
            return DOWNLOAD_CANCELLED
        return DOWNLOAD_OK

    def finish(status: str, message: str) -> None:
        global _ACTIVE_TASK
        with _ACTIVE_LOCK:
            _ACTIVE_TASK = None
        if status == DOWNLOAD_ERROR:
            _LOGGER.warning(
                "download failed for %s/%s: %s", family_id, quant_label, message
            )
        elif status == DOWNLOAD_CANCELLED:
            _LOGGER.info("download cancelled for %s/%s", family_id, quant_label)
        hub.finished.emit(family_id, quant_label, status, message)

    run_async(
        pool if pool is not None else get_download_pool(),
        work,
        on_done=lambda status: finish(str(status), ""),
        on_error=lambda message: finish(DOWNLOAD_ERROR, message),
    )
    return True


def cancel_active_download() -> None:
    """Signal the process-wide download to stop (partial files are kept)."""
    with _ACTIVE_LOCK:
        task = _ACTIVE_TASK
    if task is not None:
        task.cancel.set()
