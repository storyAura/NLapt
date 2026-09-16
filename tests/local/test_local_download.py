"""Tests for nlapt.local.download: streaming, resume, cancel, size checks."""

from __future__ import annotations

import io
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from nlapt.core.errors import DownloadCancelledError, DownloadError, ValidationError
from nlapt.local.download import (
    MSG_GATED_DENIED,
    USER_AGENT,
    _AuthStrippingRedirectHandler,
    download_file,
    part_path,
)

PAYLOAD = b"hello local world"
URL = "https://example.test/model.gguf"


class FakeResponse:
    def __init__(
        self, data: bytes, *, status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self._buffer = io.BytesIO(data)
        self.status = status
        self.headers = (
            headers if headers is not None else {"Content-Length": str(len(data))}
        )
        self.closed = False

    def read(self, size: int) -> bytes:
        return self._buffer.read(size)

    def close(self) -> None:
        self.closed = True


def make_opener(*responses: Any) -> tuple[Any, list[Any]]:
    """Opener yielding queued responses (a response, or an exception to raise)."""
    queue = list(responses)
    requests: list[Any] = []

    def opener(request: Any, timeout: float) -> Any:
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return opener, requests


class TestHappyPath:
    def test_downloads_and_renames(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        opener, _ = make_opener(FakeResponse(PAYLOAD))
        seen: list[tuple[int, int | None]] = []
        result = download_file(
            URL,
            dest,
            expected_bytes=len(PAYLOAD),
            progress=lambda done, total: seen.append((done, total)),
            opener=opener,
            chunk_size=4,
        )
        assert result == dest
        assert dest.read_bytes() == PAYLOAD
        assert not part_path(dest).exists()
        assert seen[0] == (0, len(PAYLOAD))
        assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))

    def test_already_downloaded_skips_network(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        dest.write_bytes(PAYLOAD)

        def opener(request: Any, timeout: float) -> Any:
            raise AssertionError("network must not be touched")

        result = download_file(URL, dest, expected_bytes=len(PAYLOAD), opener=opener)
        assert result == dest

    def test_response_always_closed(self, tmp_path: Path) -> None:
        response = FakeResponse(PAYLOAD)
        opener, _ = make_opener(response)
        download_file(URL, tmp_path / "m.gguf", opener=opener)
        assert response.closed


class TestValidation:
    @pytest.mark.parametrize(
        "bad_url", ["ftp://example.test/x", "http://example.test/x", "file:///x"]
    )
    def test_rejects_non_https_urls(self, tmp_path: Path, bad_url: str) -> None:
        with pytest.raises(ValidationError):
            download_file(bad_url, tmp_path / "x", opener=None)

    def test_rejects_bad_chunk_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            download_file(URL, tmp_path / "x", chunk_size=0)


class TestSha256:
    def test_matching_digest_accepted(self, tmp_path: Path) -> None:
        import hashlib

        dest = tmp_path / "model.gguf"
        opener, _ = make_opener(FakeResponse(PAYLOAD))
        download_file(
            URL,
            dest,
            expected_bytes=len(PAYLOAD),
            expected_sha256=hashlib.sha256(PAYLOAD).hexdigest().upper(),
            opener=opener,
        )
        assert dest.read_bytes() == PAYLOAD

    def test_wrong_digest_discards_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        opener, _ = make_opener(FakeResponse(PAYLOAD))
        with pytest.raises(DownloadError, match="SHA256"):
            download_file(
                URL,
                dest,
                expected_bytes=len(PAYLOAD),
                expected_sha256="0" * 64,
                opener=opener,
            )
        assert not dest.exists()
        assert not part_path(dest).exists()


class TestResume:
    def test_resume_sends_range_and_appends(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        part_path(dest).write_bytes(PAYLOAD[:5])
        remaining = PAYLOAD[5:]
        opener, requests = make_opener(
            FakeResponse(
                remaining,
                status=206,
                headers={"Content-Length": str(len(remaining))},
            )
        )
        seen: list[tuple[int, int | None]] = []
        download_file(
            URL,
            dest,
            expected_bytes=len(PAYLOAD),
            progress=lambda done, total: seen.append((done, total)),
            opener=opener,
        )
        assert requests[0].get_header("Range") == "bytes=5-"
        assert dest.read_bytes() == PAYLOAD
        assert seen[0] == (5, len(PAYLOAD))

    def test_server_ignoring_range_restarts(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        part_path(dest).write_bytes(b"stale-junk")
        opener, _ = make_opener(FakeResponse(PAYLOAD, status=200))
        download_file(URL, dest, expected_bytes=len(PAYLOAD), opener=opener)
        assert dest.read_bytes() == PAYLOAD

    def test_http_416_restarts_from_scratch(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        part_path(dest).write_bytes(b"junk-larger-than-file-somehow")
        error = urllib.error.HTTPError(URL, 416, "range", None, None)  # type: ignore[arg-type]
        opener, _ = make_opener(error, FakeResponse(PAYLOAD))
        download_file(URL, dest, expected_bytes=len(PAYLOAD), opener=opener)
        assert dest.read_bytes() == PAYLOAD


class TestFailures:
    def test_http_error_wrapped(self, tmp_path: Path) -> None:
        error = urllib.error.HTTPError(URL, 404, "nf", None, None)  # type: ignore[arg-type]
        opener, _ = make_opener(error)
        with pytest.raises(DownloadError, match="404"):
            download_file(URL, tmp_path / "x.gguf", opener=opener)

    def test_network_error_wrapped(self, tmp_path: Path) -> None:
        opener, _ = make_opener(urllib.error.URLError("boom"))
        with pytest.raises(DownloadError):
            download_file(URL, tmp_path / "x.gguf", opener=opener)

    def test_bad_status_wrapped(self, tmp_path: Path) -> None:
        opener, _ = make_opener(FakeResponse(b"", status=500))
        with pytest.raises(DownloadError, match="500"):
            download_file(URL, tmp_path / "x.gguf", opener=opener)

    def test_short_download_keeps_part_for_resume(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        opener, _ = make_opener(
            FakeResponse(PAYLOAD[:4], headers={"Content-Length": str(len(PAYLOAD))})
        )
        with pytest.raises(DownloadError, match="不完整"):
            download_file(URL, dest, expected_bytes=len(PAYLOAD), opener=opener)
        assert part_path(dest).exists()
        assert not dest.exists()

    def test_oversized_download_aborts_and_discards_part(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        opener, _ = make_opener(FakeResponse(PAYLOAD + b"extra"))
        with pytest.raises(DownloadError, match="超出预期"):
            download_file(URL, dest, expected_bytes=len(PAYLOAD), opener=opener)
        assert not part_path(dest).exists()

    def test_overrun_capped_mid_stream_even_in_small_chunks(
        self, tmp_path: Path
    ) -> None:
        dest = tmp_path / "model.gguf"
        opener, _ = make_opener(FakeResponse(b"x" * 64))
        with pytest.raises(DownloadError, match="超出预期"):
            download_file(
                URL, dest, expected_bytes=10, opener=opener, chunk_size=4
            )
        assert not part_path(dest).exists()


class TestCancel:
    def test_cancel_raises_and_keeps_part(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        cancel = threading.Event()
        cancel.set()
        opener, _ = make_opener(FakeResponse(PAYLOAD))
        with pytest.raises(DownloadCancelledError):
            download_file(URL, dest, cancel=cancel, opener=opener)
        assert part_path(dest).exists()
        assert not dest.exists()


class TestAuthHeaders:
    def test_headers_reach_the_opener(self, tmp_path: Path) -> None:
        dest = tmp_path / "gated.onnx"
        opener, requests = make_opener(FakeResponse(PAYLOAD))
        download_file(
            URL,
            dest,
            opener=opener,
            headers={"Authorization": "Bearer hf_test", "User-Agent": "spoof"},
        )
        assert requests[0].get_header("Authorization") == "Bearer hf_test"
        assert requests[0].get_header("User-agent") == USER_AGENT

    @pytest.mark.parametrize("code", [401, 403])
    def test_gated_http_error_is_actionable(self, tmp_path: Path, code: int) -> None:
        error = urllib.error.HTTPError(URL, code, "denied", None, None)  # type: ignore[arg-type]
        opener, _ = make_opener(error)
        with pytest.raises(DownloadError, match="受限模型"):
            download_file(URL, tmp_path / "x.onnx", opener=opener)
        assert str(code) in MSG_GATED_DENIED.format(code=code)

    @pytest.mark.parametrize("code", [401, 403])
    def test_gated_status_on_response_is_actionable(
        self, tmp_path: Path, code: int
    ) -> None:
        opener, _ = make_opener(FakeResponse(b"", status=code))
        with pytest.raises(DownloadError, match="设置▸CHA标注"):
            download_file(URL, tmp_path / "x.onnx", opener=opener)


class TestAuthStrippingRedirect:
    def test_cross_host_drops_authorization(self) -> None:
        req = urllib.request.Request(
            "https://huggingface.co/repo/resolve/main/w.bin",
            headers={"Authorization": "Bearer hf_secret"},
        )
        handler = _AuthStrippingRedirectHandler()
        redirected = handler.redirect_request(
            req,
            None,
            302,
            "Found",
            {},
            "https://cas-bridge.xethub.hf.co/x/w.bin",
        )
        assert redirected is not None
        assert redirected.get_header("Authorization") is None

    def test_same_host_keeps_authorization(self) -> None:
        req = urllib.request.Request(
            "https://huggingface.co/repo/resolve/main/w.bin",
            headers={"Authorization": "Bearer hf_secret"},
        )
        handler = _AuthStrippingRedirectHandler()
        redirected = handler.redirect_request(
            req,
            None,
            302,
            "Found",
            {},
            "https://huggingface.co/repo/resolve/main/other.bin",
        )
        assert redirected is not None
        assert redirected.get_header("Authorization") == "Bearer hf_secret"
