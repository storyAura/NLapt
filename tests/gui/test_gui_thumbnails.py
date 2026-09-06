"""Tests for nlapt_gui.widgets.thumbnails (async LRU + disk thumbnail loader)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QPixmap

from nlapt_gui.resources import app_data_dir
from nlapt_gui.widgets.thumbnails import (
    HEIGHT_BUCKET_PX,
    THUMB_CACHE_LIMIT,
    THUMB_DISK_DIR_NAME,
    ThumbnailLoader,
    bucket_height,
    get_decode_pool,
)


def _write_png(path: Path, size: tuple[int, int]) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 60, 20)).save(path, format="PNG")
    return path


@pytest.fixture()
def small_png(tmp_path: Path) -> Path:
    return _write_png(tmp_path / "small.png", (40, 30))


@pytest.fixture()
def large_png(tmp_path: Path) -> Path:
    return _write_png(tmp_path / "large.png", (400, 300))


class TestBucketHeight:
    def test_rounds_up_to_next_bucket(self) -> None:
        assert bucket_height(1) == HEIGHT_BUCKET_PX
        assert bucket_height(HEIGHT_BUCKET_PX) == HEIGHT_BUCKET_PX
        assert bucket_height(HEIGHT_BUCKET_PX + 1) == 2 * HEIGHT_BUCKET_PX

    def test_never_below_one_bucket(self) -> None:
        assert bucket_height(0) == HEIGHT_BUCKET_PX
        assert bucket_height(-5) == HEIGHT_BUCKET_PX


class TestThumbnailLoader:
    def test_contract_cache_limit(self) -> None:
        assert THUMB_CACHE_LIMIT == 512

    def test_default_pool_is_decode_pool(self) -> None:
        loader = ThumbnailLoader()
        assert loader._pool is get_decode_pool()
        assert loader._pool is not QThreadPool.globalInstance()

    def test_async_load_emits_ready(self, qtbot, small_png: Path) -> None:
        loader = ThumbnailLoader()
        with qtbot.waitSignal(loader.ready, timeout=2000) as blocker:
            assert loader.request("k1", small_png, 44) is None
        key, pix = blocker.args
        assert key == "k1"
        assert isinstance(pix, QPixmap)
        assert not pix.isNull()
        # image smaller than the bucket is never upscaled
        assert pix.height() == 30

    def test_scaled_read_caps_height_at_bucket(self, qtbot, large_png: Path) -> None:
        loader = ThumbnailLoader()
        with qtbot.waitSignal(loader.ready, timeout=2000) as blocker:
            loader.request("big", large_png, 44)
        _key, pix = blocker.args
        assert pix.height() == bucket_height(44)

    def test_cache_hit_returns_synchronously(self, qtbot, small_png: Path) -> None:
        loader = ThumbnailLoader()
        with qtbot.waitSignal(loader.ready, timeout=2000):
            loader.request("k1", small_png, 44)
        cached = loader.request("k1", small_png, 44)
        assert cached is not None and not cached.isNull()
        assert loader.pixmap("k1", 44) is cached

    def test_duplicate_inflight_request_not_restarted(self, qtbot, small_png: Path) -> None:
        loader = ThumbnailLoader()
        emitted: list[str] = []
        loader.ready.connect(lambda key, _pix: emitted.append(key))
        with qtbot.waitSignal(loader.ready, timeout=2000):
            loader.request("k1", small_png, 44)
            loader.request("k1", small_png, 44)  # in flight -> ignored
        qtbot.wait(100)
        assert emitted == ["k1"]

    def test_lru_evicts_oldest(self, qtbot, tmp_path: Path) -> None:
        loader = ThumbnailLoader(cache_limit=2)
        for name in ("a", "b", "c"):
            png = _write_png(tmp_path / f"{name}.png", (8, 6))
            with qtbot.waitSignal(loader.ready, timeout=2000):
                loader.request(name, png, 44)
        assert loader.cache_size() == 2
        assert loader.pixmap("a", 44) is None
        assert loader.pixmap("b", 44) is not None
        assert loader.pixmap("c", 44) is not None

    def test_missing_file_logs_and_never_emits(self, qtbot, tmp_path: Path) -> None:
        loader = ThumbnailLoader()
        with qtbot.assertNotEmitted(loader.ready, wait=300):
            loader.request("nope", tmp_path / "missing.png", 44)

    def test_clear_drops_cache(self, qtbot, small_png: Path) -> None:
        loader = ThumbnailLoader()
        with qtbot.waitSignal(loader.ready, timeout=2000):
            loader.request("k1", small_png, 44)
        loader.clear()
        assert loader.cache_size() == 0
        assert loader.pixmap("k1", 44) is None

    def test_clear_invalidates_inflight_generation(self, qtbot, small_png: Path) -> None:
        loader = ThumbnailLoader()
        callbacks: list[object] = []
        loader.ready.connect(lambda key, _pix: callbacks.append(key))
        loader.request("k1", small_png, 44)
        loader.clear()
        with qtbot.waitSignal(loader.ready, timeout=2000):
            loader.request("k1", small_png, 44)
        assert callbacks == ["k1"]

    def test_failed_request_is_not_requeued_by_repaint(self, qtbot, tmp_path: Path) -> None:
        loader = ThumbnailLoader()
        missing = tmp_path / "missing.png"
        with qtbot.assertNotEmitted(loader.ready, wait=250):
            loader.request("missing", missing, 44)
        before = len(loader._in_flight)
        loader.request("missing", missing, 44)
        assert len(loader._in_flight) == before


class TestDiskCache:
    """加载提速: decoded thumbnails persist across loaders and sessions."""

    def _decode_once(self, qtbot, png: Path) -> None:
        loader = ThumbnailLoader()
        with qtbot.waitSignal(loader.ready, timeout=2000):
            loader.request("k1", png, 44)

    def test_thumbnail_persisted_to_disk(self, qtbot, small_png: Path) -> None:
        self._decode_once(qtbot, small_png)
        thumbs = list((app_data_dir() / THUMB_DISK_DIR_NAME).glob("*.png"))
        assert len(thumbs) == 1

    def test_fresh_loader_serves_from_disk_without_redecoding(
        self, qtbot, small_png: Path
    ) -> None:
        self._decode_once(qtbot, small_png)
        # Corrupt the ORIGINAL while keeping mtime + size identical: only the
        # persisted thumbnail can produce a valid image now.
        stat = small_png.stat()
        small_png.write_bytes(b"x" * stat.st_size)
        os.utime(small_png, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        fresh = ThumbnailLoader()
        with qtbot.waitSignal(fresh.ready, timeout=2000) as blocker:
            fresh.request("k1", small_png, 44)
        _key, pix = blocker.args
        assert not pix.isNull()
        assert pix.height() == 30  # the cached decode, not the garbage bytes

    def test_changed_file_gets_a_fresh_entry(self, qtbot, small_png: Path) -> None:
        self._decode_once(qtbot, small_png)
        _write_png(small_png, (60, 50))  # replaced image: new mtime/size key
        fresh = ThumbnailLoader()
        with qtbot.waitSignal(fresh.ready, timeout=2000) as blocker:
            fresh.request("k1", small_png, 44)
        assert blocker.args[1].height() == 50
