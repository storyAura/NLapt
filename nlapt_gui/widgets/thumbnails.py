"""ThumbnailLoader - async, LRU + disk cached thumbnails for the file panel.

Images are decoded on a ``QThreadPool`` worker with :class:`QImageReader`
scaled reads (the file is never fully decoded at native size when a smaller
thumbnail is requested), so the GUI thread never blocks on disk I/O.
Decoded ``QImage`` results hop back to the GUI thread via
:func:`nlapt_gui.workers.run_async`, where they are converted to ``QPixmap``
(pixmaps must only be created on the GUI thread), cached and announced with
the ``ready`` signal.

Decoded thumbnails are additionally persisted under
``app_data_dir()/thumbs`` keyed by source path + mtime + size + bucket
height: shrinking a multi-MB PNG to a 64px row still costs a full-size
decode, which made large datasets paint slowly on every (re)open — with the
disk cache, dataset refreshes and later sessions reload thumbnails from
tiny PNGs instead.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import shiboken6
from PySide6.QtCore import QObject, QSize, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap

from nlapt.diagnostics import get_logger

from nlapt_gui.resources import app_data_dir
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

# Maximum number of cached pixmaps (contract: LRU <= 512).
THUMB_CACHE_LIMIT = 512
# Requested heights are quantized to this bucket so continuous panel
# resizes reuse cache entries instead of re-decoding every pixel step.
HEIGHT_BUCKET_PX = 64
# Directory (under the per-user data dir) holding persisted thumbnails.
THUMB_DISK_DIR_NAME = "thumbs"
# Cache files are PNG so alpha survives (thumbs are tiny, size is fine).
_DISK_FORMAT = "PNG"


def bucket_height(target_h: int) -> int:
    """Quantize a requested pixel height up to the next bucket boundary."""
    clamped = max(1, int(target_h))
    buckets = (clamped + HEIGHT_BUCKET_PX - 1) // HEIGHT_BUCKET_PX
    return buckets * HEIGHT_BUCKET_PX


def default_disk_cache_dir() -> Path:
    """Per-user persisted-thumbnail directory (not created until first write)."""
    return app_data_dir() / THUMB_DISK_DIR_NAME


def _disk_cache_file(cache_dir: Path, path: str, height: int) -> Path | None:
    """Cache path for one (source, height); None when the source is gone.

    The key hashes mtime + size, so an edited/replaced image naturally maps
    to a fresh entry instead of serving a stale thumbnail.
    ponytail: stale entries are never evicted (tiny PNGs); add a sweep if
    the thumbs dir ever measurably grows.
    """
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    key = f"{path}|{stat.st_mtime_ns}|{stat.st_size}|{height}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.png"


def _read_scaled(path: str, height: int, cache_dir: Path) -> QImage:
    """Worker-side decode: disk-cache hit, else scaled read + cache write."""
    cache_file = _disk_cache_file(cache_dir, path, height)
    if cache_file is not None and cache_file.is_file():
        cached = QImage(str(cache_file))
        if not cached.isNull():
            return cached
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and size.height() > height:
        width = max(1, round(size.width() * height / size.height()))
        reader.setScaledSize(QSize(width, height))
    image = reader.read()
    if image.isNull():
        raise OSError(f"could not read image {path!r}: {reader.errorString()}")
    if cache_file is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if not image.save(str(cache_file), _DISK_FORMAT):
                _LOGGER.debug("could not persist thumbnail for %r", path)
        except OSError:
            _LOGGER.debug("could not persist thumbnail for %r", path, exc_info=True)
    return image


class ThumbnailLoader(QObject):
    """QThreadPool-backed thumbnail cache shared by grid cells and list rows.

    ``request`` returns the cached pixmap immediately on a hit; on a miss it
    schedules an async load, returns ``None`` and later emits
    ``ready(key, pixmap)`` on the GUI thread. Failed loads are logged and
    never emit (callers keep their placeholder).
    """

    ready = Signal(str, QPixmap)

    def __init__(
        self,
        pool: QThreadPool | None = None,
        *,
        cache_limit: int = THUMB_CACHE_LIMIT,
        disk_cache_dir: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._cache_limit = max(1, cache_limit)
        self._disk_cache_dir = (
            disk_cache_dir if disk_cache_dir is not None else default_disk_cache_dir()
        )
        self._cache: OrderedDict[tuple[str, int], QPixmap] = OrderedDict()
        self._in_flight: set[tuple[str, int]] = set()

    def pixmap(self, key: str, target_h: int) -> QPixmap | None:
        """Cached pixmap for ``key`` at the bucketed height, or ``None``."""
        cache_key = (key, bucket_height(target_h))
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
        return cached

    def request(self, key: str, path: Path, target_h: int) -> QPixmap | None:
        """Return the cached pixmap or start an async load (returns None).

        The eventual result is announced through ``ready(key, pixmap)``;
        duplicate requests while a load is in flight are ignored.
        """
        height = bucket_height(target_h)
        cache_key = (key, height)
        cached = self.pixmap(key, target_h)
        if cached is not None:
            return cached
        if cache_key in self._in_flight:
            return None
        self._in_flight.add(cache_key)

        def _done(image: object) -> None:
            if not shiboken6.isValid(self):
                return  # loader torn down while the decode was in flight
            self._in_flight.discard(cache_key)
            if not isinstance(image, QImage):  # defensive; worker returns QImage
                _LOGGER.warning("unexpected thumbnail payload for %r", key)
                return
            pix = QPixmap.fromImage(image)
            self._store(cache_key, pix)
            self.ready.emit(key, pix)

        def _failed(message: str) -> None:
            if not shiboken6.isValid(self):
                return
            self._in_flight.discard(cache_key)
            _LOGGER.warning("thumbnail load failed for %r: %s", key, message)

        run_async(
            self._pool,
            _read_scaled,
            str(path),
            height,
            self._disk_cache_dir,
            on_done=_done,
            on_error=_failed,
        )
        return None

    def clear(self) -> None:
        """Drop every cached pixmap (memory only — the disk cache stays, so
        a dataset refresh re-fills from tiny PNGs instead of re-decoding)."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Number of cached entries (test/diagnostic helper)."""
        return len(self._cache)

    def _store(self, cache_key: tuple[str, int], pix: QPixmap) -> None:
        self._cache[cache_key] = pix
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._cache_limit:
            evicted_key, _ = self._cache.popitem(last=False)
            _LOGGER.debug("thumbnail cache evicted %r", evicted_key)
