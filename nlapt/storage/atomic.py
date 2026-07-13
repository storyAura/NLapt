"""Atomic file writes (spec 2.3): temp file + fsync + os.replace.

A partially written file can never replace the target: data is written to a
temp file in the same directory, flushed and fsync'd, then atomically renamed
over the target. The temp file is removed if anything fails.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from nlapt.core.errors import StorageError, ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

TEMP_PREFIX = ".nlapt-tmp-"
TEMP_SUFFIX = ".tmp"
TEXT_ENCODING = "utf-8"  # UTF-8 without BOM


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path``. Raises StorageError on failure."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValidationError(
            f"atomic_write_bytes requires bytes, got {type(data).__name__}"
        )
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise StorageError(f"parent directory does not exist: {parent}")

    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=TEMP_PREFIX, suffix=TEMP_SUFFIX)
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(bytes(data))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except OSError as exc:
        _cleanup_temp(tmp_path)
        raise StorageError(f"atomic write to {target} failed: {exc}") from exc


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` as UTF-8 (no BOM). Newlines are written verbatim.

    Callers that need normalized ``\\n`` newlines should normalize first (see
    :func:`nlapt.storage.text_io.write_caption`).
    """
    if not isinstance(text, str):
        raise ValidationError(f"atomic_write_text requires str, got {type(text).__name__}")
    atomic_write_bytes(path, text.encode(TEXT_ENCODING))


def _cleanup_temp(tmp_path: Path | None) -> None:
    """Best-effort removal of a leftover temp file; never raises."""
    if tmp_path is None:
        return
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        _LOGGER.warning("could not remove temp file %s", tmp_path)
