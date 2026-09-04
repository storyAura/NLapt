"""Export an opened dataset (images + paired txt) as a zip archive."""

from __future__ import annotations

import os
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

from nlapt.core.errors import StorageError, ValidationError
from nlapt.core.models import ImageFile
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import TEMP_PREFIX
from nlapt.storage.scanner import TXT_EXTENSION

_LOGGER = get_logger(__name__)

TEMP_SUFFIX = ".zip"


def export_dataset_zip(
    root: Path,
    dest: Path,
    files: Sequence[ImageFile],
) -> int:
    """Write ``files`` (image + existing txt) to ``dest`` as a zip.

    Streams each member from disk (datasets can be large — the snapshot
    helper that buffers the whole archive in memory is the wrong model).
    Hidden trees such as ``.backups`` / ``.nlapt`` are never included
    because they are not part of the scanned :class:`ImageFile` list.
    Returns the number of archive members written.
    """
    if not files:
        raise ValidationError("dataset has no images to export")
    target = Path(dest)
    parent = target.parent
    if not parent.is_dir():
        raise StorageError(f"parent directory does not exist: {parent}")

    tmp_path: Path | None = None
    try:
        handle, tmp_name = tempfile.mkstemp(
            dir=parent, prefix=TEMP_PREFIX, suffix=TEMP_SUFFIX
        )
        os.close(handle)
        tmp_path = Path(tmp_name)
        count = _write_members(tmp_path, files)
        os.replace(tmp_path, target)
        tmp_path = None
    except Exception as exc:
        raise StorageError(f"dataset export to {target} failed: {exc}") from exc
    finally:
        _cleanup_temp(tmp_path)
    _LOGGER.info("exported %d files from %s to %s", count, root, target)
    return count


def _write_members(tmp_path: Path, files: Sequence[ImageFile]) -> int:
    count = 0
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            if file.image_path.is_file():
                archive.write(file.image_path, file.key)
                count += 1
            if file.txt_exists and file.txt_path.is_file():
                archive.write(file.txt_path, _txt_arcname(file))
                count += 1
    return count


def _txt_arcname(file: ImageFile) -> str:
    """POSIX archive name for the caption next to ``file.key``."""
    return str(Path(file.key).with_suffix(TXT_EXTENSION).as_posix())


def _cleanup_temp(tmp_path: Path | None) -> None:
    if tmp_path is None:
        return
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        _LOGGER.warning("could not remove temp export %s", tmp_path)
