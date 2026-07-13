"""Dataset scanning and image/txt pairing (spec 2.1).

``scan_dataset`` walks a dataset root, finds supported images, pairs each
image with a ``.txt`` caption file in the same directory sharing the same
stem (both stem and extension matched case-insensitively), and collects
orphan txt files (txt without a matching image). Recursive scanning supports
kohya-style ``10_conceptname`` repeat directories; ``.backups/``, ``.nlapt/``
and other dot-directories are skipped.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from nlapt.core.errors import StorageError, ValidationError
from nlapt.core.models import IMAGE_EXTENSIONS, DatasetScanResult, ImageFile
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

TXT_EXTENSION = ".txt"
HIDDEN_DIR_PREFIX = "."

_DIGIT_RUN = re.compile(r"(\d+)")


def natural_sort_key(name: str) -> tuple[object, ...]:
    """Case-insensitive natural sort key; digit runs compare numerically.

    Splitting on digit runs guarantees strings at even tuple positions and
    ints at odd positions, so comparing two keys never mixes types.
    """
    parts = _DIGIT_RUN.split(name.lower())
    return tuple(
        int(part) if index % 2 else part for index, part in enumerate(parts)
    )


def scan_dataset(root: Path, *, recursive: bool = True) -> DatasetScanResult:
    """Scan ``root`` and return paired images plus orphan txt files.

    Raises ValidationError when ``root`` is missing or not a directory, and
    StorageError when a directory or file cannot be read during the scan.
    Results are deterministic: images natural-sorted by key, orphans by
    their root-relative POSIX path.
    """
    base = _validated_root(root)

    images: list[ImageFile] = []
    orphans: list[Path] = []
    for directory in _iter_directories(base, recursive=recursive):
        dir_images, dir_orphans = _scan_directory(base, directory)
        images.extend(dir_images)
        orphans.extend(dir_orphans)

    images.sort(key=lambda item: natural_sort_key(item.key))
    orphans.sort(key=lambda path: natural_sort_key(_relative_posix(base, path)))
    _LOGGER.info(
        "scanned dataset %s: %d images, %d orphan txts", base, len(images), len(orphans)
    )
    return DatasetScanResult(root=base, images=tuple(images), orphan_txts=tuple(orphans))


def _validated_root(root: Path) -> Path:
    base = Path(root)
    if not base.exists():
        raise ValidationError(f"dataset root does not exist: {base}")
    if not base.is_dir():
        raise ValidationError(f"dataset root is not a directory: {base}")
    if not base.is_absolute():
        base = base.absolute()
    return base


def _iter_directories(base: Path, *, recursive: bool):
    """Yield directories to scan, pruning dot-directories (.backups, .nlapt, ...)."""
    if not recursive:
        yield base
        return
    for current, dirnames, _filenames in os.walk(base):
        dirnames[:] = sorted(
            name for name in dirnames if not name.startswith(HIDDEN_DIR_PREFIX)
        )
        yield Path(current)


def _scan_directory(base: Path, directory: Path) -> tuple[list[ImageFile], list[Path]]:
    """Pair images with same-stem txts inside one directory."""
    try:
        entries = [entry for entry in directory.iterdir() if entry.is_file()]
    except OSError as exc:
        raise StorageError(f"cannot list dataset directory {directory}: {exc}") from exc

    txt_by_stem: dict[str, list[Path]] = {}
    for entry in entries:
        if entry.suffix.lower() == TXT_EXTENSION:
            txt_by_stem.setdefault(entry.stem.lower(), []).append(entry)
    for candidates in txt_by_stem.values():
        candidates.sort(key=lambda path: natural_sort_key(path.name))

    images: list[ImageFile] = []
    used_stems: set[str] = set()
    for entry in entries:
        if entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem_key = entry.stem.lower()
        candidates = txt_by_stem.get(stem_key, [])
        txt_path = _choose_txt(entry, candidates)
        if candidates:
            used_stems.add(stem_key)
        images.append(
            ImageFile(
                key=_relative_posix(base, entry),
                image_path=entry,
                txt_path=txt_path if txt_path is not None else entry.with_suffix(TXT_EXTENSION),
                txt_exists=txt_path is not None,
                mtime=_stat_mtime(entry),
            )
        )

    orphans = [
        txt
        for stem_key, candidates in txt_by_stem.items()
        if stem_key not in used_stems
        for txt in candidates
    ]
    return images, orphans


def _choose_txt(image: Path, candidates: list[Path]) -> Path | None:
    """Pick the txt paired with ``image``: exact stem first, else first sorted."""
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.stem == image.stem:
            return candidate
    return candidates[0]


def _stat_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError as exc:
        raise StorageError(f"cannot stat dataset file {path}: {exc}") from exc


def _relative_posix(base: Path, path: Path) -> str:
    return path.relative_to(base).as_posix()
