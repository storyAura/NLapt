"""Immutable dataset models shared across modules (spec 2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True)
class ImageFile:
    """One image in the dataset and its paired caption txt.

    ``key`` is the POSIX-style path of the image relative to the dataset root
    and serves as the unique identifier everywhere in the application.
    """

    key: str
    image_path: Path
    txt_path: Path
    txt_exists: bool
    mtime: float


@dataclass(frozen=True)
class DatasetScanResult:
    """Result of scanning a dataset root directory."""

    root: Path
    images: tuple[ImageFile, ...]
    orphan_txts: tuple[Path, ...]
