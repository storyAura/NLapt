"""Pixel-level image tools (alpha flatten, perceptual hashing, image backups).

Pillow is optional: helpers raise :class:`ImageProcessingError` with an
install hint when it is missing. Public types are frozen dataclasses.
"""

from __future__ import annotations

from nlapt.images.alpha import (
    FlattenReport,
    FlattenSpec,
    flatten_alpha,
    has_transparency,
    pick_color,
)
from nlapt.images.backup import ImageBackupInfo, ImageBackupManager
from nlapt.images.hashing import (
    DuplicateGroup,
    ImageFingerprint,
    dhash,
    fingerprint,
    group_similar,
    hamming,
    sha256_file,
    suggest_keep,
)
from nlapt.images.pil import PILLOW_MISSING_MESSAGE, load_pil

__all__ = [
    "PILLOW_MISSING_MESSAGE",
    "DuplicateGroup",
    "FlattenReport",
    "FlattenSpec",
    "ImageBackupInfo",
    "ImageBackupManager",
    "ImageFingerprint",
    "dhash",
    "fingerprint",
    "flatten_alpha",
    "group_similar",
    "hamming",
    "has_transparency",
    "load_pil",
    "pick_color",
    "sha256_file",
    "suggest_keep",
]
