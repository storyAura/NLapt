"""Optional Pillow loader for the image-tools package."""

from __future__ import annotations

import importlib
from types import ModuleType

from nlapt.core.errors import ImageProcessingError

PILLOW_MISSING_MESSAGE = (
    "The 'Pillow' package is required for image processing. "
    "Install it with: pip install nlapt[images]"
)


def load_pil() -> ModuleType:
    """Import ``PIL.Image`` or raise :class:`ImageProcessingError`.

    Optional dependency: imported here (not at module top) so the rest of
    ``nlapt.images`` stays importable without Pillow installed.
    """
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise ImageProcessingError(PILLOW_MISSING_MESSAGE) from exc
