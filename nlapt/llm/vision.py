"""Image preparation for vision-enabled LLM requests (spec 7.4 / 8).

Images are downscaled so their longest edge does not exceed ``max_edge``
(default 1024 px per spec 8) and re-encoded as RGB JPEG before being sent.
Pillow is optional: its absence raises ``LLMConfigError`` with an install hint.
"""

from __future__ import annotations

import importlib
import io
from pathlib import Path
from types import ModuleType
from typing import Any

from nlapt.core.errors import LLMConfigError, ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

DEFAULT_MAX_EDGE = 1024
DEFAULT_JPEG_QUALITY = 85
MIN_JPEG_QUALITY = 1
MAX_JPEG_QUALITY = 100
JPEG_FORMAT = "JPEG"
RGB_MODE = "RGB"
# Modes that carry alpha and must be composited onto a background first.
ALPHA_MODES = ("RGBA", "LA", "PA")
PALETTE_MODE = "P"
TRANSPARENCY_INFO_KEY = "transparency"
BACKGROUND_COLOR = (255, 255, 255)  # white, so transparency doesn't turn black

PILLOW_MISSING_MESSAGE = (
    "The 'Pillow' package is required for image preparation. "
    "Install it with: pip install nlapt[images]"
)


def _require_pillow() -> ModuleType:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise LLMConfigError(PILLOW_MISSING_MESSAGE) from exc


def _flatten_to_rgb(image: Any, image_module: ModuleType) -> Any:
    """Convert any Pillow image to RGB, compositing alpha over white."""
    mode = image.mode
    if mode == PALETTE_MODE and TRANSPARENCY_INFO_KEY in image.info:
        image = image.convert("RGBA")
        mode = "RGBA"
    if mode in ALPHA_MODES:
        rgba = image.convert("RGBA")
        background = image_module.new(RGB_MODE, rgba.size, BACKGROUND_COLOR)
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    if mode != RGB_MODE:
        return image.convert(RGB_MODE)
    return image


def prepare_image(
    path: Path, *, max_edge: int = DEFAULT_MAX_EDGE, quality: int = DEFAULT_JPEG_QUALITY
) -> bytes:
    """Load, downscale, and re-encode an image as RGB JPEG bytes.

    Raises ValidationError for missing/unreadable files or bad parameters,
    LLMConfigError when Pillow is not installed.
    """
    if max_edge <= 0:
        raise ValidationError(f"max_edge must be > 0, got {max_edge}")
    if not (MIN_JPEG_QUALITY <= quality <= MAX_JPEG_QUALITY):
        raise ValidationError(
            f"quality must be between {MIN_JPEG_QUALITY} and {MAX_JPEG_QUALITY}, "
            f"got {quality}"
        )
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"image file not found: {source}")
    image_module = _require_pillow()
    try:
        with image_module.open(source) as image:
            image.load()
            original_size = image.size
            flattened = _flatten_to_rgb(image, image_module)
            if max(flattened.size) > max_edge:
                flattened.thumbnail((max_edge, max_edge))
            buffer = io.BytesIO()
            flattened.save(buffer, format=JPEG_FORMAT, quality=quality)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot read image {source}: {exc}") from exc
    data = buffer.getvalue()
    _LOGGER.debug(
        "prepared image %s: %s -> JPEG %d bytes (max_edge=%d)",
        source.name,
        original_size,
        len(data),
        max_edge,
    )
    return data
