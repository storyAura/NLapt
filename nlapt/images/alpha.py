"""Flatten transparent backgrounds onto a solid (or per-file random) color."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from nlapt.core.errors import ImageProcessingError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.images.pil import load_pil
from nlapt.storage.atomic import atomic_write_bytes

if TYPE_CHECKING:
    from nlapt.images.backup import ImageBackupInfo

_LOGGER = get_logger(__name__)

RGB = tuple[int, int, int]
MODE_FIXED = "fixed"
MODE_RANDOM = "random"
FLATTEN_MODES = frozenset({MODE_FIXED, MODE_RANDOM})
ALPHA_MODES = frozenset({"RGBA", "LA", "PA"})
PALETTE_MODE = "P"
TRANSPARENCY_INFO_KEY = "transparency"
RGB_MODE = "RGB"
DEFAULT_COLOR: RGB = (255, 255, 255)
KEEP_FORMATS = frozenset({"PNG", "WEBP", "TIFF", "BMP", "GIF"})
FALLBACK_FORMAT = "PNG"


@dataclass(frozen=True)
class FlattenSpec:
    """How to pick the replacement color for transparent pixels."""

    mode: str = MODE_FIXED
    color: RGB = DEFAULT_COLOR
    palette: tuple[RGB, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in FLATTEN_MODES:
            raise ValidationError(f"flatten mode must be fixed|random, got {self.mode!r}")
        _require_rgb(self.color)
        for swatch in self.palette:
            _require_rgb(swatch)


@dataclass(frozen=True)
class FlattenReport:
    """Outcome of a flatten-alpha batch."""

    changed: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]
    backup: ImageBackupInfo | None = None


def has_transparency(path: Path) -> bool:
    """True when the image carries an alpha channel or palette transparency."""
    image_module = load_pil()
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"image file not found: {source}")
    try:
        with image_module.open(source) as image:
            if image.mode in ALPHA_MODES:
                return True
            if image.mode == PALETTE_MODE and TRANSPARENCY_INFO_KEY in image.info:
                return True
            return False
    except (OSError, ValueError) as exc:
        raise ImageProcessingError(f"could not read image {source}: {exc}") from exc


def flatten_alpha(path: Path, color: RGB) -> bool:
    """Composite transparent pixels onto ``color`` and write the file back.

    Keeps the original container (PNG / lossless WEBP). Returns False when
    the file has no transparency (JPEG / BMP / opaque PNG).
    """
    _require_rgb(color)
    if not has_transparency(path):
        return False
    image_module = load_pil()
    source = Path(path)
    try:
        with image_module.open(source) as image:
            fmt = (image.format or source.suffix.lstrip(".")).upper()
            working = image
            if working.mode == PALETTE_MODE and TRANSPARENCY_INFO_KEY in working.info:
                working = working.convert("RGBA")
            rgba = working.convert("RGBA")
            background = image_module.new(RGB_MODE, rgba.size, color)
            background.paste(rgba, mask=rgba.getchannel("A"))
            buffer = io.BytesIO()
            out_fmt = fmt if fmt in KEEP_FORMATS else FALLBACK_FORMAT
            save_kwargs: dict[str, object] = {}
            if out_fmt == "WEBP":
                save_kwargs["lossless"] = True
            background.save(buffer, format=out_fmt, **save_kwargs)
            payload = buffer.getvalue()
    except (OSError, ValueError) as exc:
        raise ImageProcessingError(f"could not flatten {source}: {exc}") from exc
    atomic_write_bytes(source, payload)
    _LOGGER.info("flattened alpha of %s onto %s", source, color)
    return True


def pick_color(
    spec: FlattenSpec,
    key: str,
    rng: object | None = None,
) -> RGB:
    """Resolve the fill color for one image under ``spec``.

    Random mode picks from ``palette`` (or ``color`` when the palette is
    empty). A provided ``rng`` must expose ``randrange``; otherwise the
    choice is a stable hash of ``seed:key``.
    """
    if spec.mode == MODE_FIXED:
        return spec.color
    palette: Sequence[RGB] = spec.palette if spec.palette else (spec.color,)
    if rng is not None:
        picker = getattr(rng, "randrange", None)
        if not callable(picker):
            raise ValidationError("rng must provide randrange(n)")
        return palette[int(picker(len(palette)))]
    digest = hashlib.sha256(f"{spec.seed}:{key}".encode("utf-8")).digest()
    return palette[digest[0] % len(palette)]


def _require_rgb(color: RGB) -> None:
    if (
        not isinstance(color, tuple)
        or len(color) != 3
        or any(not isinstance(channel, int) or isinstance(channel, bool) for channel in color)
        or any(channel < 0 or channel > 255 for channel in color)
    ):
        raise ValidationError(f"RGB color must be three ints 0-255, got {color!r}")
