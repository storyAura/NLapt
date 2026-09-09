"""Tests for nlapt.images.alpha (transparency detect + flatten + spec)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image

from nlapt.core.errors import ImageProcessingError, ValidationError
from nlapt.images.alpha import (
    MODE_FIXED,
    MODE_RANDOM,
    FlattenSpec,
    flatten_alpha,
    has_transparency,
    pick_color,
)
from nlapt.images.pil import PILLOW_MISSING_MESSAGE, load_pil


def _rgba(path: Path, color: tuple[int, int, int, int] = (10, 20, 30, 0)) -> Path:
    Image.new("RGBA", (4, 4), color).save(path, format="PNG")
    return path


def test_load_pil_returns_image_module() -> None:
    module = load_pil()
    assert hasattr(module, "new")
    assert hasattr(module, "open")


def test_has_transparency_true_for_rgba(tmp_path: Path) -> None:
    path = _rgba(tmp_path / "a.png")
    assert has_transparency(path) is True


def test_has_transparency_false_for_rgb(tmp_path: Path) -> None:
    path = tmp_path / "b.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="PNG")
    assert has_transparency(path) is False


def test_flatten_alpha_composites_and_returns_true(tmp_path: Path) -> None:
    path = _rgba(tmp_path / "c.png", (255, 0, 0, 0))
    assert flatten_alpha(path, (0, 255, 0)) is True
    with Image.open(path) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (0, 255, 0)


def test_flatten_alpha_skips_opaque(tmp_path: Path) -> None:
    path = tmp_path / "d.png"
    Image.new("RGB", (2, 2), (9, 9, 9)).save(path, format="PNG")
    before = path.read_bytes()
    assert flatten_alpha(path, (1, 2, 3)) is False
    assert path.read_bytes() == before


def test_flatten_spec_rejects_bad_mode() -> None:
    with pytest.raises(ValidationError):
        FlattenSpec(mode="stripe")


def test_pick_color_fixed() -> None:
    spec = FlattenSpec(mode=MODE_FIXED, color=(1, 2, 3))
    assert pick_color(spec, "a.png") == (1, 2, 3)


def test_pick_color_random_is_stable_per_key() -> None:
    spec = FlattenSpec(
        mode=MODE_RANDOM,
        palette=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        seed=7,
    )
    first = pick_color(spec, "one.png")
    second = pick_color(spec, "one.png")
    other = pick_color(spec, "two.png")
    assert first == second
    assert first in spec.palette
    assert other in spec.palette


def test_pick_color_uses_rng() -> None:
    spec = FlattenSpec(mode=MODE_RANDOM, palette=((9, 9, 9), (8, 8, 8)))

    class _Rng:
        def randrange(self, n: int) -> int:
            return 1

    assert pick_color(spec, "k", rng=_Rng()) == (8, 8, 8)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        has_transparency(tmp_path / "nope.png")


def test_pillow_missing_message() -> None:
    assert "pip install" in PILLOW_MISSING_MESSAGE
    assert isinstance(ImageProcessingError(PILLOW_MISSING_MESSAGE), ImageProcessingError)
