"""Tests for nlapt.llm.vision.prepare_image (Pillow-backed)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from nlapt.core.errors import ValidationError
from nlapt.llm.vision import prepare_image

RED = (255, 0, 0)


def write_png(path: Path, size: tuple[int, int], mode: str = "RGB") -> Path:
    color: tuple[int, ...] = RED if mode == "RGB" else (255, 0, 0, 128)
    image = Image.new(mode, size, color)
    image.save(path, format="PNG")
    return path


def decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


class TestPrepareImage:
    def test_downscales_to_max_edge_keeping_aspect(self, tmp_path: Path) -> None:
        source = write_png(tmp_path / "wide.png", (200, 100))
        data = prepare_image(source, max_edge=100)
        result = decode(data)
        assert result.format == "JPEG"
        assert max(result.size) <= 100
        assert result.size == (100, 50)

    def test_tall_image_downscaled(self, tmp_path: Path) -> None:
        source = write_png(tmp_path / "tall.png", (50, 400))
        data = prepare_image(source, max_edge=100)
        result = decode(data)
        assert result.size == (13, 100) or result.size == (12, 100)
        assert max(result.size) <= 100

    def test_small_image_not_upscaled(self, tmp_path: Path) -> None:
        source = write_png(tmp_path / "small.png", (50, 40))
        data = prepare_image(source, max_edge=1024)
        result = decode(data)
        assert result.size == (50, 40)
        assert result.format == "JPEG"

    def test_rgba_converted_to_rgb_jpeg(self, tmp_path: Path) -> None:
        source = write_png(tmp_path / "alpha.png", (30, 30), mode="RGBA")
        data = prepare_image(source, max_edge=1024)
        result = decode(data)
        assert result.format == "JPEG"
        assert result.mode == "RGB"

    def test_output_is_jpeg_bytes(self, tmp_path: Path) -> None:
        source = write_png(tmp_path / "img.png", (10, 10))
        data = prepare_image(source)
        assert isinstance(data, bytes)
        assert data.startswith(b"\xff\xd8")  # JPEG SOI marker

    def test_quality_affects_size(self, tmp_path: Path) -> None:
        # A noisy gradient image compresses differently at different qualities.
        source = tmp_path / "gradient.png"
        image = Image.new("RGB", (128, 128))
        image.putdata(
            [((x * 2) % 256, (y * 2) % 256, (x * y) % 256) for y in range(128) for x in range(128)]
        )
        image.save(source, format="PNG")
        high = prepare_image(source, quality=95)
        low = prepare_image(source, quality=10)
        assert len(low) < len(high)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            prepare_image(tmp_path / "missing.png")

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            prepare_image(tmp_path)

    def test_unreadable_image_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.png"
        bogus.write_bytes(b"this is not an image at all")
        with pytest.raises(ValidationError):
            prepare_image(bogus)

    def test_invalid_max_edge_raises(self, tmp_path: Path) -> None:
        source = write_png(tmp_path / "img.png", (10, 10))
        with pytest.raises(ValidationError):
            prepare_image(source, max_edge=0)

    @pytest.mark.parametrize("quality", [0, 101, -5])
    def test_invalid_quality_raises(self, tmp_path: Path, quality: int) -> None:
        source = write_png(tmp_path / "img.png", (10, 10))
        with pytest.raises(ValidationError):
            prepare_image(source, quality=quality)
