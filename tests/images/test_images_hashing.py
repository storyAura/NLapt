"""Tests for nlapt.images.hashing (dHash, SHA-256, grouping)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image

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


def _solid(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (16, 16)) -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _pattern(path: Path, kind: str, size: tuple[int, int] = (16, 16)) -> Path:
    """Images with spatial variation so dHash is not the all-zero solid-color hash."""
    image = Image.new("RGB", size)
    pixels = image.load()
    width, height = size
    for y in range(height):
        for x in range(width):
            if kind == "gradient":
                pixels[x, y] = (x * 16, y * 16, 80)
            elif kind == "stripe":
                pixels[x, y] = (255, 255, 255) if x < width // 2 else (0, 0, 0)
            else:
                pixels[x, y] = (255, 0, 0) if (x + y) % 2 == 0 else (0, 0, 255)
    image.save(path, format="PNG")
    return path


def test_identical_files_have_zero_hamming(tmp_path: Path) -> None:
    a = _pattern(tmp_path / "a.png", "gradient")
    b = _pattern(tmp_path / "b.png", "gradient")
    assert dhash(a) == dhash(b)
    assert hamming(dhash(a), dhash(b)) == 0
    assert sha256_file(a) == sha256_file(b)


def test_different_images_have_nonzero_distance(tmp_path: Path) -> None:
    a = _pattern(tmp_path / "a.png", "stripe")
    b = _pattern(tmp_path / "b.png", "checker")
    assert hamming(dhash(a), dhash(b)) > 0


def test_group_similar_clusters_exact_and_leaves_uniques(tmp_path: Path) -> None:
    same_a = _pattern(tmp_path / "same_a.png", "gradient")
    same_b = _pattern(tmp_path / "same_b.png", "gradient")
    unique = _pattern(tmp_path / "unique.png", "stripe")
    fps = (
        fingerprint("same_a.png", same_a),
        fingerprint("same_b.png", same_b),
        fingerprint("unique.png", unique),
    )
    groups = group_similar(fps, max_distance=0)
    assert len(groups) == 1
    assert groups[0].exact is True
    keys = {member.key for member in groups[0].members}
    assert keys == {"same_a.png", "same_b.png"}


def test_suggest_keep_prefers_resolution(tmp_path: Path) -> None:
    small = _solid(tmp_path / "s.png", (1, 1, 1), (8, 8))
    large = _solid(tmp_path / "l.png", (1, 1, 1), (32, 32))
    group = DuplicateGroup(
        members=(fingerprint("s.png", small), fingerprint("l.png", large)),
        exact=False,
    )
    assert suggest_keep(group).key == "l.png"


def test_fingerprint_fields(tmp_path: Path) -> None:
    path = _solid(tmp_path / "f.png", (3, 4, 5), (6, 7))
    item = fingerprint("f.png", path)
    assert isinstance(item, ImageFingerprint)
    assert item.width == 6
    assert item.height == 7
    assert item.size == path.stat().st_size
    assert len(item.sha256) == 64
