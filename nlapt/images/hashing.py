"""Perceptual dHash plus exact SHA-256 grouping for near-duplicate images."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from nlapt.core.errors import ImageProcessingError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.images.pil import load_pil

_LOGGER = get_logger(__name__)

DEFAULT_DHASH_SIZE = 8
READ_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ImageFingerprint:
    """One image's identity for duplicate clustering."""

    key: str
    path: Path
    dhash: int
    sha256: str
    width: int
    height: int
    size: int


@dataclass(frozen=True)
class DuplicateGroup:
    """A cluster of exact (same SHA-256) or visually similar images."""

    members: tuple[ImageFingerprint, ...]
    exact: bool


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of the file bytes."""
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"image file not found: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            while True:
                chunk = handle.read(READ_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ImageProcessingError(f"could not hash {source}: {exc}") from exc
    return digest.hexdigest()


def dhash(path: Path, size: int = DEFAULT_DHASH_SIZE) -> int:
    """Difference hash: grayscale ``(size+1) x size``, adjacent-pixel bits."""
    if size < 2:
        raise ValidationError(f"dhash size must be >= 2, got {size}")
    image_module = load_pil()
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"image file not found: {source}")
    try:
        with image_module.open(source) as image:
            resample = getattr(image_module, "Resampling", image_module).LANCZOS
            gray = image.convert("L").resize((size + 1, size), resample)
            pixels = list(gray.getdata())
    except (OSError, ValueError) as exc:
        raise ImageProcessingError(f"could not dhash {source}: {exc}") from exc
    bits = 0
    row_width = size + 1
    for row in range(size):
        base = row * row_width
        for col in range(size):
            bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
    return bits


def hamming(left: int, right: int) -> int:
    """Hamming distance between two 64-bit (or smaller) hashes."""
    if not isinstance(left, int) or not isinstance(right, int):
        raise ValidationError("hamming expects two ints")
    if left < 0 or right < 0:
        raise ValidationError("hamming hashes must be non-negative")
    return (left ^ right).bit_count()


def fingerprint(key: str, path: Path, *, size: int = DEFAULT_DHASH_SIZE) -> ImageFingerprint:
    """Build a fingerprint (dHash + SHA-256 + pixel size) for one file."""
    image_module = load_pil()
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"image file not found: {source}")
    try:
        with image_module.open(source) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise ImageProcessingError(f"could not fingerprint {source}: {exc}") from exc
    return ImageFingerprint(
        key=key,
        path=source,
        dhash=dhash(source, size=size),
        sha256=sha256_file(source),
        width=int(width),
        height=int(height),
        size=source.stat().st_size,
    )


def group_similar(
    fingerprints: tuple[ImageFingerprint, ...] | list[ImageFingerprint],
    max_distance: int,
) -> tuple[DuplicateGroup, ...]:
    """Union-find clusters: same SHA-256 first, then dHash Hamming distance."""
    if max_distance < 0:
        raise ValidationError(f"max_distance must be >= 0, got {max_distance}")
    items = tuple(fingerprints)
    if len(items) < 2:
        return ()
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    by_sha: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        by_sha.setdefault(item.sha256, []).append(index)
    for indexes in by_sha.values():
        for extra in indexes[1:]:
            union(indexes[0], extra)

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if hamming(items[i].dhash, items[j].dhash) <= max_distance:
                union(i, j)

    clusters: dict[int, list[ImageFingerprint]] = {}
    for index, item in enumerate(items):
        clusters.setdefault(find(index), []).append(item)

    groups: list[DuplicateGroup] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        shas = {member.sha256 for member in members}
        groups.append(DuplicateGroup(members=tuple(members), exact=len(shas) == 1))
    groups.sort(key=lambda group: (-len(group.members), group.members[0].key))
    _LOGGER.debug("grouped %d fingerprints into %d clusters", len(items), len(groups))
    return tuple(groups)


def suggest_keep(group: DuplicateGroup) -> ImageFingerprint:
    """Prefer the highest-resolution member, then the largest file, then key."""
    if not group.members:
        raise ValidationError("cannot suggest a keep member for an empty group")
    return max(
        group.members,
        key=lambda member: (member.width * member.height, member.size, member.key),
    )
