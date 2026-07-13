"""Chip (tag capsule) text model for the chips editing mode (spec 6.1).

A caption is split into chips on half-width (``,``) and full-width (``，``)
commas. Chips are plain strings; all functions are pure and return new
tuples. Rejoining always uses ``", "`` and drops empty fragments.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

CHIP_SEPARATORS: tuple[str, ...] = (",", "，")
SENTENCE_ENDINGS: tuple[str, ...] = (".", "!", "?", "。", "！", "？")

# Canonical joiner used when writing chips back to caption text (spec 6.1).
CHIP_JOINER = ", "

_SPLIT_PATTERN = re.compile("|".join(re.escape(sep) for sep in CHIP_SEPARATORS))
# Minimum number of occurrences for a chip text to be reported as duplicated.
_DUPLICATE_THRESHOLD = 2


def _validate_str(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be a string, got {type(value).__name__}")


def _validate_index(index: int, length: int, label: str) -> None:
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValidationError(f"{label} must be an integer, got {index!r}")
    if not 0 <= index < length:
        raise ValidationError(f"{label} {index} out of range for {length} chips")


def split_chips(text: str) -> tuple[str, ...]:
    """Split caption text into chips on half- and full-width commas.

    Each fragment is stripped; empty fragments are dropped.
    """
    _validate_str(text, "text")
    return tuple(part for part in (p.strip() for p in _SPLIT_PATTERN.split(text)) if part)


def join_chips(chips: Sequence[str]) -> str:
    """Rejoin chips into caption text with ``", "``, dropping empties."""
    cleaned = []
    for chip in chips:
        _validate_str(chip, "chip")
        stripped = chip.strip()
        if stripped:
            cleaned.append(stripped)
    return CHIP_JOINER.join(cleaned)


def is_long_sentence(chip: str) -> bool:
    """True when the chip contains sentence-ending punctuation (spec 6.1)."""
    _validate_str(chip, "chip")
    return any(ending in chip for ending in SENTENCE_ENDINGS)


def find_duplicates(chips: Sequence[str]) -> Mapping[str, tuple[int, ...]]:
    """Map exact chip text to its indices, for texts occurring >= 2 times."""
    positions: dict[str, list[int]] = {}
    for index, chip in enumerate(chips):
        _validate_str(chip, "chip")
        positions.setdefault(chip, []).append(index)
    return {
        text: tuple(indices)
        for text, indices in positions.items()
        if len(indices) >= _DUPLICATE_THRESHOLD
    }


def dedup_chips(chips: Sequence[str]) -> tuple[str, ...]:
    """Remove exact duplicates, keeping the first occurrence of each chip."""
    seen: set[str] = set()
    result: list[str] = []
    for chip in chips:
        _validate_str(chip, "chip")
        if chip not in seen:
            seen.add(chip)
            result.append(chip)
    return tuple(result)


def edit_chip(chips: Sequence[str], index: int, new_text: str) -> tuple[str, ...]:
    """Replace the chip at ``index`` with ``new_text``.

    Text containing commas is split into multiple chips inserted in place
    (spec 6.1 inline-edit rule); empty/whitespace text removes the chip.
    Raises ``ValidationError`` when ``index`` is out of range.
    """
    _validate_index(index, len(chips), "index")
    _validate_str(new_text, "new_text")
    replacement = split_chips(new_text)
    return tuple(chips[:index]) + replacement + tuple(chips[index + 1 :])


def move_chip(chips: Sequence[str], src: int, dst: int) -> tuple[str, ...]:
    """Move the chip at ``src`` so it ends up at position ``dst``.

    Both indices must be valid positions in ``chips``; raises
    ``ValidationError`` otherwise.
    """
    _validate_index(src, len(chips), "src")
    _validate_index(dst, len(chips), "dst")
    items = list(chips)
    item = items.pop(src)
    items.insert(dst, item)
    return tuple(items)


def count_tag_in_texts(tag: str, texts: Mapping[str, str]) -> int:
    """Number of files whose chip list contains ``tag`` as a whole chip.

    Matching is exact against whole chips (never substrings), per the
    spec 6.1 right-click tag statistic. A file counts at most once.
    Raises ``ValidationError`` for an empty/whitespace tag.
    """
    _validate_str(tag, "tag")
    needle = tag.strip()
    if not needle:
        raise ValidationError("tag must not be empty or whitespace")
    count = 0
    for text in texts.values():
        _validate_str(text, "caption text")
        if needle in split_chips(text):
            count += 1
    return count
