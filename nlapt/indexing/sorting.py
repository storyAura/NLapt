"""Key sorting for the image list panel (spec 4.2).

Provides natural (human) filename ordering plus modified-time and
token-count orderings. All functions are pure and return new tuples.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Fallback values when a key is missing from the supplied mapping; missing
# entries sort as 0 while preserving the input order (stable sort).
DEFAULT_MTIME: float = 0.0
DEFAULT_TOKEN_COUNT: int = 0

_DIGIT_RUN = re.compile(r"(\d+)")


class SortBy(str, Enum):
    """Available sort orders for the image list (spec 4.2)."""

    NAME = "name"
    MTIME = "mtime"
    TOKENS = "tokens"


def _validate_str(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be a string, got {type(value).__name__}")


def natural_key(name: str) -> tuple:
    """Sort key with case-insensitive text and numeric digit runs.

    ``natural_key("img2") < natural_key("img10")``. Splitting with a
    capturing digit-run pattern yields text at even indices and digit runs
    at odd indices, so tuple comparison never mixes ``str`` and ``int``.
    """
    _validate_str(name, "name")
    parts = _DIGIT_RUN.split(name.lower())
    return tuple(int(part) if index % 2 else part for index, part in enumerate(parts))


def sort_keys(
    keys: Sequence[str],
    by: SortBy,
    *,
    mtimes: Mapping[str, float] | None = None,
    token_counts: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    """Return ``keys`` sorted according to ``by``.

    ``SortBy.MTIME`` / ``SortBy.TOKENS`` read from the corresponding
    mapping; keys missing from the mapping (or a ``None`` mapping) are
    treated as 0 and keep their relative input order (stable sort).
    Raises ``ValidationError`` for an unknown sort mode or non-string keys.
    """
    for key in keys:
        _validate_str(key, "key")
    try:
        mode = SortBy(by)
    except ValueError as exc:
        raise ValidationError(f"unknown sort mode: {by!r}") from exc

    if mode is SortBy.NAME:
        return tuple(sorted(keys, key=natural_key))
    if mode is SortBy.MTIME:
        mtime_map: Mapping[str, float] = mtimes if mtimes is not None else {}
        return tuple(sorted(keys, key=lambda k: mtime_map.get(k, DEFAULT_MTIME)))
    token_map: Mapping[str, int] = token_counts if token_counts is not None else {}
    return tuple(sorted(keys, key=lambda k: token_map.get(k, DEFAULT_TOKEN_COUNT)))
