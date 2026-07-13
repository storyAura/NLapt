"""Review-flow navigation helpers (spec 10: 批量审核节奏).

After accepting/rejecting a suggestion (or confirming a caption) the UI jumps
to the next file that still needs attention. Both helpers scan the key list
in display order starting *after* ``current``, wrap around the end, and check
``current`` itself last — so the only remaining match can be the current file.
``current=None`` (or a key not in the list) starts from the beginning.
Returns None when nothing matches.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from nlapt.core.errors import ValidationError
from nlapt.core.states import CaptionState, FileStatus
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)


def next_unconfirmed(
    keys: Sequence[str],
    states: Mapping[str, FileStatus],
    current: str | None,
) -> str | None:
    """Next key (wrap-around) whose state is not CONFIRMED; None if all are."""
    return _next_matching(
        keys, states, current, lambda status: status.state is not CaptionState.CONFIRMED
    )


def next_pending(
    keys: Sequence[str],
    states: Mapping[str, FileStatus],
    current: str | None,
) -> str | None:
    """Next key (wrap-around) with a pending suggestion; None when none is left."""
    return _next_matching(keys, states, current, lambda status: status.has_pending)


def _next_matching(
    keys: Sequence[str],
    states: Mapping[str, FileStatus],
    current: str | None,
    predicate: Callable[[FileStatus], bool],
) -> str | None:
    """Wrap-around scan of ``keys`` after ``current`` for the first match."""
    _validate_args(keys, states, current)
    key_list = list(keys)
    if not key_list:
        return None
    if current is None or current not in key_list:
        # Full pass from the beginning (also covers a stale/unknown current).
        offsets = range(len(key_list))
        start = 0
    else:
        # Positions after current, wrapping; current itself is checked last.
        offsets = range(1, len(key_list) + 1)
        start = key_list.index(current)
    for offset in offsets:
        key = key_list[(start + offset) % len(key_list)]
        status = states.get(key)
        if status is None:
            _LOGGER.debug("no status for key %r; skipped in review navigation", key)
            continue
        if predicate(status):
            return key
    return None


def _validate_args(
    keys: Sequence[str],
    states: Mapping[str, FileStatus],
    current: str | None,
) -> None:
    if isinstance(keys, str) or not isinstance(keys, Sequence):
        raise ValidationError(f"keys must be a sequence of strings, got {type(keys).__name__}")
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ValidationError(f"keys must all be non-empty strings, got {key!r}")
    if not isinstance(states, Mapping):
        raise ValidationError(f"states must be a mapping, got {type(states).__name__}")
    if current is not None and not isinstance(current, str):
        raise ValidationError(f"current must be a string or None, got {type(current).__name__}")
