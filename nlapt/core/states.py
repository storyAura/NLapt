"""Caption state machine (spec 4.1).

States: UNLABELED (no txt or empty content), DRAFT (edited, not human-confirmed),
CONFIRMED (human reviewed). A pending AI suggestion is an overlay flag carried by
:class:`FileStatus`, not a state of its own — it coexists with the body state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nlapt.core.errors import ValidationError


class CaptionState(str, Enum):
    """Body text state of a caption file."""

    UNLABELED = "unlabeled"
    DRAFT = "draft"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class FileStatus:
    """Combined display status: body state plus the pending-suggestion overlay."""

    state: CaptionState
    has_pending: bool = False


def state_after_edit(
    current: CaptionState, new_text: str, *, revert_confirmed: bool = True
) -> CaptionState:
    """Return the state resulting from a manual edit.

    Empty/whitespace text -> UNLABELED. Non-empty text: CONFIRMED stays
    CONFIRMED when ``revert_confirmed`` is False (spec 4.1 supplementary rule
    disabled), otherwise reverts to DRAFT. UNLABELED/DRAFT + text -> DRAFT.
    """
    if not isinstance(current, CaptionState):
        raise ValidationError(f"current must be a CaptionState, got {current!r}")
    if not isinstance(new_text, str):
        raise ValidationError(f"new_text must be a string, got {type(new_text).__name__}")
    if not new_text.strip():
        return CaptionState.UNLABELED
    if current is CaptionState.CONFIRMED and not revert_confirmed:
        return CaptionState.CONFIRMED
    return CaptionState.DRAFT


def initial_state(text: str) -> CaptionState:
    """State assigned at scan/load time: UNLABELED if empty/whitespace else DRAFT."""
    if not isinstance(text, str):
        raise ValidationError(f"text must be a string, got {type(text).__name__}")
    return CaptionState.UNLABELED if not text.strip() else CaptionState.DRAFT
