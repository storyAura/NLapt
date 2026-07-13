"""Tests for nlapt.core.states (spec 4.1 state machine)."""

from __future__ import annotations

import dataclasses

import pytest

from nlapt.core.errors import ValidationError
from nlapt.core.states import CaptionState, FileStatus, initial_state, state_after_edit


def test_caption_state_values() -> None:
    assert CaptionState.UNLABELED.value == "unlabeled"
    assert CaptionState.DRAFT.value == "draft"
    assert CaptionState.CONFIRMED.value == "confirmed"
    assert isinstance(CaptionState.DRAFT, str)


@pytest.mark.parametrize(
    ("current", "text", "revert", "expected"),
    [
        # empty/whitespace always -> UNLABELED
        (CaptionState.UNLABELED, "", True, CaptionState.UNLABELED),
        (CaptionState.DRAFT, "   ", True, CaptionState.UNLABELED),
        (CaptionState.CONFIRMED, "\n\t ", True, CaptionState.UNLABELED),
        (CaptionState.CONFIRMED, "", False, CaptionState.UNLABELED),
        # non-empty from UNLABELED/DRAFT -> DRAFT regardless of flag
        (CaptionState.UNLABELED, "a girl", True, CaptionState.DRAFT),
        (CaptionState.UNLABELED, "a girl", False, CaptionState.DRAFT),
        (CaptionState.DRAFT, "a girl", True, CaptionState.DRAFT),
        (CaptionState.DRAFT, "a girl", False, CaptionState.DRAFT),
        # CONFIRMED + edit: reverts to DRAFT unless the rule is disabled
        (CaptionState.CONFIRMED, "edited", True, CaptionState.DRAFT),
        (CaptionState.CONFIRMED, "edited", False, CaptionState.CONFIRMED),
    ],
)
def test_state_after_edit_table(
    current: CaptionState, text: str, revert: bool, expected: CaptionState
) -> None:
    assert state_after_edit(current, text, revert_confirmed=revert) is expected


def test_state_after_edit_rejects_bad_inputs() -> None:
    with pytest.raises(ValidationError):
        state_after_edit("draft", "text")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        state_after_edit(CaptionState.DRAFT, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", CaptionState.UNLABELED),
        ("   \n ", CaptionState.UNLABELED),
        ("1girl, smile", CaptionState.DRAFT),
    ],
)
def test_initial_state(text: str, expected: CaptionState) -> None:
    assert initial_state(text) is expected


def test_initial_state_rejects_non_string() -> None:
    with pytest.raises(ValidationError):
        initial_state(None)  # type: ignore[arg-type]


def test_file_status_frozen_with_pending_overlay() -> None:
    status = FileStatus(state=CaptionState.DRAFT)
    assert status.has_pending is False
    overlaid = FileStatus(state=CaptionState.CONFIRMED, has_pending=True)
    assert overlaid.has_pending is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        status.state = CaptionState.CONFIRMED  # type: ignore[misc]
