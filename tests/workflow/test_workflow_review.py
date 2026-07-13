"""Tests for nlapt.workflow.review: wrap-around review navigation."""

from __future__ import annotations

import pytest

from nlapt.core.errors import ValidationError
from nlapt.core.states import CaptionState, FileStatus
from nlapt.workflow.review import next_pending, next_unconfirmed

CONFIRMED = FileStatus(state=CaptionState.CONFIRMED)
DRAFT = FileStatus(state=CaptionState.DRAFT)
UNLABELED = FileStatus(state=CaptionState.UNLABELED)
CONFIRMED_PENDING = FileStatus(state=CaptionState.CONFIRMED, has_pending=True)
DRAFT_PENDING = FileStatus(state=CaptionState.DRAFT, has_pending=True)

KEYS = ("a", "b", "c", "d")


# -- next_unconfirmed --------------------------------------------------------------


def test_finds_next_after_current() -> None:
    states = {"a": CONFIRMED, "b": DRAFT, "c": UNLABELED, "d": CONFIRMED}
    assert next_unconfirmed(KEYS, states, "b") == "c"


def test_wraps_around_the_end() -> None:
    states = {"a": DRAFT, "b": CONFIRMED, "c": CONFIRMED, "d": CONFIRMED}
    assert next_unconfirmed(KEYS, states, "c") == "a"


def test_current_none_starts_from_beginning() -> None:
    states = {"a": CONFIRMED, "b": UNLABELED, "c": DRAFT, "d": CONFIRMED}
    assert next_unconfirmed(KEYS, states, None) == "b"


def test_none_when_all_confirmed() -> None:
    states = {key: CONFIRMED for key in KEYS}
    assert next_unconfirmed(KEYS, states, "a") is None
    assert next_unconfirmed(KEYS, states, None) is None


def test_current_itself_is_last_resort() -> None:
    states = {"a": CONFIRMED, "b": DRAFT, "c": CONFIRMED, "d": CONFIRMED}
    # Only the current file remains unconfirmed: the wrap-around finds it.
    assert next_unconfirmed(KEYS, states, "b") == "b"


def test_unlabeled_counts_as_unconfirmed() -> None:
    states = {"a": CONFIRMED, "b": CONFIRMED, "c": UNLABELED, "d": CONFIRMED}
    assert next_unconfirmed(KEYS, states, "a") == "c"


def test_unknown_current_starts_from_beginning() -> None:
    states = {"a": DRAFT, "b": CONFIRMED, "c": CONFIRMED, "d": CONFIRMED}
    assert next_unconfirmed(KEYS, states, "not-in-list") == "a"


def test_empty_keys_returns_none() -> None:
    assert next_unconfirmed((), {}, None) is None
    assert next_pending((), {}, None) is None


def test_keys_without_status_are_skipped() -> None:
    states = {"a": CONFIRMED, "c": DRAFT}  # b and d have no status entry
    assert next_unconfirmed(KEYS, states, "a") == "c"


# -- next_pending -------------------------------------------------------------------


def test_next_pending_finds_pending_after_current() -> None:
    states = {"a": DRAFT, "b": DRAFT_PENDING, "c": DRAFT, "d": CONFIRMED_PENDING}
    assert next_pending(KEYS, states, "b") == "d"


def test_next_pending_wraps_around() -> None:
    states = {"a": DRAFT_PENDING, "b": DRAFT, "c": DRAFT, "d": CONFIRMED}
    assert next_pending(KEYS, states, "c") == "a"


def test_next_pending_ignores_state_only_uses_overlay() -> None:
    # CONFIRMED with pending still needs review; DRAFT without pending does not.
    states = {"a": DRAFT, "b": CONFIRMED_PENDING, "c": DRAFT, "d": DRAFT}
    assert next_pending(KEYS, states, None) == "b"


def test_next_pending_none_when_no_pending_left() -> None:
    states = {key: DRAFT for key in KEYS}
    assert next_pending(KEYS, states, "a") is None


def test_next_pending_current_none_starts_from_beginning() -> None:
    states = {"a": DRAFT_PENDING, "b": DRAFT_PENDING, "c": DRAFT, "d": DRAFT}
    assert next_pending(KEYS, states, None) == "a"


# -- validation ---------------------------------------------------------------------


@pytest.mark.parametrize("helper", [next_unconfirmed, next_pending])
def test_validates_arguments(helper) -> None:
    states = {"a": DRAFT}
    with pytest.raises(ValidationError):
        helper("abc", states, None)  # bare string is not a key sequence
    with pytest.raises(ValidationError):
        helper(("a", ""), states, None)
    with pytest.raises(ValidationError):
        helper((1,), states, None)
    with pytest.raises(ValidationError):
        helper(("a",), "not-a-mapping", None)
    with pytest.raises(ValidationError):
        helper(("a",), states, 42)
