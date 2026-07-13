"""Tests for nlapt.workflow.suggestions: accept / reject / accept-for-edit."""

from __future__ import annotations

import pytest

from nlapt.captions.store import CaptionStore, PendingSuggestion
from nlapt.core.errors import OperationError, ValidationError
from nlapt.core.events import EVT_CAPTION_CHANGED, EVT_PENDING_CLEARED, Event, EventBus
from nlapt.core.states import CaptionState
from nlapt.workflow.suggestions import (
    accept_for_edit,
    accept_suggestion,
    reject_suggestion,
)

KEY = "img/a.jpg"
ORIGINAL = "a girl with red hair"
SUGGESTED = "a woman with red hair"
SOURCE = "rewrite:polish"
CREATED_AT = 123.0


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def store(bus: EventBus) -> CaptionStore:
    store = CaptionStore(bus)
    store.load(KEY, ORIGINAL)
    return store


def _set_pending(store: CaptionStore, text: str = SUGGESTED) -> None:
    store.set_pending(KEY, PendingSuggestion(text=text, source=SOURCE, created_at=CREATED_AT))


# -- accept ------------------------------------------------------------------------


def test_accept_writes_pending_into_body_and_clears_marker(store: CaptionStore) -> None:
    _set_pending(store)
    record = accept_suggestion(store, KEY)
    assert record.text == SUGGESTED
    assert record.pending is None
    assert record.state is CaptionState.DRAFT
    assert record.dirty is True
    assert store.get(KEY).pending is None


def test_accept_applies_state_transition_on_confirmed(store: CaptionStore) -> None:
    store.confirm(KEY)
    _set_pending(store)
    record = accept_suggestion(store, KEY)
    # Default revert_confirmed_on_edit=True: editing a CONFIRMED body -> DRAFT.
    assert record.state is CaptionState.DRAFT
    assert record.text == SUGGESTED


def test_accept_publishes_change_and_cleared_events(
    bus: EventBus, store: CaptionStore
) -> None:
    _set_pending(store)
    events: list[Event] = []
    bus.subscribe(None, events.append)
    accept_suggestion(store, KEY)
    names = [event.name for event in events]
    assert EVT_CAPTION_CHANGED in names
    assert EVT_PENDING_CLEARED in names
    assert names.index(EVT_CAPTION_CHANGED) < names.index(EVT_PENDING_CLEARED)


def test_accept_without_pending_raises(store: CaptionStore) -> None:
    with pytest.raises(OperationError):
        accept_suggestion(store, KEY)


# -- reject ------------------------------------------------------------------------


def test_reject_clears_pending_and_keeps_body(store: CaptionStore) -> None:
    _set_pending(store)
    record = reject_suggestion(store, KEY)
    assert record.text == ORIGINAL
    assert record.pending is None
    assert record.dirty is False  # body untouched, nothing to save
    assert record.state is CaptionState.DRAFT


def test_reject_keeps_confirmed_state(store: CaptionStore) -> None:
    store.confirm(KEY)
    _set_pending(store)
    record = reject_suggestion(store, KEY)
    assert record.state is CaptionState.CONFIRMED
    assert record.text == ORIGINAL


def test_reject_without_pending_raises(store: CaptionStore) -> None:
    with pytest.raises(OperationError):
        reject_suggestion(store, KEY)


# -- accept for edit ----------------------------------------------------------------


def test_accept_for_edit_loads_pending_as_draft(store: CaptionStore) -> None:
    store.confirm(KEY)
    _set_pending(store)
    record = accept_for_edit(store, KEY)
    assert record.text == SUGGESTED
    assert record.pending is None
    assert record.state is CaptionState.DRAFT  # NOT confirmed (spec 10 编辑后接受)
    assert record.dirty is True


def test_accept_for_edit_without_pending_raises(store: CaptionStore) -> None:
    with pytest.raises(OperationError):
        accept_for_edit(store, KEY)


# -- shared validation ---------------------------------------------------------------


@pytest.mark.parametrize("action", [accept_suggestion, reject_suggestion, accept_for_edit])
def test_unknown_key_raises_key_error(store: CaptionStore, action) -> None:
    with pytest.raises(KeyError):
        action(store, "unknown/key.jpg")


@pytest.mark.parametrize("action", [accept_suggestion, reject_suggestion, accept_for_edit])
def test_invalid_store_or_key_raise_validation_error(store: CaptionStore, action) -> None:
    with pytest.raises(ValidationError):
        action("not-a-store", KEY)
    with pytest.raises(ValidationError):
        action(store, "")
