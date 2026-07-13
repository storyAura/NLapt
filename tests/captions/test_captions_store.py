"""Tests for nlapt.captions.store (CaptionStore, CaptionRecord, PendingSuggestion)."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor

import pytest

from nlapt.captions.store import CaptionRecord, CaptionStore, PendingSuggestion
from nlapt.core.errors import ValidationError
from nlapt.core.events import (
    EVT_CAPTION_CHANGED,
    EVT_PENDING_CLEARED,
    EVT_PENDING_SET,
    EVT_STATE_CHANGED,
    Event,
    EventBus,
)
from nlapt.core.states import CaptionState

THREAD_COUNT = 8
EDITS_PER_THREAD = 25


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def events(bus: EventBus) -> list[Event]:
    collected: list[Event] = []
    bus.subscribe(None, collected.append)
    return collected


@pytest.fixture()
def store(bus: EventBus) -> CaptionStore:
    return CaptionStore(bus)


def make_pending(text: str = "a red cat", source: str = "rewrite:polish") -> PendingSuggestion:
    return PendingSuggestion(text=text, source=source, created_at=1234.5)


class TestDataTypes:
    def test_records_are_frozen(self) -> None:
        record = CaptionRecord(key="a.png", text="cat", state=CaptionState.DRAFT)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.text = "dog"  # type: ignore[misc]

    def test_pending_suggestion_is_frozen(self) -> None:
        pending = make_pending()
        with pytest.raises(dataclasses.FrozenInstanceError):
            pending.text = "other"  # type: ignore[misc]


class TestLoadAndLookup:
    def test_load_nonempty_text_is_draft_not_dirty(self, store: CaptionStore) -> None:
        record = store.load("a.png", "a cat")
        assert record.state is CaptionState.DRAFT
        assert record.dirty is False
        assert record.pending is None

    def test_load_empty_text_is_unlabeled(self, store: CaptionStore) -> None:
        assert store.load("a.png", "").state is CaptionState.UNLABELED
        assert store.load("b.png", "   \n").state is CaptionState.UNLABELED

    def test_load_publishes_no_events(self, store: CaptionStore, events: list[Event]) -> None:
        store.load("a.png", "a cat")
        assert events == []

    def test_get_unknown_key_raises_key_error(self, store: CaptionStore) -> None:
        with pytest.raises(KeyError):
            store.get("missing.png")

    def test_keys_preserve_insertion_order(self, store: CaptionStore) -> None:
        for key in ("b.png", "a.png", "c.png"):
            store.load(key, "x")
        assert store.keys() == ("b.png", "a.png", "c.png")

    def test_load_validates_key_and_text(self, store: CaptionStore) -> None:
        with pytest.raises(ValidationError):
            store.load("", "text")
        with pytest.raises(ValidationError):
            store.load("a.png", None)  # type: ignore[arg-type]


class TestSetText:
    def test_edit_sets_dirty_and_draft(self, store: CaptionStore) -> None:
        store.load("a.png", "")
        record = store.set_text("a.png", "a cat")
        assert record.state is CaptionState.DRAFT
        assert record.dirty is True

    def test_clearing_text_reverts_to_unlabeled(self, store: CaptionStore) -> None:
        store.load("a.png", "a cat")
        assert store.set_text("a.png", "   ").state is CaptionState.UNLABELED

    def test_edit_confirmed_reverts_to_draft_by_default(self, store: CaptionStore) -> None:
        store.load("a.png", "a cat")
        store.confirm("a.png")
        assert store.set_text("a.png", "a dog").state is CaptionState.DRAFT

    def test_edit_confirmed_keeps_confirmed_when_revert_disabled(self, bus: EventBus) -> None:
        keeping = CaptionStore(bus, revert_confirmed_on_edit=False)
        keeping.load("a.png", "a cat")
        keeping.confirm("a.png")
        record = keeping.set_text("a.png", "a dog")
        assert record.state is CaptionState.CONFIRMED
        assert record.dirty is True

    def test_publishes_caption_changed_with_key(
        self, store: CaptionStore, events: list[Event]
    ) -> None:
        store.load("a.png", "")
        store.set_text("a.png", "a cat")
        assert [e.name for e in events] == [EVT_CAPTION_CHANGED]
        payload = events[0].payload
        assert payload["key"] == "a.png"
        assert payload["text"] == "a cat"
        assert payload["state"] == CaptionState.DRAFT.value

    def test_unknown_key_raises_key_error(self, store: CaptionStore) -> None:
        with pytest.raises(KeyError):
            store.set_text("missing.png", "x")

    def test_non_string_text_raises(self, store: CaptionStore) -> None:
        store.load("a.png", "x")
        with pytest.raises(ValidationError):
            store.set_text("a.png", 42)  # type: ignore[arg-type]


class TestConfirm:
    def test_confirm_sets_confirmed_and_publishes(
        self, store: CaptionStore, events: list[Event]
    ) -> None:
        store.load("a.png", "a cat")
        record = store.confirm("a.png")
        assert record.state is CaptionState.CONFIRMED
        assert [e.name for e in events] == [EVT_STATE_CHANGED]
        assert events[0].payload["key"] == "a.png"
        assert events[0].payload["state"] == CaptionState.CONFIRMED.value

    def test_confirm_empty_text_raises_validation_error(
        self, store: CaptionStore, events: list[Event]
    ) -> None:
        store.load("a.png", "   ")
        with pytest.raises(ValidationError):
            store.confirm("a.png")
        assert store.get("a.png").state is CaptionState.UNLABELED
        assert events == []

    def test_confirm_does_not_touch_dirty_flag(self, store: CaptionStore) -> None:
        store.load("a.png", "old")
        store.set_text("a.png", "new")
        assert store.confirm("a.png").dirty is True


class TestPending:
    def test_set_pending_keeps_body_text_and_state(self, store: CaptionStore) -> None:
        store.load("a.png", "a cat")
        record = store.set_pending("a.png", make_pending("a fluffy cat"))
        assert record.text == "a cat"
        assert record.state is CaptionState.DRAFT
        assert record.pending is not None
        assert record.pending.text == "a fluffy cat"

    def test_set_pending_publishes_event_with_key(
        self, store: CaptionStore, events: list[Event]
    ) -> None:
        store.load("a.png", "a cat")
        store.set_pending("a.png", make_pending(source="vision:initial"))
        assert [e.name for e in events] == [EVT_PENDING_SET]
        assert events[0].payload["key"] == "a.png"
        assert events[0].payload["source"] == "vision:initial"

    def test_clear_pending_publishes_event_with_key(
        self, store: CaptionStore, events: list[Event]
    ) -> None:
        store.load("a.png", "a cat")
        store.set_pending("a.png", make_pending())
        record = store.clear_pending("a.png")
        assert record.pending is None
        assert [e.name for e in events] == [EVT_PENDING_SET, EVT_PENDING_CLEARED]
        assert events[-1].payload["key"] == "a.png"

    def test_clear_pending_without_pending_is_idempotent_and_silent(
        self, store: CaptionStore, events: list[Event]
    ) -> None:
        store.load("a.png", "a cat")
        record = store.clear_pending("a.png")
        assert record.pending is None
        assert events == []

    def test_set_pending_rejects_non_suggestion(self, store: CaptionStore) -> None:
        store.load("a.png", "a cat")
        with pytest.raises(ValidationError):
            store.set_pending("a.png", "just text")  # type: ignore[arg-type]

    def test_pending_keys(self, store: CaptionStore) -> None:
        store.load("a.png", "x")
        store.load("b.png", "y")
        store.set_pending("b.png", make_pending())
        assert store.pending_keys() == ("b.png",)


class TestSavedAndDirty:
    def test_mark_saved_clears_dirty(self, store: CaptionStore) -> None:
        store.load("a.png", "x")
        store.set_text("a.png", "y")
        assert store.dirty_keys() == ("a.png",)
        assert store.mark_saved("a.png").dirty is False
        assert store.dirty_keys() == ()


class TestStatusCountsTexts:
    def test_status_combines_state_and_pending_overlay(self, store: CaptionStore) -> None:
        store.load("a.png", "a cat")
        store.confirm("a.png")
        store.set_pending("a.png", make_pending())
        status = store.status("a.png")
        assert status.state is CaptionState.CONFIRMED
        assert status.has_pending is True

    def test_counts_include_orthogonal_pending(self, store: CaptionStore) -> None:
        store.load("u.png", "")
        store.load("d.png", "draft text")
        store.load("c.png", "confirmed text")
        store.confirm("c.png")
        store.set_pending("c.png", make_pending())
        store.set_pending("d.png", make_pending())
        counts = store.counts()
        assert counts == {"unlabeled": 1, "draft": 1, "confirmed": 1, "pending": 2}

    def test_texts_returns_snapshot_copy(self, store: CaptionStore) -> None:
        store.load("a.png", "cat")
        snapshot = store.texts()
        assert snapshot == {"a.png": "cat"}
        snapshot["a.png"] = "mutated"
        snapshot["b.png"] = "injected"
        assert store.texts() == {"a.png": "cat"}
        assert store.get("a.png").text == "cat"


class TestConstructionAndThreadSafety:
    def test_requires_event_bus(self) -> None:
        with pytest.raises(ValidationError):
            CaptionStore("not a bus")  # type: ignore[arg-type]

    def test_concurrent_edits_keep_store_consistent(self, store: CaptionStore) -> None:
        keys = [f"img{i}.png" for i in range(THREAD_COUNT)]
        for key in keys:
            store.load(key, "")

        def hammer(key: str) -> None:
            for i in range(EDITS_PER_THREAD):
                store.set_text(key, f"text {i}")
                store.mark_saved(key)

        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as pool:
            list(pool.map(hammer, keys))

        final = f"text {EDITS_PER_THREAD - 1}"
        assert all(store.get(key).text == final for key in keys)
        assert store.counts()["draft"] == THREAD_COUNT
        assert store.dirty_keys() == ()
