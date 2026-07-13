"""Tests for nlapt_gui.history_model (cursor-based per-file caption history)."""

from __future__ import annotations

import pytest

from nlapt.core.errors import ValidationError

from nlapt_gui.history_model import (
    HISTORY_KEEP_NEWER,
    HISTORY_KEEP_OLDER,
    HISTORY_LIMIT,
    FileHistory,
    HistoryEntry,
)


@pytest.fixture()
def history(qapp) -> FileHistory:
    ticks = iter(f"10:{i // 60:02d}:{i % 60:02d}" for i in range(600))
    return FileHistory(now=lambda: next(ticks))


class TestSeed:
    def test_seed_creates_initial_entry(self, history: FileHistory) -> None:
        history.seed("a.png", "1girl, solo")
        entries = history.entries("a.png")
        assert len(entries) == 1
        assert entries[0].label == "载入原始标注"
        assert entries[0].caption == "1girl, solo"
        assert entries[0].time_label == "10:00:00"
        assert history.current_index("a.png") == 0

    def test_seed_resets_previous_history(self, history: FileHistory) -> None:
        history.seed("a.png", "old")
        history.push("a.png", "编辑", "v2")
        history.seed("a.png", "fresh")
        entries = history.entries("a.png")
        assert len(entries) == 1
        assert entries[0].caption == "fresh"
        assert history.current_index("a.png") == 0

    def test_seed_emits_changed(self, history: FileHistory, qtbot) -> None:
        with qtbot.waitSignal(history.changed, timeout=1000) as blocker:
            history.seed("a.png", "x")
        assert blocker.args == ["a.png"]


class TestPush:
    def test_push_prepends_newest_first(self, history: FileHistory) -> None:
        history.seed("a.png", "v1")
        history.push("a.png", "编辑分段「long hair」", "v2")
        entries = history.entries("a.png")
        assert [e.caption for e in entries] == ["v2", "v1"]
        assert entries[0].label == "编辑分段「long hair」"
        assert history.current_index("a.png") == 0

    def test_push_noop_when_caption_unchanged(self, history: FileHistory, qtbot) -> None:
        history.seed("a.png", "same")
        with qtbot.assertNotEmitted(history.changed, wait=50):
            history.push("a.png", "编辑", "same")
        assert len(history.entries("a.png")) == 1

    def test_push_trims_older_side(self, history: FileHistory) -> None:
        history.seed("a.png", "v0")
        for i in range(1, 30):
            history.push("a.png", "编辑", f"v{i}")
        entries = history.entries("a.png")
        # Cursor at the newest: at most 1 + KEEP_OLDER entries survive.
        assert len(entries) == 1 + HISTORY_KEEP_OLDER
        assert entries[0].caption == "v29"
        assert HISTORY_LIMIT == HISTORY_KEEP_NEWER + HISTORY_KEEP_OLDER + 1

    def test_push_without_seed_starts_history(self, history: FileHistory) -> None:
        history.push("b.png", "编辑", "first")
        assert history.entries("b.png") == (
            HistoryEntry("10:00:00", "编辑", "first"),
        )

    def test_push_in_the_past_cuts_the_newer_branch(self, history: FileHistory) -> None:
        history.seed("a.png", "v1")
        history.push("a.png", "编辑", "v2")
        history.push("a.png", "编辑", "v3")
        history.revert_caption("a.png", 2)  # cursor -> v1
        history.push("a.png", "编辑", "v4")  # new edit from the past
        entries = history.entries("a.png")
        assert [e.caption for e in entries] == ["v4", "v1"]
        assert history.current_index("a.png") == 0


class TestRevert:
    def test_revert_keeps_newer_entries(self, history: FileHistory) -> None:
        """回退 must NOT delete newer snapshots (jump both ways)."""
        history.seed("a.png", "v1")
        history.push("a.png", "编辑", "v2")
        history.push("a.png", "编辑", "v3")
        result = history.revert_caption("a.png", 2)
        assert result == "v1"
        assert [e.caption for e in history.entries("a.png")] == ["v3", "v2", "v1"]
        assert history.current_index("a.png") == 2
        # ...and the jump forward works again.
        assert history.revert_caption("a.png", 0) == "v3"
        assert history.current_index("a.png") == 0

    def test_revert_same_index_is_noop(self, history: FileHistory, qtbot) -> None:
        history.seed("a.png", "v1")
        history.push("a.png", "编辑", "v2")
        with qtbot.assertNotEmitted(history.changed, wait=50):
            assert history.revert_caption("a.png", 0) == "v2"
        assert len(history.entries("a.png")) == 2

    def test_revert_unknown_key_raises(self, history: FileHistory) -> None:
        with pytest.raises(ValidationError):
            history.revert_caption("nope.png", 1)

    def test_revert_bad_index_raises(self, history: FileHistory) -> None:
        history.seed("a.png", "v1")
        with pytest.raises(ValidationError):
            history.revert_caption("a.png", 5)

    def test_window_trims_around_cursor(self, history: FileHistory) -> None:
        """Jumping deep into the past keeps at most ±5 around the cursor."""
        history.seed("a.png", "v0")
        for i in range(1, 6):  # v1..v5 -> entries v5..v0 (6 total)
            history.push("a.png", "编辑", f"v{i}")
        history.revert_caption("a.png", 5)  # cursor -> v0 (oldest)
        entries = history.entries("a.png")
        assert len(entries) <= HISTORY_KEEP_NEWER + HISTORY_KEEP_OLDER + 1
        cursor = history.current_index("a.png")
        assert cursor <= HISTORY_KEEP_NEWER
        assert entries[cursor].caption == "v0"


class TestStepOlder:
    def test_step_older_moves_cursor_without_deleting(self, history: FileHistory) -> None:
        history.seed("a.png", "v1")
        history.push("a.png", "编辑", "v2")
        assert history.step_older("a.png") == "v1"
        assert [e.caption for e in history.entries("a.png")] == ["v2", "v1"]
        assert history.current_index("a.png") == 1

    def test_step_older_at_oldest_returns_none(self, history: FileHistory) -> None:
        history.seed("a.png", "v1")
        assert history.step_older("a.png") is None

    def test_step_older_unknown_key_returns_none(self, history: FileHistory) -> None:
        assert history.step_older("nope.png") is None


class TestClear:
    def test_clear_keep_current(self, history: FileHistory) -> None:
        history.seed("a.png", "v1")
        history.push("a.png", "编辑", "v2")
        history.push("a.png", "编辑", "v3")
        history.clear_keep_current("a.png")
        entries = history.entries("a.png")
        assert len(entries) == 1
        assert entries[0].caption == "v3"

    def test_clear_keeps_the_cursor_entry(self, history: FileHistory) -> None:
        history.seed("a.png", "v1")
        history.push("a.png", "编辑", "v2")
        history.revert_caption("a.png", 1)  # cursor -> v1
        history.clear_keep_current("a.png")
        entries = history.entries("a.png")
        assert [e.caption for e in entries] == ["v1"]
        assert history.current_index("a.png") == 0

    def test_clear_unknown_key_is_noop(self, history: FileHistory) -> None:
        history.clear_keep_current("nope.png")
        assert history.entries("nope.png") == ()

    def test_entries_unknown_key_empty(self, history: FileHistory) -> None:
        assert history.entries("nope.png") == ()
