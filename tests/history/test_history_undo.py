"""Tests for nlapt.history.undo: grouping window, limits, redo semantics."""

from __future__ import annotations

import pytest

from nlapt.core.errors import ValidationError
from nlapt.history.undo import GROUP_WINDOW_SECONDS, UNDO_LIMIT, UndoStack

STEP = 1.0  # comfortably >= GROUP_WINDOW_SECONDS: always a separate undo step


class FakeClock:
    """Deterministic injected clock; tests advance ``now`` explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


def test_constants_match_contract() -> None:
    assert UNDO_LIMIT == 50
    assert GROUP_WINDOW_SECONDS == 0.6


def test_initial_state(clock: FakeClock) -> None:
    stack = UndoStack("hello", clock=clock)
    assert stack.current == "hello"
    assert stack.can_undo() is False
    assert stack.can_redo() is False
    assert stack.undo() is None
    assert stack.redo() is None


def test_push_undo_redo_roundtrip(clock: FakeClock) -> None:
    stack = UndoStack("v0", clock=clock)
    clock.now += STEP
    stack.push("v1")
    clock.now += STEP
    stack.push("v2")
    assert stack.current == "v2"
    assert stack.undo() == "v1"
    assert stack.undo() == "v0"
    assert stack.can_undo() is False
    assert stack.redo() == "v1"
    assert stack.redo() == "v2"
    assert stack.can_redo() is False


def test_push_identical_text_is_noop(clock: FakeClock) -> None:
    stack = UndoStack("same", clock=clock)
    clock.now += STEP
    stack.push("same")
    assert stack.can_undo() is False
    assert stack.current == "same"


def test_rapid_pushes_collapse_into_one_step(clock: FakeClock) -> None:
    stack = UndoStack("a", clock=clock)
    clock.now = 10.0
    stack.push("ab")
    clock.now = 10.3  # < 0.6s after previous push -> same group
    stack.push("abc")
    clock.now = 10.5  # still within the sliding window
    stack.push("abcd")
    assert stack.undo() == "a"  # the whole burst is one step
    assert stack.can_undo() is False


def test_pushes_at_or_beyond_window_are_separate_steps(clock: FakeClock) -> None:
    stack = UndoStack("a", clock=clock)
    clock.now = 0.0
    stack.push("ab")
    clock.now = GROUP_WINDOW_SECONDS  # exactly 0.6s later -> NOT grouped
    stack.push("abc")
    assert stack.undo() == "ab"
    assert stack.undo() == "a"


def test_noop_push_does_not_extend_group_window(clock: FakeClock) -> None:
    stack = UndoStack("", clock=clock)
    clock.now = 0.0
    stack.push("x")
    clock.now = 0.5
    stack.push("x")  # identical: complete no-op, must not touch the timer
    clock.now = 0.7  # 0.7s after the real push -> separate step
    stack.push("y")
    assert stack.undo() == "x"  # a broken timer would collapse into one step
    assert stack.undo() == ""


def test_undo_breaks_grouping_chain(clock: FakeClock) -> None:
    stack = UndoStack("v0", clock=clock)
    clock.now = 1.0
    stack.push("v1")
    assert stack.undo() == "v0"
    clock.now = 1.1  # within 0.6s of the last push, but undo broke the chain
    stack.push("v2")
    assert stack.undo() == "v0"


def test_new_push_clears_redo(clock: FakeClock) -> None:
    stack = UndoStack("v0", clock=clock)
    clock.now += STEP
    stack.push("v1")
    assert stack.undo() == "v0"
    assert stack.can_redo() is True
    clock.now += STEP
    stack.push("v2")
    assert stack.can_redo() is False
    assert stack.redo() is None


def test_limit_evicts_oldest_step(clock: FakeClock) -> None:
    stack = UndoStack("s0", clock=clock, limit=3)
    for index in range(1, 6):  # s1..s5, each a separate step
        clock.now += STEP
        stack.push(f"s{index}")
    assert stack.current == "s5"
    assert stack.undo() == "s4"
    assert stack.undo() == "s3"
    assert stack.undo() == "s2"  # s0/s1 were evicted
    assert stack.can_undo() is False


def test_default_limit_keeps_fifty_steps(clock: FakeClock) -> None:
    stack = UndoStack("t0", clock=clock)
    for index in range(1, 61):  # 60 separate steps
        clock.now += STEP
        stack.push(f"t{index}")
    undone = 0
    while stack.undo() is not None:
        undone += 1
    assert undone == UNDO_LIMIT
    assert stack.current == "t10"  # 60 - 50 oldest steps evicted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial": 42},
        {"initial": "", "limit": 0},
        {"initial": "", "limit": True},
        {"initial": "", "clock": "not-callable"},
    ],
)
def test_constructor_validation(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        UndoStack(**kwargs)


def test_push_validates_text(clock: FakeClock) -> None:
    stack = UndoStack("", clock=clock)
    with pytest.raises(ValidationError):
        stack.push(None)  # type: ignore[arg-type]
