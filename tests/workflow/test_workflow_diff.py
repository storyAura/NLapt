"""Tests for nlapt.workflow.diff: mixed CJK/English word-level diff."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from nlapt.core.errors import ValidationError
from nlapt.workflow.diff import DiffOp, DiffSegment, word_diff


def _join(segments: tuple[DiffSegment, ...], ops: set[DiffOp]) -> str:
    return "".join(segment.text for segment in segments if segment.op in ops)


def _assert_invariants(old: str, new: str) -> tuple[DiffSegment, ...]:
    segments = word_diff(old, new)
    assert _join(segments, {DiffOp.EQUAL, DiffOp.DELETE}) == old
    assert _join(segments, {DiffOp.EQUAL, DiffOp.INSERT}) == new
    # Adjacent segments never share the same op (they must be merged).
    for previous, current in zip(segments, segments[1:]):
        assert previous.op is not current.op
    assert all(segment.text for segment in segments)  # no empty segments
    return segments


def test_english_word_replacement() -> None:
    segments = word_diff("a quick brown fox", "a slow brown fox")
    assert segments == (
        DiffSegment(DiffOp.EQUAL, "a "),
        DiffSegment(DiffOp.DELETE, "quick"),
        DiffSegment(DiffOp.INSERT, "slow"),
        DiffSegment(DiffOp.EQUAL, " brown fox"),
    )


def test_cjk_character_level_replacement() -> None:
    segments = word_diff("一只黑猫", "一只白猫")
    assert segments == (
        DiffSegment(DiffOp.EQUAL, "一只"),
        DiffSegment(DiffOp.DELETE, "黑"),
        DiffSegment(DiffOp.INSERT, "白"),
        DiffSegment(DiffOp.EQUAL, "猫"),
    )


def test_mixed_cjk_english() -> None:
    old = "photo of 一只黑猫 sitting"
    new = "photo of 一只白猫 sleeping"
    segments = _assert_invariants(old, new)
    ops = [segment.op for segment in segments]
    assert DiffOp.DELETE in ops
    assert DiffOp.INSERT in ops
    equal_text = _join(segments, {DiffOp.EQUAL})
    assert "photo of " in equal_text
    assert "一只" in equal_text


def test_insertion_only() -> None:
    segments = word_diff("black cat", "black fluffy cat")
    assert segments == (
        DiffSegment(DiffOp.EQUAL, "black "),
        DiffSegment(DiffOp.INSERT, "fluffy "),
        DiffSegment(DiffOp.EQUAL, "cat"),
    )


def test_deletion_only() -> None:
    segments = word_diff("black fluffy cat", "black cat")
    assert segments == (
        DiffSegment(DiffOp.EQUAL, "black "),
        DiffSegment(DiffOp.DELETE, "fluffy "),
        DiffSegment(DiffOp.EQUAL, "cat"),
    )


def test_identical_texts_yield_single_equal_segment() -> None:
    segments = word_diff("same text", "same text")
    assert segments == (DiffSegment(DiffOp.EQUAL, "same text"),)


def test_both_empty_yield_no_segments() -> None:
    assert word_diff("", "") == ()


def test_empty_old_is_pure_insert() -> None:
    assert word_diff("", "new text") == (DiffSegment(DiffOp.INSERT, "new text"),)


def test_empty_new_is_pure_delete() -> None:
    assert word_diff("old text", "") == (DiffSegment(DiffOp.DELETE, "old text"),)


def test_completely_different_texts() -> None:
    segments = _assert_invariants("黑猫", "white cat")
    assert segments == (
        DiffSegment(DiffOp.DELETE, "黑猫"),
        DiffSegment(DiffOp.INSERT, "white cat"),
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("a girl, red hair", "a woman, red hair"),
        ("一个女孩，红色头发", "一个女人，红色头发"),
        ("mixed 文本 with spaces", "different 文字 with  spaces"),
        ("tabs\tand\nnewlines", "tabs and newlines"),
        ("", "全新的中文标注"),
        ("trailing space ", "trailing space"),
        ("日本語のテキスト", "日本語のテスト"),
    ],
)
def test_invariants_hold_for_varied_inputs(old: str, new: str) -> None:
    _assert_invariants(old, new)


def test_whitespace_preserved_in_reconstruction() -> None:
    old = "word1  word2\nword3"
    new = "word1 word2\nword3"
    _assert_invariants(old, new)


def test_segment_is_frozen() -> None:
    segment = DiffSegment(DiffOp.EQUAL, "text")
    with pytest.raises(FrozenInstanceError):
        segment.text = "other"  # type: ignore[misc]


def test_diff_op_values() -> None:
    assert DiffOp.EQUAL.value == "equal"
    assert DiffOp.INSERT.value == "insert"
    assert DiffOp.DELETE.value == "delete"


@pytest.mark.parametrize(("old", "new"), [(None, "x"), ("x", 42)])
def test_non_string_inputs_raise(old, new) -> None:
    with pytest.raises(ValidationError):
        word_diff(old, new)
