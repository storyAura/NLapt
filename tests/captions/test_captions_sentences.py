"""Tests for nlapt.captions.sentences (sentence split/join and row operations)."""

from __future__ import annotations

import pytest

from nlapt.captions.sentences import (
    delete_sentence,
    join_sentences,
    merge_with_previous,
    move_sentence,
    replace_sentence,
    split_sentences,
)
from nlapt.core.errors import ValidationError


class TestSplitSentences:
    def test_splits_after_ascii_terminals_keeping_punct(self) -> None:
        assert split_sentences("A cat sits. It sleeps! Really?") == (
            "A cat sits.",
            "It sleeps!",
            "Really?",
        )

    def test_splits_after_cjk_terminals_keeping_punct(self) -> None:
        assert split_sentences("一只猫。它在睡觉！真的吗？") == (
            "一只猫。",
            "它在睡觉！",
            "真的吗？",
        )

    def test_trailing_text_without_terminal_kept(self) -> None:
        assert split_sentences("First. second without end") == (
            "First.",
            "second without end",
        )

    def test_run_of_terminals_stays_attached(self) -> None:
        assert split_sentences("Wow!! Next.") == ("Wow!!", "Next.")
        assert split_sentences("Wait... done.") == ("Wait...", "done.")

    def test_empty_and_whitespace(self) -> None:
        assert split_sentences("") == ()
        assert split_sentences("   \n ") == ()

    def test_known_v1_limitation_decimal_over_splits(self) -> None:
        # Documented v1 limitation: decimals over-split (see module docstring).
        assert split_sentences("weight 3.5 kg") == ("weight 3.", "5 kg")

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValidationError):
            split_sentences(None)  # type: ignore[arg-type]


class TestJoinSentences:
    def test_ascii_endings_join_with_single_space(self) -> None:
        assert join_sentences(("A cat.", "It sleeps!")) == "A cat. It sleeps!"

    def test_cjk_endings_join_without_space(self) -> None:
        assert join_sentences(("一只猫。", "它在睡觉！")) == "一只猫。它在睡觉！"

    def test_mixed_endings(self) -> None:
        assert join_sentences(("A cat.", "一只猫。", "It sleeps.")) == "A cat. 一只猫。It sleeps."

    def test_no_terminal_punct_joined_with_space(self) -> None:
        assert join_sentences(("hello", "world.")) == "hello world."

    def test_drops_empty_fragments_and_strips(self) -> None:
        assert join_sentences((" A cat. ", "", "  ", "Dog.")) == "A cat. Dog."

    def test_empty_sequence(self) -> None:
        assert join_sentences(()) == ""

    def test_round_trip_ascii(self) -> None:
        text = "A cat sits. It sleeps! Really?"
        assert join_sentences(split_sentences(text)) == text

    def test_round_trip_cjk(self) -> None:
        text = "一只猫。它在睡觉！"
        assert join_sentences(split_sentences(text)) == text


class TestMergeWithPrevious:
    def test_merges_ascii_with_single_space(self) -> None:
        assert merge_with_previous(("A cat.", "It sleeps.", "End."), 1) == (
            "A cat. It sleeps.",
            "End.",
        )

    def test_merges_cjk_directly(self) -> None:
        assert merge_with_previous(("一只猫。", "它在睡觉。"), 1) == ("一只猫。它在睡觉。",)

    @pytest.mark.parametrize("index", [0, -1])
    def test_index_zero_or_negative_raises(self, index: int) -> None:
        with pytest.raises(ValidationError):
            merge_with_previous(("a.", "b."), index)

    def test_index_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            merge_with_previous(("a.", "b."), 2)


class TestMoveSentence:
    def test_move_forward_and_backward(self) -> None:
        sentences = ("a.", "b.", "c.")
        assert move_sentence(sentences, 0, 2) == ("b.", "c.", "a.")
        assert move_sentence(sentences, 2, 0) == ("c.", "a.", "b.")

    @pytest.mark.parametrize(("src", "dst"), [(-1, 0), (3, 0), (0, 3)])
    def test_bounds_validated(self, src: int, dst: int) -> None:
        with pytest.raises(ValidationError):
            move_sentence(("a.", "b.", "c."), src, dst)


class TestReplaceSentence:
    def test_replaces_and_strips(self) -> None:
        assert replace_sentence(("a.", "b.", "c."), 1, "  new text. ") == (
            "a.",
            "new text.",
            "c.",
        )

    def test_empty_replacement_raises(self) -> None:
        with pytest.raises(ValidationError):
            replace_sentence(("a.", "b."), 0, "   ")

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            replace_sentence(("a.",), 1, "x.")


class TestDeleteSentence:
    def test_deletes_at_index(self) -> None:
        assert delete_sentence(("a.", "b.", "c."), 1) == ("a.", "c.")

    def test_delete_only_sentence(self) -> None:
        assert delete_sentence(("a.",), 0) == ()

    @pytest.mark.parametrize("index", [-1, 2])
    def test_out_of_range_raises(self, index: int) -> None:
        with pytest.raises(ValidationError):
            delete_sentence(("a.", "b."), index)
