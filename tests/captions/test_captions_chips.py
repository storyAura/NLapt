"""Tests for nlapt.captions.chips (chip splitting/joining and tag stats)."""

from __future__ import annotations

import pytest

from nlapt.captions.chips import (
    CHIP_SEPARATORS,
    SENTENCE_ENDINGS,
    count_tag_in_texts,
    dedup_chips,
    edit_chip,
    find_duplicates,
    is_long_sentence,
    join_chips,
    move_chip,
    split_chips,
)
from nlapt.core.errors import ValidationError


class TestConstants:
    def test_separator_and_ending_constants(self) -> None:
        assert CHIP_SEPARATORS == (",", "，")
        assert SENTENCE_ENDINGS == (".", "!", "?", "。", "！", "？")


class TestSplitChips:
    def test_splits_half_width_commas(self) -> None:
        assert split_chips("1girl, red hair, smile") == ("1girl", "red hair", "smile")

    def test_splits_full_width_commas(self) -> None:
        assert split_chips("女孩，红发，微笑") == ("女孩", "红发", "微笑")

    def test_splits_mixed_width_commas(self) -> None:
        assert split_chips("1girl，red hair, smile") == ("1girl", "red hair", "smile")

    def test_strips_and_drops_empties(self) -> None:
        assert split_chips("  a  ,, ,b ,，  ") == ("a", "b")

    def test_empty_text_yields_no_chips(self) -> None:
        assert split_chips("") == ()
        assert split_chips("  , ，  ") == ()

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValidationError):
            split_chips(None)  # type: ignore[arg-type]


class TestJoinChips:
    def test_joins_with_comma_space(self) -> None:
        assert join_chips(("a", "b", "c")) == "a, b, c"

    def test_strips_each_and_drops_empties(self) -> None:
        assert join_chips((" a ", "", "  ", "b")) == "a, b"

    def test_empty_sequence_gives_empty_string(self) -> None:
        assert join_chips(()) == ""

    def test_round_trip(self) -> None:
        text = "1girl, red hair，smile,  , long dress"
        assert join_chips(split_chips(text)) == "1girl, red hair, smile, long dress"

    def test_non_string_chip_raises(self) -> None:
        with pytest.raises(ValidationError):
            join_chips(["a", 3])  # type: ignore[list-item]


class TestIsLongSentence:
    @pytest.mark.parametrize("ending", [".", "!", "?", "。", "！", "？"])
    def test_true_for_each_terminal_punct(self, ending: str) -> None:
        assert is_long_sentence(f"the cat sits{ending}") is True

    def test_false_for_short_tag(self) -> None:
        assert is_long_sentence("red hair") is False

    def test_true_when_punct_is_embedded(self) -> None:
        assert is_long_sentence("weight 3.5 kg") is True


class TestDuplicates:
    def test_find_duplicates_reports_indices(self) -> None:
        chips = ("a", "b", "a", "c", "b", "a")
        assert find_duplicates(chips) == {"a": (0, 2, 5), "b": (1, 4)}

    def test_find_duplicates_only_for_count_two_or_more(self) -> None:
        assert find_duplicates(("a", "b", "c")) == {}

    def test_find_duplicates_is_exact_match(self) -> None:
        assert find_duplicates(("cat", "cats", "Cat")) == {}

    def test_dedup_keeps_first_occurrence(self) -> None:
        assert dedup_chips(("a", "b", "a", "c", "b")) == ("a", "b", "c")

    def test_dedup_empty(self) -> None:
        assert dedup_chips(()) == ()


class TestEditChip:
    def test_simple_replacement(self) -> None:
        assert edit_chip(("a", "b", "c"), 1, "x") == ("a", "x", "c")

    def test_embedded_commas_split_in_place(self) -> None:
        assert edit_chip(("a", "b", "c"), 1, "x, y，z") == ("a", "x", "y", "z", "c")

    def test_empty_text_removes_chip(self) -> None:
        assert edit_chip(("a", "b", "c"), 1, "   ") == ("a", "c")

    @pytest.mark.parametrize("index", [-1, 3])
    def test_out_of_range_index_raises(self, index: int) -> None:
        with pytest.raises(ValidationError):
            edit_chip(("a", "b", "c"), index, "x")


class TestMoveChip:
    def test_move_forward(self) -> None:
        assert move_chip(("a", "b", "c", "d"), 0, 2) == ("b", "c", "a", "d")

    def test_move_backward(self) -> None:
        assert move_chip(("a", "b", "c", "d"), 3, 0) == ("d", "a", "b", "c")

    def test_move_to_same_position_is_identity(self) -> None:
        assert move_chip(("a", "b"), 1, 1) == ("a", "b")

    @pytest.mark.parametrize(("src", "dst"), [(-1, 0), (2, 0), (0, -1), (0, 2)])
    def test_out_of_bounds_raises(self, src: int, dst: int) -> None:
        with pytest.raises(ValidationError):
            move_chip(("a", "b"), src, dst)


class TestCountTagInTexts:
    def test_counts_files_with_whole_chip_match(self) -> None:
        texts = {
            "a.png": "1girl, red hair, smile",
            "b.png": "1girl, blue hair",
            "c.png": "2girls, smile",
        }
        assert count_tag_in_texts("1girl", texts) == 2

    def test_substring_does_not_match(self) -> None:
        texts = {"a.png": "2girls, smiles", "b.png": "girl"}
        assert count_tag_in_texts("girl", texts) == 1
        assert count_tag_in_texts("smile", texts) == 0

    def test_file_counts_once_even_with_repeated_chip(self) -> None:
        texts = {"a.png": "smile, smile, smile"}
        assert count_tag_in_texts("smile", texts) == 1

    def test_tag_is_stripped_before_matching(self) -> None:
        texts = {"a.png": "red hair, smile"}
        assert count_tag_in_texts("  smile  ", texts) == 1

    def test_full_width_comma_texts(self) -> None:
        texts = {"a.png": "女孩，红发"}
        assert count_tag_in_texts("红发", texts) == 1

    def test_empty_tag_raises(self) -> None:
        with pytest.raises(ValidationError):
            count_tag_in_texts("   ", {"a.png": "x"})

    def test_empty_texts_mapping(self) -> None:
        assert count_tag_in_texts("smile", {}) == 0
