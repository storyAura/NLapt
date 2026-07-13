"""Tests for nlapt.indexing.sorting."""

from __future__ import annotations

import pytest

from nlapt.core.errors import ValidationError
from nlapt.indexing import SortBy, natural_key, sort_keys


class TestNaturalKey:
    def test_digit_runs_compare_numerically(self) -> None:
        assert natural_key("img2") < natural_key("img10")

    def test_case_insensitive(self) -> None:
        assert natural_key("IMG2") == natural_key("img2")

    def test_plain_text_ordering(self) -> None:
        assert natural_key("apple") < natural_key("banana")

    def test_mixed_structures_do_not_raise(self) -> None:
        # str/int alternation keeps parity aligned; comparison stays legal.
        assert natural_key("1a") < natural_key("a1")

    def test_leading_zeros_equal_numeric_value(self) -> None:
        assert natural_key("img002") == natural_key("img2")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValidationError):
            natural_key(42)  # type: ignore[arg-type]


class TestSortByName:
    def test_natural_case_insensitive_order(self) -> None:
        keys = ["img10.png", "IMG2.png", "img1.png"]
        assert sort_keys(keys, SortBy.NAME) == ("img1.png", "IMG2.png", "img10.png")

    def test_accepts_string_mode(self) -> None:
        assert sort_keys(["b", "a"], "name") == ("a", "b")  # type: ignore[arg-type]

    def test_empty_input(self) -> None:
        assert sort_keys([], SortBy.NAME) == ()


class TestSortByMtime:
    def test_sorts_ascending_by_mtime(self) -> None:
        keys = ["a", "b", "c"]
        mtimes = {"a": 30.0, "b": 10.0, "c": 20.0}
        assert sort_keys(keys, SortBy.MTIME, mtimes=mtimes) == ("b", "c", "a")

    def test_missing_entries_treated_as_zero_and_stable(self) -> None:
        keys = ["a", "b", "c"]
        mtimes = {"b": 5.0}
        # a and c fall back to 0.0 and keep their relative input order.
        assert sort_keys(keys, SortBy.MTIME, mtimes=mtimes) == ("a", "c", "b")

    def test_none_mapping_preserves_input_order(self) -> None:
        keys = ["z", "a", "m"]
        assert sort_keys(keys, SortBy.MTIME) == ("z", "a", "m")


class TestSortByTokens:
    def test_sorts_ascending_by_token_count(self) -> None:
        keys = ["a", "b", "c"]
        counts = {"a": 77, "b": 5, "c": 40}
        assert sort_keys(keys, SortBy.TOKENS, token_counts=counts) == ("b", "c", "a")

    def test_missing_entries_treated_as_zero_and_stable(self) -> None:
        keys = ["x", "y", "z"]
        counts = {"y": 3}
        assert sort_keys(keys, SortBy.TOKENS, token_counts=counts) == ("x", "z", "y")

    def test_none_mapping_preserves_input_order(self) -> None:
        keys = ["c", "b", "a"]
        assert sort_keys(keys, SortBy.TOKENS) == ("c", "b", "a")


class TestValidation:
    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValidationError):
            sort_keys(["a"], "size")  # type: ignore[arg-type]

    def test_non_string_key_raises(self) -> None:
        with pytest.raises(ValidationError):
            sort_keys([1, 2], SortBy.NAME)  # type: ignore[list-item]

    def test_input_not_mutated(self) -> None:
        keys = ["b", "a"]
        result = sort_keys(keys, SortBy.NAME)
        assert keys == ["b", "a"]
        assert result == ("a", "b")


class TestSortByEnum:
    def test_values(self) -> None:
        assert SortBy.NAME.value == "name"
        assert SortBy.MTIME.value == "mtime"
        assert SortBy.TOKENS.value == "tokens"

    def test_is_str_enum(self) -> None:
        assert isinstance(SortBy.NAME, str)
