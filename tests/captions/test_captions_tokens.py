"""Tests for nlapt.captions.tokens (heuristic CLIP token estimation)."""

from __future__ import annotations

import pytest

from nlapt.captions.tokens import (
    CLIP_TOKEN_LIMIT,
    WORD_CHARS_PER_TOKEN,
    HeuristicClipEstimator,
    TokenEstimator,
    get_default_estimator,
    is_over_limit,
)
from nlapt.core.errors import ValidationError


@pytest.fixture()
def estimator() -> HeuristicClipEstimator:
    return HeuristicClipEstimator()


class TestHeuristicEstimator:
    def test_empty_string_is_zero(self, estimator: HeuristicClipEstimator) -> None:
        assert estimator.estimate("") == 0

    def test_whitespace_only_is_zero(self, estimator: HeuristicClipEstimator) -> None:
        assert estimator.estimate("  \n\t ") == 0

    def test_short_words_are_one_token_each(self, estimator: HeuristicClipEstimator) -> None:
        assert estimator.estimate("a cat") == 2
        assert estimator.estimate("one two three") == 3

    def test_long_word_splits_into_subword_units(
        self, estimator: HeuristicClipEstimator
    ) -> None:
        word = "extraordinarily"  # 15 chars -> ceil(15 / WORD_CHARS_PER_TOKEN)
        expected = -(-len(word) // WORD_CHARS_PER_TOKEN)
        assert estimator.estimate(word) == expected
        assert expected > 1

    def test_cjk_counts_roughly_per_character(
        self, estimator: HeuristicClipEstimator
    ) -> None:
        assert estimator.estimate("一只猫") == 3
        assert estimator.estimate("これは猫") == 4

    def test_punctuation_counts_one_each(self, estimator: HeuristicClipEstimator) -> None:
        # 3 words + 2 comma separators
        assert estimator.estimate("red, blue, green") == 5

    def test_full_width_comma_counts_as_separator(
        self, estimator: HeuristicClipEstimator
    ) -> None:
        # 2 CJK chars + full-width comma + 2 CJK chars
        assert estimator.estimate("红发，微笑") == 5

    def test_long_tag_string_counts_separators(
        self, estimator: HeuristicClipEstimator
    ) -> None:
        tags = ", ".join(["tag"] * 10)
        assert estimator.estimate(tags) == 10 + 9

    def test_mixed_cjk_and_ascii(self, estimator: HeuristicClipEstimator) -> None:
        # "cat" word (1) + 2 CJK chars
        assert estimator.estimate("cat 猫咪") == 3

    def test_deterministic(self, estimator: HeuristicClipEstimator) -> None:
        text = "1girl, long red hair, 微笑。 photorealistic!"
        results = {estimator.estimate(text) for _ in range(5)}
        assert len(results) == 1

    def test_non_string_raises(self, estimator: HeuristicClipEstimator) -> None:
        with pytest.raises(ValidationError):
            estimator.estimate(None)  # type: ignore[arg-type]


class TestDefaultEstimator:
    def test_returns_token_estimator(self) -> None:
        est = get_default_estimator()
        assert isinstance(est, TokenEstimator)
        assert est.estimate("a cat") == 2

    def test_matches_heuristic_estimator(self) -> None:
        text = "1girl, red hair, 一只猫。"
        assert get_default_estimator().estimate(text) == HeuristicClipEstimator().estimate(text)


class TestIsOverLimit:
    def test_default_limit_is_77(self) -> None:
        assert CLIP_TOKEN_LIMIT == 77
        assert is_over_limit(CLIP_TOKEN_LIMIT) is False
        assert is_over_limit(CLIP_TOKEN_LIMIT + 1) is True

    def test_zero_is_not_over(self) -> None:
        assert is_over_limit(0) is False

    def test_custom_limit(self) -> None:
        assert is_over_limit(11, limit=10) is True
        assert is_over_limit(10, limit=10) is False

    def test_negative_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            is_over_limit(-1)

    def test_invalid_limit_raises(self) -> None:
        with pytest.raises(ValidationError):
            is_over_limit(5, limit=0)

    def test_non_int_raises(self) -> None:
        with pytest.raises(ValidationError):
            is_over_limit("77")  # type: ignore[arg-type]
