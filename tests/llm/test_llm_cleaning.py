"""Tests for nlapt.llm.cleaning: fences, quotes, label prefixes, combinations."""

from __future__ import annotations

import pytest

from nlapt.core.errors import LLMOutputError, ValidationError
from nlapt.llm.cleaning import OUTPUT_CONSTRAINT, clean_llm_output


class TestOutputConstraint:
    def test_exact_constraint_text(self) -> None:
        assert OUTPUT_CONSTRAINT == (
            "Output only the final caption text. "
            "No explanations, quotes, or code blocks."
        )


class TestCleanBasic:
    def test_plain_text_unchanged(self) -> None:
        assert clean_llm_output("a red fox in the snow") == "a red fox in the snow"

    def test_surrounding_whitespace_stripped(self) -> None:
        assert clean_llm_output("  a cat  \n\n") == "a cat"

    def test_internal_whitespace_preserved(self) -> None:
        assert clean_llm_output("a cat,  a dog") == "a cat,  a dog"

    def test_non_string_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            clean_llm_output(123)  # type: ignore[arg-type]


class TestCodeFences:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("```\na red fox\n```", "a red fox"),
            ("```text\na red fox\n```", "a red fox"),
            ("```json\na red fox\n```", "a red fox"),
            ("```markdown\nline one\nline two\n```", "line one\nline two"),
            ("```a red fox```", "a red fox"),
        ],
    )
    def test_fences_removed(self, raw: str, expected: str) -> None:
        assert clean_llm_output(raw) == expected

    def test_unbalanced_fence_kept(self) -> None:
        # No closing fence: not a wrapper, leave text intact.
        assert clean_llm_output("```json\na red fox") == "```json\na red fox"

    def test_fence_in_middle_untouched(self) -> None:
        assert clean_llm_output("a ``` b") == "a ``` b"


class TestQuotes:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('"a red fox"', "a red fox"),
            ("'a red fox'", "a red fox"),
            ("“一只红色的狐狸”", "一只红色的狐狸"),
            ("‘a red fox’", "a red fox"),
            ("「猫が座っている」", "猫が座っている"),
        ],
    )
    def test_matching_quotes_stripped(self, raw: str, expected: str) -> None:
        assert clean_llm_output(raw) == expected

    def test_nested_quotes_stripped(self) -> None:
        assert clean_llm_output("\"'a red fox'\"") == "a red fox"
        assert clean_llm_output("“\"你好\"”") == "你好"

    def test_mismatched_quotes_kept(self) -> None:
        assert clean_llm_output('"a red fox\'') == '"a red fox\''

    def test_inner_quotes_preserved(self) -> None:
        assert clean_llm_output('a "quoted" word') == 'a "quoted" word'


class TestLabelPrefixes:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Caption: a red fox", "a red fox"),
            ("caption: a red fox", "a red fox"),
            ("CAPTION: a red fox", "a red fox"),
            ("Caption： a red fox", "a red fox"),
            ("标注：一只红色的狐狸", "一只红色的狐狸"),
            ("标注: 一只红色的狐狸", "一只红色的狐狸"),
            ("输出：结果文本", "结果文本"),
            ("Output: a red fox", "a red fox"),
            ("Result: a red fox", "a red fox"),
        ],
    )
    def test_label_prefix_stripped(self, raw: str, expected: str) -> None:
        assert clean_llm_output(raw) == expected

    def test_label_in_middle_untouched(self) -> None:
        assert clean_llm_output("the caption: style") == "the caption: style"


class TestCombinations:
    def test_fence_quote_label_combination(self) -> None:
        raw = '```\n"Caption: a red fox"\n```'
        assert clean_llm_output(raw) == "a red fox"

    def test_label_then_quotes(self) -> None:
        assert clean_llm_output('Caption: "a red fox"') == "a red fox"

    def test_fence_with_lang_and_fullwidth_quotes(self) -> None:
        raw = "```text\n“标注：一只猫”\n```"
        assert clean_llm_output(raw) == "一只猫"


class TestEmptyResults:
    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   \n\t ",
            "```\n```",
            '""',
            "“”",
            "Caption:",
            "Caption:   ",
            '```\n"Caption: "\n```',
        ],
    )
    def test_empty_after_cleaning_raises(self, raw: str) -> None:
        with pytest.raises(LLMOutputError):
            clean_llm_output(raw)
