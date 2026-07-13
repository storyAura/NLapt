"""Tests for nlapt.ops.find_replace (spec 7.1)."""

from __future__ import annotations

import pytest

from nlapt.core.errors import RegexPatternError, ValidationError
from nlapt.ops import (
    CONTEXT_CHARS,
    FindReplaceOperation,
    FindReplaceSpec,
    compile_spec,
    scope_stats,
)


def make_op(**kwargs) -> FindReplaceOperation:
    return FindReplaceOperation(FindReplaceSpec(**kwargs))


class TestCompileSpec:
    def test_empty_find_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            compile_spec(FindReplaceSpec(find="", replace="x"))

    def test_bad_regex_raises_regex_pattern_error_with_attrs(self) -> None:
        with pytest.raises(RegexPatternError) as excinfo:
            compile_spec(FindReplaceSpec(find="[unclosed", replace="", regex=True))
        assert excinfo.value.pattern == "[unclosed"
        assert excinfo.value.detail  # engine message carried through

    def test_non_string_find_raises(self) -> None:
        with pytest.raises(ValidationError):
            compile_spec(FindReplaceSpec(find=1, replace="x"))  # type: ignore[arg-type]

    def test_spec_property_exposes_input(self) -> None:
        spec = FindReplaceSpec(find="a", replace="b")
        assert FindReplaceOperation(spec).spec is spec

    def test_non_regex_escapes_metacharacters(self) -> None:
        pattern = compile_spec(FindReplaceSpec(find="a.b", replace=""))
        assert pattern.search("a.b")
        assert not pattern.search("axb")

    def test_case_insensitive_by_default(self) -> None:
        pattern = compile_spec(FindReplaceSpec(find="Cat", replace=""))
        assert pattern.search("CAT")

    def test_case_sensitive_flag(self) -> None:
        pattern = compile_spec(FindReplaceSpec(find="Cat", replace="", case_sensitive=True))
        assert not pattern.search("CAT")
        assert pattern.search("Cat")

    def test_whole_word_wraps_word_boundaries(self) -> None:
        pattern = compile_spec(FindReplaceSpec(find="cat", replace="", whole_word=True))
        assert pattern.search("a cat here")
        assert not pattern.search("concatenate")

    def test_regex_whole_word_also_wraps(self) -> None:
        pattern = compile_spec(
            FindReplaceSpec(find="cat|dog", replace="", regex=True, whole_word=True)
        )
        assert pattern.search("the dog barks")
        assert not pattern.search("hotdogs")


class TestApply:
    def test_replaces_all_matches(self) -> None:
        assert make_op(find="girl", replace="woman").apply("girl and GIRL") == (
            "woman and woman"
        )

    def test_regex_group_references(self) -> None:
        op = make_op(find=r"(\w+)@(\w+)", replace=r"\2@\1", regex=True)
        assert op.apply("user@host") == "host@user"

    def test_regex_named_style_group_reference(self) -> None:
        op = make_op(find=r"(\d+)px", replace=r"\g<1>em", regex=True)
        assert op.apply("10px and 20px") == "10em and 20em"

    def test_non_regex_replacement_is_literal(self) -> None:
        # Backslash sequences in the replacement must not be treated as
        # group references outside regex mode.
        op = make_op(find="x", replace=r"\1")
        assert op.apply("x") == r"\1"

    def test_non_regex_backslash_replacement_roundtrip(self) -> None:
        op = make_op(find="sep", replace="C:\\new\\path")
        assert op.apply("sep") == "C:\\new\\path"

    def test_invalid_group_reference_raises_regex_pattern_error(self) -> None:
        op = make_op(find="(a)", replace=r"\5", regex=True)
        with pytest.raises(RegexPatternError) as excinfo:
            op.apply("a")
        assert excinfo.value.pattern == r"\5"

    def test_no_match_returns_text_unchanged(self) -> None:
        assert make_op(find="zebra", replace="x").apply("no hits") == "no hits"


class TestApplyOne:
    def test_replaces_exactly_the_nth_match(self) -> None:
        op = make_op(find="a", replace="X")
        assert op.apply_one("a-a-a", 0) == "X-a-a"
        assert op.apply_one("a-a-a", 1) == "a-X-a"
        assert op.apply_one("a-a-a", 2) == "a-a-X"

    def test_out_of_range_index_raises(self) -> None:
        op = make_op(find="a", replace="X")
        with pytest.raises(ValidationError):
            op.apply_one("a-a", 2)
        with pytest.raises(ValidationError):
            op.apply_one("a-a", -1)

    def test_regex_groups_in_single_replace(self) -> None:
        op = make_op(find=r"(\d+)", replace=r"[\1]", regex=True)
        assert op.apply_one("1 2 3", 1) == "1 [2] 3"

    def test_non_integer_index_raises(self) -> None:
        op = make_op(find="a", replace="X")
        with pytest.raises(ValidationError):
            op.apply_one("a-a", "0")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            op.apply_one("a-a", True)  # type: ignore[arg-type]


class TestPreview:
    def test_preview_positions_and_replacement(self) -> None:
        op = make_op(find="cat", replace="dog")
        previews = op.preview("the cat sat")
        assert len(previews) == 1
        preview = previews[0]
        assert (preview.start, preview.end) == (4, 7)
        assert preview.matched == "cat"
        assert preview.replacement == "dog"
        assert preview.context_before == "the "
        assert preview.context_after == " sat"

    def test_context_capped_at_20_chars(self) -> None:
        text = "b" * 40 + "cat" + "a" * 40
        preview = make_op(find="cat", replace="x").preview(text)[0]
        assert len(preview.context_before) == CONTEXT_CHARS
        assert len(preview.context_after) == CONTEXT_CHARS

    def test_preview_shows_expanded_regex_replacement(self) -> None:
        op = make_op(find=r"(\w+)!", replace=r"\1?", regex=True)
        previews = op.preview("wow! nice!")
        assert [p.replacement for p in previews] == ["wow?", "nice?"]

    def test_no_matches_gives_empty_tuple(self) -> None:
        assert make_op(find="zzz", replace="x").preview("abc") == ()

    def test_preview_rejects_non_string(self) -> None:
        with pytest.raises(ValidationError):
            make_op(find="a", replace="b").preview(None)  # type: ignore[arg-type]


class TestScopeStats:
    TEXTS = {
        "a.jpg": "cat cat cat",
        "b.jpg": "one cat",
        "c.jpg": "no hits here",
    }

    def test_counts_hits_and_files(self) -> None:
        spec = FindReplaceSpec(find="cat", replace="dog")
        stats = scope_stats(spec, "a.jpg", self.TEXTS)
        assert stats.current_file_hits == 3
        assert stats.total_hits == 4
        assert stats.files_with_hits == 2

    def test_current_key_none_gives_zero_current_hits(self) -> None:
        spec = FindReplaceSpec(find="cat", replace="dog")
        stats = scope_stats(spec, None, self.TEXTS)
        assert stats.current_file_hits == 0
        assert stats.total_hits == 4

    def test_current_key_absent_gives_zero_current_hits(self) -> None:
        spec = FindReplaceSpec(find="cat", replace="dog")
        assert scope_stats(spec, "missing.jpg", self.TEXTS).current_file_hits == 0

    def test_empty_find_propagates_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            scope_stats(FindReplaceSpec(find="", replace=""), None, self.TEXTS)

    def test_rejects_non_mapping_texts(self) -> None:
        spec = FindReplaceSpec(find="x", replace="y")
        with pytest.raises(ValidationError):
            scope_stats(spec, None, ["not", "a", "mapping"])  # type: ignore[arg-type]
