"""Tests for nlapt.ops.dedup (tag deduplication)."""

from __future__ import annotations

import pytest

from nlapt.core.errors import ValidationError
from nlapt.ops import CONTEXT_CHARS, DedupTagsOperation, MatchPreview


@pytest.fixture()
def op() -> DedupTagsOperation:
    return DedupTagsOperation()


class TestApply:
    def test_keeps_first_occurrences_and_rejoins(self, op: DedupTagsOperation) -> None:
        assert op.apply("a, b, a, c, b") == "a, b, c"

    def test_fullwidth_commas_are_separators(self, op: DedupTagsOperation) -> None:
        assert op.apply("a，b，a") == "a, b"

    def test_normalizes_joiner_even_without_duplicates(self, op: DedupTagsOperation) -> None:
        assert op.apply("a,b ,  c") == "a, b, c"

    def test_duplicates_are_exact_matches_only(self, op: DedupTagsOperation) -> None:
        # Case differs -> not a duplicate.
        assert op.apply("Cat, cat") == "Cat, cat"

    def test_empty_text(self, op: DedupTagsOperation) -> None:
        assert op.apply("") == ""

    def test_only_separators(self, op: DedupTagsOperation) -> None:
        assert op.apply(", ,，") == ""

    def test_rejects_non_string(self, op: DedupTagsOperation) -> None:
        with pytest.raises(ValidationError):
            op.apply(None)  # type: ignore[arg-type]


class TestPreview:
    def test_lists_each_removed_duplicate(self, op: DedupTagsOperation) -> None:
        text = "a, b, a, c, b, a"
        previews = op.preview(text)
        assert [p.matched for p in previews] == ["a", "b", "a"]
        assert all(isinstance(p, MatchPreview) for p in previews)
        assert all(p.replacement == "" for p in previews)

    def test_spans_point_into_original_text(self, op: DedupTagsOperation) -> None:
        text = "tag, other, tag"
        previews = op.preview(text)
        assert len(previews) == 1
        preview = previews[0]
        assert text[preview.start : preview.end] == "tag"
        assert preview.start == 12

    def test_span_excludes_surrounding_whitespace(self, op: DedupTagsOperation) -> None:
        text = "a,   a   , b"
        preview = op.preview(text)[0]
        assert text[preview.start : preview.end] == "a"
        assert preview.matched == "a"

    def test_no_duplicates_gives_empty_preview(self, op: DedupTagsOperation) -> None:
        assert op.preview("a, b, c") == ()

    def test_context_capped(self, op: DedupTagsOperation) -> None:
        long_tag = "x" * 60
        text = f"dup, {long_tag}, dup"
        preview = op.preview(text)[0]
        assert len(preview.context_before) == CONTEXT_CHARS
        assert preview.context_after == ""

    def test_preview_consistent_with_apply(self, op: DedupTagsOperation) -> None:
        text = "a, b, a, c, b"
        removed = {p.matched for p in op.preview(text)}
        assert removed == {"a", "b"}
        result = op.apply(text)
        assert result == "a, b, c"

    def test_rejects_non_string(self, op: DedupTagsOperation) -> None:
        with pytest.raises(ValidationError):
            op.preview(42)  # type: ignore[arg-type]


class TestOperationContract:
    def test_name(self, op: DedupTagsOperation) -> None:
        assert op.name == "dedup_tags"

    def test_apply_is_deterministic_and_pure(self, op: DedupTagsOperation) -> None:
        text = "a, b, a"
        assert op.apply(text) == op.apply(text)
        assert text == "a, b, a"
