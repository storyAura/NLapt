"""Tests for nlapt.ops.prefix_suffix (spec 7.2)."""

from __future__ import annotations

import pytest

from nlapt.core.errors import ValidationError
from nlapt.ops import (
    JOINERS,
    PrefixSuffixOperation,
    PrefixSuffixSpec,
    make_trigger_add_op,
    make_trigger_remove_op,
    remove_exact_prefix,
    remove_exact_suffix,
)


def make_op(**kwargs) -> PrefixSuffixOperation:
    return PrefixSuffixOperation(PrefixSuffixSpec(**kwargs))


class TestSpecValidation:
    def test_invalid_joiner_raises(self) -> None:
        with pytest.raises(ValidationError):
            PrefixSuffixSpec(prefix="x", joiner=" - ")

    def test_allowed_joiners(self) -> None:
        assert JOINERS == (", ", " ", "")
        for joiner in JOINERS:
            PrefixSuffixSpec(prefix="x", joiner=joiner)  # must not raise

    def test_non_string_prefix_raises(self) -> None:
        with pytest.raises(ValidationError):
            PrefixSuffixSpec(prefix=1)  # type: ignore[arg-type]

    def test_operation_rejects_non_spec(self) -> None:
        with pytest.raises(ValidationError):
            PrefixSuffixOperation("not a spec")  # type: ignore[arg-type]

    def test_operation_exposes_spec(self) -> None:
        spec = PrefixSuffixSpec(prefix="x")
        assert PrefixSuffixOperation(spec).spec is spec


class TestApplyPrefix:
    def test_adds_prefix_with_joiner(self) -> None:
        assert make_op(prefix="mina").apply("1girl, solo") == "mina, 1girl, solo"

    def test_space_joiner(self) -> None:
        assert make_op(prefix="mina", joiner=" ").apply("solo") == "mina solo"

    def test_empty_joiner(self) -> None:
        assert make_op(prefix="mina", joiner="").apply("solo") == "minasolo"

    def test_skip_if_present_prevents_double_trigger(self) -> None:
        op = make_op(prefix="mina")
        once = op.apply("solo")
        assert op.apply(once) == once

    def test_skip_disabled_adds_again(self) -> None:
        op = make_op(prefix="mina", skip_if_present=False)
        assert op.apply("mina, solo") == "mina, mina, solo"

    def test_empty_text_gets_bare_prefix_without_joiner(self) -> None:
        assert make_op(prefix="mina").apply("") == "mina"


class TestApplySuffix:
    def test_adds_suffix_with_joiner(self) -> None:
        assert make_op(suffix="masterpiece").apply("1girl") == "1girl, masterpiece"

    def test_skip_if_present_requires_joiner_boundary(self) -> None:
        op = make_op(suffix="end")
        # Exact "joiner + suffix" ending -> skipped.
        assert op.apply("text, end") == "text, end"
        # A bare-substring ending no longer suppresses the suffix.
        assert op.apply("text end") == "text end, end"
        # Space joiner matches its own boundary.
        space_op = make_op(suffix="end", joiner=" ")
        assert space_op.apply("text end") == "text end"

    def test_empty_text_gets_bare_suffix(self) -> None:
        assert make_op(suffix="end").apply("") == "end"

    def test_prefix_and_suffix_together(self) -> None:
        op = make_op(prefix="pre", suffix="post")
        assert op.apply("mid") == "pre, mid, post"

    def test_prefix_and_suffix_on_empty_text(self) -> None:
        op = make_op(prefix="pre", suffix="post")
        assert op.apply("") == "pre, post"

    def test_no_prefix_no_suffix_is_noop(self) -> None:
        assert make_op().apply("unchanged") == "unchanged"


class TestPreview:
    def test_prefix_preview_is_zero_width_at_start(self) -> None:
        previews = make_op(prefix="mina").preview("solo")
        assert len(previews) == 1
        assert (previews[0].start, previews[0].end) == (0, 0)
        assert previews[0].matched == ""
        assert previews[0].replacement == "mina, "

    def test_suffix_preview_at_end(self) -> None:
        previews = make_op(suffix="end").preview("body")
        assert (previews[0].start, previews[0].end) == (4, 4)
        assert previews[0].replacement == ", end"

    def test_skipped_insertions_produce_no_previews(self) -> None:
        assert make_op(prefix="mina").preview("mina, solo") == ()

    def test_preview_matches_apply_on_empty_text(self) -> None:
        op = make_op(prefix="pre", suffix="post")
        previews = op.preview("")
        combined = previews[0].replacement + previews[1].replacement
        assert combined == op.apply("")


class TestRemoveExact:
    def test_removes_prefix_with_joiner(self) -> None:
        assert remove_exact_prefix("mina, solo", "mina", ", ") == "solo"

    def test_removes_bare_prefix_when_whole_text(self) -> None:
        assert remove_exact_prefix("mina", "mina", ", ") == ""

    def test_partial_prefix_not_removed(self) -> None:
        assert remove_exact_prefix("minahamu, solo", "mina", ", ") == "minahamu, solo"

    def test_prefix_without_joiner_not_removed(self) -> None:
        assert remove_exact_prefix("mina solo", "mina", ", ") == "mina solo"

    def test_removes_suffix_with_joiner(self) -> None:
        assert remove_exact_suffix("solo, end", "end", ", ") == "solo"

    def test_removes_bare_suffix_when_whole_text(self) -> None:
        assert remove_exact_suffix("end", "end", ", ") == ""

    def test_suffix_without_joiner_not_removed(self) -> None:
        assert remove_exact_suffix("solo end", "end", ", ") == "solo end"

    def test_empty_prefix_raises(self) -> None:
        with pytest.raises(ValidationError):
            remove_exact_prefix("text", "", ", ")

    def test_empty_suffix_raises(self) -> None:
        with pytest.raises(ValidationError):
            remove_exact_suffix("text", "", ", ")

    def test_invalid_joiner_raises(self) -> None:
        with pytest.raises(ValidationError):
            remove_exact_prefix("text", "x", "; ")


class TestTriggerOps:
    def test_add_then_remove_roundtrip(self) -> None:
        add = make_trigger_add_op("minahamu")
        remove = make_trigger_remove_op("minahamu")
        text = "1girl, solo"
        assert remove.apply(add.apply(text)) == text

    def test_add_skips_when_already_present(self) -> None:
        add = make_trigger_add_op("minahamu")
        assert add.apply("minahamu, solo") == "minahamu, solo"

    def test_remove_on_missing_trigger_is_noop(self) -> None:
        remove = make_trigger_remove_op("minahamu")
        assert remove.apply("1girl, solo") == "1girl, solo"

    def test_remove_bare_trigger_only_text(self) -> None:
        remove = make_trigger_remove_op("minahamu")
        assert remove.apply("minahamu") == ""

    def test_remove_preview_shows_removed_span(self) -> None:
        remove = make_trigger_remove_op("mina")
        previews = remove.preview("mina, solo")
        assert len(previews) == 1
        assert previews[0].matched == "mina, "
        assert previews[0].replacement == ""

    def test_remove_preview_empty_when_no_match(self) -> None:
        assert make_trigger_remove_op("mina").preview("solo") == ()

    def test_empty_trigger_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_trigger_add_op("  ")
        with pytest.raises(ValidationError):
            make_trigger_remove_op("")

    def test_custom_joiner(self) -> None:
        add = make_trigger_add_op("mina", joiner=" ")
        remove = make_trigger_remove_op("mina", joiner=" ")
        assert add.apply("solo") == "mina solo"
        assert remove.apply("mina solo") == "solo"
