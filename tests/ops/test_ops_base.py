"""Tests for nlapt.ops.base (registry, previews, protocol)."""

from __future__ import annotations

import dataclasses

import pytest

from nlapt.core.errors import ValidationError
from nlapt.ops import (
    CONTEXT_CHARS,
    MatchPreview,
    OperationRegistry,
    TextOperation,
    get_default_registry,
)
from nlapt.ops.base import build_match_preview


class _UpperOperation:
    name = "upper"

    def preview(self, text: str) -> tuple[MatchPreview, ...]:
        return ()

    def apply(self, text: str) -> str:
        return text.upper()


class TestMatchPreview:
    def test_is_frozen(self) -> None:
        preview = MatchPreview(0, 1, "a", "b", "", "")
        with pytest.raises(dataclasses.FrozenInstanceError):
            preview.matched = "x"  # type: ignore[misc]

    def test_build_match_preview_caps_context(self) -> None:
        text = "x" * 50 + "HIT" + "y" * 50
        preview = build_match_preview(text, 50, 53, "NEW")
        assert preview.matched == "HIT"
        assert preview.replacement == "NEW"
        assert preview.context_before == "x" * CONTEXT_CHARS
        assert preview.context_after == "y" * CONTEXT_CHARS

    def test_build_match_preview_short_context(self) -> None:
        preview = build_match_preview("abc", 1, 2, "")
        assert preview.context_before == "a"
        assert preview.context_after == "c"

    def test_build_match_preview_rejects_bad_span(self) -> None:
        with pytest.raises(ValidationError):
            build_match_preview("abc", 2, 1, "")
        with pytest.raises(ValidationError):
            build_match_preview("abc", 0, 99, "")


class TestOperationRegistry:
    def test_register_create_and_names(self) -> None:
        registry = OperationRegistry()
        registry.register("upper", lambda: _UpperOperation())
        assert registry.names() == ("upper",)
        op = registry.create("upper")
        assert op.apply("abc") == "ABC"

    def test_duplicate_registration_raises(self) -> None:
        registry = OperationRegistry()
        registry.register("upper", lambda: _UpperOperation())
        with pytest.raises(ValidationError):
            registry.register("upper", lambda: _UpperOperation())

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValidationError):
            OperationRegistry().create("nope")

    def test_register_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            OperationRegistry().register("", lambda: _UpperOperation())

    def test_register_non_callable_raises(self) -> None:
        with pytest.raises(ValidationError):
            OperationRegistry().register("bad", "not-callable")  # type: ignore[arg-type]

    def test_create_with_bad_params_raises_validation_error(self) -> None:
        registry = get_default_registry()
        with pytest.raises(ValidationError):
            registry.create("dedup_tags", bogus=True)


class TestDefaultRegistry:
    def test_preregistered_names(self) -> None:
        registry = get_default_registry()
        assert set(registry.names()) == {"find_replace", "prefix_suffix", "dedup_tags"}

    def test_find_replace_factory_works(self) -> None:
        op = get_default_registry().create("find_replace", find="a", replace="b")
        assert op.apply("aaa") == "bbb"
        assert isinstance(op, TextOperation)

    def test_prefix_suffix_factory_works(self) -> None:
        op = get_default_registry().create("prefix_suffix", prefix="x")
        assert op.apply("y") == "x, y"

    def test_dedup_factory_works(self) -> None:
        op = get_default_registry().create("dedup_tags")
        assert op.apply("a, b, a") == "a, b"

    def test_registries_are_independent(self) -> None:
        first = get_default_registry()
        second = get_default_registry()
        first.register("upper", lambda: _UpperOperation())
        assert "upper" not in second.names()

    def test_operations_satisfy_protocol(self) -> None:
        registry = get_default_registry()
        ops = [
            registry.create("find_replace", find="a", replace="b"),
            registry.create("prefix_suffix", prefix="x"),
            registry.create("dedup_tags"),
        ]
        for op in ops:
            assert isinstance(op, TextOperation)
            assert isinstance(op.name, str) and op.name
