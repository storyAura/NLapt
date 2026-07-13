"""Tests for nlapt.llm.templates: rendering, defaults, TemplateStore."""

from __future__ import annotations

import pytest

from nlapt.core.errors import ValidationError
from nlapt.llm.templates import (
    ALLOWED_VARIABLES,
    DEFAULT_TEMPLATES,
    TemplateStore,
    render_template,
)

EXPECTED_DEFAULT_KEYS = {
    "polish",
    "rewrite",
    "expand",
    "condense",
    "translate_en_zh",
    "translate_zh_en",
    "translate_to_zh",
    "translate_to_en",
    "translate_to_ja",
}


class TestRenderTemplate:
    def test_all_allowed_variables(self) -> None:
        result = render_template(
            "cap={caption} file={filename} trig={trigger}",
            caption="a fox",
            filename="img_001.png",
            trigger="minahamu",
        )
        assert result == "cap=a fox file=img_001.png trig=minahamu"

    def test_allowed_variables_constant(self) -> None:
        assert ALLOWED_VARIABLES == frozenset({"caption", "filename", "trigger"})

    def test_missing_values_default_to_empty(self) -> None:
        assert render_template("[{caption}]") == "[]"

    def test_unknown_variable_raises(self) -> None:
        with pytest.raises(ValidationError):
            render_template("hello {unknown_var}")

    def test_positional_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            render_template("hello {0}", caption="x")

    def test_attribute_access_raises(self) -> None:
        with pytest.raises(ValidationError):
            render_template("{caption.upper}", caption="x")

    def test_format_spec_raises(self) -> None:
        with pytest.raises(ValidationError):
            render_template("{caption:>10}", caption="x")

    def test_literal_braces(self) -> None:
        assert render_template("{{not_a_var}}") == "{not_a_var}"
        assert render_template("{{{caption}}}", caption="fox") == "{fox}"

    def test_unbalanced_brace_raises(self) -> None:
        with pytest.raises(ValidationError):
            render_template("hello {caption")

    def test_extra_variables(self) -> None:
        result = render_template(
            "condense to {target_tokens} tokens: {caption}",
            caption="a fox",
            extra={"target_tokens": "60"},
        )
        assert result == "condense to 60 tokens: a fox"

    def test_extra_variable_not_allowed_without_extra(self) -> None:
        with pytest.raises(ValidationError):
            render_template("condense to {target_tokens} tokens")

    def test_non_string_template_raises(self) -> None:
        with pytest.raises(ValidationError):
            render_template(42)  # type: ignore[arg-type]


class TestDefaultTemplates:
    def test_expected_keys(self) -> None:
        assert set(DEFAULT_TEMPLATES) == EXPECTED_DEFAULT_KEYS

    def test_condense_uses_target_tokens(self) -> None:
        assert "{target_tokens}" in DEFAULT_TEMPLATES["condense"]

    def test_all_templates_reference_caption(self) -> None:
        for name, template in DEFAULT_TEMPLATES.items():
            assert "{caption}" in template, f"template {name!r} lacks {{caption}}"

    def test_all_templates_render(self) -> None:
        for name, template in DEFAULT_TEMPLATES.items():
            extra = {"target_tokens": "60"} if name == "condense" else None
            rendered = render_template(template, caption="a fox", extra=extra)
            assert "a fox" in rendered

    def test_mapping_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            DEFAULT_TEMPLATES["polish"] = "hacked"  # type: ignore[index]


class TestTemplateStore:
    def test_save_get_roundtrip(self) -> None:
        store = TemplateStore()
        store.save("mine", "do something with {caption}")
        assert store.get("mine") == "do something with {caption}"

    def test_initial_mapping(self) -> None:
        store = TemplateStore({"a": "template a", "b": "template b"})
        assert store.names() == ("a", "b")

    def test_save_replaces_existing(self) -> None:
        store = TemplateStore({"a": "old"})
        store.save("a", "new")
        assert store.get("a") == "new"
        assert store.names() == ("a",)

    def test_empty_name_raises(self) -> None:
        store = TemplateStore()
        with pytest.raises(ValidationError):
            store.save("", "body")
        with pytest.raises(ValidationError):
            store.save("   ", "body")

    def test_empty_template_raises(self) -> None:
        store = TemplateStore()
        with pytest.raises(ValidationError):
            store.save("name", "")
        with pytest.raises(ValidationError):
            store.save("name", "  \n ")

    def test_get_unknown_raises_key_error(self) -> None:
        store = TemplateStore()
        with pytest.raises(KeyError):
            store.get("nope")

    def test_delete(self) -> None:
        store = TemplateStore({"a": "x", "b": "y"})
        store.delete("a")
        assert store.names() == ("b",)
        with pytest.raises(KeyError):
            store.delete("a")

    def test_as_dict_returns_independent_copy(self) -> None:
        store = TemplateStore({"a": "x"})
        exported = store.as_dict()
        exported["a"] = "mutated"
        exported["b"] = "added"
        assert store.get("a") == "x"
        assert store.names() == ("a",)

    def test_invalid_initial_entry_raises(self) -> None:
        with pytest.raises(ValidationError):
            TemplateStore({"": "body"})
