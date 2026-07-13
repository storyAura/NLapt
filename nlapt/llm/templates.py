"""Prompt templates with variable interpolation (spec 7.4).

Templates use ``{caption}`` / ``{filename}`` / ``{trigger}`` placeholders;
``{{`` and ``}}`` are literal braces. The condense template additionally uses
``{target_tokens}``, supplied by the rewrite service via ``extra``.
"""

from __future__ import annotations

import string
import threading
from collections.abc import Mapping
from types import MappingProxyType

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

ALLOWED_VARIABLES: frozenset[str] = frozenset({"caption", "filename", "trigger"})

# Extra variable used only by the condense template (substituted via `extra`).
TARGET_TOKENS_VARIABLE = "target_tokens"

DEFAULT_TEMPLATES: Mapping[str, str] = MappingProxyType(
    {
        "polish": (
            "Polish the following image caption. Fix grammar and improve fluency "
            "while preserving as much of the original information as possible.\n\n"
            "Caption: {caption}"
        ),
        "rewrite": (
            "Rewrite the following image caption. Restructure the sentences and "
            "unify the style of expression while keeping the original meaning.\n\n"
            "Caption: {caption}"
        ),
        "expand": (
            "Expand the following image caption by adding descriptive detail such "
            "as composition, lighting, and atmosphere.\n\n"
            "Caption: {caption}"
        ),
        "condense": (
            "Condense the following image caption to at most {target_tokens} "
            "tokens, keeping the most important visual information.\n\n"
            "Caption: {caption}"
        ),
        "translate_en_zh": (
            "Translate the following image caption from English to Chinese.\n\n"
            "Caption: {caption}"
        ),
        "translate_zh_en": (
            "Translate the following image caption from Chinese to English.\n\n"
            "Caption: {caption}"
        ),
        # Target-language translation (中/英/日 caption workspace, spec module 1):
        # the source language is whatever the caption currently is.
        "translate_to_zh": (
            "Translate the following image caption into Simplified Chinese. "
            "Keep the meaning, tag structure and comma separation intact.\n\n"
            "Caption: {caption}"
        ),
        "translate_to_en": (
            "Translate the following image caption into English. "
            "Keep the meaning, tag structure and comma separation intact.\n\n"
            "Caption: {caption}"
        ),
        "translate_to_ja": (
            "Translate the following image caption into Japanese. "
            "Keep the meaning, tag structure and comma separation intact.\n\n"
            "Caption: {caption}"
        ),
    }
)


def render_template(
    template: str,
    *,
    caption: str = "",
    filename: str = "",
    trigger: str = "",
    extra: Mapping[str, str] | None = None,
) -> str:
    """Interpolate ``{var}`` placeholders; unknown variables -> ValidationError."""
    if not isinstance(template, str):
        raise ValidationError(
            f"template must be a string, got {type(template).__name__}"
        )
    values: dict[str, str] = {
        "caption": caption,
        "filename": filename,
        "trigger": trigger,
    }
    if extra:
        for key, value in extra.items():
            if not isinstance(key, str) or not key:
                raise ValidationError("extra variable names must be non-empty strings")
            values[key] = str(value)
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValidationError(f"invalid template syntax: {exc}") from exc
    parts: list[str] = []
    for literal, field_name, format_spec, conversion in parsed:
        parts.append(literal)
        if field_name is None:
            continue
        if conversion or format_spec:
            raise ValidationError(
                f"template variable {{{field_name}}} must not use conversions "
                "or format specs"
            )
        if field_name not in values:
            allowed = ", ".join(sorted(values))
            raise ValidationError(
                f"unknown template variable {{{field_name}}}; allowed: {allowed}"
            )
        parts.append(values[field_name])
    return "".join(parts)


class TemplateStore:
    """Named custom templates (persisted through ``AppConfig.custom_templates``).

    A sanctioned mutable store: entries are replaced, never mutated in place.
    """

    def __init__(self, initial: Mapping[str, str] | None = None) -> None:
        self._lock = threading.Lock()
        self._templates: dict[str, str] = {}
        if initial:
            for name, template in initial.items():
                self.save(name, template)

    def save(self, name: str, template: str) -> None:
        """Add or replace a named template. Empty name/template -> ValidationError."""
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("template name must be a non-empty string")
        if not isinstance(template, str) or not template.strip():
            raise ValidationError("template text must be a non-empty string")
        with self._lock:
            self._templates[name] = template
        _LOGGER.debug("saved template %r (%d chars)", name, len(template))

    def get(self, name: str) -> str:
        with self._lock:
            return self._templates[name]

    def delete(self, name: str) -> None:
        with self._lock:
            del self._templates[name]
        _LOGGER.debug("deleted template %r", name)

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._templates)

    def as_dict(self) -> dict[str, str]:
        with self._lock:
            return dict(self._templates)
