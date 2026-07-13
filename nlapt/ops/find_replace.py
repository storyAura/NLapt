"""Find/replace text operation (spec 7.1).

Supports case sensitivity, whole-word matching, and regular expressions.
Non-regex mode escapes all metacharacters; regex replacement templates
(``\\1`` / ``\\g<1>``) are honored ONLY in regex mode — otherwise the
replacement string is inserted literally.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from nlapt.core.errors import RegexPatternError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.ops.base import (
    OP_FIND_REPLACE,
    MatchPreview,
    build_match_preview,
    validate_operation_text,
)

_LOGGER = get_logger(__name__)

# Wrapper enforcing whole-word matches; the non-capturing group keeps user
# regex group numbering intact.
_WHOLE_WORD_TEMPLATE = r"\b(?:{pattern})\b"


@dataclass(frozen=True)
class FindReplaceSpec:
    """User input for a find/replace operation."""

    find: str
    replace: str
    case_sensitive: bool = False
    whole_word: bool = False
    regex: bool = False


def compile_spec(spec: FindReplaceSpec) -> re.Pattern[str]:
    """Compile ``spec`` into a regex pattern.

    Raises ``ValidationError`` if ``find`` is empty and
    ``RegexPatternError`` (carrying pattern and detail) if the regex does
    not compile. Non-regex input is escaped; ``whole_word`` wraps the
    pattern in ``\\b...\\b`` in both modes.
    """
    if not isinstance(spec.find, str) or not isinstance(spec.replace, str):
        raise ValidationError("find and replace must be strings")
    if not spec.find:
        raise ValidationError("find text must not be empty")

    pattern = spec.find if spec.regex else re.escape(spec.find)
    if spec.whole_word:
        pattern = _WHOLE_WORD_TEMPLATE.format(pattern=pattern)
    flags = 0 if spec.case_sensitive else re.IGNORECASE
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise RegexPatternError(
            f"invalid regex pattern {spec.find!r}: {exc}",
            pattern=spec.find,
            detail=str(exc),
        ) from exc


class FindReplaceOperation:
    """``TextOperation`` implementation for find/replace (spec 7.1)."""

    name = OP_FIND_REPLACE

    def __init__(self, spec: FindReplaceSpec) -> None:
        self._spec = spec
        self._pattern = compile_spec(spec)  # fail fast on bad input

    @property
    def spec(self) -> FindReplaceSpec:
        return self._spec

    def _expand(self, match: re.Match[str]) -> str:
        """Replacement text for one match.

        Regex mode expands group references (``\\1`` / ``\\g<1>``);
        otherwise the replacement is literal.
        """
        if not self._spec.regex:
            return self._spec.replace
        try:
            return match.expand(self._spec.replace)
        except (re.error, IndexError) as exc:
            raise RegexPatternError(
                f"invalid replacement template {self._spec.replace!r}: {exc}",
                pattern=self._spec.replace,
                detail=str(exc),
            ) from exc

    def preview(self, text: str) -> tuple[MatchPreview, ...]:
        """All matches with their computed replacements and capped context."""
        validate_operation_text(text)
        return tuple(
            build_match_preview(text, match.start(), match.end(), self._expand(match))
            for match in self._pattern.finditer(text)
        )

    def apply(self, text: str) -> str:
        """Replace every match in ``text``."""
        validate_operation_text(text)
        return self._pattern.sub(self._expand, text)

    def apply_one(self, text: str, match_index: int) -> str:
        """Replace exactly the ``match_index``-th match (0-based).

        Raises ``ValidationError`` when the index does not address a match.
        """
        validate_operation_text(text)
        if not isinstance(match_index, int) or isinstance(match_index, bool):
            raise ValidationError(f"match_index must be an integer, got {match_index!r}")
        matches = list(self._pattern.finditer(text))
        if not 0 <= match_index < len(matches):
            raise ValidationError(
                f"match_index {match_index} out of range for {len(matches)} matches"
            )
        match = matches[match_index]
        return text[: match.start()] + self._expand(match) + text[match.end() :]


@dataclass(frozen=True)
class ScopeHitStats:
    """Two-level hit statistics for the live scope indicator (spec 7.1)."""

    current_file_hits: int
    total_hits: int
    files_with_hits: int


def scope_stats(
    spec: FindReplaceSpec,
    current_key: str | None,
    texts: Mapping[str, str],
) -> ScopeHitStats:
    """Count hits for ``spec`` across ``texts`` (key -> caption).

    ``current_file_hits`` is 0 when ``current_key`` is ``None`` or absent
    from ``texts``.
    """
    if not isinstance(texts, Mapping):
        raise ValidationError(f"texts must be a mapping, got {type(texts).__name__}")
    pattern = compile_spec(spec)
    current_hits = 0
    total_hits = 0
    files_with_hits = 0
    for key, text in texts.items():
        validate_operation_text(text)
        hits = sum(1 for _ in pattern.finditer(text))
        total_hits += hits
        if hits:
            files_with_hits += 1
        if key == current_key:
            current_hits = hits
    return ScopeHitStats(
        current_file_hits=current_hits,
        total_hits=total_hits,
        files_with_hits=files_with_hits,
    )
