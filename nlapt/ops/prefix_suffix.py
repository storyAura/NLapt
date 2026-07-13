"""Prefix/suffix text operation and trigger-word helpers (spec 7.2).

Adds a prefix and/or suffix with a selectable joiner. ``skip_if_present``
(default on) prevents trigger words from being stacked twice. Removal
helpers match the exact "prefix + joiner" text, and also the bare
prefix/suffix when it constitutes the whole caption.
"""

from __future__ import annotations

from dataclasses import dataclass

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger
from nlapt.ops.base import (
    OP_PREFIX_SUFFIX,
    MatchPreview,
    TextOperation,
    build_match_preview,
    validate_operation_text,
)

_LOGGER = get_logger(__name__)

# Allowed joiners between the added text and the caption body (spec 7.2).
JOINERS: tuple[str, ...] = (", ", " ", "")

# Registry name for the trigger-word removal operation.
OP_TRIGGER_REMOVE = "trigger_remove"


def _validate_joiner(joiner: str) -> None:
    if joiner not in JOINERS:
        raise ValidationError(f"joiner must be one of {JOINERS!r}, got {joiner!r}")


def _validate_fragment(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be a string, got {type(value).__name__}")


@dataclass(frozen=True)
class PrefixSuffixSpec:
    """User input for a prefix/suffix operation."""

    prefix: str = ""
    suffix: str = ""
    joiner: str = ", "
    skip_if_present: bool = True

    def __post_init__(self) -> None:
        _validate_fragment(self.prefix, "prefix")
        _validate_fragment(self.suffix, "suffix")
        _validate_fragment(self.joiner, "joiner")
        _validate_joiner(self.joiner)


def _prefix_present(text: str, prefix: str, joiner: str) -> bool:
    """Boundary-aware presence test for ``skip_if_present`` (spec 7.2).

    With a non-empty joiner the prefix counts as present only when it is the
    whole caption or is followed by the joiner — a caption merely beginning
    with a longer word (``caterpillar`` vs trigger ``cat``) must still receive
    the trigger. An empty joiner keeps the raw ``startswith`` semantics
    (explicitly chosen by the user).
    """
    if joiner == "":
        return text.startswith(prefix)
    return text == prefix or text.startswith(prefix + joiner)


def _suffix_present(text: str, suffix: str, joiner: str) -> bool:
    """Boundary-aware suffix counterpart of :func:`_prefix_present`."""
    if joiner == "":
        return text.endswith(suffix)
    return text == suffix or text.endswith(joiner + suffix)


class PrefixSuffixOperation:
    """``TextOperation`` implementation for prefix/suffix (spec 7.2).

    Empty text receives the bare prefix (or suffix) without a joiner.
    """

    name = OP_PREFIX_SUFFIX

    def __init__(self, spec: PrefixSuffixSpec) -> None:
        if not isinstance(spec, PrefixSuffixSpec):
            raise ValidationError(
                f"spec must be a PrefixSuffixSpec, got {type(spec).__name__}"
            )
        self._spec = spec

    @property
    def spec(self) -> PrefixSuffixSpec:
        return self._spec

    def _prefix_insertion(self, text: str) -> str | None:
        """Text to prepend, or ``None`` when nothing should be added."""
        spec = self._spec
        if not spec.prefix:
            return None
        if spec.skip_if_present and _prefix_present(text, spec.prefix, spec.joiner):
            return None
        if not text:
            return spec.prefix
        return spec.prefix + spec.joiner

    def _suffix_insertion(self, text: str) -> str | None:
        """Text to append, or ``None`` when nothing should be added."""
        spec = self._spec
        if not spec.suffix:
            return None
        if spec.skip_if_present and _suffix_present(text, spec.suffix, spec.joiner):
            return None
        if not text:
            return spec.suffix
        return spec.joiner + spec.suffix

    def preview(self, text: str) -> tuple[MatchPreview, ...]:
        """Zero-width insertion previews at the start and/or end of ``text``.

        Skipped insertions (already present) produce no preview. The
        suffix insertion is computed against the prefixed intermediate
        text so replacements mirror ``apply`` exactly.
        """
        validate_operation_text(text)
        previews: list[MatchPreview] = []
        prefix_insertion = self._prefix_insertion(text)
        intermediate = text if prefix_insertion is None else prefix_insertion + text
        if prefix_insertion is not None:
            previews.append(build_match_preview(text, 0, 0, prefix_insertion))
        suffix_insertion = self._suffix_insertion(intermediate)
        if suffix_insertion is not None:
            previews.append(build_match_preview(text, len(text), len(text), suffix_insertion))
        return tuple(previews)

    def apply(self, text: str) -> str:
        """Add prefix then suffix according to the spec."""
        validate_operation_text(text)
        prefix_insertion = self._prefix_insertion(text)
        result = text if prefix_insertion is None else prefix_insertion + text
        suffix_insertion = self._suffix_insertion(result)
        if suffix_insertion is not None:
            result = result + suffix_insertion
        return result


def _removed_prefix_length(text: str, prefix: str, joiner: str) -> int:
    """Length of the leading span an exact-prefix removal would delete (0 if none)."""
    full = prefix + joiner
    if text.startswith(full):
        return len(full)
    if text == prefix:
        return len(text)
    return 0


def remove_exact_prefix(text: str, prefix: str, joiner: str) -> str:
    """Remove a leading exact ``prefix + joiner`` match from ``text``.

    Also removes a bare ``prefix`` when it is the whole text. Returns
    ``text`` unchanged when there is no exact match.
    """
    validate_operation_text(text)
    _validate_fragment(joiner, "joiner")
    _validate_joiner(joiner)
    if not isinstance(prefix, str) or not prefix:
        raise ValidationError(f"prefix must be a non-empty string, got {prefix!r}")
    return text[_removed_prefix_length(text, prefix, joiner) :]


def remove_exact_suffix(text: str, suffix: str, joiner: str) -> str:
    """Remove a trailing exact ``joiner + suffix`` match from ``text``.

    Also removes a bare ``suffix`` when it is the whole text. Returns
    ``text`` unchanged when there is no exact match.
    """
    validate_operation_text(text)
    _validate_fragment(joiner, "joiner")
    _validate_joiner(joiner)
    if not isinstance(suffix, str) or not suffix:
        raise ValidationError(f"suffix must be a non-empty string, got {suffix!r}")
    full = joiner + suffix
    if text.endswith(full):
        return text[: len(text) - len(full)]
    if text == suffix:
        return ""
    return text


class _TriggerRemoveOperation:
    """``TextOperation`` removing an exact leading trigger word."""

    name = OP_TRIGGER_REMOVE

    def __init__(self, trigger: str, joiner: str) -> None:
        self._trigger = trigger
        self._joiner = joiner

    def preview(self, text: str) -> tuple[MatchPreview, ...]:
        validate_operation_text(text)
        removed = _removed_prefix_length(text, self._trigger, self._joiner)
        if removed == 0:
            return ()
        return (build_match_preview(text, 0, removed, ""),)

    def apply(self, text: str) -> str:
        return remove_exact_prefix(text, self._trigger, self._joiner)


def _validate_trigger(trigger: str) -> None:
    if not isinstance(trigger, str) or not trigger.strip():
        raise ValidationError(f"trigger must be a non-empty string, got {trigger!r}")


def make_trigger_add_op(trigger: str, joiner: str = ", ") -> TextOperation:
    """Operation that prepends ``trigger`` once (skips when already present)."""
    _validate_trigger(trigger)
    spec = PrefixSuffixSpec(prefix=trigger, joiner=joiner, skip_if_present=True)
    return PrefixSuffixOperation(spec)


def make_trigger_remove_op(trigger: str, joiner: str = ", ") -> TextOperation:
    """Operation that removes an exact leading ``trigger + joiner`` (or bare trigger)."""
    _validate_trigger(trigger)
    _validate_fragment(joiner, "joiner")
    _validate_joiner(joiner)
    return _TriggerRemoveOperation(trigger, joiner)
