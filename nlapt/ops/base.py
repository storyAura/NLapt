"""Text operation protocol, match previews, and the operation registry.

Contract: docs/ARCHITECTURE.md, section ``nlapt.ops.base``. A
``TextOperation`` is a pure, deterministic text transform; the
``OperationRegistry`` is the extensibility point for creating operations
by name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Maximum characters of surrounding context captured in a MatchPreview.
CONTEXT_CHARS = 20

# Names of the built-in operations pre-registered by get_default_registry().
OP_FIND_REPLACE = "find_replace"
OP_PREFIX_SUFFIX = "prefix_suffix"
OP_DEDUP_TAGS = "dedup_tags"


@dataclass(frozen=True)
class MatchPreview:
    """One hit of an operation inside a text, with capped context."""

    start: int
    end: int
    matched: str
    replacement: str
    context_before: str  # up to CONTEXT_CHARS chars
    context_after: str


@runtime_checkable
class TextOperation(Protocol):
    """Pure text transform. ``apply()`` must be deterministic and side-effect free."""

    name: str

    def preview(self, text: str) -> tuple[MatchPreview, ...]: ...

    def apply(self, text: str) -> str: ...


def validate_operation_text(text: str) -> None:
    """Boundary check shared by operation implementations."""
    if not isinstance(text, str):
        raise ValidationError(f"text must be a string, got {type(text).__name__}")


def build_match_preview(text: str, start: int, end: int, replacement: str) -> MatchPreview:
    """Build a ``MatchPreview`` for ``text[start:end]`` with capped context."""
    if not 0 <= start <= end <= len(text):
        raise ValidationError(
            f"invalid preview span ({start}, {end}) for text of length {len(text)}"
        )
    return MatchPreview(
        start=start,
        end=end,
        matched=text[start:end],
        replacement=replacement,
        context_before=text[max(0, start - CONTEXT_CHARS) : start],
        context_after=text[end : end + CONTEXT_CHARS],
    )


class OperationRegistry:
    """Extensibility point: register/create text operations by name."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., TextOperation]] = {}

    def register(self, name: str, factory: Callable[..., TextOperation]) -> None:
        """Register ``factory`` under ``name``.

        Raises ``ValidationError`` for an empty name, a non-callable
        factory, or a duplicate registration.
        """
        if not isinstance(name, str) or not name:
            raise ValidationError(f"operation name must be a non-empty string, got {name!r}")
        if not callable(factory):
            raise ValidationError(f"factory for {name!r} must be callable")
        if name in self._factories:
            raise ValidationError(f"operation {name!r} is already registered")
        self._factories[name] = factory
        _LOGGER.debug("registered operation %r", name)

    def create(self, name: str, **params: Any) -> TextOperation:
        """Instantiate the operation registered under ``name``.

        Raises ``ValidationError`` for an unknown name or parameters the
        factory rejects.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise ValidationError(f"unknown operation: {name!r}")
        try:
            return factory(**params)
        except TypeError as exc:
            raise ValidationError(
                f"invalid parameters for operation {name!r}: {exc}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        """Registered operation names in registration order."""
        return tuple(self._factories)


def get_default_registry() -> OperationRegistry:
    """Fresh registry pre-registered with the built-in operations.

    Imports are local to keep ``nlapt.ops.base`` free of circular imports
    (the operation modules import this module for ``MatchPreview``).
    """
    from nlapt.ops.dedup import DedupTagsOperation
    from nlapt.ops.find_replace import FindReplaceOperation, FindReplaceSpec
    from nlapt.ops.prefix_suffix import PrefixSuffixOperation, PrefixSuffixSpec

    def _make_find_replace(**params: Any) -> TextOperation:
        return FindReplaceOperation(FindReplaceSpec(**params))

    def _make_prefix_suffix(**params: Any) -> TextOperation:
        return PrefixSuffixOperation(PrefixSuffixSpec(**params))

    def _make_dedup(**params: Any) -> TextOperation:
        return DedupTagsOperation(**params)

    registry = OperationRegistry()
    registry.register(OP_FIND_REPLACE, _make_find_replace)
    registry.register(OP_PREFIX_SUFFIX, _make_prefix_suffix)
    registry.register(OP_DEDUP_TAGS, _make_dedup)
    return registry
