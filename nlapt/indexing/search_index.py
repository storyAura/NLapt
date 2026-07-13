"""In-memory full-text search index over captions and filenames (spec 4.2).

Case-insensitive substring matching. A query is whitespace-separated terms
combined with AND. Terms prefixed with ``-`` are negations that check the
caption content only (never the filename); positive terms match filename OR
caption. Updates are thread-safe and reflected immediately.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Prefix marking a negated query term ("-foo" -> caption must NOT contain foo).
NEGATION_PREFIX = "-"
# POSIX-style key separator; the filename is the last path segment.
KEY_SEPARATOR = "/"


@dataclass(frozen=True)
class _IndexEntry:
    """Pre-lowered searchable fields for one key."""

    caption_lower: str
    filename_lower: str


def _validate_key(key: str) -> None:
    if not isinstance(key, str) or not key:
        raise ValidationError(f"index key must be a non-empty string, got {key!r}")


def _validate_text(text: str, label: str) -> None:
    if not isinstance(text, str):
        raise ValidationError(f"{label} must be a string, got {type(text).__name__}")


def _make_entry(key: str, text: str) -> _IndexEntry:
    filename = key.rsplit(KEY_SEPARATOR, 1)[-1]
    return _IndexEntry(caption_lower=text.lower(), filename_lower=filename.lower())


def _parse_query(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ``query`` into (positive, negated) lowered terms.

    A lone ``-`` (negation with an empty body) is ignored so that a
    half-typed query never filters everything out.
    """
    positives: list[str] = []
    negations: list[str] = []
    for term in query.split():
        if term.startswith(NEGATION_PREFIX):
            body = term[len(NEGATION_PREFIX) :]
            if body:
                negations.append(body.lower())
        else:
            positives.append(term.lower())
    return tuple(positives), tuple(negations)


def _matches(entry: _IndexEntry, positives: tuple[str, ...], negations: tuple[str, ...]) -> bool:
    for term in positives:
        if term not in entry.caption_lower and term not in entry.filename_lower:
            return False
    for term in negations:
        if term in entry.caption_lower:
            return False
    return True


class SearchIndex:
    """In-memory full-text index over captions + filenames.

    Keys are POSIX-style relative image paths; the filename searched by
    positive terms is the last path segment. All mutating and querying
    methods are guarded by an ``RLock``; entries are replaced, never
    mutated in place.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, _IndexEntry] = {}

    def build(self, entries: Mapping[str, str]) -> None:
        """Replace the whole index with ``entries`` (key -> caption text)."""
        if not isinstance(entries, Mapping):
            raise ValidationError(
                f"entries must be a mapping, got {type(entries).__name__}"
            )
        built: dict[str, _IndexEntry] = {}
        for key, text in entries.items():
            _validate_key(key)
            _validate_text(text, f"caption for {key!r}")
            built[key] = _make_entry(key, text)
        with self._lock:
            self._entries = built
        _LOGGER.debug("search index built with %d entries", len(built))

    def update(self, key: str, text: str) -> None:
        """Insert or replace the caption text for ``key``.

        An existing key keeps its position in index order; a new key is
        appended at the end.
        """
        _validate_key(key)
        _validate_text(text, f"caption for {key!r}")
        entry = _make_entry(key, text)
        with self._lock:
            self._entries[key] = entry

    def remove(self, key: str) -> None:
        """Remove ``key`` from the index. Removing an absent key is a no-op."""
        _validate_key(key)
        with self._lock:
            removed = self._entries.pop(key, None)
        if removed is None:
            _LOGGER.debug("remove ignored for unknown key %r", key)

    def query(self, query: str) -> tuple[str, ...]:
        """Return matching keys in index order.

        Whitespace-separated terms; all must match (AND). ``-foo`` means
        the caption must NOT contain ``foo`` (negation checks caption
        content only); positive terms match filename OR caption. An
        empty/blank query returns all keys.
        """
        _validate_text(query, "query")
        positives, negations = _parse_query(query)
        with self._lock:
            if not positives and not negations:
                return tuple(self._entries)
            return tuple(
                key
                for key, entry in self._entries.items()
                if _matches(entry, positives, negations)
            )
