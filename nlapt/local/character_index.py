"""Danbooru character-tag index (parent / child / copyright lookup).

Loaded from ``danbooru_character_tags.csv`` (stdlib csv). Tagger tags use
spaces; the CSV uses underscores — :func:`normalize_tag` bridges them.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from nlapt.core.errors import LocalInferenceError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

MSG_BAD_CSV = "角色对照表损坏: {path}"
_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*$")
_SERIES_SUFFIX_RE = re.compile(r"\s*\(series\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class CharacterEntry:
    """One Danbooru character tag and its wiki fields."""

    tag: str
    other_names: str = ""
    copyright: str = ""
    parent_tag: str = ""
    post_count: int = 0


class CharacterIndex:
    """In-memory lookup of character tags, parents, and children."""

    def __init__(self, entries: tuple[CharacterEntry, ...]) -> None:
        self._by_tag: dict[str, CharacterEntry] = {}
        children: dict[str, list[CharacterEntry]] = {}
        for entry in entries:
            self._by_tag[entry.tag] = entry
            parent = entry.parent_tag
            if parent:
                children.setdefault(parent, []).append(entry)
        self._children = {key: tuple(value) for key, value in children.items()}

    @classmethod
    def load(cls, csv_path: Path) -> CharacterIndex:
        """Parse the character CSV (header required)."""
        path = Path(csv_path)
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = [_entry_from_row(row) for row in reader]
        except (OSError, csv.Error, UnicodeError) as exc:
            raise LocalInferenceError(MSG_BAD_CSV.format(path=path)) from exc
        index = cls(tuple(row for row in rows if row is not None))
        _LOGGER.info("character index loaded (%s tags) from %s", len(index._by_tag), path)
        return index

    def lookup(self, tag: str) -> CharacterEntry | None:
        """Exact lookup after :func:`normalize_tag`."""
        return self._by_tag.get(normalize_tag(tag))

    def children_of(self, tag: str) -> tuple[CharacterEntry, ...]:
        """Direct children of ``tag`` (already normalized)."""
        return self._children.get(normalize_tag(tag), ())

    def family(self, tag: str) -> tuple[CharacterEntry, ...]:
        """Root parent + every descendant + self, by ``post_count`` desc."""
        start = self.lookup(tag)
        if start is None:
            return ()
        root = start
        seen: set[str] = {start.tag}
        while root.parent_tag:
            parent = self.lookup(root.parent_tag)
            if parent is None or parent.tag in seen:
                break
            seen.add(parent.tag)
            root = parent
        collected: dict[str, CharacterEntry] = {}
        stack = [root]
        while stack:
            current = stack.pop()
            if current.tag in collected:
                continue
            collected[current.tag] = current
            stack.extend(self._children.get(current.tag, ()))
        if start.tag not in collected:
            collected[start.tag] = start
        return tuple(sorted(collected.values(), key=lambda item: item.post_count, reverse=True))


def normalize_tag(text: str) -> str:
    """Danbooru form: lowercase, spaces → underscores."""
    return text.strip().lower().replace(" ", "_")


def display_name(tag: str) -> str:
    """Human name: underscores → spaces, drop a trailing ``(qualifier)``."""
    text = tag.strip().replace("_", " ")
    text = _QUALIFIER_RE.sub("", text).strip()
    return text.title() if text else tag


def display_series(copyright: str) -> str:
    """Human series name; strips a trailing ``_(series)`` / `` (series)``."""
    text = copyright.strip().replace("_", " ")
    text = _SERIES_SUFFIX_RE.sub("", text).strip()
    return text.title() if text else ""


def _entry_from_row(row: dict[str, str | None]) -> CharacterEntry | None:
    tag = normalize_tag(str(row.get("character_tag") or ""))
    if not tag:
        return None
    raw_count = str(row.get("post_count") or "0").strip() or "0"
    try:
        count = int(raw_count)
    except ValueError:
        count = 0
    return CharacterEntry(
        tag=tag,
        other_names=str(row.get("other_names") or "").strip(),
        copyright=normalize_tag(str(row.get("copyright") or "")),
        parent_tag=normalize_tag(str(row.get("parent_tag") or "")),
        post_count=count,
    )
