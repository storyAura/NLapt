"""Tag deduplication operation over comma-separated chips (spec 6.1, 7).

``apply`` pipes the text through ``split_chips -> dedup_chips ->
join_chips`` (keeping the first occurrence of each chip and normalizing
the joiner to ``", "``). ``preview`` lists every removed duplicate as a
``MatchPreview`` located in the ORIGINAL text.
"""

from __future__ import annotations

import re

from nlapt.captions.chips import CHIP_SEPARATORS, dedup_chips, join_chips, split_chips
from nlapt.diagnostics import get_logger
from nlapt.ops.base import (
    OP_DEDUP_TAGS,
    MatchPreview,
    build_match_preview,
    validate_operation_text,
)

_LOGGER = get_logger(__name__)

_SEPARATOR_PATTERN = re.compile("|".join(re.escape(sep) for sep in CHIP_SEPARATORS))


def _chip_spans(text: str) -> tuple[tuple[str, int, int], ...]:
    """Chips with their (start, end) spans in ``text``.

    Mirrors ``split_chips`` semantics (strip each fragment, drop empties)
    while tracking original character offsets.
    """
    spans: list[tuple[str, int, int]] = []
    position = 0
    boundaries = [match.span() for match in _SEPARATOR_PATTERN.finditer(text)]
    for boundary_start, boundary_end in [*boundaries, (len(text), len(text))]:
        fragment = text[position:boundary_start]
        stripped = fragment.strip()
        if stripped:
            leading = len(fragment) - len(fragment.lstrip())
            start = position + leading
            spans.append((stripped, start, start + len(stripped)))
        position = boundary_end
    return tuple(spans)


class DedupTagsOperation:
    """``TextOperation`` removing exact duplicate chips (first occurrence wins)."""

    name = OP_DEDUP_TAGS

    def preview(self, text: str) -> tuple[MatchPreview, ...]:
        """One ``MatchPreview`` per removed duplicate chip, in text order."""
        validate_operation_text(text)
        seen: set[str] = set()
        previews: list[MatchPreview] = []
        for chip, start, end in _chip_spans(text):
            if chip in seen:
                previews.append(build_match_preview(text, start, end, ""))
            else:
                seen.add(chip)
        return tuple(previews)

    def apply(self, text: str) -> str:
        """Deduplicated text, rejoined with ``", "``."""
        validate_operation_text(text)
        return join_chips(dedup_chips(split_chips(text)))
