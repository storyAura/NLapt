"""Word-level diff for the suggestion review bar (spec 10).

The suggestion bar renders original and suggested text merged: deleted
fragments red/struck-through, inserted fragments green. ``word_diff``
tokenizes mixed CJK/English text — whitespace runs, single CJK characters,
and runs of other non-space characters — and aligns the token sequences with
:class:`difflib.SequenceMatcher`. Adjacent segments with the same op are
merged.

Invariants (guaranteed, tested):

- ``"".join(EQUAL + DELETE segments) == old``
- ``"".join(EQUAL + INSERT segments) == new``
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from enum import Enum

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# CJK ideographs, kana, CJK punctuation, and full/half-width forms are diffed
# per character (there is no whitespace word boundary in CJK text).
_CJK_CHAR_CLASS = (
    "["
    "　-〿"  # CJK symbols and punctuation
    "぀-ヿ"  # hiragana + katakana
    "㐀-䶿"  # CJK extension A
    "一-鿿"  # CJK unified ideographs
    "豈-﫿"  # CJK compatibility ideographs
    "＀-￯"  # full-width forms
    "]"
)
# Token = whitespace run | single CJK char | run of non-space non-CJK chars.
# The alternation covers every character, so joining tokens restores the text.
_TOKEN_PATTERN = re.compile(rf"\s+|{_CJK_CHAR_CLASS}|(?:(?!{_CJK_CHAR_CLASS})\S)+")


class DiffOp(str, Enum):
    """Kind of a diff segment."""

    EQUAL = "equal"
    INSERT = "insert"
    DELETE = "delete"


@dataclass(frozen=True)
class DiffSegment:
    """A run of text sharing one diff op; segments never have empty text."""

    op: DiffOp
    text: str


def word_diff(old: str, new: str) -> tuple[DiffSegment, ...]:
    """Word-level diff between ``old`` and ``new`` (mixed CJK/English).

    Returns segments in display order with adjacent same-op segments merged.
    Both empty inputs yield an empty tuple.
    """
    if not isinstance(old, str):
        raise ValidationError(f"old must be a string, got {type(old).__name__}")
    if not isinstance(new, str):
        raise ValidationError(f"new must be a string, got {type(new).__name__}")

    old_tokens = _TOKEN_PATTERN.findall(old)
    new_tokens = _TOKEN_PATTERN.findall(new)
    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)

    segments: list[DiffSegment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            _append_merged(segments, DiffOp.EQUAL, "".join(old_tokens[i1:i2]))
        elif tag == "delete":
            _append_merged(segments, DiffOp.DELETE, "".join(old_tokens[i1:i2]))
        elif tag == "insert":
            _append_merged(segments, DiffOp.INSERT, "".join(new_tokens[j1:j2]))
        else:  # "replace": old fragment removed, new fragment added
            _append_merged(segments, DiffOp.DELETE, "".join(old_tokens[i1:i2]))
            _append_merged(segments, DiffOp.INSERT, "".join(new_tokens[j1:j2]))
    return tuple(segments)


def _append_merged(segments: list[DiffSegment], op: DiffOp, text: str) -> None:
    """Append a segment, merging into the previous one when the op matches."""
    if not text:
        return
    if segments and segments[-1].op is op:
        segments[-1] = DiffSegment(op=op, text=segments[-1].text + text)
        return
    segments.append(DiffSegment(op=op, text=text))
