"""Sentence-level text model for the sentences editing mode (spec 6.2).

Splitting is naive by design (v1): text is cut after runs of terminal
punctuation (``.``, ``!``, ``?``, ``。``, ``！``, ``？``), with the
punctuation kept attached to its sentence and surrounding whitespace
stripped.

Known v1 limitation (documented, not "fixed"): decimals ("3.5 kg") and
abbreviations ("e.g.") contain terminal punctuation and therefore
over-split into multiple pieces. Users repair such splits manually with
:func:`merge_with_previous`.

Join rule (spec 6.2): a sentence ending with ASCII terminal punctuation is
followed by a single space; one ending with CJK terminal punctuation is
followed directly by the next sentence. Sentences without terminal
punctuation are joined with a single space.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

ASCII_ENDINGS: tuple[str, ...] = (".", "!", "?")
CJK_ENDINGS: tuple[str, ...] = ("。", "！", "？")
TERMINAL_ENDINGS: tuple[str, ...] = ASCII_ENDINGS + CJK_ENDINGS

ASCII_JOIN_SEPARATOR = " "
CJK_JOIN_SEPARATOR = ""

_ENDINGS_CLASS = "".join(re.escape(ch) for ch in TERMINAL_ENDINGS)
# A sentence is either "body + run of terminal punctuation" or a trailing
# punctuation-free remainder.
_SENTENCE_PATTERN = re.compile(
    f"[^{_ENDINGS_CLASS}]*[{_ENDINGS_CLASS}]+|[^{_ENDINGS_CLASS}]+"
)


def _validate_str(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be a string, got {type(value).__name__}")


def _validate_index(index: int, length: int, label: str) -> None:
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValidationError(f"{label} must be an integer, got {index!r}")
    if not 0 <= index < length:
        raise ValidationError(f"{label} {index} out of range for {length} sentences")


def _separator_after(sentence: str) -> str:
    """Separator inserted after ``sentence`` when joining (spec 6.2 rule)."""
    if sentence and sentence[-1] in CJK_ENDINGS:
        return CJK_JOIN_SEPARATOR
    return ASCII_JOIN_SEPARATOR


def split_sentences(text: str) -> tuple[str, ...]:
    """Split text after runs of terminal punctuation, keeping it attached.

    Fragments are stripped; empty fragments are dropped. See the module
    docstring for the known v1 over-splitting limitation.
    """
    _validate_str(text, "text")
    pieces = (match.group(0).strip() for match in _SENTENCE_PATTERN.finditer(text))
    return tuple(piece for piece in pieces if piece)


def join_sentences(sentences: Sequence[str]) -> str:
    """Rejoin sentences: single space after ASCII endings, none after CJK."""
    cleaned: list[str] = []
    for sentence in sentences:
        _validate_str(sentence, "sentence")
        stripped = sentence.strip()
        if stripped:
            cleaned.append(stripped)
    if not cleaned:
        return ""
    parts: list[str] = [cleaned[0]]
    for sentence in cleaned[1:]:
        parts.append(_separator_after(parts[-1][-1:]))
        parts.append(sentence)
    return "".join(parts)


def merge_with_previous(sentences: Sequence[str], index: int) -> tuple[str, ...]:
    """Merge the sentence at ``index`` into the one before it.

    The two pieces are joined with the standard join rule (space after an
    ASCII ending, nothing after a CJK ending). Raises ``ValidationError``
    when ``index <= 0`` (there is no previous sentence) or out of range.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValidationError(f"index must be an integer, got {index!r}")
    if index <= 0:
        raise ValidationError(f"cannot merge sentence {index} with previous: no previous sentence")
    _validate_index(index, len(sentences), "index")
    previous = sentences[index - 1]
    current = sentences[index]
    _validate_str(previous, "sentence")
    _validate_str(current, "sentence")
    merged = f"{previous}{_separator_after(previous)}{current}"
    return tuple(sentences[: index - 1]) + (merged,) + tuple(sentences[index + 1 :])


def move_sentence(sentences: Sequence[str], src: int, dst: int) -> tuple[str, ...]:
    """Move the sentence at ``src`` so it ends up at position ``dst``."""
    _validate_index(src, len(sentences), "src")
    _validate_index(dst, len(sentences), "dst")
    items = list(sentences)
    item = items.pop(src)
    items.insert(dst, item)
    return tuple(items)


def replace_sentence(sentences: Sequence[str], index: int, text: str) -> tuple[str, ...]:
    """Replace the sentence at ``index`` with ``text`` (stripped).

    The replacement is kept as a single sentence (no re-splitting). Raises
    ``ValidationError`` for an out-of-range index or empty/whitespace text
    (use :func:`delete_sentence` to remove a sentence).
    """
    _validate_index(index, len(sentences), "index")
    _validate_str(text, "text")
    stripped = text.strip()
    if not stripped:
        raise ValidationError("replacement text must not be empty; use delete_sentence to remove")
    return tuple(sentences[:index]) + (stripped,) + tuple(sentences[index + 1 :])


def delete_sentence(sentences: Sequence[str], index: int) -> tuple[str, ...]:
    """Remove the sentence at ``index``."""
    _validate_index(index, len(sentences), "index")
    return tuple(sentences[:index]) + tuple(sentences[index + 1 :])
