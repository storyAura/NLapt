"""CLIP token estimation for the caption editor footer (spec 6.5).

Provides a deterministic heuristic approximation of the CLIP BPE tokenizer.
The estimate is advisory only ("~ n tokens"): exceeding
:data:`CLIP_TOKEN_LIMIT` warns the user but never blocks editing.

Heuristic rules of :class:`HeuristicClipEstimator` (documented, deterministic):

1. CJK characters (Han ideographs, Hiragana, Katakana) count 1 token each —
   CLIP BPE has no CJK merges, so CJK text tokenizes roughly per character.
2. Alphanumeric word runs (``[A-Za-z0-9]+``) count
   ``max(1, ceil(len(word) / WORD_CHARS_PER_TOKEN))`` tokens — short common
   words are single BPE tokens, long words split into several sub-word units.
3. Every other non-whitespace character (commas — half- and full-width —
   periods, symbols, non-CJK unicode letters) counts 1 token each, so
   comma-separated tag strings include their separators in the estimate.
4. Whitespace contributes nothing; the empty string estimates to 0.

``get_default_estimator`` is the future hook for plugging in a real CLIP
tokenizer without changing call sites.
"""

from __future__ import annotations

import math
import re
from typing import Protocol, runtime_checkable

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# CLIP text encoders truncate at 77 tokens (spec 6.5).
CLIP_TOKEN_LIMIT = 77

# Average characters per BPE sub-word unit for alphanumeric words (heuristic).
WORD_CHARS_PER_TOKEN = 5

# CJK ranges treated as one-token-per-character:
# Han (base + extension A + compatibility) and Japanese kana.
_CJK_RANGES = "一-鿿㐀-䶿豈-﫿぀-ヿ"
# Order matters: single CJK char, then alphanumeric word run, then any other
# single non-whitespace character.
_TOKEN_PATTERN = re.compile(f"(?P<cjk>[{_CJK_RANGES}])|(?P<word>[A-Za-z0-9]+)|(?P<other>\\S)")


@runtime_checkable
class TokenEstimator(Protocol):
    """Anything that maps caption text to an estimated token count."""

    def estimate(self, text: str) -> int: ...


class HeuristicClipEstimator:
    """Deterministic CLIP-BPE approximation (rules in the module docstring)."""

    def estimate(self, text: str) -> int:
        """Estimated CLIP token count for ``text``; ``estimate("") == 0``."""
        if not isinstance(text, str):
            raise ValidationError(f"text must be a string, got {type(text).__name__}")
        total = 0
        for match in _TOKEN_PATTERN.finditer(text):
            if match.lastgroup == "word":
                total += max(1, math.ceil(len(match.group("word")) / WORD_CHARS_PER_TOKEN))
            else:
                # One token per CJK character and per punctuation/symbol char.
                total += 1
        return total


_DEFAULT_ESTIMATOR: TokenEstimator = HeuristicClipEstimator()


def get_default_estimator() -> TokenEstimator:
    """Return the default estimator (heuristic; future real-CLIP plugin hook)."""
    return _DEFAULT_ESTIMATOR


def is_over_limit(count: int, limit: int = CLIP_TOKEN_LIMIT) -> bool:
    """True when ``count`` exceeds ``limit`` (advisory warning only)."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValidationError(f"count must be a non-negative integer, got {count!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValidationError(f"limit must be a positive integer, got {limit!r}")
    return count > limit
