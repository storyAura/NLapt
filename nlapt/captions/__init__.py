"""Caption text model: store, chips, sentences, and token estimation.

Re-exports the public API of the ``nlapt.captions`` package (contract:
docs/ARCHITECTURE.md, sections ``nlapt.captions.*``).
"""

from __future__ import annotations

from nlapt.captions.chips import (
    CHIP_SEPARATORS,
    SENTENCE_ENDINGS,
    count_tag_in_texts,
    dedup_chips,
    edit_chip,
    find_duplicates,
    is_long_sentence,
    join_chips,
    move_chip,
    split_chips,
)
from nlapt.captions.sentences import (
    delete_sentence,
    join_sentences,
    merge_with_previous,
    move_sentence,
    replace_sentence,
    split_sentences,
)
from nlapt.captions.store import CaptionRecord, CaptionStore, PendingSuggestion
from nlapt.captions.tokens import (
    CLIP_TOKEN_LIMIT,
    HeuristicClipEstimator,
    TokenEstimator,
    get_default_estimator,
    is_over_limit,
)

__all__ = [
    # store
    "CaptionRecord",
    "CaptionStore",
    "PendingSuggestion",
    # chips
    "CHIP_SEPARATORS",
    "SENTENCE_ENDINGS",
    "count_tag_in_texts",
    "dedup_chips",
    "edit_chip",
    "find_duplicates",
    "is_long_sentence",
    "join_chips",
    "move_chip",
    "split_chips",
    # sentences
    "delete_sentence",
    "join_sentences",
    "merge_with_previous",
    "move_sentence",
    "replace_sentence",
    "split_sentences",
    # tokens
    "CLIP_TOKEN_LIMIT",
    "HeuristicClipEstimator",
    "TokenEstimator",
    "get_default_estimator",
    "is_over_limit",
]
