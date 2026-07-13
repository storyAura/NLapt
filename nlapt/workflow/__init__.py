"""Review workflow (spec 10): word-level suggestion diff, accept/reject glue
over the caption store, and review-flow navigation helpers.

Re-exports the public API of the ``nlapt.workflow`` package (contract:
docs/ARCHITECTURE.md, section ``nlapt.workflow``).
"""

from __future__ import annotations

from nlapt.workflow.diff import DiffOp, DiffSegment, word_diff
from nlapt.workflow.review import next_pending, next_unconfirmed
from nlapt.workflow.suggestions import (
    accept_for_edit,
    accept_suggestion,
    reject_suggestion,
)

__all__ = [
    "DiffOp",
    "DiffSegment",
    "accept_for_edit",
    "accept_suggestion",
    "next_pending",
    "next_unconfirmed",
    "reject_suggestion",
    "word_diff",
]
