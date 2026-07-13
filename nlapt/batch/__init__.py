"""Batch processing engine (spec 9): progress primitives, resume checkpoints,
and the shared execution engine every batch capability reuses.

Re-exports the public API of the ``nlapt.batch`` package (contract:
docs/ARCHITECTURE.md, section ``nlapt.batch``).
"""

from __future__ import annotations

from nlapt.batch.checkpoint import CheckpointStore, make_checkpoint_id
from nlapt.batch.engine import BatchEngine, ProgressCallback
from nlapt.batch.progress import (
    BatchController,
    BatchItemResult,
    BatchReport,
    BatchStatus,
)

__all__ = [
    "BatchController",
    "BatchEngine",
    "BatchItemResult",
    "BatchReport",
    "BatchStatus",
    "CheckpointStore",
    "ProgressCallback",
    "make_checkpoint_id",
]
