"""History layer (spec 7.5): per-file undo stacks and the session-level
operation log with whole-operation rollback.

Re-exports the public API of the ``nlapt.history`` package (contract:
docs/ARCHITECTURE.md, section ``nlapt.history``).
"""

from __future__ import annotations

from nlapt.history.oplog import OperationLog, OperationRecord
from nlapt.history.undo import GROUP_WINDOW_SECONDS, UNDO_LIMIT, UndoStack

__all__ = [
    "GROUP_WINDOW_SECONDS",
    "OperationLog",
    "OperationRecord",
    "UNDO_LIMIT",
    "UndoStack",
]
