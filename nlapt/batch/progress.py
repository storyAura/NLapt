"""Batch progress primitives (spec 9): status, per-item results, report,
and the cooperative pause/resume/cancel controller.

``BatchController`` is the handle a UI shares with :class:`~nlapt.batch.engine.BatchEngine`.
The engine calls :meth:`BatchController.wait_if_paused` before dispatching each
item, so pausing blocks new dispatches while items already in flight finish.
Cancelling keeps the completed portion (spec 9).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

from nlapt.diagnostics import get_logger
from nlapt.storage.snapshots import SnapshotInfo

_LOGGER = get_logger(__name__)


class BatchStatus(str, Enum):
    """Lifecycle status of a batch run. Final reports are COMPLETED or CANCELLED."""

    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


@dataclass(frozen=True)
class BatchItemResult:
    """Outcome for a single key processed by a batch worker."""

    key: str
    ok: bool
    detail: str = ""  # e.g. "3 replacements" or a suggestion text summary
    error: str = ""   # populated when ok is False


@dataclass(frozen=True)
class BatchReport:
    """Aggregated outcome of one batch run.

    ``results`` contains only the items processed in this run — keys skipped
    via a resume checkpoint are not repeated. A CANCELLED report keeps the
    completed portion (spec 9).
    """

    operation: str
    status: BatchStatus
    results: tuple[BatchItemResult, ...]
    snapshot: SnapshotInfo | None
    succeeded: int
    failed: int
    failed_keys: tuple[str, ...]


class BatchController:
    """Thread-safe cooperative pause/resume/cancel switch for a batch run.

    All methods may be called from any thread. ``wait_if_paused`` blocks the
    calling (engine) thread while paused and returns immediately once the run
    is resumed or cancelled — cancellation always unblocks waiters.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paused = False
        self._cancelled = False

    def pause(self) -> None:
        """Request that the engine stop dispatching new items."""
        with self._condition:
            self._paused = True
        _LOGGER.debug("batch controller paused")

    def resume(self) -> None:
        """Allow the engine to continue dispatching items."""
        with self._condition:
            self._paused = False
            self._condition.notify_all()
        _LOGGER.debug("batch controller resumed")

    def cancel(self) -> None:
        """Request cancellation; also wakes any thread blocked in ``wait_if_paused``."""
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()
        _LOGGER.debug("batch controller cancelled")

    def wait_if_paused(self) -> None:
        """Block while paused; return immediately when running or cancelled."""
        with self._condition:
            while self._paused and not self._cancelled:
                self._condition.wait()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""
        with self._condition:
            return self._cancelled
