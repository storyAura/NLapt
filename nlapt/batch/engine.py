"""Shared batch execution engine (spec 9).

Every batch capability (replace, prefix/suffix, translate, AI ops, dedup, ...)
reuses this pipeline:

1. automatic snapshot **before** execution (when enabled and a
   :class:`~nlapt.storage.snapshots.SnapshotManager` is wired in);
2. skip keys already recorded under the resume checkpoint;
3. run the worker over the remaining keys with a
   :class:`concurrent.futures.ThreadPoolExecutor` — worker exceptions become
   failed :class:`~nlapt.batch.progress.BatchItemResult` items and never crash
   the run;
4. honor pause/cancel between item dispatches — pause blocks only NEW
   dispatches while finished in-flight items keep being collected (progress,
   events, checkpoint marks); cancel keeps the completed portion;
5. mark the checkpoint after each success, clear it on full completion;
6. publish ``EVT_BATCH_STARTED`` / ``EVT_BATCH_PROGRESS`` / ``EVT_BATCH_FINISHED``.

Progress accounting: ``total`` is the full requested key count; ``done``
starts at the number of checkpoint-skipped keys and increases by one per
processed item, so callbacks always see monotonically increasing counts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as _wait_futures

from nlapt.batch.checkpoint import CheckpointStore
from nlapt.batch.progress import (
    BatchController,
    BatchItemResult,
    BatchReport,
    BatchStatus,
)
from nlapt.core.errors import StorageError, ValidationError
from nlapt.core.events import (
    EVT_BATCH_FINISHED,
    EVT_BATCH_PROGRESS,
    EVT_BATCH_STARTED,
    EVT_SNAPSHOT_CREATED,
    EventBus,
)
from nlapt.diagnostics import get_logger
from nlapt.storage.snapshots import SnapshotInfo, SnapshotManager

_LOGGER = get_logger(__name__)

ProgressCallback = Callable[[int, int, str], None]  # (done, total, current_key)

DEFAULT_CONCURRENCY = 1
THREAD_NAME_PREFIX = "nlapt-batch"
# Bounded future-wait used while paused with items in flight: finished futures
# keep being collected (progress/checkpoint marks fire) and a resume is picked
# up within this interval — no unbounded block, no busy spin.
PAUSED_WAIT_TIMEOUT_SECONDS = 0.05


class BatchEngine:
    """Run a worker callable over many keys with snapshot/pause/resume support."""

    def __init__(
        self,
        *,
        snapshots: SnapshotManager | None,
        bus: EventBus,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        if not isinstance(bus, EventBus):
            raise ValidationError(f"bus must be an EventBus, got {type(bus).__name__}")
        if snapshots is not None and not isinstance(snapshots, SnapshotManager):
            raise ValidationError(
                f"snapshots must be a SnapshotManager or None, got {type(snapshots).__name__}"
            )
        if checkpoints is not None and not isinstance(checkpoints, CheckpointStore):
            raise ValidationError(
                f"checkpoints must be a CheckpointStore or None, got {type(checkpoints).__name__}"
            )
        self._snapshots = snapshots
        self._bus = bus
        self._checkpoints = checkpoints

    def run(
        self,
        *,
        operation: str,
        keys: Sequence[str],
        worker: Callable[[str], BatchItemResult],
        concurrency: int = DEFAULT_CONCURRENCY,
        controller: BatchController | None = None,
        on_progress: ProgressCallback | None = None,
        take_snapshot: bool = True,
        checkpoint_id: str | None = None,
    ) -> BatchReport:
        """Execute ``worker`` over ``keys`` and return an aggregated report.

        Raises ValidationError for bad arguments and SnapshotError when the
        pre-execution snapshot cannot be created (the run is not started
        without its safety snapshot).
        """
        _validate_run_args(operation, keys, worker, concurrency, controller, on_progress)
        if checkpoint_id is not None and self._checkpoints is None:
            raise ValidationError("checkpoint_id requires a CheckpointStore to be configured")

        snapshot = self._create_snapshot(operation) if take_snapshot else None
        already_done = (
            self._checkpoints.completed(checkpoint_id)
            if checkpoint_id is not None and self._checkpoints is not None
            else frozenset()
        )
        pending = [key for key in keys if key not in already_done]
        skipped = len(keys) - len(pending)
        total = len(keys)
        if skipped:
            _LOGGER.info(
                "batch %r resumes from checkpoint: %d of %d keys already done",
                operation, skipped, total,
            )
        self._bus.publish(EVT_BATCH_STARTED, operation=operation, total=total, skipped=skipped)

        results, cancelled = self._execute(
            operation=operation,
            pending=pending,
            worker=worker,
            concurrency=concurrency,
            controller=controller,
            on_progress=on_progress,
            checkpoint_id=checkpoint_id,
            done_offset=skipped,
            total=total,
        )

        status = BatchStatus.CANCELLED if cancelled else BatchStatus.COMPLETED
        if checkpoint_id is not None and status is BatchStatus.COMPLETED:
            self._clear_checkpoint(checkpoint_id)
        failed_keys = tuple(item.key for item in results if not item.ok)
        report = BatchReport(
            operation=operation,
            status=status,
            results=tuple(results),
            snapshot=snapshot,
            succeeded=len(results) - len(failed_keys),
            failed=len(failed_keys),
            failed_keys=failed_keys,
        )
        self._bus.publish(
            EVT_BATCH_FINISHED,
            operation=operation,
            status=status.value,
            succeeded=report.succeeded,
            failed=report.failed,
            total=total,
        )
        _LOGGER.info(
            "batch %r finished: status=%s succeeded=%d failed=%d skipped=%d",
            operation, status.value, report.succeeded, report.failed, skipped,
        )
        return report

    # -- internals ---------------------------------------------------------------

    def _execute(
        self,
        *,
        operation: str,
        pending: list[str],
        worker: Callable[[str], BatchItemResult],
        concurrency: int,
        controller: BatchController | None,
        on_progress: ProgressCallback | None,
        checkpoint_id: str | None,
        done_offset: int,
        total: int,
    ) -> tuple[list[BatchItemResult], bool]:
        """Dispatch/collect loop. Returns (results, cancelled)."""
        results: list[BatchItemResult] = []
        done = done_offset
        cancelled = controller.cancelled if controller is not None else False
        key_iter = iter(pending)
        next_key: str | None = next(key_iter, None)
        in_flight: dict[Future[BatchItemResult], str] = {}

        with ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix=THREAD_NAME_PREFIX
        ) as pool:
            while next_key is not None or in_flight:
                paused = False
                # Dispatch up to `concurrency` items; pause blocks only NEW
                # dispatches — in-flight futures keep being collected below.
                while next_key is not None and len(in_flight) < concurrency and not cancelled:
                    if controller is not None:
                        if in_flight:
                            # Never block with futures in flight: finished
                            # items must be drained while paused.
                            if _controller_paused(controller):
                                paused = True
                                break
                        else:
                            controller.wait_if_paused()
                        if controller.cancelled:
                            cancelled = True
                            break
                    future = pool.submit(_run_worker, worker, next_key)
                    in_flight[future] = next_key
                    next_key = next(key_iter, None)
                if cancelled:
                    next_key = None  # stop dispatching; drain what is in flight
                if not in_flight:
                    break
                # Bounded wait while pause skipped dispatching, so a resume is
                # noticed promptly even when no in-flight future completes.
                timeout = PAUSED_WAIT_TIMEOUT_SECONDS if paused else None
                finished, _ = _wait_futures(
                    in_flight, timeout=timeout, return_when=FIRST_COMPLETED
                )
                for future in finished:
                    key = in_flight.pop(future)
                    item = future.result()  # _run_worker never raises
                    results.append(item)
                    done += 1
                    if item.ok and checkpoint_id is not None:
                        self._mark_checkpoint(checkpoint_id, key)
                    self._bus.publish(
                        EVT_BATCH_PROGRESS,
                        operation=operation, done=done, total=total, key=key, ok=item.ok,
                    )
                    _notify_progress(on_progress, done, total, key)
                if controller is not None and controller.cancelled:
                    cancelled = True
                    next_key = None
        return results, cancelled

    def _create_snapshot(self, operation: str) -> SnapshotInfo | None:
        """Pre-execution safety snapshot; None when no manager is configured."""
        if self._snapshots is None:
            return None
        info = self._snapshots.create(operation)  # SnapshotError propagates by design
        self._bus.publish(
            EVT_SNAPSHOT_CREATED,
            operation=info.operation, path=str(info.path), file_count=info.file_count,
        )
        return info

    def _mark_checkpoint(self, checkpoint_id: str, key: str) -> None:
        """Best-effort checkpoint mark: a persistence failure must not kill the
        run mid-way (losing a mark only means redoing that one item on resume)."""
        assert self._checkpoints is not None
        try:
            self._checkpoints.mark(checkpoint_id, key)
        except StorageError:
            _LOGGER.exception("could not persist checkpoint mark for %r", key)

    def _clear_checkpoint(self, checkpoint_id: str) -> None:
        """Best-effort checkpoint clear after full completion (logged on failure)."""
        assert self._checkpoints is not None
        try:
            self._checkpoints.clear(checkpoint_id)
        except StorageError:
            _LOGGER.exception("could not clear checkpoint %s", checkpoint_id)


def _controller_paused(controller: BatchController) -> bool:
    """Non-blocking paused query for the dispatch loop.

    ``BatchController`` exposes no public paused accessor (its contract is
    the blocking ``wait_if_paused``), so this intra-package helper peeks at
    the flag under the controller's own condition lock. A cancelled
    controller reports not-paused so cancellation handling takes over.
    """
    with controller._condition:  # noqa: SLF001 - same-package friend access
        return controller._paused and not controller._cancelled


def _run_worker(worker: Callable[[str], BatchItemResult], key: str) -> BatchItemResult:
    """Invoke the worker; convert exceptions/bad returns into failed items."""
    try:
        result = worker(key)
    except Exception as exc:  # noqa: BLE001 - a worker failure must not stop the batch
        _LOGGER.exception("batch worker raised for key %r", key)
        return BatchItemResult(key=key, ok=False, error=f"{type(exc).__name__}: {exc}")
    if not isinstance(result, BatchItemResult):
        _LOGGER.warning(
            "batch worker returned %s for key %r, expected BatchItemResult",
            type(result).__name__, key,
        )
        return BatchItemResult(
            key=key, ok=False,
            error=f"worker returned {type(result).__name__}, expected BatchItemResult",
        )
    return result


def _notify_progress(on_progress: ProgressCallback | None, done: int, total: int, key: str) -> None:
    """Call the progress callback; its failures are logged, never propagated."""
    if on_progress is None:
        return
    try:
        on_progress(done, total, key)
    except Exception:  # noqa: BLE001 - observer failures must not stop the batch
        _LOGGER.exception("progress callback failed at %d/%d (%r)", done, total, key)


def _validate_run_args(
    operation: str,
    keys: Sequence[str],
    worker: Callable[[str], BatchItemResult],
    concurrency: int,
    controller: BatchController | None,
    on_progress: ProgressCallback | None,
) -> None:
    if not isinstance(operation, str) or not operation.strip():
        raise ValidationError(f"operation must be a non-empty string, got {operation!r}")
    if isinstance(keys, str) or not isinstance(keys, Sequence):
        raise ValidationError(f"keys must be a sequence of strings, got {type(keys).__name__}")
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ValidationError(f"keys must all be non-empty strings, got {key!r}")
    if not callable(worker):
        raise ValidationError(f"worker must be callable, got {type(worker).__name__}")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValidationError(f"concurrency must be an int >= 1, got {concurrency!r}")
    if controller is not None and not isinstance(controller, BatchController):
        raise ValidationError(
            f"controller must be a BatchController or None, got {type(controller).__name__}"
        )
    if on_progress is not None and not callable(on_progress):
        raise ValidationError(
            f"on_progress must be callable or None, got {type(on_progress).__name__}"
        )
