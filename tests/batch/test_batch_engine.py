"""Tests for nlapt.batch.engine: snapshot-first execution, failure isolation,
pause/cancel, concurrency, checkpoint resume, progress, and events."""

from __future__ import annotations

import threading
import zipfile
from pathlib import Path
from typing import Any

import pytest

from nlapt.batch.checkpoint import CheckpointStore, make_checkpoint_id
from nlapt.batch.engine import BatchEngine
from nlapt.batch.progress import BatchController, BatchItemResult, BatchStatus
from nlapt.core.errors import ValidationError
from nlapt.core.events import (
    EVT_BATCH_FINISHED,
    EVT_BATCH_PROGRESS,
    EVT_BATCH_STARTED,
    EVT_SNAPSHOT_CREATED,
    Event,
    EventBus,
)
from nlapt.storage.snapshots import SnapshotManager

TINY_WAIT = 0.05    # negative checks only ("did NOT start")
JOIN_TIMEOUT = 5.0  # positive checks; consumed only on failure
UTF8 = "utf-8"


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def events(bus: EventBus) -> list[Event]:
    collected: list[Event] = []
    bus.subscribe(None, collected.append)
    return collected


def _ok_worker(key: str) -> BatchItemResult:
    return BatchItemResult(key=key, ok=True, detail="done")


def _gate_worker(
    started: dict[str, threading.Event], gates: dict[str, threading.Event]
):
    """Worker that signals start and blocks until its per-key gate opens."""

    def worker(key: str) -> BatchItemResult:
        started[key].set()
        if not gates[key].wait(JOIN_TIMEOUT):
            raise TimeoutError(f"gate for {key} never opened")
        return BatchItemResult(key=key, ok=True)

    return worker


def _run_in_thread(engine: BatchEngine, **kwargs: Any) -> tuple[threading.Thread, dict]:
    box: dict[str, Any] = {}

    def target() -> None:
        box["report"] = engine.run(**kwargs)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


# -- basic runs ----------------------------------------------------------------


def test_successful_run_report_and_events(bus: EventBus, events: list[Event]) -> None:
    engine = BatchEngine(snapshots=None, bus=bus)
    keys = ("a.txt", "b.txt", "c.txt")
    report = engine.run(operation="noop", keys=keys, worker=_ok_worker)

    assert report.operation == "noop"
    assert report.status is BatchStatus.COMPLETED
    assert report.snapshot is None
    assert report.succeeded == 3
    assert report.failed == 0
    assert report.failed_keys == ()
    assert tuple(item.key for item in report.results) == keys
    assert all(item.ok and item.detail == "done" for item in report.results)

    names = [event.name for event in events]
    assert names[0] == EVT_BATCH_STARTED
    assert names.count(EVT_BATCH_PROGRESS) == 3
    assert names[-1] == EVT_BATCH_FINISHED
    started = events[0].payload
    assert started["operation"] == "noop"
    assert started["total"] == 3
    assert started["skipped"] == 0
    finished = events[-1].payload
    assert finished["status"] == BatchStatus.COMPLETED.value
    assert finished["succeeded"] == 3
    assert finished["failed"] == 0


def test_empty_keys_completes(bus: EventBus) -> None:
    engine = BatchEngine(snapshots=None, bus=bus)
    report = engine.run(operation="noop", keys=(), worker=_ok_worker)
    assert report.status is BatchStatus.COMPLETED
    assert report.results == ()
    assert report.succeeded == 0


# -- snapshot before execution ---------------------------------------------------


def test_snapshot_taken_before_execution(
    tmp_path: Path, bus: EventBus, events: list[Event]
) -> None:
    caption = tmp_path / "a.txt"
    caption.write_text("old", encoding=UTF8)
    manager = SnapshotManager(tmp_path)
    engine = BatchEngine(snapshots=manager, bus=bus)

    def worker(key: str) -> BatchItemResult:
        (tmp_path / key).write_text("new", encoding=UTF8)
        return BatchItemResult(key=key, ok=True)

    report = engine.run(operation="rewrite", keys=("a.txt",), worker=worker)

    assert caption.read_text(encoding=UTF8) == "new"
    assert report.snapshot is not None
    with zipfile.ZipFile(report.snapshot.path) as archive:
        assert archive.read("a.txt") == b"old"  # pre-execution content
    assert EVT_SNAPSHOT_CREATED in [event.name for event in events]


def test_no_snapshot_when_disabled(tmp_path: Path, bus: EventBus) -> None:
    manager = SnapshotManager(tmp_path)
    engine = BatchEngine(snapshots=manager, bus=bus)
    report = engine.run(
        operation="noop", keys=("a.txt",), worker=_ok_worker, take_snapshot=False
    )
    assert report.snapshot is None
    assert not any((tmp_path / ".backups").glob("*.zip")) or not (
        tmp_path / ".backups"
    ).exists()


# -- failure isolation ------------------------------------------------------------


def test_worker_exception_becomes_failed_result(bus: EventBus) -> None:
    def worker(key: str) -> BatchItemResult:
        if key == "bad.txt":
            raise RuntimeError("boom")
        return BatchItemResult(key=key, ok=True)

    engine = BatchEngine(snapshots=None, bus=bus)
    report = engine.run(
        operation="mixed", keys=("a.txt", "bad.txt", "c.txt"), worker=worker
    )
    assert report.status is BatchStatus.COMPLETED
    assert report.succeeded == 2
    assert report.failed == 1
    assert report.failed_keys == ("bad.txt",)
    failed = next(item for item in report.results if not item.ok)
    assert "RuntimeError" in failed.error
    assert "boom" in failed.error


def test_worker_bad_return_type_becomes_failed_result(bus: EventBus) -> None:
    engine = BatchEngine(snapshots=None, bus=bus)
    report = engine.run(
        operation="badret", keys=("a.txt",), worker=lambda key: "not-a-result"
    )
    assert report.failed == 1
    assert report.failed_keys == ("a.txt",)
    assert "BatchItemResult" in report.results[0].error


def test_progress_callback_errors_are_swallowed(bus: EventBus) -> None:
    def broken_progress(done: int, total: int, key: str) -> None:
        raise ValueError("observer boom")

    engine = BatchEngine(snapshots=None, bus=bus)
    report = engine.run(
        operation="noop", keys=("a.txt",), worker=_ok_worker, on_progress=broken_progress
    )
    assert report.status is BatchStatus.COMPLETED
    assert report.succeeded == 1


# -- progress ---------------------------------------------------------------------


def test_progress_monotonic_with_concurrency(bus: EventBus) -> None:
    keys = tuple(f"file_{index}.txt" for index in range(12))
    seen: list[tuple[int, int, str]] = []
    lock = threading.Lock()

    def on_progress(done: int, total: int, key: str) -> None:
        with lock:
            seen.append((done, total, key))

    engine = BatchEngine(snapshots=None, bus=bus)
    report = engine.run(
        operation="par", keys=keys, worker=_ok_worker, concurrency=4,
        on_progress=on_progress,
    )
    assert report.succeeded == len(keys)
    assert [done for done, _, _ in seen] == list(range(1, len(keys) + 1))
    assert all(total == len(keys) for _, total, _ in seen)
    assert {key for _, _, key in seen} == set(keys)


def test_concurrency_runs_workers_in_parallel(bus: EventBus) -> None:
    barrier = threading.Barrier(2)

    def worker(key: str) -> BatchItemResult:
        barrier.wait(timeout=JOIN_TIMEOUT)  # needs both workers alive at once
        return BatchItemResult(key=key, ok=True)

    engine = BatchEngine(snapshots=None, bus=bus)
    report = engine.run(
        operation="par", keys=("a.txt", "b.txt"), worker=worker, concurrency=2
    )
    assert report.status is BatchStatus.COMPLETED
    assert report.succeeded == 2


# -- pause / cancel ----------------------------------------------------------------


def test_cancel_keeps_completed_portion(bus: EventBus) -> None:
    keys = ("k1", "k2", "k3")
    started = {key: threading.Event() for key in keys}
    gates = {key: threading.Event() for key in keys}
    controller = BatchController()
    engine = BatchEngine(snapshots=None, bus=bus)
    thread, box = _run_in_thread(
        engine, operation="cancelme", keys=keys,
        worker=_gate_worker(started, gates), controller=controller,
    )
    assert started["k1"].wait(JOIN_TIMEOUT)
    controller.cancel()
    gates["k1"].set()
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()

    report = box["report"]
    assert report.status is BatchStatus.CANCELLED
    assert tuple(item.key for item in report.results) == ("k1",)
    assert report.succeeded == 1
    assert not started["k2"].is_set()
    assert not started["k3"].is_set()


def test_pause_blocks_dispatch_until_resume(bus: EventBus) -> None:
    keys = ("k1", "k2")
    started = {key: threading.Event() for key in keys}
    gates = {key: threading.Event() for key in keys}
    controller = BatchController()
    engine = BatchEngine(snapshots=None, bus=bus)
    thread, box = _run_in_thread(
        engine, operation="pauseme", keys=keys,
        worker=_gate_worker(started, gates), controller=controller,
    )
    assert started["k1"].wait(JOIN_TIMEOUT)
    controller.pause()
    gates["k1"].set()  # in-flight item finishes while paused
    assert not started["k2"].wait(TINY_WAIT), "paused engine dispatched a new item"
    controller.resume()
    assert started["k2"].wait(JOIN_TIMEOUT), "resume did not restart dispatch"
    gates["k2"].set()
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()

    report = box["report"]
    assert report.status is BatchStatus.COMPLETED
    assert report.succeeded == 2


def test_cancel_while_paused_unblocks_and_cancels(bus: EventBus) -> None:
    keys = ("k1", "k2")
    started = {key: threading.Event() for key in keys}
    gates = {key: threading.Event() for key in keys}
    controller = BatchController()
    engine = BatchEngine(snapshots=None, bus=bus)
    thread, box = _run_in_thread(
        engine, operation="pausecancel", keys=keys,
        worker=_gate_worker(started, gates), controller=controller,
    )
    assert started["k1"].wait(JOIN_TIMEOUT)
    controller.pause()
    gates["k1"].set()  # k1 finishes; engine now blocks before dispatching k2
    assert not started["k2"].wait(TINY_WAIT)
    controller.cancel()  # cancel must unblock the paused engine
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()

    report = box["report"]
    assert report.status is BatchStatus.CANCELLED
    assert tuple(item.key for item in report.results) == ("k1",)
    assert not started["k2"].is_set()


def test_pre_cancelled_controller_processes_nothing(bus: EventBus) -> None:
    controller = BatchController()
    controller.cancel()
    calls: list[str] = []

    def worker(key: str) -> BatchItemResult:
        calls.append(key)
        return BatchItemResult(key=key, ok=True)

    engine = BatchEngine(snapshots=None, bus=bus)
    report = engine.run(
        operation="never", keys=("a.txt",), worker=worker, controller=controller
    )
    assert report.status is BatchStatus.CANCELLED
    assert report.results == ()
    assert calls == []


# -- checkpoint resume ---------------------------------------------------------------


def test_cancelled_run_keeps_checkpoint_marks(tmp_path: Path, bus: EventBus) -> None:
    keys = ("k1", "k2", "k3")
    store = CheckpointStore(tmp_path)
    cid = make_checkpoint_id("rewrite", keys, "params")
    started = {key: threading.Event() for key in keys}
    gates = {key: threading.Event() for key in keys}
    controller = BatchController()
    engine = BatchEngine(snapshots=None, bus=bus, checkpoints=store)
    thread, box = _run_in_thread(
        engine, operation="rewrite", keys=keys,
        worker=_gate_worker(started, gates), controller=controller,
        checkpoint_id=cid,
    )
    assert started["k1"].wait(JOIN_TIMEOUT)
    controller.cancel()
    gates["k1"].set()
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()

    assert box["report"].status is BatchStatus.CANCELLED
    assert store.has(cid) is True
    assert store.completed(cid) == frozenset({"k1"})


def test_checkpoint_skips_completed_and_clears_after_completion(
    tmp_path: Path, bus: EventBus, events: list[Event]
) -> None:
    keys = ("k1", "k2", "k3")
    store = CheckpointStore(tmp_path)
    cid = make_checkpoint_id("rewrite", keys, "params")
    store.mark(cid, "k1")  # as left behind by an interrupted earlier run
    calls: list[str] = []
    progress: list[tuple[int, int, str]] = []

    def worker(key: str) -> BatchItemResult:
        calls.append(key)
        return BatchItemResult(key=key, ok=True)

    engine = BatchEngine(snapshots=None, bus=bus, checkpoints=store)
    report = engine.run(
        operation="rewrite", keys=keys, worker=worker, checkpoint_id=cid,
        on_progress=lambda done, total, key: progress.append((done, total, key)),
    )

    assert calls == ["k2", "k3"]  # completed key skipped
    assert tuple(item.key for item in report.results) == ("k2", "k3")
    assert report.status is BatchStatus.COMPLETED
    # done counts resume above the skipped portion and stay monotonic.
    assert progress == [(2, 3, "k2"), (3, 3, "k3")]
    assert events[0].payload["skipped"] == 1
    # Fully completed -> checkpoint cleared for the next fresh run.
    assert store.has(cid) is False


def test_checkpoint_id_requires_store(bus: EventBus) -> None:
    engine = BatchEngine(snapshots=None, bus=bus)
    with pytest.raises(ValidationError):
        engine.run(
            operation="noop", keys=("a.txt",), worker=_ok_worker, checkpoint_id="cid"
        )


# -- validation -------------------------------------------------------------------


def test_constructor_validates_collaborators(bus: EventBus) -> None:
    with pytest.raises(ValidationError):
        BatchEngine(snapshots=None, bus="not-a-bus")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BatchEngine(snapshots="nope", bus=bus)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BatchEngine(snapshots=None, bus=bus, checkpoints="nope")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operation": ""},
        {"operation": "   "},
        {"keys": "a.txt"},
        {"keys": ("a.txt", "")},
        {"keys": (1,)},
        {"worker": "not-callable"},
        {"concurrency": 0},
        {"concurrency": True},
        {"controller": "nope"},
        {"on_progress": "nope"},
    ],
)
def test_run_validates_arguments(bus: EventBus, kwargs: dict[str, Any]) -> None:
    engine = BatchEngine(snapshots=None, bus=bus)
    base: dict[str, Any] = {
        "operation": "noop",
        "keys": ("a.txt",),
        "worker": _ok_worker,
    }
    with pytest.raises(ValidationError):
        engine.run(**{**base, **kwargs})
