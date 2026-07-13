"""Tests for nlapt.batch.progress: status enum, result records, controller."""

from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError

import pytest

from nlapt.batch.progress import (
    BatchController,
    BatchItemResult,
    BatchReport,
    BatchStatus,
)

# Tiny wait used only to observe that a thread is still blocked (negative check).
TINY_WAIT = 0.05
# Generous timeout for positive checks; only ever consumed on failure.
JOIN_TIMEOUT = 5.0


def test_batch_status_values() -> None:
    assert BatchStatus.RUNNING.value == "running"
    assert BatchStatus.PAUSED.value == "paused"
    assert BatchStatus.CANCELLED.value == "cancelled"
    assert BatchStatus.COMPLETED.value == "completed"
    assert issubclass(BatchStatus, str)


def test_item_result_defaults() -> None:
    item = BatchItemResult(key="a.txt", ok=True)
    assert item.detail == ""
    assert item.error == ""


def test_item_result_is_frozen() -> None:
    item = BatchItemResult(key="a.txt", ok=True)
    with pytest.raises(FrozenInstanceError):
        item.ok = False  # type: ignore[misc]


def test_report_is_frozen() -> None:
    report = BatchReport(
        operation="op",
        status=BatchStatus.COMPLETED,
        results=(),
        snapshot=None,
        succeeded=0,
        failed=0,
        failed_keys=(),
    )
    with pytest.raises(FrozenInstanceError):
        report.failed = 1  # type: ignore[misc]


def test_controller_initial_state() -> None:
    controller = BatchController()
    assert controller.cancelled is False
    # Not paused, not cancelled: must return immediately (no deadlock).
    controller.wait_if_paused()


def test_cancel_sets_cancelled_property() -> None:
    controller = BatchController()
    controller.cancel()
    assert controller.cancelled is True
    # wait_if_paused returns immediately once cancelled, even when paused.
    controller.pause()
    controller.wait_if_paused()


def _start_waiter(controller: BatchController) -> tuple[threading.Thread, threading.Event]:
    """Run wait_if_paused on a thread; the event fires once it returns."""
    released = threading.Event()

    def target() -> None:
        controller.wait_if_paused()
        released.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, released


def test_pause_blocks_wait_until_resume() -> None:
    controller = BatchController()
    controller.pause()
    thread, released = _start_waiter(controller)
    assert not released.wait(TINY_WAIT), "wait_if_paused returned while paused"
    controller.resume()
    assert released.wait(JOIN_TIMEOUT), "resume did not unblock the waiter"
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()


def test_cancel_unblocks_paused_waiter() -> None:
    controller = BatchController()
    controller.pause()
    thread, released = _start_waiter(controller)
    assert not released.wait(TINY_WAIT), "wait_if_paused returned while paused"
    controller.cancel()
    assert released.wait(JOIN_TIMEOUT), "cancel did not unblock the waiter"
    thread.join(JOIN_TIMEOUT)
    assert not thread.is_alive()
    assert controller.cancelled is True


def test_pause_resume_cycle_is_repeatable() -> None:
    controller = BatchController()
    for _ in range(3):
        controller.pause()
        controller.resume()
    controller.wait_if_paused()  # running: returns immediately
    assert controller.cancelled is False
