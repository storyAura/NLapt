"""Tests for nlapt_gui.workers (FunctionWorker + run_async)."""

from __future__ import annotations

import threading

from PySide6.QtCore import QThread, QThreadPool

from nlapt_gui.workers import run_async


def _pool() -> QThreadPool:
    return QThreadPool.globalInstance()


class TestRunAsync:
    def test_result_delivered(self, qtbot, qapp) -> None:
        results: list[int] = []
        run_async(_pool(), lambda a, b: a + b, 2, 3, on_done=results.append)
        qtbot.waitUntil(lambda: results == [5], timeout=2000)

    def test_runs_off_gui_thread_but_delivers_on_gui_thread(self, qtbot, qapp) -> None:
        seen: dict[str, object] = {}

        def work() -> int:
            seen["worker_thread"] = threading.get_ident()
            return 42

        def done(value: int) -> None:
            seen["value"] = value
            seen["done_thread"] = QThread.currentThread()

        run_async(_pool(), work, on_done=done)
        qtbot.waitUntil(lambda: "value" in seen, timeout=2000)
        assert seen["value"] == 42
        assert seen["worker_thread"] != threading.get_ident()
        assert seen["done_thread"] is qapp.thread()

    def test_error_delivered_as_message(self, qtbot, qapp) -> None:
        errors: list[str] = []

        def boom() -> None:
            raise RuntimeError("kaboom")

        run_async(_pool(), boom, on_error=errors.append)
        qtbot.waitUntil(lambda: errors == ["kaboom"], timeout=2000)

    def test_error_without_message_uses_type_name(self, qtbot, qapp) -> None:
        errors: list[str] = []

        def boom() -> None:
            raise ValueError()

        run_async(_pool(), boom, on_error=errors.append)
        qtbot.waitUntil(lambda: errors == ["ValueError"], timeout=2000)

    def test_workers_not_garbage_collected(self, qtbot, qapp) -> None:
        """Many short-lived workers all report back (references are kept)."""
        import gc

        results: list[int] = []
        for i in range(24):
            run_async(_pool(), lambda v: v, i, on_done=results.append)
        gc.collect()
        qtbot.waitUntil(lambda: len(results) == 24, timeout=2000)
        assert sorted(results) == list(range(24))

    def test_no_callbacks_is_safe(self, qtbot, qapp) -> None:
        done = threading.Event()

        def work() -> None:
            done.set()

        run_async(_pool(), work)
        qtbot.waitUntil(done.is_set, timeout=2000)
