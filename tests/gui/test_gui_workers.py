"""Tests for nlapt_gui.workers (FunctionWorker + run_async)."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QThread, QThreadPool

from nlapt_gui.workers import FunctionWorker, _ACTIVE_WORKERS, run_async, set_debug

_WORKER_LOGGER = "nlapt.nlapt_gui.workers"


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

    def test_error_logs_warning_without_traceback_when_debug_off(
        self, qtbot, qapp
    ) -> None:
        set_debug(False)
        errors: list[str] = []
        records: list[logging.LogRecord] = []

        def boom() -> None:
            raise RuntimeError("quiet-fail")

        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = logging.getLogger(_WORKER_LOGGER)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            run_async(_pool(), boom, on_error=errors.append)
            qtbot.waitUntil(lambda: errors == ["quiet-fail"], timeout=2000)
        finally:
            logger.removeHandler(handler)
            set_debug(False)
        assert any("quiet-fail" in record.getMessage() for record in records)
        assert all(record.exc_info in (None, (None, None, None)) for record in records)

    def test_error_logs_exception_when_debug_on(self, qtbot, qapp) -> None:
        set_debug(True)
        errors: list[str] = []
        records: list[logging.LogRecord] = []

        def boom() -> None:
            raise RuntimeError("loud-fail")

        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger = logging.getLogger(_WORKER_LOGGER)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            run_async(_pool(), boom, on_error=errors.append)
            qtbot.waitUntil(lambda: errors == ["loud-fail"], timeout=2000)
            assert any(
                record.exc_info not in (None, (None, None, None))
                and record.exc_info is not None
                and "loud-fail" in str(record.exc_info[1])
                for record in records
            )
        finally:
            logger.removeHandler(handler)
            set_debug(False)

    def test_error_emit_gone_does_not_raise(self) -> None:
        def boom() -> None:
            raise RuntimeError("kaboom")

        worker = FunctionWorker(boom)

        class _Dead:
            def emit(self, *_args: object) -> None:
                raise RuntimeError("Signal source has been deleted")

        worker.signals.error = _Dead()  # type: ignore[method-assign]
        _ACTIVE_WORKERS.add(worker)
        worker.run()
        assert worker not in _ACTIVE_WORKERS

    def test_done_emit_gone_does_not_raise(self) -> None:
        worker = FunctionWorker(lambda: 1)

        class _Dead:
            def emit(self, *_args: object) -> None:
                raise RuntimeError("Signal source has been deleted")

        worker.signals.done = _Dead()  # type: ignore[method-assign]
        worker.run()

    def test_no_callbacks_is_safe(self, qtbot, qapp) -> None:
        done = threading.Event()

        def work() -> None:
            done.set()

        run_async(_pool(), work)
        qtbot.waitUntil(done.is_set, timeout=2000)
