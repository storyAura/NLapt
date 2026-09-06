"""QThreadPool workers - run blocking work off the GUI thread.

``run_async`` keeps a module-level reference to every in-flight worker so
neither the QRunnable nor its signal object is garbage-collected before the
result lands. Callbacks are connected with an explicit queued connection;
because the signal holder object is created on the caller's (GUI) thread,
results are always delivered through the GUI event loop.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal

from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Strong references to in-flight workers (released when done/error fires).
_ACTIVE_WORKERS: set["FunctionWorker"] = set()
# When False (default), failures log one warning line — no traceback in the
# launcher console. 调试模式 turns this on via :func:`set_debug`.
_DEBUG = False


def set_debug(enabled: bool) -> None:
    """Gate worker exception tracebacks (on = log full stacks)."""
    global _DEBUG
    _DEBUG = bool(enabled)


def debug_enabled() -> bool:
    """True when worker failures should log a full traceback."""
    return _DEBUG


class _WorkerSignals(QObject):
    """Signal holder for FunctionWorker (QRunnable cannot own signals)."""

    done = Signal(object)
    error = Signal(str)


class FunctionWorker(QRunnable):
    """Run ``fn(*args)`` on a QThreadPool thread.

    Emits ``signals.done(result)`` on success or ``signals.error(str)``
    when ``fn`` raises. Failures log a one-line warning unless debug
    mode is on, in which case the full traceback is logged.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any) -> None:
        super().__init__()
        self.signals = _WorkerSignals()
        self._fn = fn
        self._args = args
        # Lifetime is managed by _ACTIVE_WORKERS, not by the pool.
        self.setAutoDelete(False)

    def run(self) -> None:  # noqa: D102 - QRunnable entry point
        try:
            result = self._fn(*self._args)
        except Exception as exc:
            name = getattr(self._fn, "__name__", self._fn)
            if _DEBUG:
                _LOGGER.exception("worker %r failed", name)
            else:
                _LOGGER.warning("worker %r failed: %s", name, exc)
            self._emit(self.signals.error, str(exc) or type(exc).__name__)
        else:
            self._emit(self.signals.done, result)

    def _emit(self, signal: Signal, payload: object) -> None:
        """Deliver ``payload``; a torn-down QObject must not become a crash."""
        try:
            signal.emit(payload)
        except RuntimeError:
            _LOGGER.debug("worker signal dropped (receiver gone)")
            _ACTIVE_WORKERS.discard(self)


def run_async(
    pool: QThreadPool,
    fn: Callable[..., Any],
    *args: Any,
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[str], None] | None = None,
) -> None:
    """Start ``fn(*args)`` on ``pool``; deliver the result via queued signals.

    ``on_done(result)`` / ``on_error(message)`` run on the GUI thread. The
    worker is kept referenced until one of them has been dispatched.
    """
    worker = FunctionWorker(fn, *args)
    queued = Qt.ConnectionType.QueuedConnection
    if on_done is not None:
        worker.signals.done.connect(on_done, queued)
    if on_error is not None:
        worker.signals.error.connect(on_error, queued)

    def _release(_payload: object = None) -> None:
        _ACTIVE_WORKERS.discard(worker)

    # Connected after the user callbacks so release happens last.
    worker.signals.done.connect(_release, queued)
    worker.signals.error.connect(_release, queued)
    _ACTIVE_WORKERS.add(worker)
    pool.start(worker)
