"""Idle auto-stop for llama-server after inference finishes.

Inference that started the server used to stop it the moment it finished;
repeated runs then reloaded multi-GB weights every time. The stop is now
armed on an idle timer: any new local request cancels it, and only a full
idle window actually unloads the model. Servers the USER started
(启动本地服务) are never auto-stopped.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from nlapt.diagnostics import get_logger
from nlapt.local.server import LocalServerManager

_LOGGER = get_logger(__name__)

IDLE_STOP_DELAY_SECONDS = 30.0


class IdleServerStopper:
    """Delays the post-inference llama-server stop by an idle window.

    ``note_request``/``note_finished`` bracket every local inference run
    (single request or whole batch). The timer only arms once no run is in
    flight AND the server was loaded by inference (not by the user), and any
    new run cancels it — so back-to-back runs keep the model warm.
    ``timer_factory`` is injectable so tests never wait.
    """

    def __init__(
        self,
        manager: LocalServerManager,
        *,
        delay: float = IDLE_STOP_DELAY_SECONDS,
        timer_factory: Callable[[float, Callable[[], None]], threading.Timer] = (
            threading.Timer
        ),
    ) -> None:
        self._manager = manager
        self._delay = delay
        self._timer_factory = timer_factory
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._active = 0
        self._auto_started = False

    def note_request(self) -> None:
        """A local inference run is starting: cancel any pending stop.

        When the server is not running yet, the coming run is the one that
        loads the model — remember that so only inference-loaded servers get
        idle-stopped.
        """
        with self._lock:
            self._cancel_locked()
            if self._active == 0 and not self._manager.is_running():
                self._auto_started = True
            self._active += 1

    def note_finished(self) -> None:
        """A run ended: arm the idle stop once nothing else is in flight."""
        with self._lock:
            self._active = max(0, self._active - 1)
            if self._active > 0 or not self._auto_started:
                return
            self._cancel_locked()
            timer = self._timer_factory(self._delay, self._fire)
            timer.daemon = True  # never delay interpreter exit by the window
            timer.start()
            self._timer = timer

    def note_user_control(self) -> None:
        """The user started/stopped the server manually: no auto stop."""
        with self._lock:
            self._cancel_locked()
            self._auto_started = False

    def pending(self) -> bool:
        """Whether an idle stop is currently armed (tests/diagnostics)."""
        with self._lock:
            return self._timer is not None

    def _cancel_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _fire(self) -> None:
        with self._lock:
            self._timer = None
            if self._active > 0:
                return  # a run raced in; it re-arms on finish
            self._auto_started = False
        try:
            if self._manager.is_running():
                _LOGGER.info(
                    "stopping llama-server after %.0fs idle", self._delay
                )
            self._manager.stop()
        except Exception:  # timer thread: never let a stop failure propagate
            _LOGGER.exception("idle llama-server stop failed")


_IDLE_STOPPER: IdleServerStopper | None = None


def get_idle_stopper() -> IdleServerStopper:
    """Process-wide idle stopper bound to the shared caption-server manager."""
    global _IDLE_STOPPER
    if _IDLE_STOPPER is None:
        # Circular: local_bridge re-exports this and owns get_server_manager.
        from nlapt_gui.local_bridge import get_server_manager

        _IDLE_STOPPER = IdleServerStopper(get_server_manager())
    return _IDLE_STOPPER
