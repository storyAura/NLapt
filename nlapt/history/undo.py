"""Per-file snapshot undo stack (spec 7.5).

Each caption file owns one :class:`UndoStack` holding up to ``UNDO_LIMIT``
previous text snapshots. Rapid consecutive edits (pushes less than
``GROUP_WINDOW_SECONDS`` apart) collapse into a single undo step, matching
the input-grouping rule of the free-text editor (spec 6.3). The clock is
injected so tests control time without sleeping.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

UNDO_LIMIT = 50
GROUP_WINDOW_SECONDS = 0.6


class UndoStack:
    """Snapshot-based undo/redo for one caption's text.

    Semantics:

    - ``push(text)`` with ``text == current`` is a complete no-op (the group
      timer is not touched).
    - A push less than ``GROUP_WINDOW_SECONDS`` after the previous push
      collapses into the same undo step (sliding window); pushes spaced
      ``>= GROUP_WINDOW_SECONDS`` apart create separate steps.
    - ``undo``/``redo`` break the grouping chain: the next push always starts
      a new step.
    - Any push clears the redo history.
    - At most ``limit`` undo steps are kept; the oldest is evicted first.

    Thread-safe (RLock), although a stack is normally driven by one UI thread.
    """

    def __init__(
        self,
        initial: str,
        *,
        limit: int = UNDO_LIMIT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(initial, str):
            raise ValidationError(f"initial text must be a string, got {type(initial).__name__}")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValidationError(f"undo limit must be an int >= 1, got {limit!r}")
        if not callable(clock):
            raise ValidationError(f"clock must be callable, got {type(clock).__name__}")
        self._lock = threading.RLock()
        self._clock = clock
        self._limit = limit
        self._current = initial
        self._undo: list[str] = []
        self._redo: list[str] = []
        self._last_push: float | None = None  # None breaks grouping

    def push(self, text: str) -> None:
        """Record an edit. Identical text is a no-op; rapid pushes group."""
        if not isinstance(text, str):
            raise ValidationError(f"text must be a string, got {type(text).__name__}")
        with self._lock:
            if text == self._current:
                return
            now = self._clock()
            grouped = (
                self._last_push is not None
                and (now - self._last_push) < GROUP_WINDOW_SECONDS
            )
            if not grouped:
                self._undo.append(self._current)
                if len(self._undo) > self._limit:
                    evicted = self._undo.pop(0)
                    _LOGGER.debug("undo limit reached; evicted oldest step %r", evicted[:40])
            self._current = text
            self._redo.clear()
            self._last_push = now

    def undo(self) -> str | None:
        """Step back one snapshot; returns the new current text or None."""
        with self._lock:
            if not self._undo:
                return None
            self._redo.append(self._current)
            self._current = self._undo.pop()
            self._last_push = None  # an undo always ends the current edit group
            return self._current

    def redo(self) -> str | None:
        """Reapply the last undone snapshot; returns the new current text or None."""
        with self._lock:
            if not self._redo:
                return None
            self._undo.append(self._current)
            if len(self._undo) > self._limit:
                self._undo.pop(0)
            self._current = self._redo.pop()
            self._last_push = None
            return self._current

    @property
    def current(self) -> str:
        """The text the stack currently points at."""
        with self._lock:
            return self._current

    def can_undo(self) -> bool:
        with self._lock:
            return bool(self._undo)

    def can_redo(self) -> bool:
        with self._lock:
            return bool(self._redo)
