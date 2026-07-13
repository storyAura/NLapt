"""In-process event bus decoupling the core from any future UI layer.

Handlers must never break publishers: exceptions raised by handlers are
logged and swallowed by :meth:`EventBus.publish`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Well-known event names published by the core stores and engines.
EVT_CAPTION_CHANGED = "caption_changed"
EVT_STATE_CHANGED = "state_changed"
EVT_PENDING_SET = "pending_set"
EVT_PENDING_CLEARED = "pending_cleared"
EVT_FILE_SAVED = "file_saved"
EVT_SAVE_FAILED = "save_failed"
EVT_BATCH_STARTED = "batch_started"
EVT_BATCH_PROGRESS = "batch_progress"
EVT_BATCH_FINISHED = "batch_finished"
EVT_SNAPSHOT_CREATED = "snapshot_created"
EVT_OPLOG_APPENDED = "oplog_appended"
# Published by the app facade when several same-stem images resolve to one
# caption txt: only the canonical key stays editable (payload: key, excluded, txt).
EVT_TXT_CONFLICT = "txt_conflict"
# Published by the app facade when opened caption files are not plain UTF-8
# and can be converted (payload: keys, encodings).
EVT_ENCODING_ISSUES = "encoding_issues"

EventHandler = Callable[["Event"], None]


@dataclass(frozen=True)
class Event:
    """A named event with an immutable payload mapping."""

    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError(f"event name must be a non-empty string, got {self.name!r}")
        # Freeze the payload: copy then wrap so later caller mutation is impossible.
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


class EventBus:
    """Thread-safe publish/subscribe hub.

    ``subscribe(None, handler)`` receives every event (wildcard). Handler
    exceptions are logged and swallowed so the bus never breaks the caller.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[str | None, list[EventHandler]] = {}

    def subscribe(self, name: str | None, handler: EventHandler) -> Callable[[], None]:
        """Register ``handler`` for events named ``name`` (None = all events).

        Returns an idempotent unsubscribe callable.
        """
        if name is not None and (not isinstance(name, str) or not name):
            raise ValidationError(f"subscription name must be None or a non-empty string, got {name!r}")
        if not callable(handler):
            raise ValidationError(f"handler must be callable, got {handler!r}")
        with self._lock:
            self._handlers.setdefault(name, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(name, [])
                if handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def publish(self, name: str, **payload: Any) -> None:
        """Publish an event to name-specific and wildcard subscribers."""
        event = Event(name=name, payload=payload)
        with self._lock:
            handlers = list(self._handlers.get(name, ())) + list(self._handlers.get(None, ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - bus isolates handler failures by design
                _LOGGER.exception("event handler %r failed for event %r", handler, name)
