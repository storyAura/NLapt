"""Central in-memory caption store (spec 4.1, 10 state machine context).

``CaptionStore`` maps ``key -> CaptionRecord``. Records are frozen dataclasses;
every mutation replaces the record with a new one under an ``RLock`` and
publishes an event on the injected :class:`~nlapt.core.events.EventBus`.

The pending AI suggestion is an orthogonal overlay (spec 4.1 supplementary
rule): it never overwrites the body text and coexists with the body state.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace

from nlapt.core.errors import ValidationError
from nlapt.core.events import (
    EVT_CAPTION_CHANGED,
    EVT_PENDING_CLEARED,
    EVT_PENDING_SET,
    EVT_STATE_CHANGED,
    EventBus,
)
from nlapt.core.states import CaptionState, FileStatus, initial_state, state_after_edit
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Keys of the mapping returned by CaptionStore.counts().
COUNT_UNLABELED = "unlabeled"
COUNT_DRAFT = "draft"
COUNT_CONFIRMED = "confirmed"
COUNT_PENDING = "pending"


@dataclass(frozen=True)
class PendingSuggestion:
    """An unreviewed AI suggestion attached to a caption (spec 4.1, 10)."""

    text: str
    source: str
    created_at: float


@dataclass(frozen=True)
class CaptionRecord:
    """Immutable snapshot of one caption file's in-memory editing state."""

    key: str
    text: str
    state: CaptionState
    dirty: bool = False
    pending: PendingSuggestion | None = None


def _validate_key(key: str) -> None:
    if not isinstance(key, str) or not key:
        raise ValidationError(f"caption key must be a non-empty string, got {key!r}")


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise ValidationError(f"caption text must be a string, got {type(text).__name__}")


class CaptionStore:
    """Thread-safe in-memory store of caption records.

    Records are replaced, never mutated. State transitions follow
    :func:`nlapt.core.states.state_after_edit`; ``revert_confirmed_on_edit``
    controls whether editing a CONFIRMED caption reverts it to DRAFT
    (spec 4.1 supplementary rule).
    """

    def __init__(self, bus: EventBus, *, revert_confirmed_on_edit: bool = True) -> None:
        if not isinstance(bus, EventBus):
            raise ValidationError(f"bus must be an EventBus, got {type(bus).__name__}")
        self._bus = bus
        self._revert_confirmed_on_edit = bool(revert_confirmed_on_edit)
        self._lock = threading.RLock()
        self._records: dict[str, CaptionRecord] = {}

    # -- loading / lookup ---------------------------------------------------

    def load(self, key: str, text: str) -> CaptionRecord:
        """Register a caption as read from disk (not dirty, no events)."""
        _validate_key(key)
        _validate_text(text)
        record = CaptionRecord(key=key, text=text, state=initial_state(text))
        with self._lock:
            self._records[key] = record
        return record

    def get(self, key: str) -> CaptionRecord:
        """Return the record for ``key``; raises ``KeyError`` if unknown."""
        with self._lock:
            return self._records[key]

    def keys(self) -> tuple[str, ...]:
        """All known keys in insertion order."""
        with self._lock:
            return tuple(self._records)

    # -- editing ------------------------------------------------------------

    def set_text(self, key: str, text: str) -> CaptionRecord:
        """Apply a manual edit: new text, state transition, dirty flag."""
        _validate_text(text)
        with self._lock:
            current = self.get(key)
            new_state = state_after_edit(
                current.state, text, revert_confirmed=self._revert_confirmed_on_edit
            )
            record = replace(current, text=text, state=new_state, dirty=True)
            self._records[key] = record
        self._bus.publish(
            EVT_CAPTION_CHANGED,
            key=key,
            text=record.text,
            state=record.state.value,
            dirty=record.dirty,
        )
        return record

    def set_text_forced_state(
        self, key: str, text: str, state: CaptionState
    ) -> CaptionRecord:
        """Set new text with an explicit target state (bypasses transitions).

        Used by workflows that must pin the resulting state regardless of the
        ``revert_confirmed_on_edit`` setting — e.g. "accept for edit" loads AI
        text as a DRAFT body (spec 10 编辑后接受). Marks the record dirty and
        publishes ``EVT_CAPTION_CHANGED`` (plus ``EVT_STATE_CHANGED`` when the
        state actually changed). Raises ``ValidationError`` for empty text
        with a non-UNLABELED target state.
        """
        _validate_text(text)
        if not isinstance(state, CaptionState):
            raise ValidationError(
                f"state must be a CaptionState, got {type(state).__name__}"
            )
        if state is not CaptionState.UNLABELED and not text.strip():
            raise ValidationError(
                f"cannot force state {state.value!r} for {key!r}: text is empty"
            )
        with self._lock:
            current = self.get(key)
            state_changed = current.state is not state
            record = replace(current, text=text, state=state, dirty=True)
            self._records[key] = record
        self._bus.publish(
            EVT_CAPTION_CHANGED,
            key=key,
            text=record.text,
            state=record.state.value,
            dirty=record.dirty,
        )
        if state_changed:
            self._bus.publish(EVT_STATE_CHANGED, key=key, state=record.state.value)
        return record

    def confirm(self, key: str) -> CaptionRecord:
        """Mark the caption human-confirmed (spec 6.4).

        Raises ``ValidationError`` when the text is empty/whitespace —
        an empty caption cannot be confirmed.
        """
        with self._lock:
            current = self.get(key)
            if not current.text.strip():
                raise ValidationError(f"cannot confirm {key!r}: caption text is empty")
            record = replace(current, state=CaptionState.CONFIRMED)
            self._records[key] = record
        self._bus.publish(EVT_STATE_CHANGED, key=key, state=record.state.value)
        return record

    # -- pending suggestions --------------------------------------------------

    def set_pending(self, key: str, suggestion: PendingSuggestion) -> CaptionRecord:
        """Attach a pending AI suggestion (overlay; body text untouched)."""
        if not isinstance(suggestion, PendingSuggestion):
            raise ValidationError(
                f"suggestion must be a PendingSuggestion, got {type(suggestion).__name__}"
            )
        _validate_text(suggestion.text)
        with self._lock:
            record = replace(self.get(key), pending=suggestion)
            self._records[key] = record
        self._bus.publish(EVT_PENDING_SET, key=key, source=suggestion.source)
        return record

    def clear_pending(self, key: str) -> CaptionRecord:
        """Remove the pending suggestion. Idempotent: publishes
        ``EVT_PENDING_CLEARED`` only when a suggestion was actually present."""
        with self._lock:
            current = self.get(key)
            had_pending = current.pending is not None
            record = replace(current, pending=None) if had_pending else current
            self._records[key] = record
        if had_pending:
            self._bus.publish(EVT_PENDING_CLEARED, key=key)
        return record

    # -- persistence bookkeeping ----------------------------------------------

    def mark_saved(self, key: str) -> CaptionRecord:
        """Clear the dirty flag after the caption was written to disk."""
        with self._lock:
            record = replace(self.get(key), dirty=False)
            self._records[key] = record
        return record

    # -- queries ---------------------------------------------------------------

    def dirty_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(k for k, r in self._records.items() if r.dirty)

    def pending_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(k for k, r in self._records.items() if r.pending is not None)

    def status(self, key: str) -> FileStatus:
        record = self.get(key)
        return FileStatus(state=record.state, has_pending=record.pending is not None)

    def counts(self) -> Mapping[str, int]:
        """Per-state counts plus the orthogonal ``pending`` overlay count."""
        with self._lock:
            records = tuple(self._records.values())
        counts = {
            COUNT_UNLABELED: 0,
            COUNT_DRAFT: 0,
            COUNT_CONFIRMED: 0,
            COUNT_PENDING: 0,
        }
        for record in records:
            counts[record.state.value] += 1
            if record.pending is not None:
                counts[COUNT_PENDING] += 1
        return counts

    def texts(self) -> Mapping[str, str]:
        """Snapshot copy of ``key -> current text`` (safe to mutate by caller)."""
        with self._lock:
            return {k: r.text for k, r in self._records.items()}
