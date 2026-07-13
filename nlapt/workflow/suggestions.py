"""Accept/reject glue over :class:`~nlapt.captions.store.CaptionStore` (spec 10).

The suggestion bar offers three actions on a pending AI suggestion:

- **accept**: pending text becomes the body (normal edit-state transitions
  apply), then the pending marker is cleared. Persisting to disk and jumping
  to the next pending file is the facade's job.
- **reject**: the pending suggestion is discarded; the body is untouched.
- **accept for edit** ("编辑后接受"): pending text is loaded as a DRAFT body
  for further manual editing — it is NOT confirmed — and the marker cleared.

All three raise :class:`~nlapt.core.errors.OperationError` when the key has
no pending suggestion. Unknown keys propagate the store's ``KeyError``.
"""

from __future__ import annotations

from nlapt.captions.store import CaptionRecord, CaptionStore, PendingSuggestion
from nlapt.core.errors import OperationError, ValidationError
from nlapt.core.states import CaptionState
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)


def accept_suggestion(store: CaptionStore, key: str) -> CaptionRecord:
    """Write the pending text into the body, then clear the pending marker."""
    pending = _require_pending(store, key)
    store.set_text(key, pending.text)
    record = store.clear_pending(key)
    _LOGGER.info("suggestion accepted for %r (source=%s)", key, pending.source)
    return record


def reject_suggestion(store: CaptionStore, key: str) -> CaptionRecord:
    """Discard the pending suggestion; the body text stays untouched."""
    pending = _require_pending(store, key)
    record = store.clear_pending(key)
    _LOGGER.info("suggestion rejected for %r (source=%s)", key, pending.source)
    return record


def accept_for_edit(store: CaptionStore, key: str) -> CaptionRecord:
    """Load the pending text as a DRAFT body for manual editing (NOT confirmed).

    The DRAFT state is forced explicitly: even with
    ``revert_confirmed_on_edit=False`` the unreviewed AI text must never land
    as CONFIRMED. The user edits and confirms explicitly afterwards
    (spec 10 "编辑后接受").
    """
    pending = _require_pending(store, key)
    store.set_text_forced_state(key, pending.text, CaptionState.DRAFT)
    record = store.clear_pending(key)
    _LOGGER.info("suggestion loaded for editing on %r (source=%s)", key, pending.source)
    return record


def _require_pending(store: CaptionStore, key: str) -> PendingSuggestion:
    """Return the pending suggestion for ``key`` or raise OperationError."""
    if not isinstance(store, CaptionStore):
        raise ValidationError(f"store must be a CaptionStore, got {type(store).__name__}")
    if not isinstance(key, str) or not key:
        raise ValidationError(f"key must be a non-empty string, got {key!r}")
    record = store.get(key)  # KeyError for unknown keys, per store contract
    if record.pending is None:
        raise OperationError(f"no pending suggestion for {key!r}")
    return record.pending
