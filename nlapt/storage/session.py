"""Crash-recovery session persistence (spec 12 reliability).

``SessionStore`` saves the volatile editing state (unsaved drafts, pending
AI suggestions and per-file states) to ``session.json``. The default path
is ``<root>/.nlapt/session.json``; pass ``state_dir`` to store it outside
the dataset (the facade uses the per-user dataset state folder).
Writes are atomic; corrupt files raise :class:`SessionError`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from nlapt.core.errors import SessionError, StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

_LOGGER = get_logger(__name__)

SESSION_DIR_NAME = ".nlapt"
SESSION_FILE_NAME = "session.json"
SESSION_ENCODING = "utf-8"
JSON_INDENT = 2

_MAPPING_FIELDS = ("drafts", "pending", "pending_sources", "states")
_SAVED_AT_FIELD = "saved_at"


@dataclass(frozen=True)
class SessionSnapshot:
    """Immutable snapshot of recoverable in-memory editing state."""

    drafts: Mapping[str, str]          # key -> unsaved caption text
    pending: Mapping[str, str]         # key -> pending suggestion text
    pending_sources: Mapping[str, str]  # key -> source label (e.g. "rewrite:polish")
    states: Mapping[str, str]          # key -> CaptionState value
    saved_at: float                    # unix timestamp


class SessionStore:
    """Persist/load a :class:`SessionSnapshot` under ``<root>/.nlapt/session.json``."""

    def __init__(self, root: Path, *, state_dir: Path | None = None) -> None:
        base = Path(root)
        if not base.is_dir():
            raise ValidationError(f"session root is not a directory: {base}")
        if not base.is_absolute():
            base = base.absolute()
        self._root = base
        if state_dir is None:
            self._path = base / SESSION_DIR_NAME / SESSION_FILE_NAME
        else:
            dest = Path(state_dir)
            if not dest.is_absolute():
                dest = dest.absolute()
            self._path = dest / SESSION_FILE_NAME

    def save(self, snapshot: SessionSnapshot) -> None:
        """Atomically write ``snapshot`` as UTF-8 JSON.

        Raises ValidationError for malformed snapshot data and SessionError
        when the file cannot be written.
        """
        payload = _snapshot_to_payload(snapshot)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT)
            atomic_write_text(self._path, text)
        except (OSError, StorageError) as exc:
            raise SessionError(f"cannot save session to {self._path}: {exc}") from exc
        _LOGGER.debug("session saved to %s", self._path)

    def load(self) -> SessionSnapshot | None:
        """Return the stored snapshot, or None when no session file exists.

        Raises SessionError when the file exists but is unreadable or corrupt.
        """
        try:
            data = self._path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SessionError(f"cannot read session file {self._path}: {exc}") from exc
        try:
            payload = json.loads(data.decode(SESSION_ENCODING))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionError(f"corrupt session file {self._path}: {exc}") from exc
        return _payload_to_snapshot(payload, self._path)

    def clear(self) -> None:
        """Remove the session file; no-op when it does not exist."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            raise SessionError(f"cannot remove session file {self._path}: {exc}") from exc
        _LOGGER.debug("session file cleared: %s", self._path)


def _snapshot_to_payload(snapshot: SessionSnapshot) -> dict[str, Any]:
    """Validate a snapshot at the write boundary and convert it to plain JSON data."""
    payload: dict[str, Any] = {}
    for field_name in _MAPPING_FIELDS:
        mapping = getattr(snapshot, field_name)
        if not isinstance(mapping, Mapping):
            raise ValidationError(
                f"session field {field_name!r} must be a mapping, "
                f"got {type(mapping).__name__}"
            )
        converted: dict[str, str] = {}
        for key, value in mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValidationError(
                    f"session field {field_name!r} must map str to str, "
                    f"got {key!r} -> {value!r}"
                )
            converted[key] = value
        payload[field_name] = converted
    if isinstance(snapshot.saved_at, bool) or not isinstance(snapshot.saved_at, (int, float)):
        raise ValidationError(
            f"session saved_at must be a number, got {snapshot.saved_at!r}"
        )
    payload[_SAVED_AT_FIELD] = float(snapshot.saved_at)
    return payload


def _payload_to_snapshot(payload: Any, path: Path) -> SessionSnapshot:
    """Validate loaded JSON and build an immutable snapshot (SessionError on corruption)."""
    if not isinstance(payload, dict):
        raise SessionError(f"corrupt session file {path}: top level is not an object")
    mappings: dict[str, Mapping[str, str]] = {}
    for field_name in _MAPPING_FIELDS:
        raw = payload.get(field_name)
        if not isinstance(raw, dict):
            raise SessionError(
                f"corrupt session file {path}: field {field_name!r} missing or not an object"
            )
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SessionError(
                    f"corrupt session file {path}: field {field_name!r} "
                    f"has a non-string entry {key!r} -> {value!r}"
                )
        mappings[field_name] = MappingProxyType(dict(raw))
    saved_at = payload.get(_SAVED_AT_FIELD)
    if isinstance(saved_at, bool) or not isinstance(saved_at, (int, float)):
        raise SessionError(
            f"corrupt session file {path}: field {_SAVED_AT_FIELD!r} missing or not a number"
        )
    return SessionSnapshot(
        drafts=mappings["drafts"],
        pending=mappings["pending"],
        pending_sources=mappings["pending_sources"],
        states=mappings["states"],
        saved_at=float(saved_at),
    )
