"""Session-level operation log with whole-operation rollback (spec 7.5).

Every batch/edit/restore operation appends an immutable
:class:`OperationRecord`. Records carry the pre-execution snapshot (when one
was taken), which enables :meth:`OperationLog.rollback`: only the affected
keys are restored through ``SnapshotManager.restore(only=...)`` and the
rollback itself is appended as a new record — carrying the pre-restore
snapshot, so a rollback can itself be rolled back.

The log is session-scoped (in-memory); cross-session recovery is provided by
the snapshot system (spec 2.4).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

from nlapt.core.errors import OperationError, ValidationError
from nlapt.core.events import EVT_OPLOG_APPENDED, EventBus
from nlapt.diagnostics import get_logger
from nlapt.storage.snapshots import RestoreResult, SnapshotInfo, SnapshotManager

_LOGGER = get_logger(__name__)

FIRST_OP_ID = 1
ROLLBACK_KIND = "rollback"


def _now() -> float:
    """Record timestamp source; module-level so tests can monkeypatch it."""
    return time.time()


@dataclass(frozen=True)
class OperationRecord:
    """One logged operation of the current session."""

    op_id: int
    timestamp: float
    description: str  # e.g. 'batch replace "girl" -> "woman" · 9 files 17 hits'
    affected_keys: tuple[str, ...]
    snapshot: SnapshotInfo | None  # pre-execution snapshot; enables rollback
    kind: str  # "batch" | "edit" | "rollback" | "restore" | ...


class OperationLog:
    """Thread-safe, append-only in-memory log; newest records first."""

    def __init__(self, bus: EventBus) -> None:
        if not isinstance(bus, EventBus):
            raise ValidationError(f"bus must be an EventBus, got {type(bus).__name__}")
        self._bus = bus
        self._lock = threading.RLock()
        self._records: list[OperationRecord] = []  # append order (oldest first)
        self._next_id = FIRST_OP_ID

    def append(
        self,
        *,
        description: str,
        affected_keys: Sequence[str],
        snapshot: SnapshotInfo | None,
        kind: str,
    ) -> OperationRecord:
        """Append a record, publish ``EVT_OPLOG_APPENDED``, and return it."""
        _validate_append_args(description, affected_keys, snapshot, kind)
        with self._lock:
            record = OperationRecord(
                op_id=self._next_id,
                timestamp=_now(),
                description=description,
                affected_keys=tuple(affected_keys),
                snapshot=snapshot,
                kind=kind,
            )
            self._next_id += 1
            self._records.append(record)
        self._bus.publish(
            EVT_OPLOG_APPENDED,
            op_id=record.op_id,
            description=record.description,
            kind=record.kind,
            affected=len(record.affected_keys),
        )
        _LOGGER.info("oplog #%d [%s]: %s", record.op_id, record.kind, record.description)
        return record

    def records(self) -> tuple[OperationRecord, ...]:
        """All records, newest first (spec 7.5: 按时间倒序)."""
        with self._lock:
            return tuple(reversed(self._records))

    def rollback(self, op_id: int, snapshots: SnapshotManager) -> RestoreResult:
        """Restore ONLY the affected keys of record ``op_id`` from its snapshot.

        Raises OperationError when the id is unknown or the record carries no
        snapshot. Keys created by the operation (absent from the snapshot) are
        not deleted — a documented v1 limitation. The rollback is appended as
        a new record whose snapshot is the pre-restore snapshot, so it can be
        rolled back in turn.
        """
        if not isinstance(op_id, int) or isinstance(op_id, bool):
            raise ValidationError(f"op_id must be an int, got {op_id!r}")
        if not isinstance(snapshots, SnapshotManager):
            raise ValidationError(
                f"snapshots must be a SnapshotManager, got {type(snapshots).__name__}"
            )
        with self._lock:
            record = next((r for r in self._records if r.op_id == op_id), None)
        if record is None:
            raise OperationError(f"cannot roll back: unknown operation id {op_id}")
        if record.snapshot is None:
            raise OperationError(
                f"cannot roll back operation #{op_id} ({record.description!r}): "
                "no snapshot was taken for it"
            )
        result = snapshots.restore(record.snapshot, only=record.affected_keys)
        self.append(
            description=f"rollback of #{op_id}: {record.description}",
            affected_keys=result.restored_files,
            snapshot=result.pre_restore_snapshot,
            kind=ROLLBACK_KIND,
        )
        _LOGGER.info(
            "rolled back operation #%d, restored %d file(s)",
            op_id, len(result.restored_files),
        )
        return result


def _validate_append_args(
    description: str,
    affected_keys: Sequence[str],
    snapshot: SnapshotInfo | None,
    kind: str,
) -> None:
    if not isinstance(description, str) or not description.strip():
        raise ValidationError(f"description must be a non-empty string, got {description!r}")
    if not isinstance(kind, str) or not kind.strip():
        raise ValidationError(f"kind must be a non-empty string, got {kind!r}")
    if isinstance(affected_keys, str) or not isinstance(affected_keys, Sequence):
        raise ValidationError(
            f"affected_keys must be a sequence of strings, got {type(affected_keys).__name__}"
        )
    for key in affected_keys:
        if not isinstance(key, str) or not key:
            raise ValidationError(f"affected_keys must all be non-empty strings, got {key!r}")
    if snapshot is not None and not isinstance(snapshot, SnapshotInfo):
        raise ValidationError(
            f"snapshot must be a SnapshotInfo or None, got {type(snapshot).__name__}"
        )
