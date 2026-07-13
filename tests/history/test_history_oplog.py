"""Tests for nlapt.history.oplog: append/order/events and snapshot rollback."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.errors import OperationError, ValidationError
from nlapt.core.events import EVT_OPLOG_APPENDED, Event, EventBus
from nlapt.history.oplog import OperationLog, OperationRecord
from nlapt.storage.snapshots import SnapshotManager

UTF8 = "utf-8"
FIXED_TIME = 1_700_000_000.0


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def log(bus: EventBus) -> OperationLog:
    return OperationLog(bus)


def _append(log: OperationLog, description: str = "batch replace", **overrides):
    kwargs = {
        "description": description,
        "affected_keys": ("a.txt", "b.txt"),
        "snapshot": None,
        "kind": "batch",
    }
    kwargs.update(overrides)
    return log.append(**kwargs)


# -- append / records --------------------------------------------------------------


def test_append_assigns_incrementing_ids_and_timestamp(
    log: OperationLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nlapt.history.oplog._now", lambda: FIXED_TIME)
    first = _append(log, "first")
    second = _append(log, "second")
    assert isinstance(first, OperationRecord)
    assert first.op_id == 1
    assert second.op_id == 2
    assert first.timestamp == FIXED_TIME
    assert first.affected_keys == ("a.txt", "b.txt")
    assert first.kind == "batch"


def test_records_are_newest_first(log: OperationLog) -> None:
    _append(log, "first")
    _append(log, "second")
    _append(log, "third")
    descriptions = [record.description for record in log.records()]
    assert descriptions == ["third", "second", "first"]


def test_append_publishes_event(bus: EventBus, log: OperationLog) -> None:
    events: list[Event] = []
    bus.subscribe(EVT_OPLOG_APPENDED, events.append)
    record = _append(log, "batch replace")
    assert len(events) == 1
    payload = events[0].payload
    assert payload["op_id"] == record.op_id
    assert payload["description"] == "batch replace"
    assert payload["kind"] == "batch"
    assert payload["affected"] == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"description": ""},
        {"description": "   "},
        {"kind": ""},
        {"affected_keys": "a.txt"},
        {"affected_keys": ("a.txt", "")},
        {"affected_keys": (1,)},
        {"snapshot": "not-a-snapshot"},
    ],
)
def test_append_validates_arguments(log: OperationLog, overrides: dict) -> None:
    with pytest.raises(ValidationError):
        _append(log, **overrides)


def test_constructor_requires_event_bus() -> None:
    with pytest.raises(ValidationError):
        OperationLog("not-a-bus")  # type: ignore[arg-type]


# -- rollback -----------------------------------------------------------------------


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(f"original {name}", encoding=UTF8)
    return tmp_path


def test_rollback_restores_only_affected_keys(
    dataset: Path, log: OperationLog
) -> None:
    manager = SnapshotManager(dataset)
    snapshot = manager.create("batch replace")
    for name in ("a.txt", "b.txt", "c.txt"):
        (dataset / name).write_text(f"modified {name}", encoding=UTF8)
    record = _append(
        log, "batch replace", affected_keys=("a.txt", "b.txt"), snapshot=snapshot
    )

    result = log.rollback(record.op_id, manager)

    assert result.restored_files == ("a.txt", "b.txt")
    assert (dataset / "a.txt").read_text(encoding=UTF8) == "original a.txt"
    assert (dataset / "b.txt").read_text(encoding=UTF8) == "original b.txt"
    # c.txt was not affected by the operation: it keeps its modified content.
    assert (dataset / "c.txt").read_text(encoding=UTF8) == "modified c.txt"


def test_rollback_appends_rollback_record(dataset: Path, log: OperationLog) -> None:
    manager = SnapshotManager(dataset)
    snapshot = manager.create("batch replace")
    (dataset / "a.txt").write_text("modified", encoding=UTF8)
    record = _append(log, "batch replace", affected_keys=("a.txt",), snapshot=snapshot)

    result = log.rollback(record.op_id, manager)

    newest = log.records()[0]
    assert newest.kind == "rollback"
    assert str(record.op_id) in newest.description
    assert newest.affected_keys == ("a.txt",)
    # The rollback record carries the pre-restore snapshot: it can be rolled back too.
    assert newest.snapshot == result.pre_restore_snapshot
    assert newest.snapshot is not None


def test_rollback_of_rollback_restores_modified_state(
    dataset: Path, log: OperationLog
) -> None:
    manager = SnapshotManager(dataset)
    snapshot = manager.create("batch replace")
    (dataset / "a.txt").write_text("modified", encoding=UTF8)
    record = _append(log, "batch replace", affected_keys=("a.txt",), snapshot=snapshot)
    log.rollback(record.op_id, manager)
    rollback_record = log.records()[0]

    log.rollback(rollback_record.op_id, manager)

    assert (dataset / "a.txt").read_text(encoding=UTF8) == "modified"


def test_rollback_unknown_id_raises(dataset: Path, log: OperationLog) -> None:
    manager = SnapshotManager(dataset)
    with pytest.raises(OperationError):
        log.rollback(999, manager)


def test_rollback_without_snapshot_raises(dataset: Path, log: OperationLog) -> None:
    manager = SnapshotManager(dataset)
    record = _append(log, "manual edit", snapshot=None, kind="edit")
    with pytest.raises(OperationError):
        log.rollback(record.op_id, manager)


def test_rollback_validates_arguments(dataset: Path, log: OperationLog) -> None:
    manager = SnapshotManager(dataset)
    with pytest.raises(ValidationError):
        log.rollback("1", manager)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        log.rollback(True, manager)
    with pytest.raises(ValidationError):
        log.rollback(1, "not-a-manager")  # type: ignore[arg-type]
