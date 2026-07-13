"""Tests for nlapt.storage.snapshots (spec 2.4)."""

from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from nlapt.core.errors import SnapshotError, ValidationError
from nlapt.storage import snapshots as snapshots_module
from nlapt.storage.snapshots import (
    BACKUP_DIR_NAME,
    SnapshotInfo,
    SnapshotManager,
)

FIXED_TIME = datetime(2026, 7, 11, 14, 30, 45)
FIXED_STAMP = "2026-07-11_1430"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    (tmp_path / "img1.txt").write_bytes(b"caption one\n")
    (tmp_path / "img2.txt").write_bytes(b"caption two\n")
    sub = tmp_path / "10_concept"
    sub.mkdir()
    (sub / "img3.txt").write_bytes(b"caption three\n")
    (tmp_path / "img1.png").write_bytes(b"fake")
    return tmp_path


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch):
    """Freeze snapshots._now; returns a setter to advance the clock."""
    holder = {"value": FIXED_TIME}
    monkeypatch.setattr(snapshots_module, "_now", lambda: holder["value"])

    def set_time(value: datetime) -> None:
        holder["value"] = value

    return set_time


class TestInit:
    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            SnapshotManager(tmp_path / "nope")

    def test_bad_retention_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            SnapshotManager(tmp_path, retention=0)

    def test_real_clock_produces_valid_name(self, dataset: Path) -> None:
        import re

        info = SnapshotManager(dataset).create("op")
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}_\d{4}_op(_\d+)?\.zip", info.path.name
        )


class TestCreate:
    def test_zip_contains_all_txts_recursively(self, dataset: Path, frozen_now) -> None:
        info = SnapshotManager(dataset).create("批量替换")
        assert info.path.parent == dataset / BACKUP_DIR_NAME
        assert info.path.name == f"{FIXED_STAMP}_批量替换.zip"
        assert info.file_count == 3
        assert info.created == datetime(2026, 7, 11, 14, 30)
        with zipfile.ZipFile(info.path) as archive:
            assert sorted(archive.namelist()) == [
                "10_concept/img3.txt",
                "img1.txt",
                "img2.txt",
            ]
            assert archive.read("img1.txt") == b"caption one\n"

    def test_excludes_backups_and_nlapt_dirs(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        manager.create("first")  # populates .backups
        nlapt_dir = dataset / ".nlapt"
        nlapt_dir.mkdir()
        (nlapt_dir / "session.txt").write_bytes(b"internal")
        backup_note = dataset / BACKUP_DIR_NAME / "note.txt"
        backup_note.write_bytes(b"not a caption")
        info = SnapshotManager(dataset).create("second")
        with zipfile.ZipFile(info.path) as archive:
            names = archive.namelist()
        assert all(not name.startswith((".backups", ".nlapt")) for name in names)
        assert info.file_count == 3

    def test_empty_dataset_gives_zero_entry_zip(self, tmp_path: Path, frozen_now) -> None:
        info = SnapshotManager(tmp_path).create("noop")
        assert info.file_count == 0
        with zipfile.ZipFile(info.path) as archive:
            assert archive.namelist() == []

    def test_operation_name_sanitized(self, dataset: Path, frozen_now) -> None:
        info = SnapshotManager(dataset).create('bad/op:na*me?"<>| x')
        assert info.path.name == f"{FIXED_STAMP}_bad_op_na_me______x.zip"

    def test_empty_operation_raises(self, dataset: Path) -> None:
        with pytest.raises(ValidationError):
            SnapshotManager(dataset).create("   ")

    def test_collision_appends_numeric_suffix(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        first = manager.create("op")
        second = manager.create("op")
        third = manager.create("op")
        assert first.path.name == f"{FIXED_STAMP}_op.zip"
        assert second.path.name == f"{FIXED_STAMP}_op_2.zip"
        assert third.path.name == f"{FIXED_STAMP}_op_3.zip"

    def test_retention_prunes_oldest(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset, retention=3)
        for minute in range(5):
            frozen_now(datetime(2026, 7, 11, 10, minute))
            manager.create(f"op{minute}")
        remaining = sorted(p.name for p in (dataset / BACKUP_DIR_NAME).glob("*.zip"))
        assert remaining == [
            "2026-07-11_1002_op2.zip",
            "2026-07-11_1003_op3.zip",
            "2026-07-11_1004_op4.zip",
        ]


class TestList:
    def test_no_backup_dir_gives_empty(self, tmp_path: Path) -> None:
        assert SnapshotManager(tmp_path).list_snapshots() == ()

    def test_newest_first(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        frozen_now(datetime(2026, 7, 11, 9, 0))
        manager.create("old")
        frozen_now(datetime(2026, 7, 11, 11, 0))
        manager.create("new")
        listed = manager.list_snapshots()
        assert [info.operation for info in listed] == ["new", "old"]
        assert all(info.file_count == 3 for info in listed)

    def test_skips_corrupt_snapshot_zip(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        manager.create("good")
        corrupt = dataset / BACKUP_DIR_NAME / "2026-07-11_0800_corrupt.zip"
        corrupt.write_bytes(b"this is not a zip file")
        listed = manager.list_snapshots()
        assert [info.operation for info in listed] == ["good"]

    def test_ignores_foreign_files(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        manager.create("op")
        (dataset / BACKUP_DIR_NAME / "random.zip").write_bytes(b"not named right")
        (dataset / BACKUP_DIR_NAME / "readme.md").write_text("x", encoding="utf-8")
        listed = manager.list_snapshots()
        assert [info.operation for info in listed] == ["op"]


class TestRestore:
    def test_preview_counts_files(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        info = manager.create("op")
        assert manager.preview_restore(info) == 3

    def test_restore_reproduces_exact_bytes(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        info = manager.create("op")
        (dataset / "img1.txt").write_bytes(b"MODIFIED")
        (dataset / "img2.txt").unlink()
        result = manager.restore(info)
        assert result.restored_files == (
            "10_concept/img3.txt",
            "img1.txt",
            "img2.txt",
        )
        assert (dataset / "img1.txt").read_bytes() == b"caption one\n"
        assert (dataset / "img2.txt").read_bytes() == b"caption two\n"
        assert (dataset / "10_concept" / "img3.txt").read_bytes() == b"caption three\n"

    def test_restore_recreates_missing_directories(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        info = manager.create("op")
        (dataset / "10_concept" / "img3.txt").unlink()
        (dataset / "10_concept").rmdir()
        manager.restore(info)
        assert (dataset / "10_concept" / "img3.txt").read_bytes() == b"caption three\n"

    def test_restore_creates_pre_restore_snapshot_first(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        info = manager.create("op")
        (dataset / "img1.txt").write_bytes(b"MODIFIED")
        result = manager.restore(info)
        pre = result.pre_restore_snapshot
        assert pre.path.exists()
        assert pre.operation == snapshots_module.PRE_RESTORE_OPERATION
        # the pre-restore snapshot captures the state BEFORE restoring
        with zipfile.ZipFile(pre.path) as archive:
            assert archive.read("img1.txt") == b"MODIFIED"

    def test_restore_only_subset(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        info = manager.create("op")
        (dataset / "img1.txt").write_bytes(b"CHANGED-1")
        (dataset / "img2.txt").write_bytes(b"CHANGED-2")
        result = manager.restore(info, only=["img1.txt"])
        assert result.restored_files == ("img1.txt",)
        assert (dataset / "img1.txt").read_bytes() == b"caption one\n"
        assert (dataset / "img2.txt").read_bytes() == b"CHANGED-2"

    def test_restore_only_ignores_unknown_keys(self, dataset: Path, frozen_now) -> None:
        manager = SnapshotManager(dataset)
        info = manager.create("op")
        result = manager.restore(info, only=["img1.txt", "does-not-exist.txt"])
        assert result.restored_files == ("img1.txt",)

    def test_restore_missing_zip_raises(self, dataset: Path) -> None:
        manager = SnapshotManager(dataset)
        ghost = SnapshotInfo(
            path=dataset / BACKUP_DIR_NAME / "gone.zip",
            created=FIXED_TIME,
            operation="gone",
            file_count=0,
        )
        with pytest.raises(SnapshotError):
            manager.restore(ghost)
        with pytest.raises(SnapshotError):
            manager.preview_restore(ghost)


class TestZipSlip:
    @pytest.mark.parametrize(
        "entry",
        [
            "../evil.txt",
            "sub/../../evil.txt",
            "/abs/evil.txt",
            "C:/evil.txt",
            "..\\evil.txt",
        ],
    )
    def test_unsafe_entries_rejected(self, dataset: Path, entry: str, frozen_now) -> None:
        backup_dir = dataset / BACKUP_DIR_NAME
        backup_dir.mkdir(exist_ok=True)
        evil_zip = backup_dir / "2026-07-11_0900_evil.zip"
        with zipfile.ZipFile(evil_zip, "w") as archive:
            archive.writestr(entry, b"pwned")
        manager = SnapshotManager(dataset)
        info = SnapshotInfo(path=evil_zip, created=FIXED_TIME, operation="evil", file_count=1)
        with pytest.raises(SnapshotError):
            manager.restore(info)
        assert not (dataset.parent / "evil.txt").exists()
        assert not (dataset / "evil.txt").exists()

    def test_unsafe_entry_makes_no_pre_restore_snapshot(self, dataset: Path, frozen_now) -> None:
        backup_dir = dataset / BACKUP_DIR_NAME
        backup_dir.mkdir(exist_ok=True)
        evil_zip = backup_dir / "2026-07-11_0900_evil.zip"
        with zipfile.ZipFile(evil_zip, "w") as archive:
            archive.writestr("../evil.txt", b"pwned")
        manager = SnapshotManager(dataset)
        info = SnapshotInfo(path=evil_zip, created=FIXED_TIME, operation="evil", file_count=1)
        before = set(backup_dir.iterdir())
        with pytest.raises(SnapshotError):
            manager.restore(info)
        assert set(backup_dir.iterdir()) == before
