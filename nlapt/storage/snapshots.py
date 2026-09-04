"""Zip snapshots of all caption txts (spec 2.4).

Before any multi-file batch operation, ``SnapshotManager.create`` packs every
``*.txt`` under the dataset root (excluding ``.backups/`` and ``.nlapt/``)
into ``<root>/.backups/YYYY-MM-DD_HHMM_<operation>.zip`` by default, or into
an explicit ``backup_dir`` (the facade uses the per-user dataset state
folder). Restore first takes a pre-restore snapshot so the restore itself
can be undone, and rejects zip-slip entries (absolute paths or ``..``
escapes) with SnapshotError.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Collection

from nlapt.core.errors import SnapshotError, StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_bytes
from nlapt.storage.scanner import TXT_EXTENSION, natural_sort_key
from nlapt.storage.session import SESSION_DIR_NAME

_LOGGER = get_logger(__name__)

BACKUP_DIR_NAME = ".backups"
SNAPSHOT_SUFFIX = ".zip"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M"
DEFAULT_RETENTION = 20
DEFAULT_OPERATION_NAME = "operation"
MAX_OPERATION_LENGTH = 60
SANITIZE_REPLACEMENT = "_"
FIRST_COLLISION_SUFFIX = 2
# Operation name for the automatic snapshot taken before a restore ("恢复前快照").
PRE_RESTORE_OPERATION = "pre_restore"

_EXCLUDED_DIRS = frozenset({BACKUP_DIR_NAME, SESSION_DIR_NAME})
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RUN = re.compile(r"\s+")
_SNAPSHOT_NAME = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2}_\d{4})_(?P<operation>.+)\.zip$"
)
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class SnapshotInfo:
    """Metadata for one snapshot zip in ``.backups/``."""

    path: Path
    created: datetime
    operation: str
    file_count: int


@dataclass(frozen=True)
class RestoreResult:
    """Outcome of a restore: restored txt keys plus the safety snapshot."""

    restored_files: tuple[str, ...]  # txt paths relative to root, POSIX
    pre_restore_snapshot: SnapshotInfo


def _now() -> datetime:
    """Snapshot timestamp source; module-level so tests can monkeypatch it."""
    return datetime.now()


class SnapshotManager:
    """Create, list, and restore zip snapshots of all caption txts."""

    def __init__(
        self,
        root: Path,
        *,
        retention: int = DEFAULT_RETENTION,
        backup_dir: Path | None = None,
    ) -> None:
        base = Path(root)
        if not base.is_dir():
            raise ValidationError(f"snapshot root is not a directory: {base}")
        if not base.is_absolute():
            base = base.absolute()
        if not isinstance(retention, int) or isinstance(retention, bool) or retention < 1:
            raise ValidationError(f"snapshot retention must be an int >= 1, got {retention!r}")
        self._root = base
        if backup_dir is None:
            self._backup_dir = base / BACKUP_DIR_NAME
        else:
            dest = Path(backup_dir)
            self._backup_dir = dest if dest.is_absolute() else dest.absolute()
        self._retention = retention

    def create(self, operation: str) -> SnapshotInfo:
        """Zip all txts under root into ``.backups/`` and enforce retention.

        An empty dataset produces a valid zip with zero entries. Raises
        ValidationError for an empty operation name, SnapshotError on IO failure.
        """
        return self._create(operation, retention_exclude=None)

    def _create(
        self, operation: str, *, retention_exclude: Path | None
    ) -> SnapshotInfo:
        """Create a snapshot, optionally protecting one zip from retention pruning."""
        sanitized = _sanitize_operation(operation)
        stamp = _now().replace(second=0, microsecond=0)
        files = _collect_txt_files(self._root)
        try:
            self._backup_dir.mkdir(parents=True, exist_ok=True)
            target = self._unique_target(stamp, sanitized)
            payload = _build_zip_bytes(files)
            atomic_write_bytes(target, payload)
        except (OSError, StorageError) as exc:
            raise SnapshotError(f"snapshot creation failed for {self._root}: {exc}") from exc
        _LOGGER.info("snapshot created: %s (%d txt files)", target, len(files))
        self._enforce_retention(exclude=retention_exclude)
        return SnapshotInfo(
            path=target, created=stamp, operation=sanitized, file_count=len(files)
        )

    def list_snapshots(self) -> tuple[SnapshotInfo, ...]:
        """All snapshots in ``.backups/``, newest first. Unreadable zips are
        skipped with a warning so one corrupt file cannot break listing."""
        infos: list[SnapshotInfo] = []
        for path, stamp, operation in self._snapshot_files():
            try:
                with zipfile.ZipFile(path) as archive:
                    count = sum(1 for name in archive.namelist() if not name.endswith("/"))
            except (OSError, zipfile.BadZipFile) as exc:
                _LOGGER.warning("skipping unreadable snapshot %s: %s", path, exc)
                continue
            infos.append(
                SnapshotInfo(path=path, created=stamp, operation=operation, file_count=count)
            )
        infos.sort(key=lambda info: (info.created, natural_sort_key(info.path.name)))
        return tuple(reversed(infos))

    def preview_restore(self, snapshot: SnapshotInfo) -> int:
        """Number of txt files the snapshot would restore."""
        return len(self._read_entries(Path(snapshot.path)))

    def restore(
        self, snapshot: SnapshotInfo, *, only: Collection[str] | None = None
    ) -> RestoreResult:
        """Restore txts from ``snapshot`` (all entries, or just the ``only`` keys).

        Entries are validated against zip-slip before anything is touched;
        then a pre-restore snapshot is created so the restore can be undone.
        Restored files reproduce the archived bytes exactly.

        All selected entry bytes are read into memory BEFORE the pre-restore
        snapshot is created: creating that snapshot enforces retention, which
        must never destroy the very zip being restored. As defense in depth the
        target zip is also excluded from retention pruning during that create.
        """
        zip_path = Path(snapshot.path)
        entries = self._read_entries(zip_path)
        selected = _select_entries(entries, only)

        try:
            with zipfile.ZipFile(zip_path) as archive:
                payloads = tuple((name, archive.read(name)) for name in selected)
        except (OSError, zipfile.BadZipFile) as exc:
            raise SnapshotError(f"restore from {zip_path} failed: {exc}") from exc

        pre_restore = self._create(PRE_RESTORE_OPERATION, retention_exclude=zip_path)
        try:
            for name, data in payloads:
                target = self._root / PurePosixPath(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(target, data)
        except (OSError, StorageError) as exc:
            raise SnapshotError(f"restore from {zip_path} failed: {exc}") from exc
        _LOGGER.info("restored %d txt files from %s", len(selected), zip_path)
        return RestoreResult(
            restored_files=tuple(selected), pre_restore_snapshot=pre_restore
        )

    def _read_entries(self, zip_path: Path) -> list[str]:
        """Read and validate all restorable entry names (natural-sorted)."""
        try:
            with zipfile.ZipFile(zip_path) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/")]
        except (OSError, zipfile.BadZipFile) as exc:
            raise SnapshotError(f"cannot open snapshot {zip_path}: {exc}") from exc
        for name in names:
            self._validate_entry(name, zip_path)
        names.sort(key=natural_sort_key)
        return names

    def _validate_entry(self, name: str, zip_path: Path) -> None:
        """Reject zip-slip entries: absolute paths, drive letters, ``..``, backslashes."""
        pure = PurePosixPath(name)
        if (
            "\\" in name
            or pure.is_absolute()
            or _WINDOWS_DRIVE.match(name)
            or ".." in pure.parts
            or not name.strip()
        ):
            raise SnapshotError(
                f"snapshot {zip_path} contains an unsafe entry: {name!r}"
            )
        resolved = (self._root / pure).resolve()
        if not resolved.is_relative_to(self._root.resolve()):
            raise SnapshotError(
                f"snapshot {zip_path} entry escapes the dataset root: {name!r}"
            )

    def _unique_target(self, stamp: datetime, operation: str) -> Path:
        """Snapshot path for stamp+operation; append _2, _3... on collision."""
        base_name = f"{stamp.strftime(TIMESTAMP_FORMAT)}_{operation}"
        target = self._backup_dir / f"{base_name}{SNAPSHOT_SUFFIX}"
        counter = FIRST_COLLISION_SUFFIX
        while target.exists():
            target = self._backup_dir / f"{base_name}_{counter}{SNAPSHOT_SUFFIX}"
            counter += 1
        return target

    def _enforce_retention(self, *, exclude: Path | None = None) -> None:
        """Delete the oldest snapshots beyond the retention limit (best effort).

        ``exclude`` protects one zip (e.g. a restore target) from pruning;
        nothing newer is deleted in its place, so the count may temporarily
        stay one above the limit until the next snapshot is created.
        """
        files = sorted(
            self._snapshot_files(),
            key=lambda item: (item[1], natural_sort_key(item[0].name)),
        )
        excess = len(files) - self._retention
        excluded = exclude.resolve() if exclude is not None else None
        for path, _stamp, _operation in files[:max(excess, 0)]:
            if excluded is not None and path.resolve() == excluded:
                _LOGGER.info(
                    "retention kept snapshot %s (protected during restore)", path
                )
                continue
            try:
                path.unlink()
                _LOGGER.info("retention pruned old snapshot %s", path)
            except OSError as exc:
                _LOGGER.warning("could not prune old snapshot %s: %s", path, exc)

    def _snapshot_files(self) -> list[tuple[Path, datetime, str]]:
        """(path, created, operation) for every well-named zip in ``.backups/``."""
        if not self._backup_dir.is_dir():
            return []
        results: list[tuple[Path, datetime, str]] = []
        for path in self._backup_dir.glob(f"*{SNAPSHOT_SUFFIX}"):
            match = _SNAPSHOT_NAME.match(path.name)
            if match is None:
                continue
            try:
                stamp = datetime.strptime(match.group("stamp"), TIMESTAMP_FORMAT)
            except ValueError:
                _LOGGER.warning("ignoring zip with invalid timestamp: %s", path)
                continue
            results.append((path, stamp, match.group("operation")))
        return results


def _sanitize_operation(operation: str) -> str:
    """Make an operation name safe as a filename component (ValidationError if empty)."""
    if not isinstance(operation, str) or not operation.strip():
        raise ValidationError("snapshot operation name must be a non-empty string")
    cleaned = _INVALID_FILENAME_CHARS.sub(SANITIZE_REPLACEMENT, operation.strip())
    cleaned = _WHITESPACE_RUN.sub(SANITIZE_REPLACEMENT, cleaned)
    cleaned = cleaned[:MAX_OPERATION_LENGTH].strip(" .")
    return cleaned if cleaned else DEFAULT_OPERATION_NAME


def _collect_txt_files(root: Path) -> list[tuple[Path, str]]:
    """All txt files under root (excluding .backups/.nlapt) as (path, POSIX arcname)."""
    collected: list[tuple[Path, str]] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in _EXCLUDED_DIRS)
        for filename in filenames:
            if filename.lower().endswith(TXT_EXTENSION):
                path = Path(current) / filename
                collected.append((path, path.relative_to(root).as_posix()))
    collected.sort(key=lambda item: natural_sort_key(item[1]))
    return collected


def _build_zip_bytes(files: list[tuple[Path, str]]) -> bytes:
    """Build the snapshot zip in memory so it can be written atomically."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in files:
            archive.writestr(arcname, path.read_bytes())
    return buffer.getvalue()


def _select_entries(entries: list[str], only: Collection[str] | None) -> list[str]:
    """Filter zip entries by the ``only`` key subset (all when None)."""
    if only is None:
        return entries
    wanted = {str(key) for key in only}
    missing = wanted.difference(entries)
    if missing:
        _LOGGER.warning(
            "restore skipped %d key(s) not present in snapshot: %s",
            len(missing),
            sorted(missing),
        )
    return [name for name in entries if name in wanted]
