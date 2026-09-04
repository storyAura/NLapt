"""Storage layer: atomic writes, encoding-aware caption IO, and the
dataset-level services (scanner, zip snapshots, crash-recovery session)."""

from __future__ import annotations

from nlapt.storage.atomic import atomic_write_bytes, atomic_write_text
from nlapt.storage.export import export_dataset_zip
from nlapt.storage.paths import (
    app_state_dir,
    dataset_state_dir,
    migrate_legacy_dataset_state,
)
from nlapt.storage.scanner import scan_dataset
from nlapt.storage.session import SESSION_DIR_NAME, SessionSnapshot, SessionStore
from nlapt.storage.snapshots import (
    BACKUP_DIR_NAME,
    RestoreResult,
    SnapshotInfo,
    SnapshotManager,
)
from nlapt.storage.text_io import (
    CANDIDATE_ENCODINGS,
    TextReadResult,
    read_text_detect,
    write_caption,
)

__all__ = [
    "BACKUP_DIR_NAME",
    "CANDIDATE_ENCODINGS",
    "RestoreResult",
    "SESSION_DIR_NAME",
    "SessionSnapshot",
    "SessionStore",
    "SnapshotInfo",
    "SnapshotManager",
    "TextReadResult",
    "app_state_dir",
    "atomic_write_bytes",
    "atomic_write_text",
    "dataset_state_dir",
    "export_dataset_zip",
    "migrate_legacy_dataset_state",
    "read_text_detect",
    "scan_dataset",
    "write_caption",
]
