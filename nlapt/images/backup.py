"""Dataset-local backups and quarantine folders for image-mutating tools."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nlapt.core.errors import ImageProcessingError, StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text
from nlapt.storage.scanner import natural_sort_key

_LOGGER = get_logger(__name__)

BACKUP_DIR_NAME = ".backups"
IMAGES_SUBDIR = "images"
DUPLICATES_SUBDIR = "duplicates"
MANIFEST_NAME = "manifest.json"
KIND_BACKUP = "backup"
KIND_QUARANTINE = "quarantine"
DEFAULT_RETENTION = 5
TIMESTAMP_FORMAT = "%Y-%m-%d_%H%M"
DEFAULT_OPERATION_NAME = "operation"
MAX_OPERATION_LENGTH = 60
SANITIZE_REPLACEMENT = "_"
FIRST_COLLISION_SUFFIX = 2

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True)
class ImageBackupInfo:
    """One flatten-backup or quarantine folder under ``.backups/``."""

    path: Path
    created: datetime
    operation: str
    kind: str
    file_count: int


def _now() -> datetime:
    """Timestamp source; module-level so tests can monkeypatch it."""
    return datetime.now()


class ImageBackupManager:
    """Copy-on-write image backups and move-to-quarantine for duplicates."""

    def __init__(self, root: Path, *, retention: int = DEFAULT_RETENTION) -> None:
        base = Path(root)
        if not base.is_dir():
            raise ValidationError(f"image backup root is not a directory: {base}")
        if not base.is_absolute():
            base = base.absolute()
        if not isinstance(retention, int) or isinstance(retention, bool) or retention < 1:
            raise ValidationError(f"retention must be an int >= 1, got {retention!r}")
        self._root = base
        self._retention = retention
        self._images_dir = base / BACKUP_DIR_NAME / IMAGES_SUBDIR
        self._duplicates_dir = base / BACKUP_DIR_NAME / DUPLICATES_SUBDIR

    @property
    def root(self) -> Path:
        return self._root

    def create(self, operation: str, paths: tuple[Path, ...]) -> ImageBackupInfo:
        """Copy ``paths`` into ``.backups/images/<stamp>_<op>/`` and prune old ones."""
        sanitized = _sanitize_operation(operation)
        stamp = _now().replace(second=0, microsecond=0)
        resolved = tuple(self._require_inside(Path(path)) for path in paths)
        try:
            self._images_dir.mkdir(parents=True, exist_ok=True)
            target = self._unique_target(self._images_dir, stamp, sanitized)
            target.mkdir(parents=True, exist_ok=False)
            files: list[dict[str, str]] = []
            for source in resolved:
                rel = source.relative_to(self._root).as_posix()
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                files.append({"rel": rel, "txt": ""})
            info = ImageBackupInfo(
                path=target,
                created=stamp,
                operation=sanitized,
                kind=KIND_BACKUP,
                file_count=len(files),
            )
            _write_manifest(target, info, files)
        except (OSError, StorageError) as exc:
            raise ImageProcessingError(
                f"image backup creation failed for {self._root}: {exc}"
            ) from exc
        _LOGGER.info("image backup created: %s (%d files)", target, len(resolved))
        self._enforce_retention(self._images_dir, KIND_BACKUP)
        return info

    def quarantine(self, paths: tuple[Path, ...], *, operation: str = "duplicates") -> ImageBackupInfo:
        """Move images (and sibling ``.txt``) into ``.backups/duplicates/<stamp>/``."""
        sanitized = _sanitize_operation(operation)
        stamp = _now().replace(second=0, microsecond=0)
        resolved = tuple(self._require_inside(Path(path)) for path in paths)
        if not resolved:
            raise ValidationError("quarantine requires at least one image path")
        try:
            self._duplicates_dir.mkdir(parents=True, exist_ok=True)
            target = self._unique_target(self._duplicates_dir, stamp, sanitized)
            target.mkdir(parents=True, exist_ok=False)
            files: list[dict[str, str]] = []
            for source in resolved:
                rel = source.relative_to(self._root).as_posix()
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(dest))
                txt_rel = ""
                txt_source = source.with_suffix(".txt")
                if txt_source.is_file():
                    txt_rel = txt_source.relative_to(self._root).as_posix()
                    txt_dest = target / txt_rel
                    txt_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(txt_source), str(txt_dest))
                files.append({"rel": rel, "txt": txt_rel})
            info = ImageBackupInfo(
                path=target,
                created=stamp,
                operation=sanitized,
                kind=KIND_QUARANTINE,
                file_count=len(files),
            )
            _write_manifest(target, info, files)
        except (OSError, StorageError) as exc:
            raise ImageProcessingError(
                f"quarantine failed for {self._root}: {exc}"
            ) from exc
        _LOGGER.info("quarantined %d images into %s", len(resolved), target)
        self._enforce_retention(self._duplicates_dir, KIND_QUARANTINE)
        return info

    def list_backups(self) -> tuple[ImageBackupInfo, ...]:
        """All flatten backups and quarantines, newest first."""
        infos = list(self._scan_dir(self._images_dir)) + list(self._scan_dir(self._duplicates_dir))
        infos.sort(key=lambda info: (info.created, natural_sort_key(info.path.name)))
        return tuple(reversed(infos))

    def restore(self, info: ImageBackupInfo) -> tuple[str, ...]:
        """Copy (backup) or move (quarantine) files back onto the dataset."""
        manifest = _read_manifest(Path(info.path))
        kind = str(manifest.get("kind", info.kind))
        files = manifest.get("files")
        if not isinstance(files, list):
            raise ImageProcessingError(f"backup manifest missing files: {info.path}")
        restored: list[str] = []
        try:
            for entry in files:
                if not isinstance(entry, dict):
                    continue
                rel = entry.get("rel")
                if not isinstance(rel, str) or not rel or ".." in Path(rel).parts:
                    raise ImageProcessingError(f"unsafe backup entry {rel!r}")
                source = info.path / rel
                dest = self._root / rel
                if not source.is_file():
                    _LOGGER.warning("backup member missing: %s", source)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                if kind == KIND_QUARANTINE:
                    shutil.move(str(source), str(dest))
                else:
                    shutil.copy2(source, dest)
                restored.append(rel)
                txt_rel = entry.get("txt")
                if isinstance(txt_rel, str) and txt_rel and ".." not in Path(txt_rel).parts:
                    txt_source = info.path / txt_rel
                    txt_dest = self._root / txt_rel
                    if txt_source.is_file():
                        txt_dest.parent.mkdir(parents=True, exist_ok=True)
                        if kind == KIND_QUARANTINE:
                            shutil.move(str(txt_source), str(txt_dest))
                        else:
                            shutil.copy2(txt_source, txt_dest)
        except OSError as exc:
            raise ImageProcessingError(f"restore failed from {info.path}: {exc}") from exc
        _LOGGER.info("restored %d images from %s", len(restored), info.path)
        return tuple(restored)

    def _require_inside(self, path: Path) -> Path:
        resolved = path if path.is_absolute() else (self._root / path)
        resolved = resolved.resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as exc:
            raise ValidationError(f"path is outside the dataset: {path}") from exc
        if not resolved.is_file():
            raise ValidationError(f"image file not found: {resolved}")
        return resolved

    def _unique_target(self, parent: Path, stamp: datetime, operation: str) -> Path:
        base = f"{stamp.strftime(TIMESTAMP_FORMAT)}_{operation}"
        candidate = parent / base
        if not candidate.exists():
            return candidate
        suffix = FIRST_COLLISION_SUFFIX
        while True:
            collision = parent / f"{base}_{suffix}"
            if not collision.exists():
                return collision
            suffix += 1

    def _scan_dir(self, folder: Path) -> tuple[ImageBackupInfo, ...]:
        if not folder.is_dir():
            return ()
        found: list[ImageBackupInfo] = []
        for child in folder.iterdir():
            if not child.is_dir():
                continue
            manifest_path = child / MANIFEST_NAME
            if not manifest_path.is_file():
                continue
            try:
                data = _read_manifest(child)
                created_raw = data.get("created")
                created = (
                    datetime.fromisoformat(str(created_raw))
                    if created_raw
                    else datetime.fromtimestamp(child.stat().st_mtime)
                )
                files = data.get("files")
                count = len(files) if isinstance(files, list) else 0
                found.append(
                    ImageBackupInfo(
                        path=child,
                        created=created,
                        operation=str(data.get("operation", child.name)),
                        kind=str(data.get("kind", KIND_BACKUP)),
                        file_count=count,
                    )
                )
            except (OSError, ValueError, ImageProcessingError) as exc:
                _LOGGER.warning("skipping unreadable image backup %s: %s", child, exc)
        return tuple(found)

    def _enforce_retention(self, folder: Path, kind: str) -> None:
        infos = [info for info in self._scan_dir(folder) if info.kind == kind]
        infos.sort(key=lambda info: (info.created, natural_sort_key(info.path.name)))
        overflow = infos[: max(0, len(infos) - self._retention)]
        for info in overflow:
            try:
                shutil.rmtree(info.path)
                _LOGGER.info("pruned image backup %s", info.path)
            except OSError:
                _LOGGER.warning("could not prune image backup %s", info.path)


def _sanitize_operation(operation: str) -> str:
    if not isinstance(operation, str) or not operation.strip():
        raise ValidationError("image backup operation name must be a non-empty string")
    cleaned = _INVALID_FILENAME_CHARS.sub(SANITIZE_REPLACEMENT, operation.strip())
    cleaned = _WHITESPACE_RUN.sub(SANITIZE_REPLACEMENT, cleaned)
    cleaned = cleaned[:MAX_OPERATION_LENGTH].strip(" .")
    return cleaned if cleaned else DEFAULT_OPERATION_NAME


def _write_manifest(
    folder: Path, info: ImageBackupInfo, files: list[dict[str, str]]
) -> None:
    payload = {
        "kind": info.kind,
        "operation": info.operation,
        "created": info.created.isoformat(),
        "files": files,
    }
    atomic_write_text(folder / MANIFEST_NAME, json.dumps(payload, ensure_ascii=False, indent=2))


def _read_manifest(folder: Path) -> dict[str, object]:
    path = folder / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ImageProcessingError(f"unreadable backup manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ImageProcessingError(f"backup manifest is not an object: {path}")
    return data
