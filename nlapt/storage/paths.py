"""Per-user data directory and per-dataset state paths.

Dataset snapshots / crash-recovery session / batch checkpoints used to live
inside the opened folder (``.backups/``, ``.nlapt/``). They now live under
the per-user data dir so the dataset itself stays a clean image+txt tree.

``nlapt_gui.resources.app_data_dir`` delegates here — one source of truth
for ``NLAPT_DATA_DIR`` / ``%APPDATA%/NLapt`` / ``~/.config/nlapt``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import sys
from pathlib import Path
from uuid import UUID

from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

ENV_DATA_DIR = "NLAPT_DATA_DIR"
ENV_DOCUMENTS_DIR = "NLAPT_DOCUMENTS_DIR"
WINDOWS_DIR_NAME = "NLapt"
UNIX_DIR_NAME = "nlapt"
DOCUMENTS_FOLDER_NAME = "Documents"
DATASETS_DIR_NAME = "datasets"
STATE_BACKUPS_DIR_NAME = "backups"
LEGACY_BACKUP_DIR_NAME = ".backups"
LEGACY_SESSION_DIR_NAME = ".nlapt"
STATE_KEY_ENCODING = "utf-8"
# FOLDERID_Documents — Known Folder ID used by SHGetKnownFolderPath.
_FOLDERID_DOCUMENTS = UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7")


class _GUID(ctypes.Structure):
    """Windows GUID layout for SHGetKnownFolderPath."""

    _fields_ = (
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    )


def _guid_from_uuid(value: UUID) -> _GUID:
    packed = value.bytes_le
    return _GUID(
        int.from_bytes(packed[0:4], "little"),
        int.from_bytes(packed[4:6], "little"),
        int.from_bytes(packed[6:8], "little"),
        (ctypes.c_ubyte * 8).from_buffer_copy(packed[8:16]),
    )


def windows_known_documents_dir() -> Path | None:
    """User Documents via ``SHGetKnownFolderPath``, or None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        path_ptr = ctypes.c_wchar_p()
        folderid = _guid_from_uuid(_FOLDERID_DOCUMENTS)
        result = ctypes.windll.shell32.SHGetKnownFolderPath(  # type: ignore[attr-defined]
            ctypes.byref(folderid),
            0,
            None,
            ctypes.byref(path_ptr),
        )
        if result != 0 or not path_ptr.value:
            return None
        resolved = Path(path_ptr.value)
        ctypes.windll.ole32.CoTaskMemFree(path_ptr)  # type: ignore[attr-defined]
        return resolved
    except (OSError, AttributeError, ValueError, TypeError):
        return None


def app_state_dir() -> Path:
    """Per-user writable data directory, created on demand.

    ``%APPDATA%/NLapt`` on Windows, ``~/.config/nlapt`` elsewhere. The
    ``NLAPT_DATA_DIR`` environment variable overrides both (tests and
    portable installs).
    """
    override = os.environ.get(ENV_DATA_DIR, "").strip()
    if override:
        target = Path(override)
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "").strip()
        base = Path(appdata) if appdata else Path.home()
        target = base / WINDOWS_DIR_NAME
    else:
        target = Path.home() / ".config" / UNIX_DIR_NAME
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        _LOGGER.exception("could not create app data dir %s", target)
        raise
    return target


def user_documents_dir() -> Path:
    """User Documents folder (``NLAPT_DOCUMENTS_DIR`` overrides; tests isolate it).

    On Windows, ``SHGetKnownFolderPath(FOLDERID_Documents)`` wins over
    ``Path.home()/Documents`` so OneDrive / redirected folders resolve.
    """
    override = os.environ.get(ENV_DOCUMENTS_DIR, "").strip()
    if override:
        return Path(override)
    known = windows_known_documents_dir()
    if known is not None:
        return known
    return Path.home() / DOCUMENTS_FOLDER_NAME


def user_documents_app_dir() -> Path:
    """``<Documents>/NLapt``, created on demand — custom API keys live here."""
    target = user_documents_dir() / WINDOWS_DIR_NAME
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        _LOGGER.exception("could not create documents app dir %s", target)
        raise
    return target


def dataset_state_key(root: Path) -> str:
    """Stable hex id for one dataset root (normcased resolved path)."""
    resolved = Path(root)
    if not resolved.is_absolute():
        resolved = resolved.absolute()
    resolved = resolved.resolve()
    material = os.path.normcase(str(resolved))
    return hashlib.sha1(material.encode(STATE_KEY_ENCODING)).hexdigest()


def dataset_state_dir(root: Path) -> Path:
    """Per-dataset state folder under :func:`app_state_dir` (created on demand)."""
    target = app_state_dir() / DATASETS_DIR_NAME / dataset_state_key(root)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        _LOGGER.exception("could not create dataset state dir %s", target)
        raise
    return target


def migrate_legacy_dataset_state(root: Path, state_dir: Path) -> None:
    """Best-effort move of ``<root>/.backups`` and ``<root>/.nlapt`` into ``state_dir``.

    Existing files in the destination win (already-migrated state). Failures
    are logged and never raised — opening a dataset must not block on this.
    """
    base = Path(root)
    dest = Path(state_dir)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _LOGGER.warning("cannot prepare dataset state dir %s: %s", dest, exc)
        return
    _move_tree(base / LEGACY_BACKUP_DIR_NAME, dest / STATE_BACKUPS_DIR_NAME)
    _move_tree(base / LEGACY_SESSION_DIR_NAME, dest)


def _move_tree(source: Path, dest: Path) -> None:
    """Move ``source`` contents into ``dest``, then remove ``source``."""
    if not source.is_dir():
        return
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _LOGGER.warning("cannot create %s to migrate %s: %s", dest, source, exc)
        return
    try:
        children = list(source.iterdir())
    except OSError as exc:
        _LOGGER.warning("cannot list legacy state dir %s: %s", source, exc)
        return
    for item in children:
        target = dest / item.name
        try:
            if target.exists():
                _remove_path(item)
                continue
            shutil.move(str(item), str(target))
        except OSError as exc:
            _LOGGER.warning("could not migrate %s -> %s: %s", item, target, exc)
    _remove_path(source)


def _remove_path(path: Path) -> None:
    """Delete a leftover file or directory (best effort)."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except OSError as exc:
        _LOGGER.warning("could not remove leftover %s: %s", path, exc)
