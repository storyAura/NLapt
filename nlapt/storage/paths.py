"""Per-user data directory and per-dataset state paths.

Dataset snapshots / crash-recovery session / batch checkpoints used to live
inside the opened folder (``.backups/``, ``.nlapt/``). They now live under
the per-user data dir so the dataset itself stays a clean image+txt tree.

``nlapt_gui.resources.app_data_dir`` delegates here — one source of truth
for ``NLAPT_DATA_DIR`` / ``%APPDATA%/NLapt`` / ``~/.config/nlapt``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

ENV_DATA_DIR = "NLAPT_DATA_DIR"
WINDOWS_DIR_NAME = "NLapt"
UNIX_DIR_NAME = "nlapt"
DATASETS_DIR_NAME = "datasets"
STATE_BACKUPS_DIR_NAME = "backups"
LEGACY_BACKUP_DIR_NAME = ".backups"
LEGACY_SESSION_DIR_NAME = ".nlapt"
STATE_KEY_ENCODING = "utf-8"


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
