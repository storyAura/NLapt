"""Packaging-aware resource and per-user data directory helpers.

All GUI code resolves bundled resources through :func:`resource_path` and
user-writable state (settings, config, logs) through :func:`app_data_dir`
so PyInstaller onedir/onefile builds keep working without code changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from nlapt.storage.paths import (
    ENV_DATA_DIR,
    UNIX_DIR_NAME,
    WINDOWS_DIR_NAME,
    app_state_dir,
)

# Re-exported so existing callers keep importing these names from resources.
__all__ = [
    "CONFIG_FILE_NAME",
    "ENV_DATA_DIR",
    "UNIX_DIR_NAME",
    "WINDOWS_DIR_NAME",
    "app_data_dir",
    "config_path",
    "resource_path",
]

# File name of the persisted core AppConfig inside the data directory.
CONFIG_FILE_NAME = "config.json"


def resource_path(rel: str) -> Path:
    """Absolute path of a bundled resource, PyInstaller-aware.

    Inside a PyInstaller bundle resources are unpacked under
    ``sys._MEIPASS``; in a source checkout they live next to the
    repository root (the parent of the ``nlapt_gui`` package).
    """
    bundle_root = getattr(sys, "_MEIPASS", None)
    base = Path(bundle_root) if bundle_root else Path(__file__).resolve().parent.parent
    return base / rel


def config_path() -> Path:
    """Location of the persisted core config (neutral home — no widget deps)."""
    return app_data_dir() / CONFIG_FILE_NAME


def app_data_dir() -> Path:
    """Per-user writable data directory, created on demand.

    Delegates to :func:`nlapt.storage.paths.app_state_dir` so GUI and core
    share one ``NLAPT_DATA_DIR`` / ``%APPDATA%/NLapt`` resolution.
    """
    return app_state_dir()
