"""Packaging-aware resource and per-user data directory helpers.

All GUI code resolves bundled resources through :func:`resource_path` and
user-writable state (settings, config, logs) through :func:`app_data_dir`
so PyInstaller onedir/onefile builds keep working without code changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Environment override for the data directory (used by tests / portable mode).
ENV_DATA_DIR = "NLAPT_DATA_DIR"
# Directory name under %APPDATA% on Windows.
WINDOWS_DIR_NAME = "NLapt"
# Directory name under ~/.config on other platforms.
UNIX_DIR_NAME = "nlapt"


def resource_path(rel: str) -> Path:
    """Absolute path of a bundled resource, PyInstaller-aware.

    Inside a PyInstaller bundle resources are unpacked under
    ``sys._MEIPASS``; in a source checkout they live next to the
    repository root (the parent of the ``nlapt_gui`` package).
    """
    bundle_root = getattr(sys, "_MEIPASS", None)
    base = Path(bundle_root) if bundle_root else Path(__file__).resolve().parent.parent
    return base / rel


def app_data_dir() -> Path:
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
