"""UI settings persistence - frozen dataclass with atomic JSON round-trip.

A corrupt or partially invalid settings file never breaks startup: bad JSON
falls back to full defaults (with a warning) and individually invalid fields
fall back to their own defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui import resources
from nlapt_gui.theme.tokens import (
    DEFAULT_ACCENT,
    DEFAULT_EDITOR_H,
    DEFAULT_THEME,
    DEFAULT_THUMB_MIN,
    EDITOR_H_RANGE,
    THUMB_MIN_RANGE,
)

_LOGGER = get_logger(__name__)

SETTINGS_FILE_NAME = "ui_settings.json"
VIEW_MODES: tuple[str, ...] = ("list", "mid", "big")
DEFAULT_VIEW_MODE = "mid"


def _default_sections() -> dict[str, bool]:
    """Right-panel section open states (design defaults)."""
    return {"fr": True, "ps": False, "tr": False, "hist": True}


@dataclass(frozen=True)
class UISettings:
    """Persisted UI state (window-independent preferences)."""

    theme: str = DEFAULT_THEME
    accent: str = DEFAULT_ACCENT
    thumb_min: int = DEFAULT_THUMB_MIN
    view_mode: str = DEFAULT_VIEW_MODE
    editor_h: int = DEFAULT_EDITOR_H
    folder_open: Mapping[str, bool] = field(default_factory=dict)
    last_root: str = ""
    sections: Mapping[str, bool] = field(default_factory=_default_sections)


def _default_path() -> Path:
    return resources.app_data_dir() / SETTINGS_FILE_NAME


def _coerce_str(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _coerce_int(value: Any, default: int, bounds: tuple[int, int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    low, high = bounds
    return max(low, min(high, value))


def _coerce_bool_map(value: Any, default: Mapping[str, bool]) -> dict[str, bool]:
    if not isinstance(value, dict):
        return dict(default)
    return {k: bool(v) for k, v in value.items() if isinstance(k, str)}


def load_ui_settings(path: Path | None = None) -> UISettings:
    """Load settings from JSON; missing or corrupt file -> defaults."""
    target = path if path is not None else _default_path()
    defaults = UISettings()
    if not target.exists():
        return defaults
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("corrupt ui settings %s (%s); using defaults", target, exc)
        return defaults
    if not isinstance(data, dict):
        _LOGGER.warning("ui settings %s is not a JSON object; using defaults", target)
        return defaults
    view_mode = data.get("view_mode")
    if view_mode not in VIEW_MODES:
        view_mode = defaults.view_mode
    merged_sections = dict(_default_sections())
    merged_sections.update(_coerce_bool_map(data.get("sections"), {}))
    return replace(
        defaults,
        theme=_coerce_str(data.get("theme"), defaults.theme),
        accent=_coerce_str(data.get("accent"), defaults.accent),
        thumb_min=_coerce_int(data.get("thumb_min"), defaults.thumb_min, THUMB_MIN_RANGE),
        view_mode=view_mode,
        editor_h=_coerce_int(data.get("editor_h"), defaults.editor_h, EDITOR_H_RANGE),
        folder_open=_coerce_bool_map(data.get("folder_open"), {}),
        last_root=data.get("last_root") if isinstance(data.get("last_root"), str) else "",
        sections=merged_sections,
    )


def save_ui_settings(settings: UISettings, path: Path | None = None) -> None:
    """Persist settings atomically as UTF-8 JSON."""
    target = path if path is not None else _default_path()
    payload = {
        "theme": settings.theme,
        "accent": settings.accent,
        "thumb_min": settings.thumb_min,
        "view_mode": settings.view_mode,
        "editor_h": settings.editor_h,
        "folder_open": dict(settings.folder_open),
        "last_root": settings.last_root,
        "sections": dict(settings.sections),
    }
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
    _LOGGER.debug("ui settings saved to %s", target)
