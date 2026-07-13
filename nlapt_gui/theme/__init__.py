"""Theme package - design tokens, QSS generation, and the ThemeManager."""

from __future__ import annotations

from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.qss import build_qss
from nlapt_gui.theme.tokens import (
    ACCENT_OPTIONS,
    DEFAULT_ACCENT,
    DEFAULT_EDITOR_H,
    DEFAULT_THEME,
    DEFAULT_THUMB_MIN,
    EDITOR_H_RANGE,
    FONT_STACK,
    MIN_WINDOW,
    MONO_STACK,
    THEMES,
    THUMB_MIN_RANGE,
    WINDOWS_CLOSE_HOVER,
    ZOOM_RANGE,
    ZOOM_STEP,
    ThemeTokens,
    accent_soft,
    mix,
    with_accent,
)

__all__ = [
    "ACCENT_OPTIONS",
    "DEFAULT_ACCENT",
    "DEFAULT_EDITOR_H",
    "DEFAULT_THEME",
    "DEFAULT_THUMB_MIN",
    "EDITOR_H_RANGE",
    "FONT_STACK",
    "MIN_WINDOW",
    "MONO_STACK",
    "THEMES",
    "THUMB_MIN_RANGE",
    "WINDOWS_CLOSE_HOVER",
    "ZOOM_RANGE",
    "ZOOM_STEP",
    "ThemeManager",
    "ThemeTokens",
    "accent_soft",
    "build_qss",
    "mix",
    "with_accent",
]
