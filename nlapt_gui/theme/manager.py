"""ThemeManager - applies a theme (QSS + palette) app-wide and persists it."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QObject, QPropertyAnimation, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

from nlapt.diagnostics import get_logger

from nlapt_gui import anim
from nlapt_gui.theme.glyphs import ensure_glyphs
from nlapt_gui.theme.qss import build_qss
from nlapt_gui.theme.tokens import (
    DEFAULT_ACCENT,
    DEFAULT_THEME,
    THEMES,
    ThemeTokens,
    with_accent,
)

_LOGGER = get_logger(__name__)

# Theme cross-fade: how far the window dips before fading back to full opacity,
# and how long the restore takes. Pure visual polish (skip-safe).
_TRANSITION_DIP = 0.72
_TRANSITION_MS = 190


def _palette_for(tokens: ThemeTokens) -> QPalette:
    """Qt palette matching the tokens (native menus/dialogs follow the scheme)."""
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: tokens.bg,
        QPalette.ColorRole.WindowText: tokens.text,
        QPalette.ColorRole.Base: tokens.surface,
        QPalette.ColorRole.AlternateBase: tokens.surface2,
        QPalette.ColorRole.Text: tokens.text,
        QPalette.ColorRole.Button: tokens.surface,
        QPalette.ColorRole.ButtonText: tokens.text,
        QPalette.ColorRole.Highlight: tokens.accent,
        QPalette.ColorRole.HighlightedText: tokens.onaccent,
        QPalette.ColorRole.PlaceholderText: tokens.text3,
        QPalette.ColorRole.ToolTipBase: tokens.surface,
        QPalette.ColorRole.ToolTipText: tokens.text,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    return palette


class ThemeManager(QObject):
    """Owns the active theme: builds QSS, sets the palette, persists choice.

    ``apply`` resolves the theme name + optional accent into final
    :class:`ThemeTokens`, styles the QApplication and emits
    ``theme_changed`` so token-aware widgets (custom-painted parts) can
    repaint.
    """

    theme_changed = Signal(object)  # ThemeTokens

    def __init__(
        self,
        app: QApplication | None = None,
        *,
        settings_path: Path | None = None,
        persist: bool = True,
    ) -> None:
        super().__init__()
        self._app = app
        self._settings_path = settings_path
        self._persist = persist
        self._tokens: ThemeTokens = THEMES[DEFAULT_THEME]
        self._theme_name: str = DEFAULT_THEME
        self._accent: str = DEFAULT_ACCENT
        # Keep references to running cross-fade animations so Qt does not GC
        # them mid-flight (they are cleared when they finish).
        self._transitions: list[QPropertyAnimation] = []

    @property
    def tokens(self) -> ThemeTokens:
        """Tokens currently applied (accent override already merged)."""
        return self._tokens

    @property
    def theme_name(self) -> str:
        return self._theme_name

    @property
    def accent(self) -> str:
        return self._accent

    def apply(self, theme_name: str, accent: str | None = None) -> ThemeTokens:
        """Apply a theme by name with an optional custom accent.

        Unknown names fall back to the default theme (never raises so a
        stale settings file cannot break startup).
        """
        base = THEMES.get(theme_name)
        if base is None:
            _LOGGER.warning("unknown theme %r; falling back to %r", theme_name, DEFAULT_THEME)
            theme_name = DEFAULT_THEME
            base = THEMES[DEFAULT_THEME]
        resolved_accent = accent if accent else base.accent
        tokens = base if resolved_accent == base.accent else with_accent(base, resolved_accent)
        app = self._app if self._app is not None else QApplication.instance()
        if app is not None:
            # Clear the old stylesheet first so native controls (tabs, radios,
            # spin boxes) repolish against the new palette instead of keeping
            # the previous theme. Then QSS, which Qt prefers over the palette.
            app.setStyleSheet("")
            app.setPalette(_palette_for(tokens))
            app.setStyleSheet(build_qss(tokens, ensure_glyphs(tokens)))
        self._tokens = tokens
        self._theme_name = theme_name
        self._accent = resolved_accent
        if self._persist:
            self._save(theme_name, resolved_accent)
        self.theme_changed.emit(tokens)
        self._cross_fade(app)
        _LOGGER.info("theme applied: %s (accent %s)", theme_name, resolved_accent)
        return tokens

    def _cross_fade(self, app: QApplication | None) -> None:
        """Briefly dip then restore each top-level window's opacity.

        Skip-safe: when animations are disabled (tests) this is a no-op and no
        window opacity is ever touched, so windows stay fully opaque. The new
        QSS is already live before this runs, so the effect is a gentle fade
        into the new palette rather than an abrupt flip.
        """
        if app is None or not anim.animations_enabled():
            return
        for window in app.topLevelWidgets():
            if not isinstance(window, QWidget) or not window.isVisible():
                continue
            try:
                self._fade_window(window)
            except Exception:  # visual polish must never break theme switching
                _LOGGER.debug("theme cross-fade skipped for %r", window, exc_info=True)

    def _fade_window(self, window: QWidget) -> None:
        """Run a single dip/restore opacity animation on one window."""
        window.setWindowOpacity(_TRANSITION_DIP)
        animation = QPropertyAnimation(window, b"windowOpacity", self)
        animation.setDuration(_TRANSITION_MS)
        animation.setStartValue(_TRANSITION_DIP)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _cleanup() -> None:
            window.setWindowOpacity(1.0)
            if animation in self._transitions:
                self._transitions.remove(animation)

        animation.finished.connect(_cleanup)
        self._transitions.append(animation)
        animation.start(QAbstractAnimation.DeletionPolicy.KeepWhenStopped)

    def _save(self, theme_name: str, accent: str) -> None:
        """Merge theme/accent into the persisted UI settings on disk."""
        try:
            from dataclasses import replace

            # Imported lazily: settings depends on theme.tokens, so a
            # module-level import here would be circular.
            from nlapt_gui.settings import load_ui_settings, save_ui_settings

            current = load_ui_settings(self._settings_path)
            save_ui_settings(
                replace(current, theme=theme_name, accent=accent), self._settings_path
            )
        except Exception:  # persistence must never break theme switching
            _LOGGER.exception("could not persist theme selection")
