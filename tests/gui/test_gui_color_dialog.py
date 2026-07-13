"""Tests for nlapt_gui.widgets.color_dialog (色彩设置, spec module 4.2)."""

from __future__ import annotations

import pytest

from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import ACCENT_OPTIONS, DEFAULT_THEME, THEMES
from nlapt_gui.widgets.color_dialog import ColorSettingsDialog
from nlapt_gui.widgets.dialogs import CenteredDialog


@pytest.fixture()
def manager(qtbot) -> ThemeManager:
    mgr = ThemeManager(persist=False)
    mgr.apply(DEFAULT_THEME)
    return mgr


@pytest.fixture()
def dialog(qtbot, manager) -> ColorSettingsDialog:
    dlg = ColorSettingsDialog(manager)
    qtbot.addWidget(dlg)
    return dlg


class TestStructure:
    def test_is_a_centered_fading_dialog(self, dialog) -> None:
        assert isinstance(dialog, CenteredDialog)
        assert dialog.windowTitle() == "色彩设置"

    def test_lists_every_theme(self, dialog) -> None:
        names = {row.theme_name for row in dialog._theme_rows}
        assert names == set(THEMES)

    def test_lists_accent_presets(self, dialog) -> None:
        colors = {swatch.color.upper() for swatch in dialog._accent_swatches}
        assert {option.upper() for option in ACCENT_OPTIONS} <= colors


class TestLiveApplication:
    def test_apply_theme_switches_and_keeps_accent(self, dialog, manager) -> None:
        manager.apply(DEFAULT_THEME, "#123456")
        dialog.apply_theme("墨黑")
        assert manager.theme_name == "墨黑"
        assert manager.accent == "#123456"

    def test_apply_accent_changes_tokens(self, dialog, manager) -> None:
        dialog.apply_accent("#AB34CD")
        assert manager.accent == "#AB34CD"
        assert manager.tokens.accent == "#AB34CD"

    def test_custom_accent_appears_as_swatch(self, dialog, manager) -> None:
        dialog.apply_accent("#AB34CD")
        colors = {swatch.color.upper() for swatch in dialog._accent_swatches}
        assert "#AB34CD" in colors

    def test_reset_restores_theme_accent(self, dialog, manager) -> None:
        dialog.apply_accent("#AB34CD")
        dialog.reset_accent()
        assert manager.accent == THEMES[manager.theme_name].accent

    def test_theme_row_click_signal(self, qtbot, dialog, manager) -> None:
        target = next(row for row in dialog._theme_rows if row.theme_name == "石墨")
        target.picked.emit("石墨")
        assert manager.theme_name == "石墨"

    def test_invalid_custom_color_rejected_by_manager(self, dialog, manager) -> None:
        # A cancelled QColorDialog returns an invalid color: nothing changes.
        before = manager.accent
        from PySide6.QtGui import QColor

        class _Cancelled:
            @staticmethod
            def getColor(*_a, **_k) -> QColor:
                return QColor()  # invalid

        import nlapt_gui.widgets.color_dialog as module

        original = module.QColorDialog
        module.QColorDialog = _Cancelled  # type: ignore[assignment]
        try:
            dialog.pick_custom_accent()
        finally:
            module.QColorDialog = original
        assert manager.accent == before
