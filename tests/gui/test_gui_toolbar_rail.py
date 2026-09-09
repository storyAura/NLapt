"""Tests for the 44px left ToolbarRail."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu

from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES
from nlapt_gui.widgets.toolbar_rail import (
    ACTION_ABOUT,
    ACTION_QUIT,
    RAIL_W,
    TIP_EXPORT,
    TIP_OPEN,
    TIP_REDO,
    TIP_SAVE,
    TIP_SAVE_ALL,
    TIP_SETTINGS,
    TIP_THEME,
    TIP_TOOLS,
    TIP_TOOLS_MENU,
    TIP_UNDO,
    ToolbarRail,
)

K1 = "0001.png"


@pytest.fixture()
def manager(qtbot) -> ThemeManager:
    mgr = ThemeManager(persist=False)
    mgr.apply(DEFAULT_THEME)
    return mgr


@pytest.fixture()
def rail(qtbot, controller, manager) -> ToolbarRail:
    widget = ToolbarRail(controller, manager)
    qtbot.addWidget(widget)
    widget.show()
    return widget


class TestLayout:
    def test_fixed_width(self, rail: ToolbarRail) -> None:
        assert rail.width() == RAIL_W

    def test_tooltips(self, rail: ToolbarRail) -> None:
        assert rail.open_button.toolTip() == TIP_OPEN
        assert rail.tools_menu_button.toolTip() == TIP_TOOLS_MENU
        assert rail.save_button.toolTip() == TIP_SAVE
        assert rail.save_all_button.toolTip() == TIP_SAVE_ALL
        assert rail.export_button.toolTip() == TIP_EXPORT
        assert rail.undo_button.toolTip() == TIP_UNDO
        assert rail.redo_button.toolTip() == TIP_REDO
        assert rail.tools_button.toolTip() == TIP_TOOLS
        assert rail.theme_button.toolTip() == TIP_THEME
        assert rail.settings_button.toolTip() == TIP_SETTINGS


class TestActions:
    def test_open_emits_signal(self, qtbot, rail: ToolbarRail) -> None:
        with qtbot.waitSignal(rail.open_folder_requested, timeout=1000):
            rail.open_button.click()

    def test_settings_emits_signal(self, qtbot, rail: ToolbarRail) -> None:
        with qtbot.waitSignal(rail.settings_requested, timeout=1000):
            rail.settings_button.click()

    def test_export_emits_signal(self, qtbot, rail: ToolbarRail) -> None:
        with qtbot.waitSignal(rail.export_requested, timeout=1000):
            rail.export_button.click()

    def test_save_delegates_to_controller(self, qtbot, rail, controller) -> None:
        controller.set_caption(K1, "rail save", "编辑")
        with qtbot.waitSignal(controller.files_saved, timeout=2000):
            rail.save_button.click()
        assert not controller.record(K1).dirty

    def test_undo_and_redo(self, qtbot, rail, controller) -> None:
        original = controller.record(K1).text
        assert not rail.undo_button.isEnabled()
        assert not rail.redo_button.isEnabled()
        controller.set_caption(K1, "changed", "编辑")
        assert rail.undo_button.isEnabled()
        rail.undo_button.click()
        assert controller.record(K1).text == original
        assert rail.redo_button.isEnabled()
        rail.redo_button.click()
        assert controller.record(K1).text == "changed"


class TestToolsToggle:
    def test_click_emits_toggled(self, qtbot, rail: ToolbarRail) -> None:
        with qtbot.waitSignal(rail.tools_toggled, timeout=1000) as blocker:
            rail.tools_button.click()
        assert blocker.args == [True]
        assert rail.tools_open()
        with qtbot.waitSignal(rail.tools_toggled, timeout=1000) as blocker:
            rail.tools_button.click()
        assert blocker.args == [False]
        assert not rail.tools_open()


class TestThemePopup:
    def test_popup_rows_match_themes(self, qtbot, rail: ToolbarRail) -> None:
        popup = rail.open_theme_popup()
        qtbot.addWidget(popup)
        assert [row.theme_name for row in popup.rows()] == list(THEMES)

    def test_pick_applies_theme(self, qtbot, rail: ToolbarRail, manager) -> None:
        popup = rail.open_theme_popup()
        qtbot.addWidget(popup)
        target = next(row for row in popup.rows() if row.theme_name == "深邃")
        qtbot.mouseClick(target, Qt.MouseButton.LeftButton)
        assert manager.theme_name == "深邃"

    def test_colors_button_emits(self, qtbot, rail: ToolbarRail) -> None:
        popup = rail.open_theme_popup()
        qtbot.addWidget(popup)
        with qtbot.waitSignal(rail.colors_requested, timeout=1000):
            popup.colors_button.click()


class TestToolsMenu:
    def test_popup_carries_pool_and_forwards_switch(self, qtbot, rail, controller) -> None:
        from nlapt.core.config import AppConfig, LLMProfile, ModelRef

        from nlapt_gui.widgets.tools_menu import ACTION_MODEL_TEXT, LABEL_MODEL_TEXT

        profile = LLMProfile(
            name="alpha",
            api_type="openai",
            base_url="http://a",
            models=("m1", "m2"),
            enabled_models=("m1", "m2"),
        )
        controller.reload_config(
            AppConfig(profiles=(profile,), text_target=ModelRef("alpha", "m1"))
        )
        popup = rail.open_tools_menu()
        qtbot.addWidget(popup)
        assert popup._buttons[ACTION_MODEL_TEXT].text() == LABEL_MODEL_TEXT.format(model="m1")
        menu = popup.build_model_menu("text")
        second = next(a for a in menu.actions() if a.text().endswith("m2"))
        with qtbot.waitSignal(rail.model_switch_requested, timeout=1000) as blocker:
            second.trigger()
        assert blocker.args == ["text", ModelRef("alpha", "m2")]


class TestLogoMenu:
    def test_menu_entries(self, rail: ToolbarRail, monkeypatch) -> None:
        shown: list[str] = []

        def _capture(menu, _pos, *args, **kwargs) -> None:  # noqa: ANN001
            shown.extend(action.text() for action in menu.actions() if action.text())

        monkeypatch.setattr(QMenu, "popup", _capture)
        rail._open_logo_menu()
        assert shown == [ACTION_ABOUT, ACTION_QUIT]

    def test_about_shows_summary_and_author(self, rail: ToolbarRail, monkeypatch) -> None:
        from nlapt_gui.widgets.title_bar import ABOUT_TEXT, ABOUT_TITLE, AUTHOR

        captured: list[tuple[str, str]] = []

        def _fake_show_message(parent, title, body, *args, **kwargs):  # noqa: ANN001
            captured.append((title, body))

        monkeypatch.setattr(
            "nlapt_gui.widgets.toolbar_rail.show_message", _fake_show_message
        )
        rail.show_about()
        assert captured == [(ABOUT_TITLE, ABOUT_TEXT)]
        assert AUTHOR in captured[0][1]
        assert "使用说明" not in captured[0][1]
