"""Behavioral tests for the frameless title bar (menus, popup, controls).

Constructs :class:`TitleBar` directly against the shared ``controller``
fixture and a non-persisting :class:`ThemeManager`. Offscreen; no network.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from nlapt_gui import __version__
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES
from nlapt_gui.widgets.title_bar import (
    ACTION_OPEN_FOLDER,
    APP_NAME,
    MENU_EDIT,
    MENU_FILE,
    MENU_HELP,
    MENU_TOOLS,
    MENU_VIEW,
    TITLE_BAR_HEIGHT,
    VERSION_LABEL,
    TitleBar,
)


@pytest.fixture()
def manager(qtbot) -> ThemeManager:
    mgr = ThemeManager(persist=False)
    mgr.apply(DEFAULT_THEME)
    return mgr


@pytest.fixture()
def title_bar(qtbot, controller, manager) -> TitleBar:
    bar = TitleBar(controller, manager)
    bar.resize(1360, TITLE_BAR_HEIGHT)
    qtbot.addWidget(bar)
    bar.show()
    qtbot.waitExposed(bar)
    return bar


class TestStructure:
    def test_fixed_height(self, title_bar: TitleBar) -> None:
        assert title_bar.height() == TITLE_BAR_HEIGHT

    def test_name_and_version_labels(self, title_bar: TitleBar) -> None:
        assert title_bar.name_label.text() == APP_NAME
        assert title_bar.version_label.text() == VERSION_LABEL
        assert __version__ in VERSION_LABEL

    def test_all_menus_present(self, title_bar: TitleBar) -> None:
        assert set(title_bar.menus) == {
            MENU_FILE,
            MENU_EDIT,
            MENU_VIEW,
            MENU_TOOLS,
            MENU_HELP,
        }

    def test_file_menu_entries(self, title_bar: TitleBar) -> None:
        texts = [a.text() for a in title_bar.menus[MENU_FILE].actions() if a.text()]
        assert texts[0] == ACTION_OPEN_FOLDER
        assert any(t.startswith("保存") for t in texts)
        assert "退出" in texts


class TestSignals:
    def test_open_folder_signal(self, qtbot, title_bar: TitleBar) -> None:
        with qtbot.waitSignal(title_bar.open_folder_requested, timeout=1000):
            title_bar.action_open_folder.trigger()

    def test_settings_signal(self, qtbot, title_bar: TitleBar) -> None:
        with qtbot.waitSignal(title_bar.settings_requested, timeout=1000):
            title_bar.action_settings.trigger()


class TestThemePopup:
    def test_popup_rows_match_themes(self, qtbot, title_bar: TitleBar) -> None:
        popup = title_bar.open_theme_popup()
        qtbot.addWidget(popup)
        assert [row.theme_name for row in popup.rows()] == list(THEMES)

    def test_theme_menu_applies(self, title_bar: TitleBar, manager) -> None:
        title_bar.theme_actions["深邃"].trigger()
        assert manager.theme_name == "深邃"


class TestViewMode:
    def test_view_mode_action_sets_controller(self, title_bar, controller) -> None:
        title_bar.view_mode_actions["big"].trigger()
        assert controller.view_mode == "big"

    def test_controller_syncs_check(self, title_bar, controller) -> None:
        controller.set_view_mode("list")
        assert title_bar.view_mode_actions["list"].isChecked()


class TestWindowControls:
    def test_control_buttons_exist(self, title_bar: TitleBar) -> None:
        assert title_bar.min_button is not None
        assert title_bar.max_button is not None
        assert title_bar.close_button is not None

    def test_apply_tokens_does_not_raise(self, title_bar, manager) -> None:
        # Re-theming restyles every token-dependent part without error.
        manager.apply("墨黑")
        title_bar.apply_tokens(manager.tokens)
        assert title_bar.theme_name_label.text() == "墨黑"
