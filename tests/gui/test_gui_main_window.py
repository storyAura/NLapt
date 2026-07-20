"""Tests for title_bar / status_bar / main_window (integrator widgets)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QLineEdit

from nlapt_gui import __version__
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME, EDITOR_H_RANGE, MIN_WINDOW, THEMES
from nlapt_gui.widgets.main_window import EDGE_MARGIN_PX, MainWindow
from nlapt_gui.widgets.status_bar import StatusBar
from nlapt_gui.widgets.title_bar import (
    ACTION_OPEN_FOLDER,
    MENU_EDIT,
    MENU_FILE,
    MENU_HELP,
    MENU_TOOLS,
    MENU_VIEW,
    TITLE_BAR_HEIGHT,
    VERSION_LABEL,
)

K1 = "0001.png"
K2 = "0002.png"


@pytest.fixture()
def manager(qtbot) -> ThemeManager:
    mgr = ThemeManager(persist=False)
    mgr.apply(DEFAULT_THEME)
    return mgr


@pytest.fixture()
def window(qtbot, controller, manager) -> MainWindow:
    win = MainWindow(controller, manager)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    return win


class TestAssembly:
    def test_frameless_and_min_size(self, window) -> None:
        assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
        assert (window.minimumWidth(), window.minimumHeight()) == MIN_WINDOW

    def test_three_columns_and_bars(self, window) -> None:
        assert window.file_panel.width() == 300
        assert window.tools_panel.width() == 340
        assert window.title_bar.height() == TITLE_BAR_HEIGHT
        assert window.status_bar.height() == 26

    def test_editor_height_from_settings_and_splitter(self, window, controller) -> None:
        assert window.editor_panel.height() == controller.settings.editor_h
        window.splitter.editor_h_changed.emit(400)
        assert window.editor_panel.height() == 400
        low, high = EDITOR_H_RANGE
        window._apply_editor_height(high + 500)
        assert window.editor_panel.height() == high

    def test_editor_panel_uses_real_translate_bridge(self, window) -> None:
        assert window.editor_panel._bridge is window.translate_bridge

    def test_close_calls_controller_close(self, qtbot, controller, manager, monkeypatch) -> None:
        win = MainWindow(controller, manager)
        qtbot.addWidget(win)
        closed: list[bool] = []
        monkeypatch.setattr(controller, "close", lambda: closed.append(True))
        win.close()
        assert closed == [True]


class TestMaxRestore:
    ZOOMED = Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen

    def test_toggle_maximizes_then_restores(self, qtbot, window) -> None:
        window.toggle_max_restore()
        assert window.windowState() & Qt.WindowState.WindowMaximized
        window.toggle_max_restore()
        assert not (window.windowState() & self.ZOOMED)

    def test_toggle_recovers_from_fullscreen(self, qtbot, window) -> None:
        # A window stuck in FULLSCREEN must come back to normal too — the
        # old isMaximized()-only branch kept re-maximizing forever.
        window.showFullScreen()
        window.toggle_max_restore()
        assert not (window.windowState() & self.ZOOMED)

    def test_title_bar_button_delegates_to_window(self, qtbot, window) -> None:
        window.showMaximized()
        window.title_bar.toggle_max_restore()
        assert not (window.windowState() & self.ZOOMED)


class TestTitleBar:
    def test_menus_present(self, window) -> None:
        bar = window.title_bar
        assert set(bar.menus) == {MENU_FILE, MENU_EDIT, MENU_VIEW, MENU_TOOLS, MENU_HELP}
        file_actions = [a.text() for a in bar.menus[MENU_FILE].actions() if a.text()]
        assert file_actions[0] == ACTION_OPEN_FOLDER
        assert "刷新" in file_actions
        assert any(text.startswith("保存") for text in file_actions)
        assert "全部保存" in file_actions
        assert "退出" in file_actions
        edit_actions = [a.text() for a in bar.menus[MENU_EDIT].actions()]
        assert any(text.startswith("撤销") for text in edit_actions)
        assert "复制标注" in edit_actions

    def test_version_label_from_package(self, window) -> None:
        assert window.title_bar.version_label.text() == VERSION_LABEL
        assert __version__ in window.title_bar.version_label.text()

    def test_file_menu_save_action(self, qtbot, window, controller) -> None:
        controller.set_caption(K1, "changed via menu", "编辑")
        with qtbot.waitSignal(controller.files_saved, timeout=2000):
            window.title_bar.action_save.trigger()
        assert not controller.record(K1).dirty

    def test_view_mode_actions_sync(self, window, controller) -> None:
        window.title_bar.view_mode_actions["big"].trigger()
        assert controller.view_mode == "big"
        controller.set_view_mode("list")
        assert window.title_bar.view_mode_actions["list"].isChecked()

    def test_theme_popup_rows_and_pick(self, qtbot, window, manager) -> None:
        popup = window.title_bar.open_theme_popup()
        qtbot.addWidget(popup)
        rows = popup.rows()
        assert len(rows) == len(THEMES)
        assert [row.theme_name for row in rows] == list(THEMES)
        target = next(row for row in rows if row.theme_name == "石墨")
        qtbot.mouseClick(target, Qt.MouseButton.LeftButton)
        assert manager.theme_name == "石墨"
        assert window.title_bar.theme_name_label.text() == "石墨"

    def test_theme_change_updates_controller_settings(self, window, controller, manager) -> None:
        manager.apply("墨黑")
        assert controller.settings.theme == "墨黑"

    def test_theme_menu_applies(self, window, manager) -> None:
        window.title_bar.theme_actions["深邃"].trigger()
        assert manager.theme_name == "深邃"

    def test_open_folder_action_emits_signal(self, qtbot, window, monkeypatch) -> None:
        # Neutralize the modal directory dialog wired to this signal.
        picked: list[bool] = []
        monkeypatch.setattr(window, "pick_folder", lambda: picked.append(True))
        window.title_bar.open_folder_requested.disconnect()
        window.title_bar.open_folder_requested.connect(window.pick_folder)
        with qtbot.waitSignal(window.title_bar.open_folder_requested, timeout=1000):
            window.title_bar.action_open_folder.trigger()
        assert picked == [True]

    def test_pick_folder_opens_dataset(self, qtbot, window, controller, tmp_path, monkeypatch) -> None:
        from nlapt_gui.widgets import main_window as mw

        target = window._controller.root
        assert target is not None
        monkeypatch.setattr(
            mw.QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *args, **kwargs: str(target)),
        )
        with qtbot.waitSignal(controller.dataset_opened, timeout=2000):
            window.pick_folder()


class TestStatusBar:
    def test_left_summary(self, window, controller, demo_dataset: Path) -> None:
        text = window.status_bar.left_label.text()
        assert str(demo_dataset) in text
        assert "UTF-8" in text
        assert f"{len(controller.keys())} 个标注文件" in text

    def test_mode_label_tracks_controller(self, window, controller) -> None:
        assert window.status_bar.mode_label.text() == "模式 · 胶囊"
        controller.set_mode("text")
        assert window.status_bar.mode_label.text() == "模式 · 文本"

    def test_theme_label_tracks_manager(self, window, manager) -> None:
        manager.apply("石墨")
        assert window.status_bar.theme_label.text() == "主题 · 石墨"

    def test_no_dataset_state(self, qtbot, qapp) -> None:
        from nlapt.app import NLaptApp

        from nlapt_gui.controller import AppController
        from nlapt_gui.settings import UISettings

        ctrl = AppController(NLaptApp(), settings=UISettings())
        bar = StatusBar(ctrl)
        qtbot.addWidget(bar)
        assert bar.left_label.text() == "未打开数据集"


class TestShortcuts:
    def test_ctrl_s_saves_current(self, qtbot, window, controller) -> None:
        controller.set_caption(K1, "shortcut save text", "编辑")
        window.setFocus()
        with qtbot.waitSignal(controller.files_saved, timeout=2000):
            qtbot.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
        assert not controller.record(K1).dirty

    def test_ctrl_shift_s_saves_all(self, qtbot, window, controller) -> None:
        controller.set_caption(K1, "dirty one", "编辑")
        controller.set_caption(K2, "dirty two", "编辑")
        with qtbot.waitSignal(controller.files_saved, timeout=2000) as blocker:
            qtbot.keyClick(
                window,
                Qt.Key.Key_S,
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
            )
        assert set(blocker.args[0]) == {K1, K2}

    def test_alt_arrows_navigate(self, qtbot, window, controller) -> None:
        controller.set_current(K1)
        qtbot.keyClick(window, Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
        assert controller.current_key == K2
        qtbot.keyClick(window, Qt.Key.Key_Up, Qt.KeyboardModifier.AltModifier)
        assert controller.current_key == K1

    def test_ctrl_z_outside_text_input_undoes_caption(self, qtbot, window, controller) -> None:
        original = controller.record(K1).text
        controller.set_current(K1)
        controller.set_caption(K1, "edited for undo", "编辑")
        window.setFocus()
        qtbot.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert controller.record(K1).text == original

    def test_ctrl_z_inside_text_input_stays_local(self, qtbot, window, controller) -> None:
        controller.set_current(K1)
        controller.set_caption(K1, "edited caption text", "编辑")
        field = QLineEdit(window)
        field.show()
        field.setFocus()
        qtbot.waitUntil(lambda: field.hasFocus(), timeout=2000)
        qtbot.keyClicks(field, "abc")
        qtbot.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        # The caption edit is untouched; the line edit consumed the undo.
        assert controller.record(K1).text == "edited caption text"
        assert field.text() != "abc"


class TestEdgeResize:
    def test_edge_hit_test(self, window) -> None:
        margin = EDGE_MARGIN_PX
        assert window.edge_at(QPoint(0, window.height() // 2)) == Qt.Edge.LeftEdge
        assert window.edge_at(QPoint(window.width() - 1, window.height() // 2)) == Qt.Edge.RightEdge
        assert window.edge_at(QPoint(window.width() // 2, 0)) == Qt.Edge.TopEdge
        assert (
            window.edge_at(QPoint(window.width() // 2, window.height() - 1))
            == Qt.Edge.BottomEdge
        )
        corner = window.edge_at(QPoint(margin, margin))
        assert corner == (Qt.Edge.LeftEdge | Qt.Edge.TopEdge)
        assert not window.edge_at(QPoint(window.width() // 2, window.height() // 2))


class TestToasts:
    def test_controller_toasts_reach_overlay(self, qtbot, window, controller) -> None:
        controller.toast_requested.emit("集成冒烟提示", "ok")
        qtbot.waitUntil(
            lambda: "集成冒烟提示" in window.toast_overlay.active_texts(), timeout=2000
        )
