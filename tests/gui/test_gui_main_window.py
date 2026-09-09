"""Tests for MainWindow assembly (rail + two-column body + tools overlay)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import QLineEdit

from nlapt.app import NLaptApp

from nlapt_gui import anim
from nlapt_gui.controller import TOAST_NO_DATASET, TOAST_WARN, AppController
from nlapt_gui.settings import UISettings
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME, EDITOR_H_RANGE, MIN_WINDOW, THEMES
from nlapt_gui.widgets.main_window import (
    FILE_PANEL_DEFAULT_W,
    TOOLS_PANEL_DEFAULT_W,
    EDGE_MARGIN_PX,
    MainWindow,
)
from nlapt_gui.widgets.preview_panel import HEADER_H
from nlapt_gui.widgets.status_bar import StatusBar
from nlapt_gui.widgets.toolbar_rail import RAIL_W

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

    def test_rail_and_two_columns(self, window) -> None:
        assert window.rail.width() == RAIL_W
        assert window.file_panel.width() == FILE_PANEL_DEFAULT_W
        assert window.body_splitter.count() == 2
        assert window.body_splitter.widget(0) is window.file_panel
        assert not hasattr(window, "title_bar")
        assert not hasattr(window, "status_bar")
        assert not window.tools_panel.isVisible()
        assert window.tools_panel.width() == TOOLS_PANEL_DEFAULT_W

    def test_editor_height_from_settings_and_splitter(self, window, controller) -> None:
        assert window.editor_panel.height() == controller.settings.editor_h
        window.splitter.editor_h_changed.emit(400)
        assert window.editor_panel.height() == 400
        low, high = EDITOR_H_RANGE
        window._apply_editor_height(high + 500)
        assert window.editor_panel.height() == high

    def test_editor_panel_uses_real_translate_bridge(self, window) -> None:
        assert window.editor_panel._bridge is window.translate_bridge

    def test_owns_batch_progress_dialog(self, window) -> None:
        from nlapt_gui.widgets.batch_progress_dialog import BatchProgressDialog

        assert isinstance(window.batch_progress_dialog, BatchProgressDialog)
        assert not window.batch_progress_dialog.isVisible()

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
        window.showFullScreen()
        window.toggle_max_restore()
        assert not (window.windowState() & self.ZOOMED)

    def test_info_bar_button_delegates_to_window(self, qtbot, window) -> None:
        window.showMaximized()
        window.preview_panel.max_button.clicked.emit()
        assert not (window.windowState() & self.ZOOMED)


class TestRail:
    def test_save_button_saves_current(self, qtbot, window, controller) -> None:
        controller.set_caption(K1, "changed via rail", "编辑")
        with qtbot.waitSignal(controller.files_saved, timeout=2000):
            window.rail.save_button.click()
        assert not controller.record(K1).dirty

    def test_view_mode_stays_on_file_panel(self, window, controller) -> None:
        window.file_panel.view_buttons["big"].click()
        assert controller.view_mode == "big"
        controller.set_view_mode("list")
        assert window.file_panel.view_buttons["list"].property("segActive") is True

    def test_theme_popup_rows_and_pick(self, qtbot, window, manager) -> None:
        popup = window.rail.open_theme_popup()
        qtbot.addWidget(popup)
        rows = popup.rows()
        assert len(rows) == len(THEMES)
        assert [row.theme_name for row in rows] == list(THEMES)
        target = next(row for row in rows if row.theme_name == "石墨")
        qtbot.mouseClick(target, Qt.MouseButton.LeftButton)
        assert manager.theme_name == "石墨"

    def test_theme_change_updates_controller_settings(self, window, controller, manager) -> None:
        manager.apply("墨黑")
        assert controller.settings.theme == "墨黑"

    def test_open_folder_action_emits_signal(self, qtbot, window, monkeypatch) -> None:
        picked: list[bool] = []
        monkeypatch.setattr(window, "pick_folder", lambda: picked.append(True))
        window.rail.open_folder_requested.disconnect()
        window.rail.open_folder_requested.connect(window.pick_folder)
        with qtbot.waitSignal(window.rail.open_folder_requested, timeout=1000):
            window.rail.open_button.click()
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


class TestToolsDrawer:
    def test_starts_hidden(self, window) -> None:
        assert not window.tools_panel.isVisible()
        assert not window.rail.tools_open()

    def test_rail_toggle_shows_overlay_on_the_right(self, window) -> None:
        window.rail.set_tools_open(True)
        assert window.tools_panel.isVisible()
        assert window.tools_panel.width() == TOOLS_PANEL_DEFAULT_W
        assert window.tools_panel.x() == window.width() - TOOLS_PANEL_DEFAULT_W
        assert window.tools_panel.y() == HEADER_H
        assert window.tools_panel.height() == window.height() - HEADER_H

    def test_drawer_leaves_window_controls_uncovered(self, window) -> None:
        window.rail.set_tools_open(True)
        close_btn = window.preview_panel.close_button
        tools_rect = QRect(
            window.tools_panel.mapToGlobal(QPoint(0, 0)), window.tools_panel.size()
        )
        close_rect = QRect(close_btn.mapToGlobal(QPoint(0, 0)), close_btn.size())
        assert not tools_rect.intersects(close_rect)

    def test_escape_closes_drawer(self, qtbot, window) -> None:
        window.rail.set_tools_open(True)
        qtbot.keyClick(window, Qt.Key.Key_Escape)
        assert not window.rail.tools_open()
        assert not window.tools_panel.isVisible()

    def test_open_slide_reaches_docked_geometry(self, window) -> None:
        anim.set_animations_enabled(True)
        try:
            window.rail.set_tools_open(True)
            animation = window._tools_anim
            assert animation is not None
            animation.setCurrentTime(animation.duration())
            assert window.tools_panel.isVisible()
            assert window.tools_panel.width() == TOOLS_PANEL_DEFAULT_W
            assert window.tools_panel.x() == window.width() - TOOLS_PANEL_DEFAULT_W
            assert window.tools_panel.y() == HEADER_H
            assert window.tools_panel.height() == window.height() - HEADER_H
        finally:
            anim.set_animations_enabled(False)

    def test_close_slide_hides_at_end(self, window) -> None:
        anim.set_animations_enabled(True)
        try:
            window.rail.set_tools_open(True)
            window._tools_anim.setCurrentTime(window._tools_anim.duration())
            window.rail.set_tools_open(False)
            animation = window._tools_anim
            assert animation is not None
            assert window.tools_panel.isVisible()
            animation.setCurrentTime(animation.duration())
            assert not window.tools_panel.isVisible()
        finally:
            anim.set_animations_enabled(False)

    def test_reopen_during_close_stays_visible(self, window) -> None:
        anim.set_animations_enabled(True)
        try:
            window.rail.set_tools_open(True)
            window._tools_anim.setCurrentTime(window._tools_anim.duration())
            window.rail.set_tools_open(False)
            assert window.tools_panel.isVisible()
            window.rail.set_tools_open(True)
            animation = window._tools_anim
            assert animation is not None
            animation.setCurrentTime(animation.duration())
            assert window.tools_panel.isVisible()
            assert window.tools_panel.x() == window.width() - TOOLS_PANEL_DEFAULT_W
        finally:
            anim.set_animations_enabled(False)

    def test_resize_during_close_hides(self, window) -> None:
        anim.set_animations_enabled(True)
        try:
            window.rail.set_tools_open(True)
            window._tools_anim.setCurrentTime(window._tools_anim.duration())
            window.rail.set_tools_open(False)
            assert window._tools_anim is not None
            window.resize(window.width() + 80, window.height() + 40)
            assert not window.tools_panel.isVisible()
        finally:
            anim.set_animations_enabled(False)

    def test_resize_during_open_snaps_to_new_edge(self, window) -> None:
        anim.set_animations_enabled(True)
        try:
            window.rail.set_tools_open(True)
            assert window._tools_anim is not None
            window.resize(window.width() + 80, window.height() + 40)
            assert window.tools_panel.isVisible()
            assert window.tools_panel.x() == window.width() - TOOLS_PANEL_DEFAULT_W
            assert window.tools_panel.y() == HEADER_H
            assert window.tools_panel.height() == window.height() - HEADER_H
            assert window.tools_panel.width() == TOOLS_PANEL_DEFAULT_W
        finally:
            anim.set_animations_enabled(False)


class TestExport:
    def test_no_dataset_toasts(self, qtbot, qapp, manager) -> None:
        ctrl = AppController(NLaptApp(), settings=UISettings())
        win = MainWindow(ctrl, manager)
        qtbot.addWidget(win)
        toasts: list[tuple[str, str]] = []
        ctrl.toast_requested.connect(lambda text, kind: toasts.append((text, kind)))
        win.export_dataset()
        assert (TOAST_NO_DATASET, TOAST_WARN) in toasts

    def test_dirty_cancel_skips_dialog(self, qtbot, window, controller, monkeypatch, tmp_path) -> None:
        from nlapt_gui.widgets import main_window as mw

        controller.set_caption(K1, "dirty export", "编辑")
        monkeypatch.setattr(mw, "ask_confirm", lambda *args, **kwargs: False)
        called: list[str] = []
        monkeypatch.setattr(
            mw.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *args, **kwargs: called.append("dialog") or ("", "")),
        )
        window.export_dataset()
        assert called == []

    def test_export_writes_zip(self, qtbot, window, controller, monkeypatch, tmp_path) -> None:
        from nlapt_gui.widgets import main_window as mw

        dest = tmp_path / "dataset.zip"
        monkeypatch.setattr(
            mw.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *args, **kwargs: (str(dest), "ZIP (*.zip)")),
        )
        with qtbot.waitSignal(controller.toast_requested, timeout=2000) as blocker:
            window.export_dataset()
        assert dest.is_file()
        assert "已导出" in blocker.args[0]


class TestStatusBarStandalone:
    def test_left_summary(self, qtbot, controller, demo_dataset: Path) -> None:
        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        text = bar.left_label.text()
        assert str(demo_dataset) in text
        assert "UTF-8" in text
        assert f"{len(controller.keys())} 个标注文件" in text

    def test_mode_label_tracks_controller(self, qtbot, controller) -> None:
        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        assert bar.mode_label.text() == "模式 · 胶囊"
        controller.set_mode("text")
        assert bar.mode_label.text() == "模式 · 文本"

    def test_theme_label_tracks_manager(self, qtbot, controller, manager) -> None:
        bar = StatusBar(controller, manager)
        qtbot.addWidget(bar)
        manager.apply("石墨")
        assert bar.theme_label.text() == "主题 · 石墨"

    def test_no_dataset_state(self, qtbot, qapp) -> None:
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

    def test_ctrl_y_redoes_caption(self, qtbot, window, controller) -> None:
        controller.set_current(K1)
        controller.set_caption(K1, "edited for redo", "编辑")
        window.setFocus()
        qtbot.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        qtbot.keyClick(window, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert controller.record(K1).text == "edited for redo"

    def test_ctrl_z_inside_text_input_stays_local(self, qtbot, window, controller) -> None:
        controller.set_current(K1)
        controller.set_caption(K1, "edited caption text", "编辑")
        field = QLineEdit(window)
        field.show()
        field.setFocus()
        qtbot.waitUntil(lambda: field.hasFocus(), timeout=2000)
        qtbot.keyClicks(field, "abc")
        qtbot.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
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


class TestCompareInferAction:
    def test_dispatch_routes_scope_to_compare_launcher(
        self, window, controller, monkeypatch
    ) -> None:
        from nlapt_gui.widgets import main_window as mw
        from nlapt_gui.widgets.tools_menu import ACTION_INFER_COMPARE, LABEL_INFER_COMPARE

        picked: list[str] = []
        launched: list[tuple[tuple[str, ...], object]] = []
        monkeypatch.setattr(
            mw,
            "pick_scope_keys",
            lambda ctrl, title, parent=None: picked.append(title) or (K1, K2),
        )
        monkeypatch.setattr(
            mw,
            "open_compare_infer",
            lambda ctrl, keys, loader, parent=None: launched.append((tuple(keys), loader)),
        )
        window._dispatch_tool_action(ACTION_INFER_COMPARE)
        assert picked == [LABEL_INFER_COMPARE]
        assert launched == [((K1, K2), window.file_panel.thumbnail_loader)]
        # The file panel's right-click entry takes the same route (no scope dialog).
        window.file_panel.compare_infer_requested.emit((K2,))
        assert picked == [LABEL_INFER_COMPARE]
        assert launched[-1] == ((K2,), window.file_panel.thumbnail_loader)


class TestToasts:
    def test_controller_toasts_reach_overlay(self, qtbot, window, controller) -> None:
        controller.toast_requested.emit("集成冒烟提示", "ok")
        qtbot.waitUntil(
            lambda: "集成冒烟提示" in window.toast_overlay.active_texts(), timeout=2000
        )
