"""Tests for MainWindow resize fixes: click-safe edge resize (startSystemResize
+ hover cursor), the resizable 2-column QSplitter body, and splitter-size
persistence.

Covers issue #1 (robust, click-safe frameless window resize) and issue #4
(draggable column dividers). Runs offscreen; no real waits.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QSplitter

from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME
from nlapt_gui.widgets.main_window import (
    EDGE_MARGIN_PX,
    FILE_PANEL_DEFAULT_W,
    PANEL_MAX_W,
    PANEL_MIN_W,
    MainWindow,
    load_splitter_sizes,
    save_splitter_sizes,
)


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


class TestSplitterBody:
    def test_body_is_splitter_with_two_widgets(self, window) -> None:
        splitter = window.body_splitter
        assert isinstance(splitter, QSplitter)
        assert splitter.orientation() == Qt.Orientation.Horizontal
        assert splitter.count() == 2
        assert splitter.widget(0) is window.file_panel
        # The middle widget hosts the preview + vertical splitter + editor.
        middle = splitter.widget(1)
        assert window.preview_panel.parent() is middle
        assert window.editor_panel.parent() is middle

    def test_side_panel_width_lock_released(self, window) -> None:
        assert window.file_panel.minimumWidth() == PANEL_MIN_W
        assert window.file_panel.maximumWidth() == PANEL_MAX_W

    def test_default_side_widths_pinned_on_show(self, window) -> None:
        assert window.file_panel.width() == FILE_PANEL_DEFAULT_W

    def test_dragging_divider_changes_panel_widths(self, window) -> None:
        splitter = window.body_splitter
        left, middle = splitter.sizes()
        splitter.setSizes([left + 120, middle - 120])
        assert window.file_panel.width() == left + 120

    def test_vertical_editor_splitter_still_works(self, window) -> None:
        window.splitter.editor_h_changed.emit(410)
        assert window.editor_panel.height() == 410


class TestSplitterPersistence:
    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "body_splitter.json"
        save_splitter_sizes([280, 700], path)
        assert load_splitter_sizes(path) == [280, 700]

    def test_missing_file_returns_none(self, tmp_path) -> None:
        assert load_splitter_sizes(tmp_path / "nope.json") is None

    def test_corrupt_file_returns_none(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{ not json", encoding="utf-8")
        assert load_splitter_sizes(path) is None

    def test_wrong_shape_returns_none(self, tmp_path) -> None:
        path = tmp_path / "shape.json"
        path.write_text("[1]", encoding="utf-8")
        assert load_splitter_sizes(path) is None

    def test_legacy_three_sizes_keep_left_width(self, tmp_path) -> None:
        path = tmp_path / "legacy.json"
        save_splitter_sizes([270, 700, 380], path)
        assert load_splitter_sizes(path) == [270, 700]

    def test_saved_sizes_restore_side_widths(self, qtbot, controller, manager, tmp_path) -> None:
        path = tmp_path / "restore.json"
        save_splitter_sizes([270, 700], path)
        win = MainWindow(controller, manager, splitter_sizes_path=path)
        qtbot.addWidget(win)
        win.show()
        qtbot.waitExposed(win)
        assert win.file_panel.width() == 270

    def test_close_persists_current_sizes(self, qtbot, controller, manager, tmp_path) -> None:
        path = tmp_path / "onclose.json"
        win = MainWindow(controller, manager, splitter_sizes_path=path)
        qtbot.addWidget(win)
        win.show()
        qtbot.waitExposed(win)
        left, middle = win.body_splitter.sizes()
        win.body_splitter.setSizes([left + 40, middle - 40])
        win.close()
        saved = load_splitter_sizes(path)
        assert saved is not None
        assert saved[0] == left + 40


class TestEdgeResize:
    """Click-safe edge resize: edge detection + hover cursor, no WM_NCHITTEST."""

    def test_edge_at_detects_all_edges(self, window) -> None:
        w, h = window.width(), window.height()
        assert window.edge_at(QPoint(0, h // 2)) == Qt.Edge.LeftEdge
        assert window.edge_at(QPoint(w - 1, h // 2)) == Qt.Edge.RightEdge
        assert window.edge_at(QPoint(w // 2, 0)) == Qt.Edge.TopEdge
        assert window.edge_at(QPoint(w // 2, h - 1)) == Qt.Edge.BottomEdge
        assert window.edge_at(QPoint(0, 0)) == (Qt.Edge.LeftEdge | Qt.Edge.TopEdge)
        assert window.edge_at(QPoint(w - 1, h - 1)) == (
            Qt.Edge.RightEdge | Qt.Edge.BottomEdge
        )

    def test_center_has_no_edge(self, window) -> None:
        # The interior is NEVER treated as a resize border, so clicks are safe.
        assert not window.edge_at(QPoint(window.width() // 2, window.height() // 2))

    def test_edge_margin_boundary(self, window) -> None:
        mid = window.height() // 2
        assert window.edge_at(QPoint(EDGE_MARGIN_PX, mid)) == Qt.Edge.LeftEdge
        assert not window.edge_at(QPoint(EDGE_MARGIN_PX + 1, mid))

    def test_edge_cursor_shapes(self, window) -> None:
        assert window.edge_cursor_shape(Qt.Edge.LeftEdge) == Qt.CursorShape.SizeHorCursor
        assert window.edge_cursor_shape(Qt.Edge.TopEdge) == Qt.CursorShape.SizeVerCursor
        assert (
            window.edge_cursor_shape(Qt.Edge.LeftEdge | Qt.Edge.TopEdge)
            == Qt.CursorShape.SizeFDiagCursor
        )
        assert (
            window.edge_cursor_shape(Qt.Edge.RightEdge | Qt.Edge.TopEdge)
            == Qt.CursorShape.SizeBDiagCursor
        )
        no_edge = window.edge_at(QPoint(window.width() // 2, window.height() // 2))
        assert window.edge_cursor_shape(no_edge) is None

    def test_hover_cursor_sets_and_clears(self, window) -> None:
        from PySide6.QtWidgets import QApplication

        window._update_resize_cursor(Qt.Edge.LeftEdge)
        assert window._resize_cursor_active
        assert QApplication.overrideCursor() is not None
        interior = window.edge_at(QPoint(window.width() // 2, window.height() // 2))
        window._update_resize_cursor(interior)  # move to interior
        assert not window._resize_cursor_active


class TestFadeIn:
    def test_fade_in_finishes_synchronously(self, window) -> None:
        # Skip-safe: the end state is reachable without waiting.
        window.finish_fade_in()
        assert window.graphicsEffect() is None

    def test_fade_in_only_runs_once(self, qtbot, window) -> None:
        window.finish_fade_in()
        # A subsequent hide/show must not restart the fade-in.
        window.hide()
        window.show()
        qtbot.waitExposed(window)
        assert window._did_fade_in is True
        assert window.graphicsEffect() is None
