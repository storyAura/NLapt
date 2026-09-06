"""Tests for nlapt_gui.widgets.preview_panel (header, zoom, multi, splitter)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from nlapt.app import NLaptApp

from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings
from nlapt_gui.theme.tokens import EDITOR_H_RANGE, THEMES, ZOOM_RANGE
from nlapt_gui.widgets.preview_panel import (
    MULTI_PILL_FMT,
    OVERFLOW_FMT,
    TEXT_FIT,
    PreviewPanel,
    SplitterHandle,
)

K1 = "0001.png"
K2 = "0002.png"
K3 = "10_concept/0003.png"
K4 = "10_concept/0004.png"


@pytest.fixture()
def panel(qtbot, controller: AppController) -> PreviewPanel:
    widget = PreviewPanel(controller)
    qtbot.addWidget(widget)
    widget.resize(900, 620)
    widget.show()
    return widget


def _mouse_event(kind: QEvent.Type, global_y: float, button, buttons) -> QMouseEvent:
    point = QPointF(10.0, 4.0)
    return QMouseEvent(
        kind, point, point, QPointF(50.0, global_y), button, buttons,
        Qt.KeyboardModifier.NoModifier,
    )


class TestHeader:
    def test_name_meta_and_pos(self, panel: PreviewPanel, controller) -> None:
        assert panel.name_label.text() == "0001.png"
        assert panel.meta_pill.text() == "PNG"
        assert "×" in panel.dim_label.text()
        assert panel.size_label.text()
        assert panel.mtime_label.text().startswith("修改于 ")
        assert panel.pos_label.text() == "1 / 4"

    def test_tooltips_and_texts(self, panel: PreviewPanel) -> None:
        assert panel.prev_button.toolTip() == "上一张 (Alt+↑)"
        assert panel.next_button.toolTip() == "下一张 (Alt+↓)"
        assert panel.min_button is not None
        assert panel.max_button is not None
        assert panel.close_button is not None

    def test_dirty_pill_follows_dirty_state(self, qtbot, panel, controller) -> None:
        assert not panel.dirty_pill.isVisible()
        controller.set_caption(K1, "edited", "test")
        assert panel.dirty_pill.isVisible()
        assert panel.dirty_pill.text() == "未保存"
        with qtbot.waitSignal(controller.files_saved, timeout=2000):
            controller.save_current()
        assert not panel.dirty_pill.isVisible()

    def test_nav_buttons_and_pos_label(self, qtbot, panel, controller) -> None:
        qtbot.mouseClick(panel.next_button, Qt.MouseButton.LeftButton)
        assert controller.current_key == K2
        assert panel.pos_label.text() == "2 / 4"
        assert panel.name_label.text() == "0002.png"
        qtbot.mouseClick(panel.prev_button, Qt.MouseButton.LeftButton)
        assert controller.current_key == K1
        assert panel.pos_label.text() == "1 / 4"


class TestFilteredNavigation:
    def test_empty_filter_results_disable_navigation_and_keep_preview(
        self, qtbot, panel: PreviewPanel, controller: AppController
    ) -> None:
        qtbot.waitUntil(lambda: panel.pixmap_for(K1) is not None, timeout=2000)
        pixmap = panel.single_view._pixmap
        panel._set_zoom(200)
        controller.set_filter("no-matching-file-or-caption")

        assert controller.current_key == K1
        assert panel.name_label.text() == K1
        assert panel.pos_label.text() == "未匹配 / 0"
        assert panel.single_view._pixmap is pixmap
        assert panel.zoom_label() == "200%"
        assert not panel.prev_button.isEnabled()
        assert not panel.next_button.isEnabled()
        qtbot.mouseClick(panel.next_button, Qt.MouseButton.LeftButton)
        qtbot.mouseClick(panel.prev_button, Qt.MouseButton.LeftButton)
        assert controller.current_key == K1

        controller.set_filter("")
        assert panel.prev_button.isEnabled()
        assert panel.next_button.isEnabled()
        assert panel.pos_label.text() == "1 / 4"

    @pytest.mark.parametrize(
        ("button_name", "target", "position"),
        (("next_button", K1, "1 / 3"), ("prev_button", K3, "3 / 3")),
    )
    def test_unmatched_current_enters_result_boundary(
        self, qtbot, panel: PreviewPanel, controller: AppController,
        button_name: str, target: str, position: str,
    ) -> None:
        controller.set_current(K4)
        controller.set_filter("1girl")
        assert controller.current_key == K4
        assert panel.name_label.text() == "0004.png"
        assert panel.pos_label.text() == "未匹配 / 3"
        assert panel.prev_button.isEnabled()
        assert panel.next_button.isEnabled()

        qtbot.mouseClick(getattr(panel, button_name), Qt.MouseButton.LeftButton)
        assert controller.current_key == target
        assert panel.pos_label.text() == position

    def test_matching_multi_selection_keeps_navigation_and_wraps(
        self, qtbot, panel: PreviewPanel, controller: AppController
    ) -> None:
        for key in (K1, K2, K3):
            controller.toggle_selected(key)
        controller.set_filter("1girl")
        assert panel.current_view() == "multi"
        assert panel.prev_button.isEnabled()
        assert panel.next_button.isEnabled()

        qtbot.mouseClick(panel.prev_button, Qt.MouseButton.LeftButton)
        assert controller.current_key == K3
        assert panel.pos_label.text() == "3 / 3"
        qtbot.mouseClick(panel.next_button, Qt.MouseButton.LeftButton)
        assert controller.current_key == K1
        assert panel.pos_label.text() == "1 / 3"
        assert [cell.key for cell in panel.multi_cells()] == [K1, K2, K3]


class TestZoom:
    def test_default_is_fit(self, panel: PreviewPanel) -> None:
        assert panel.zoom_label() == TEXT_FIT
        assert panel.zoom_label_widget.text() == TEXT_FIT
        assert panel.zoom_fit_button.text() == TEXT_FIT

    def test_zoom_in_starts_from_fit_percent(self, qtbot, panel: PreviewPanel) -> None:
        qtbot.waitUntil(lambda: panel.pixmap_for(K1) is not None, timeout=2000)
        # tiny 4x3 demo image in a large viewport -> fit caps at 100%
        assert panel.single_view.fit_percent() == 100
        qtbot.mouseClick(panel.zoom_in_button, Qt.MouseButton.LeftButton)
        assert panel.zoom_label() == "120%"
        qtbot.mouseClick(panel.zoom_out_button, Qt.MouseButton.LeftButton)
        assert panel.zoom_label() == "100%"

    def test_zoom_clamps_to_range(self, qtbot, panel: PreviewPanel) -> None:
        qtbot.waitUntil(lambda: panel.pixmap_for(K1) is not None, timeout=2000)
        low, high = ZOOM_RANGE
        for _ in range(60):
            panel.zoom_in()
        assert panel.zoom_label() == f"{high}%"
        for _ in range(60):
            panel.zoom_out()
        # Tiny demo image: fit caps at 100, so the floor is ZOOM_RANGE[0].
        assert panel.zoom_label() == f"{low}%"

    def test_zoom_out_reaches_whole_image_for_large_photos(
        self, qtbot, panel: PreviewPanel
    ) -> None:
        """Spec 5 / user report: the zoom floor must allow a whole-image view."""
        from PySide6.QtGui import QPixmap

        qtbot.waitUntil(lambda: panel.pixmap_for(K1) is not None, timeout=2000)
        # Simulate a big photo whose fit scale is far below 100%.
        panel.single_view.set_pixmap(QPixmap(2894, 4093))
        fit = panel.single_view.fit_percent()
        assert fit < 100
        for _ in range(60):
            panel.zoom_out()
        floor = int(panel.zoom_label().rstrip("%"))
        # The floor is at or below the fit scale -> the entire image is visible.
        assert floor <= max(1, fit)

    def test_fit_button_resets(self, qtbot, panel: PreviewPanel) -> None:
        panel.zoom_in()
        assert panel.zoom_label() != TEXT_FIT
        qtbot.mouseClick(panel.zoom_fit_button, Qt.MouseButton.LeftButton)
        assert panel.zoom_label() == TEXT_FIT


class TestMultiMode:
    def test_single_by_default(self, panel: PreviewPanel) -> None:
        assert panel.current_view() == "single"
        assert not panel.multi_pill.isVisible()
        assert panel.zoom_pill.isVisible()

    def test_multi_pill_and_grid(self, qtbot, panel, controller) -> None:
        for key in (K1, K2, K3):
            controller.toggle_selected(key)
        assert panel.current_view() == "multi"
        assert panel.multi_pill.isVisible()
        assert panel.multi_pill.text() == MULTI_PILL_FMT.format(n=3)
        assert not panel.zoom_pill.isVisible()
        assert [cell.key for cell in panel.multi_cells()] == [K1, K2, K3]
        assert not panel.overflow_pill.isVisible()  # 3 <= 4

    def test_click_cell_sets_current(self, qtbot, panel, controller) -> None:
        for key in (K1, K2, K3):
            controller.toggle_selected(key)
        cell = panel.multi_cells()[1]
        qtbot.waitUntil(lambda: cell.width() > 20)
        qtbot.mouseClick(cell, Qt.MouseButton.LeftButton, pos=cell.rect().center())
        assert controller.current_key == K2

    def test_clear_selection_returns_to_single(self, panel, controller) -> None:
        for key in (K1, K2, K3):
            controller.toggle_selected(key)
        controller.clear_selection()
        assert panel.current_view() == "single"
        assert not panel.multi_pill.isVisible()
        assert panel.zoom_pill.isVisible()

    def test_multi_cells_render(self, panel, controller) -> None:
        for key in (K1, K2, K3):
            controller.toggle_selected(key)
        controller.set_caption(K2, "dirty now", "test")  # dirty dot path
        assert not panel.grab().isNull()


class TestOverflow:
    @pytest.fixture()
    def big_controller(self, qtbot, tmp_path: Path) -> AppController:
        from PIL import Image

        root = tmp_path / "big"
        root.mkdir()
        for index in range(6):
            image = root / f"{index:04d}.png"
            Image.new("RGB", (4, 3), (index * 20, 0, 0)).save(image, format="PNG")
            image.with_suffix(".txt").write_text(f"tag{index}", encoding="utf-8")
        ctrl = AppController(NLaptApp(), settings=UISettings())
        with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
            ctrl.open_dataset(root)
        return ctrl

    def test_overflow_pill_and_first_four(self, qtbot, big_controller) -> None:
        panel = PreviewPanel(big_controller)
        qtbot.addWidget(panel)
        panel.resize(900, 620)
        panel.show()
        big_controller.select_all()
        assert panel.current_view() == "multi"
        assert len(panel.multi_cells()) == 4  # only first four shown
        assert panel.overflow_pill.isVisible()
        assert panel.overflow_pill.text() == OVERFLOW_FMT.format(n=6)


class TestSplitterHandle:
    def test_drag_emits_and_persists(self, qtbot, controller) -> None:
        handle = SplitterHandle(controller)
        qtbot.addWidget(handle)
        handle.resize(300, 8)
        assert handle.toolTip() == "拖动调整编辑区高度"
        start = controller.settings.editor_h
        handle.mousePressEvent(
            _mouse_event(
                QEvent.Type.MouseButtonPress, 100.0,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            )
        )
        with qtbot.waitSignal(handle.editor_h_changed, timeout=1000) as blocker:
            handle.mouseMoveEvent(
                _mouse_event(
                    QEvent.Type.MouseMove, 70.0,
                    Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                )
            )
        assert blocker.args == [start + 30]
        handle.mouseReleaseEvent(
            _mouse_event(
                QEvent.Type.MouseButtonRelease, 70.0,
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            )
        )
        assert controller.settings.editor_h == start + 30
        assert handle.editor_height() == start + 30

    def test_drag_clamps_to_range(self, qtbot, controller) -> None:
        handle = SplitterHandle(controller)
        qtbot.addWidget(handle)
        low, high = EDITOR_H_RANGE
        handle.mousePressEvent(
            _mouse_event(
                QEvent.Type.MouseButtonPress, 500.0,
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            )
        )
        handle.mouseMoveEvent(
            _mouse_event(
                QEvent.Type.MouseMove, -5000.0,
                Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
            )
        )
        assert handle.editor_height() == high
        handle.mouseMoveEvent(
            _mouse_event(
                QEvent.Type.MouseMove, 9000.0,
                Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
            )
        )
        assert handle.editor_height() == low
        handle.mouseReleaseEvent(
            _mouse_event(
                QEvent.Type.MouseButtonRelease, 9000.0,
                Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            )
        )
        assert controller.settings.editor_h == low


class TestTheme:
    def test_apply_tokens_and_render(self, panel: PreviewPanel, controller) -> None:
        panel.apply_tokens(THEMES["墨黑"])
        assert panel.current_tokens().name == "墨黑"
        assert not panel.grab().isNull()


class TestProgressiveImage:
    def test_uncached_switch_clears_identity_until_current_frame(self, qtbot, panel, controller) -> None:
        qtbot.waitUntil(lambda: panel.pixmap_for(K1) is not None, timeout=2000)
        panel._pix_cache.pop(K2, None)
        panel._coarse_cache.pop(K2, None)
        panel._loading.discard(K2)
        controller.set_current(K2)
        assert panel.single_view._pixmap is None
        qtbot.waitUntil(lambda: panel.pixmap_for(K2) is not None, timeout=2000)
        assert panel.single_view._pixmap is panel._pix_cache[K2]
