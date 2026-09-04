"""Issue #2 - make single-image preview zoom/viewing comfortable (design 5).

Covers the four comfort behaviors added to :class:`PreviewPanel`:

1. **Zoom toward cursor** - a wheel zoom keeps the image point under the mouse
   fixed by shifting the scroll offset toward the cursor (not top-left anchored).
2. **Drag to pan** - when zoomed past 适应 a left-drag moves the scroll bars;
   at 适应 no pan starts.
3. **Double-click** toggles 适应 <-> 100% original size.
4. **Smoother zoom** - sub-notch wheel deltas accumulate; a full notch still
   equals one ``ZOOM_STEP``.

Plus the skip-safe image cross-fade (final state exposed synchronously via
``_SingleView.finish_image_fade``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent

from nlapt.app import NLaptApp

from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings
from nlapt_gui.widgets.preview_panel import PreviewPanel

K1 = "0001.png"


def _wheel(delta_y: int, pos: QPointF) -> QWheelEvent:
    return QWheelEvent(
        pos,
        pos,
        QPoint(0, delta_y),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _mouse(kind: QEvent.Type, local: QPointF, glob: QPointF, button, buttons) -> QMouseEvent:
    return QMouseEvent(kind, local, glob, button, buttons, Qt.KeyboardModifier.NoModifier)


def _dbl_click(pos: QPointF) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


@pytest.fixture()
def big_controller(qtbot, tmp_path: Path) -> AppController:
    """A dataset whose image is large enough to overflow the viewport when zoomed."""
    from PIL import Image

    root = tmp_path / "big"
    root.mkdir()
    for index in range(2):
        image = root / f"{index:04d}.png"
        Image.new("RGB", (600, 450), (30 + index * 40, 60, 90)).save(image, format="PNG")
        image.with_suffix(".txt").write_text(f"tag{index}", encoding="utf-8")
    ctrl = AppController(NLaptApp(), settings=UISettings())
    with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
        ctrl.open_dataset(root)
    return ctrl


@pytest.fixture()
def big_panel(qtbot, big_controller: AppController) -> PreviewPanel:
    panel = PreviewPanel(big_controller)
    qtbot.addWidget(panel)
    panel.resize(520, 460)
    panel.show()
    qtbot.waitExposed(panel)
    key = big_controller.keys()[0]
    big_controller.set_current(key)
    qtbot.waitUntil(lambda: panel.pixmap_for(key) is not None, timeout=2000)
    return panel


class TestZoomTowardCursor:
    def test_wheel_zoom_keeps_point_under_cursor(self, big_panel: PreviewPanel) -> None:
        """Zooming in with the cursor off-center preserves the image fraction there."""
        panel = big_panel
        panel._set_zoom(200)  # overflow the viewport so there is scroll headroom
        assert panel.single_pannable() is True

        rect = panel.single_view.image_rect()
        assert rect is not None
        anchor = QPointF(rect.left() + 0.75 * rect.width(), rect.top() + 0.60 * rect.height())

        hbar, vbar = panel.single_scrollbars()
        # Viewport-relative position of the anchor before the zoom.
        vp_x = anchor.x() - hbar.value()
        vp_y = anchor.y() - vbar.value()

        panel.wheel_zoom(1, anchor)  # -> 220%
        assert panel.zoom_label() == "220%"

        new_rect = panel.single_view.image_rect()
        hbar, vbar = panel.single_scrollbars()
        # The same viewport position now maps to this view coordinate.
        view_x = vp_x + hbar.value()
        view_y = vp_y + vbar.value()
        fx = (view_x - new_rect.left()) / new_rect.width()
        fy = (view_y - new_rect.top()) / new_rect.height()
        assert abs(fx - 0.75) < 0.02
        assert abs(fy - 0.60) < 0.02

    def test_offset_shifts_toward_cursor_not_top_left(self, big_panel: PreviewPanel) -> None:
        """A right/low cursor pushes the scroll offset up from the top-left origin."""
        panel = big_panel
        panel._set_zoom(200)
        hbar, vbar = panel.single_scrollbars()
        assert hbar.value() == 0 and vbar.value() == 0

        rect = panel.single_view.image_rect()
        anchor = QPointF(rect.left() + 0.9 * rect.width(), rect.top() + 0.9 * rect.height())
        panel.single_view.wheelEvent(_wheel(120, anchor))

        hbar, vbar = panel.single_scrollbars()
        assert hbar.value() > 0
        assert vbar.value() > 0


class TestDragToPan:
    def test_drag_pans_when_zoomed(self, big_panel: PreviewPanel) -> None:
        panel = big_panel
        panel._set_zoom(200)
        view = panel.single_view
        hbar, vbar = panel.single_scrollbars()
        assert hbar.value() == 0

        view.mousePressEvent(
            _mouse(
                QEvent.Type.MouseButtonPress,
                QPointF(200.0, 200.0),
                QPointF(300.0, 300.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
            )
        )
        assert view.cursor().shape() == Qt.CursorShape.ClosedHandCursor
        view.mouseMoveEvent(
            _mouse(
                QEvent.Type.MouseMove,
                QPointF(160.0, 170.0),
                QPointF(260.0, 270.0),  # dragged up-left by (40, 30)
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
            )
        )
        # Dragging content up-left scrolls the view down-right (offset grows).
        assert hbar.value() == 40
        assert vbar.value() == 30
        view.mouseReleaseEvent(
            _mouse(
                QEvent.Type.MouseButtonRelease,
                QPointF(160.0, 170.0),
                QPointF(260.0, 270.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
            )
        )
        assert view.cursor().shape() == Qt.CursorShape.OpenHandCursor

    def test_no_pan_at_fit(self, big_panel: PreviewPanel) -> None:
        panel = big_panel
        panel.zoom_fit()
        view = panel.single_view
        assert panel.single_pannable() is False
        hbar, _ = panel.single_scrollbars()
        before = hbar.value()

        view.mousePressEvent(
            _mouse(
                QEvent.Type.MouseButtonPress,
                QPointF(200.0, 200.0),
                QPointF(300.0, 300.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
            )
        )
        assert view._pan_origin is None  # pan did not start
        view.mouseMoveEvent(
            _mouse(
                QEvent.Type.MouseMove,
                QPointF(120.0, 120.0),
                QPointF(220.0, 220.0),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
            )
        )
        assert hbar.value() == before


class TestDoubleClickToggle:
    def test_double_click_toggles_fit_and_actual(self, qtbot, controller) -> None:
        controller.set_current(K1)
        panel = PreviewPanel(controller)
        qtbot.addWidget(panel)
        assert panel.zoom_label() == "适应"

        pos = QPointF(5.0, 5.0)
        panel.single_view.mouseDoubleClickEvent(_dbl_click(pos))
        assert panel.zoom_label() == "100%"

        panel.single_view.mouseDoubleClickEvent(_dbl_click(pos))
        assert panel.zoom_label() == "适应"

    def test_double_click_ignored_in_multi_mode(self, qtbot, controller) -> None:
        controller.toggle_selected(K1)
        controller.toggle_selected("0002.png")
        panel = PreviewPanel(controller)
        qtbot.addWidget(panel)
        assert controller.multi_mode() is True
        before = panel.zoom_label()
        panel.single_view.mouseDoubleClickEvent(_dbl_click(QPointF(5.0, 5.0)))
        assert panel.zoom_label() == before


class TestSmootherZoom:
    def test_sub_notch_deltas_accumulate(self, qtbot, controller) -> None:
        """Three 40-unit deltas equal one 120-unit notch (one ZOOM_STEP)."""
        controller.set_current(K1)
        panel = PreviewPanel(controller)
        qtbot.addWidget(panel)
        pos = QPointF(5.0, 5.0)

        panel.single_view.wheelEvent(_wheel(40, pos))
        assert panel.zoom_label() == "适应"  # not enough to step yet
        panel.single_view.wheelEvent(_wheel(40, pos))
        assert panel.zoom_label() == "适应"
        panel.single_view.wheelEvent(_wheel(40, pos))
        assert panel.zoom_label() == "120%"  # accumulated a full notch

    def test_full_notch_steps_once(self, qtbot, controller) -> None:
        controller.set_current(K1)
        panel = PreviewPanel(controller)
        qtbot.addWidget(panel)
        panel.single_view.wheelEvent(_wheel(120, QPointF(5.0, 5.0)))
        assert panel.zoom_label() == "120%"


class TestImageFade:
    def test_fade_disabled_is_end_state(self, big_panel: PreviewPanel) -> None:
        view = big_panel.single_view
        image = QImage(20, 20, QImage.Format.Format_RGB32)
        image.fill(0)
        view.set_pixmap(QPixmap.fromImage(image))
        assert view._fade_alpha == 1.0
        assert view.graphicsEffect() is None

    def test_fade_is_skip_safe(self, big_panel: PreviewPanel) -> None:
        """The fade exposes its final state synchronously; content stays correct."""
        from nlapt_gui import anim

        panel = big_panel
        panel.zoom_fit()
        view = panel.single_view
        anim.set_animations_enabled(True)
        try:
            image = QImage(20, 20, QImage.Format.Format_RGB32)
            image.fill(0)
            fresh = QPixmap.fromImage(image)
            view.set_pixmap(fresh)

            # Paint-level fade (no QGraphicsEffect - those crash during resizes):
            # mid-fade the alpha is below 1.0 but the pixmap is already current.
            assert view.graphicsEffect() is None
            assert view._fade_alpha < 1.0
            assert view._pixmap is fresh

            view.finish_image_fade()
            assert view._fade_alpha == 1.0
            assert not panel.grab().isNull()
        finally:
            anim.set_animations_enabled(False)

    def test_finish_fade_is_idempotent(self, big_panel: PreviewPanel) -> None:
        view = big_panel.single_view
        view.finish_image_fade()
        view.finish_image_fade()  # no-op, must not raise
        assert view._fade_alpha == 1.0
        assert view.graphicsEffect() is None
