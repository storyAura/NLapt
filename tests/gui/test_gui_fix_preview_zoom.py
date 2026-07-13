"""Regression: preview wheel-zoom + multi-cell double-click to single view.

Issue #3(a) - the mouse wheel over the single image must zoom (up = in, down =
out) reusing the zoom-pill logic (from 适应 the first step lands on 120%), and
must be ignored while the 2x2 compare grid is showing.

Issue #3(b) - double-clicking a compare-grid cell must open that image as the
standalone large single preview: it becomes current AND the selection collapses
(so ``multi_mode()`` turns false and the single view shows it).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent

from nlapt_gui.widgets.preview_panel import PreviewPanel

from tests.gui.conftest import DEMO_KEYS


def _wheel(delta_y: int) -> QWheelEvent:
    pos = QPointF(10.0, 10.0)
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


def _dbl_click() -> QMouseEvent:
    pos = QPointF(5.0, 5.0)
    return QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def test_wheel_up_zooms_in_from_fit(qtbot, controller) -> None:
    """Wheel-up over the single image steps 适应 -> 120%."""
    controller.set_current(DEMO_KEYS[0])
    panel = PreviewPanel(controller)
    qtbot.addWidget(panel)

    assert panel.current_view() == "single"
    assert panel.zoom_label() == "适应"

    panel.single_view.wheelEvent(_wheel(120))
    assert panel.zoom_label() == "120%"


def test_wheel_down_zooms_out(qtbot, controller) -> None:
    """Wheel-down steps back down (120% -> 100%)."""
    controller.set_current(DEMO_KEYS[0])
    panel = PreviewPanel(controller)
    qtbot.addWidget(panel)

    panel.single_view.wheelEvent(_wheel(120))
    assert panel.zoom_label() == "120%"
    panel.single_view.wheelEvent(_wheel(-120))
    assert panel.zoom_label() == "100%"


def test_wheel_ignored_in_multi_mode(qtbot, controller) -> None:
    """The compare grid must not respond to wheel-zoom."""
    controller.set_current(DEMO_KEYS[0])
    panel = PreviewPanel(controller)
    qtbot.addWidget(panel)
    panel.zoom_fit()

    controller.toggle_selected(DEMO_KEYS[0])
    controller.toggle_selected(DEMO_KEYS[1])
    assert controller.multi_mode() is True

    panel.single_view.wheelEvent(_wheel(120))
    assert panel.zoom_label() == "适应"


def test_double_click_multi_cell_opens_single(qtbot, controller) -> None:
    """Double-clicking a compare tile collapses to its standalone single view."""
    controller.toggle_selected(DEMO_KEYS[0])
    controller.toggle_selected(DEMO_KEYS[1])
    panel = PreviewPanel(controller)
    qtbot.addWidget(panel)

    assert controller.multi_mode() is True
    assert panel.current_view() == "multi"

    target = DEMO_KEYS[1]
    cell = next(c for c in panel.multi_cells() if c.key == target)
    cell.mouseDoubleClickEvent(_dbl_click())

    assert controller.current_key == target
    assert controller.multi_mode() is False
    assert panel.current_view() == "single"
