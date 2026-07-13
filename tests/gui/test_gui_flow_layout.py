"""Tests for nlapt_gui.widgets.flow_layout (wrapping chip layout)."""

from __future__ import annotations

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QWidget

from nlapt_gui.widgets.flow_layout import FlowLayout


def _make(qtbot, count: int, size: tuple[int, int] = (50, 20)):
    parent = QWidget()
    qtbot.addWidget(parent)
    layout = FlowLayout(parent)
    children = []
    for _ in range(count):
        child = QWidget(parent)
        child.setFixedSize(*size)
        layout.addWidget(child)
        children.append(child)
    return parent, layout, children


class TestFlowLayout:
    def test_count_and_item_access(self, qtbot, qapp) -> None:
        _parent, layout, children = _make(qtbot, 3)
        assert layout.count() == 3
        assert layout.itemAt(0).widget() is children[0]
        assert layout.itemAt(2).widget() is children[2]
        assert layout.itemAt(3) is None
        assert layout.itemAt(-1) is None

    def test_take_at_removes_item(self, qtbot, qapp) -> None:
        _parent, layout, children = _make(qtbot, 3)
        item = layout.takeAt(1)
        assert item.widget() is children[1]
        assert layout.count() == 2
        assert layout.takeAt(9) is None

    def test_single_row_when_wide(self, qtbot, qapp) -> None:
        _parent, layout, children = _make(qtbot, 3)
        layout.setGeometry(QRect(0, 0, 400, 100))
        assert children[0].geometry() == QRect(0, 0, 50, 20)
        assert children[1].geometry() == QRect(58, 0, 50, 20)  # 8px gap
        assert children[2].geometry() == QRect(116, 0, 50, 20)

    def test_wraps_to_new_row_when_narrow(self, qtbot, qapp) -> None:
        _parent, layout, children = _make(qtbot, 3)
        layout.setGeometry(QRect(0, 0, 120, 100))
        assert children[0].geometry() == QRect(0, 0, 50, 20)
        assert children[1].geometry() == QRect(58, 0, 50, 20)
        # third widget no longer fits -> wraps below with 8px vertical gap
        assert children[2].geometry() == QRect(0, 28, 50, 20)

    def test_height_for_width_grows_when_narrow(self, qtbot, qapp) -> None:
        _parent, layout, _children = _make(qtbot, 3)
        assert layout.hasHeightForWidth()
        assert layout.heightForWidth(400) == 20
        assert layout.heightForWidth(120) == 48  # two rows: 20 + 8 + 20

    def test_insert_widget_positions_item(self, qtbot, qapp) -> None:
        parent, layout, children = _make(qtbot, 2)
        inserted = QWidget(parent)
        inserted.setFixedSize(50, 20)
        layout.insert_widget(1, inserted)
        assert layout.itemAt(0).widget() is children[0]
        assert layout.itemAt(1).widget() is inserted
        assert layout.itemAt(2).widget() is children[1]
        layout.setGeometry(QRect(0, 0, 400, 100))
        assert inserted.geometry().x() == 58

    def test_insert_widget_clamps_index(self, qtbot, qapp) -> None:
        parent, layout, _children = _make(qtbot, 2)
        tail = QWidget(parent)
        tail.setFixedSize(50, 20)
        layout.insert_widget(99, tail)
        assert layout.itemAt(2).widget() is tail

    def test_minimum_size_covers_largest_child(self, qtbot, qapp) -> None:
        parent, layout, _children = _make(qtbot, 2)
        big = QWidget(parent)
        big.setFixedSize(80, 33)
        layout.addWidget(big)
        assert layout.minimumSize().width() >= 80
        assert layout.minimumSize().height() >= 33
