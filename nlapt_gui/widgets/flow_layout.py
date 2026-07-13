"""FlowLayout - a wrapping layout for the chips editing mode.

Ports Qt's classic flow-layout example to PySide6 with an extra
:meth:`insert_widget` used to place the inline chip editor at an arbitrary
position among the chips. Items flow left-to-right and wrap onto new rows
(design: ``display:flex; flex-wrap:wrap; gap:8px``).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QWidget

from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Design gap between chips ("gap:8px").
DEFAULT_SPACING = 8


class FlowLayout(QLayout):
    """Left-to-right wrapping layout with fixed horizontal/vertical gaps."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        margin: int = 0,
        h_spacing: int = DEFAULT_SPACING,
        v_spacing: int = DEFAULT_SPACING,
    ) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    # -- spacing -----------------------------------------------------------------
    def horizontal_spacing(self) -> int:
        return self._h_spacing

    def vertical_spacing(self) -> int:
        return self._v_spacing

    # -- QLayout interface -------------------------------------------------------
    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt override
        self._items.append(item)

    def insert_widget(self, index: int, widget: QWidget) -> None:
        """Add ``widget`` and move its layout item to position ``index``."""
        self.addWidget(widget)
        item = self._items.pop()
        clamped = max(0, min(index, len(self._items)))
        self._items.insert(clamped, item)
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt override
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt override
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt override
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt override
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt override
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt override
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt override
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )
        return size

    # -- layout engine -----------------------------------------------------------
    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        """Place items row by row; returns the total height used."""
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = effective.x()
        y = effective.y()
        row_height = 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if next_x - self._h_spacing > effective.right() + 1 and row_height > 0:
                x = effective.x()
                y = y + row_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                row_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y() + margins.bottom()
