"""Width-aware caption chips and their single-line inline editor."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAbstractTextDocumentLayout,
    QFocusEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QTextDocument,
    QTextOption,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)

if TYPE_CHECKING:
    from nlapt_gui.widgets.chips_editor import ChipsEditor

CHIP_EDIT_TOOLTIP = "点击编辑 · 可拖拽排序"
CHIP_DELETE_TOOLTIP = "删除"
CHIP_DELETE_TEXT = "×"
DROP_INDICATOR_PX = 3
_CHIP_FIELD_MIN_W = 64
_CHIP_FIELD_PAD_W = 34


class InlineChipField(QLineEdit):
    """Single-line inline editor: Enter commits, Esc cancels, blur commits."""

    submitted = Signal()
    cancelled = Signal()
    focus_lost = Signal()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        hint = super().sizeHint()
        text = self.text() or self.placeholderText()
        width = self.fontMetrics().horizontalAdvance(text) + _CHIP_FIELD_PAD_W
        return QSize(max(_CHIP_FIELD_MIN_W, width), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(_CHIP_FIELD_MIN_W, super().minimumSizeHint().height())

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self.submitted.emit()
            return
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            self.cancelled.emit()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802 - Qt override
        super().focusOutEvent(event)
        self.focus_lost.emit()

    def cursor_index(self) -> int:
        return self.cursorPosition()


class _ChipText(QWidget):
    """Plain text whose measured line wrapping matches its painted layout."""

    def __init__(self, text: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("chipText")
        self.setAccessibleName(text)
        self.setProperty("mono", "true")
        self.setStyleSheet("font-size: 12.5px;")
        self.setToolTip(CHIP_EDIT_TOOLTIP)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._document = QTextDocument(self)
        self._document.setDocumentMargin(0)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self._document.setDefaultTextOption(option)
        self._document.setPlainText(text)

    def _measure(self, width: int) -> QSize:
        self.ensurePolished()
        self._document.setDefaultFont(self.font())
        self._document.setTextWidth(width)
        size = self._document.size()
        return QSize(math.ceil(self._document.idealWidth()), math.ceil(size.height()))

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self._measure(-1)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(1, self.fontMetrics().height())

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt override
        return self._measure(max(1, width)).height()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        self._measure(self.width())
        context = QAbstractTextDocumentLayout.PaintContext()
        context.palette = self.palette()
        context.clip = QRectF(self.rect())
        painter = QPainter(self)
        self._document.documentLayout().draw(painter, context)
        painter.end()


class ChipWidget(QFrame):
    """Caption pill with wrapping text and a top-right delete action."""

    def __init__(
        self, editor: "ChipsEditor", index: int, text: str, *, selected: bool = False
    ) -> None:
        super().__init__(editor)
        self._editor = editor
        self._index = index
        self._text = text
        self._press_pos: QPoint | None = None
        self._drop_indicator = False
        self.setObjectName(f"chip_{index}")
        self.setProperty("chip", "true")
        self.setProperty("chipSelected", "true" if selected else "false")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(CHIP_EDIT_TOOLTIP)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 5, 7, 5)
        row.setSpacing(7)
        self._label = _ChipText(text, self)
        row.addWidget(self._label, 1)
        remove = QPushButton(CHIP_DELETE_TEXT, self)
        remove.setObjectName("chipDelete")
        remove.setProperty("variant", "danger-ghost")
        remove.setFixedSize(16, 16)
        remove.setStyleSheet("padding: 0; border-radius: 8px; font-size: 13px;")
        remove.setToolTip(CHIP_DELETE_TOOLTIP)
        remove.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove.clicked.connect(lambda: self._editor.delete_segment(self._index))
        row.addWidget(remove, 0, Qt.AlignmentFlag.AlignTop)

    @property
    def index(self) -> int:
        return self._index

    @property
    def chip_text(self) -> str:
        return self._text

    def set_drop_indicator(self, on: bool) -> None:
        if on != self._drop_indicator:
            self._drop_indicator = on
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if self._drop_indicator:
            painter = QPainter(self)
            accent = self.palette().color(QPalette.ColorRole.Highlight)
            painter.fillRect(0, 0, DROP_INDICATOR_PX, self.height(), accent)
            painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._press_pos is None:
            return
        distance = (event.position().toPoint() - self._press_pos).manhattanLength()
        if distance >= QApplication.startDragDistance():
            self._press_pos = None
            self._editor.begin_drag(self._index, self)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._press_pos is not None and event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = None
            self._editor.on_segment_clicked(self._index)
        super().mouseReleaseEvent(event)
