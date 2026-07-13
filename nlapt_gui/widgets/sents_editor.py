"""Sentences (分句) editing mode - numbered, drag-sortable segment rows.

Rows are the SAME comma segments as the chips mode (design: 按英文逗号分段),
not the core sentence splitter. Each row shows a 6-dot drag handle, a
zero-padded number badge, the segment text (click to open a multi-line
inline editor with autosize) and a delete button; ``+ 添加分段`` inserts at
the end. All commit/undo/toolbar behavior comes from
:class:`~nlapt_gui.widgets.chips_editor.SegmentEditorBase`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QMimeData, QObject, QPoint, Qt, Signal
from PySide6.QtGui import (
    QDrag,
    QFocusEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui.widgets.chips_editor import (
    DRAG_SOURCE_OPACITY,
    DROP_INDICATOR_PX,
    INSERT_PLACEHOLDER,
    MIME_SEGMENT,
    SegmentEditorBase,
)

if TYPE_CHECKING:
    from nlapt_gui.controller import AppController

_LOGGER = get_logger(__name__)

# -- exact UI strings (prototype) --------------------------------------------------
ADD_SENT_TEXT = "+ 添加分段"
ROW_EDIT_TOOLTIP = "点击编辑"
ROW_DELETE_TOOLTIP = "删除该段"
HANDLE_TOOLTIP = "拖拽排序"
INSERT_BADGE_TEXT = "+"

# Geometry from the design.
_HANDLE_W = 18
_HANDLE_H = 24
_BADGE_SIZE = 22
_ROW_GAP = 8
_ADD_BTN_INDENT = 48
_AREA_MIN_ROWS = 2
_AREA_PAD_V = 18

# Palette-driven styling (no hardcoded colors; roles come from ThemeTokens
# through the ThemeManager palette).
_ROW_QSS = """
QLabel#sentText {
    background: palette(window);
    border: 1px solid palette(mid);
    border-radius: 9px;
    padding: 8px 12px;
    font-size: 13px;
}
QLabel#sentText:hover { border-color: palette(highlight); }
QLabel#sentText[selected="true"] {
    border: 1.5px solid palette(highlight);
    background: palette(alternate-base);
}
QLabel#segBadge {
    background: palette(alternate-base);
    color: palette(text);
    border-radius: 6px;
    font-size: 10.5px;
}
QLabel#segBadgeActive {
    background: palette(alternate-base);
    color: palette(highlight);
    border-radius: 6px;
    font-size: 10.5px;
    font-weight: 600;
}
"""


def pad2(number: int) -> str:
    """Prototype ``pad2``: zero-padded row number ('01')."""
    return f"{number:02d}"


class InlineSentArea(QPlainTextEdit):
    """Multi-line inline editor: Enter commits, Shift+Enter newline, Esc cancels."""

    submitted = Signal()
    cancelled = Signal()
    focus_lost = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("editorField", "true")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.textChanged.connect(self._autosize)
        self._autosize()

    # duck-typed field API used by SegmentEditorBase.
    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str) -> None:  # noqa: N802 - mirrors QLineEdit API
        self.setPlainText(text)

    def cursor_index(self) -> int:
        return self.textCursor().position()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
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

    def _autosize(self) -> None:
        """Grow with content (prototype: style.height = scrollHeight)."""
        lines = max(float(_AREA_MIN_ROWS), self.document().size().height())
        height = int(lines * self.fontMetrics().lineSpacing()) + _AREA_PAD_V
        self.setFixedHeight(height)


class ActiveSentRow(QWidget):
    """Editing/inserting row: [spacer][accent badge][autosizing text area].

    Duck-types the inline-field API of ``SegmentEditorBase`` and proxies it
    to the inner :class:`InlineSentArea` so the base class never needs to
    know the row structure.
    """

    submitted = Signal()
    cancelled = Signal()
    focus_lost = Signal()

    def __init__(self, badge_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_ROW_QSS)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_ROW_GAP)
        spacer = QWidget(self)
        spacer.setFixedWidth(_HANDLE_W)
        row.addWidget(spacer, 0, Qt.AlignmentFlag.AlignTop)
        badge = QLabel(badge_text, self)
        badge.setObjectName("segBadgeActive")
        badge.setProperty("mono", "true")
        badge.setFixedSize(_BADGE_SIZE, _BADGE_SIZE)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        self.area = InlineSentArea(self)
        row.addWidget(self.area, 1)
        self.setFocusProxy(self.area)
        self.area.submitted.connect(self.submitted)
        self.area.cancelled.connect(self.cancelled)
        self.area.focus_lost.connect(self.focus_lost)

    def text(self) -> str:
        return self.area.text()

    def setText(self, text: str) -> None:  # noqa: N802 - mirrors QLineEdit API
        self.area.setText(text)

    def cursor_index(self) -> int:
        return self.area.cursor_index()

    def selectAll(self) -> None:  # noqa: N802 - mirrors QLineEdit API
        self.area.selectAll()


class DragHandle(QWidget):
    """6-dot drag handle; starts the row drag past the drag threshold."""

    def __init__(self, row: "SentRow") -> None:
        super().__init__(row)
        self._row = row
        self._press_pos: QPoint | None = None
        self.setFixedSize(_HANDLE_W, _HANDLE_H)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(HANDLE_TOOLTIP)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().color(QPalette.ColorRole.PlaceholderText)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        cx = self.width() // 2
        cy = self.height() // 2
        for dx in (-3, 3):
            for dy in (-5, 0, 5):
                painter.drawEllipse(QPoint(cx + dx, cy + dy), 1, 1)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._press_pos is None:
            return
        distance = (event.position().toPoint() - self._press_pos).manhattanLength()
        if distance >= QApplication.startDragDistance():
            self._press_pos = None
            self._row.request_drag()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self._press_pos = None


class SentRow(QWidget):
    """One display row: drag handle, number badge, text box, delete button."""

    def __init__(
        self, editor: "SentsEditor", index: int, text: str, *, selected: bool = False
    ) -> None:
        super().__init__(editor)
        self._editor = editor
        self._index = index
        self._drop_indicator = False
        self.setStyleSheet(_ROW_QSS)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_ROW_GAP)
        handle = DragHandle(self)
        row.addWidget(handle, 0, Qt.AlignmentFlag.AlignTop)
        badge = QLabel(pad2(index + 1), self)
        badge.setObjectName("segBadgeActive" if selected else "segBadge")
        badge.setProperty("mono", "true")
        badge.setFixedSize(_BADGE_SIZE, _BADGE_SIZE)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        self._label = QLabel(text, self)
        self._label.setObjectName("sentText")
        self._label.setProperty("selected", "true" if selected else "false")
        self._label.setWordWrap(True)
        self._label.setToolTip(ROW_EDIT_TOOLTIP)
        self._label.setCursor(Qt.CursorShape.IBeamCursor)
        self._label.installEventFilter(self)
        row.addWidget(self._label, 1)
        remove = QPushButton(self)
        remove.setProperty("variant", "danger-ghost")
        remove.setText("×")
        remove.setFixedSize(24, 24)
        remove.setToolTip(ROW_DELETE_TOOLTIP)
        remove.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove.clicked.connect(lambda: self._editor.delete_segment(self._index))
        row.addWidget(remove, 0, Qt.AlignmentFlag.AlignTop)

    @property
    def index(self) -> int:
        return self._index

    def segment_text(self) -> str:
        return self._label.text()

    def request_drag(self) -> None:
        self._editor.begin_drag(self._index, self)

    def click_text(self) -> None:
        """Select this row; a second click on it opens the inline editor."""
        self._editor.on_segment_clicked(self._index)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self._label and event.type() == QEvent.Type.MouseButtonPress:
            # Consume the press so it cannot bubble to the editor background
            # handler (which clears the selection on empty-area clicks).
            return True
        if obj is self._label and event.type() == QEvent.Type.MouseButtonRelease:
            self.click_text()
            return True
        return super().eventFilter(obj, event)

    def set_drop_indicator(self, on: bool) -> None:
        if on != self._drop_indicator:
            self._drop_indicator = on
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if self._drop_indicator:
            painter = QPainter(self)
            accent = self.palette().color(QPalette.ColorRole.Highlight)
            painter.fillRect(0, 0, self.width(), DROP_INDICATOR_PX, accent)
            painter.end()


class SentsEditor(SegmentEditorBase):
    """分句 mode: numbered autosizing rows over the same comma segments."""

    def __init__(
        self, controller: "AppController", key: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(controller, key, parent)
        self._vbox = QVBoxLayout(self)
        self._vbox.setContentsMargins(0, 0, 0, 0)
        self._vbox.setSpacing(_ROW_GAP)
        self._rows: list[SentRow] = []
        self.setAcceptDrops(True)
        self._rebuild()

    # -- view ---------------------------------------------------------------------
    def rows(self) -> tuple[SentRow, ...]:
        return tuple(self._rows)

    def _rebuild(self) -> None:
        while self._vbox.count():
            item = self._vbox.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is None or widget is self._field:
                continue
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        self._rows = []
        segs = self.segments()
        entries: list[QWidget] = []
        for i, seg in enumerate(segs):
            if self._edit_index == i and self._field is not None:
                entries.append(self._field)
                continue
            row = SentRow(self, i, seg, selected=self._selected_index == i)
            self._rows.append(row)
            entries.append(row)
        if self._insert_pos is not None and self._field is not None:
            entries.insert(min(self._insert_pos, len(entries)), self._field)
        for widget in entries:
            self._vbox.addWidget(widget)
            widget.show()
        if self._insert_pos is None:
            add = QPushButton(ADD_SENT_TEXT, self)
            add.setProperty("chipAdd", "true")
            add.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            add.clicked.connect(self.start_insert_end)
            wrap = QWidget(self)
            wrap_row = QHBoxLayout(wrap)
            wrap_row.setContentsMargins(_ADD_BTN_INDENT, 0, 0, 0)
            wrap_row.setSpacing(0)
            wrap_row.addWidget(add, 0, Qt.AlignmentFlag.AlignLeft)
            wrap_row.addStretch(1)
            self._vbox.addWidget(wrap)

    def _create_field(self, initial: str, placeholder: str) -> QWidget:
        if self._insert_pos is not None:
            badge_text = INSERT_BADGE_TEXT
        else:
            badge_text = pad2((self._edit_index or 0) + 1)
        field = ActiveSentRow(badge_text, self)
        field.setText(initial)
        if placeholder:
            field.area.setPlaceholderText(placeholder)
        return field

    # -- drag & drop -----------------------------------------------------------------
    def begin_drag(self, index: int, row: SentRow) -> None:
        self._drag_index = index
        effect = QGraphicsOpacityEffect(row)
        effect.setOpacity(DRAG_SOURCE_OPACITY)
        row.setGraphicsEffect(effect)
        drag = QDrag(row)
        mime = QMimeData()
        mime.setText(row.segment_text())
        mime.setData(MIME_SEGMENT, str(index).encode("ascii"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)
        self._drag_index = None
        self._rebuild()

    def _row_index_at(self, pos: QPoint) -> int | None:
        for row in self._rows:
            if row.geometry().contains(pos):
                return row.index
        return None

    def _segment_widget(self, index: int) -> QWidget | None:
        for row in self._rows:
            if row.index == index:
                return row
        return None

    def _set_indicator(self, index: int | None) -> None:
        for row in self._rows:
            row.set_drop_indicator(row.index == index)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag_index is not None and event.mimeData().hasFormat(MIME_SEGMENT):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag_index is None:
            return
        self._set_indicator(self._row_index_at(event.position().toPoint()))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._set_indicator(None)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag_index is None:
            return
        target = self._row_index_at(event.position().toPoint())
        to_index = target if target is not None else len(self.segments())
        event.acceptProposedAction()
        self._set_indicator(None)
        self.reorder(self._drag_index, to_index)


# INSERT_PLACEHOLDER is applied through _create_field; re-exported for tests.
__all__ = [
    "ActiveSentRow",
    "InlineSentArea",
    "SentRow",
    "SentsEditor",
    "pad2",
    "INSERT_PLACEHOLDER",
]
