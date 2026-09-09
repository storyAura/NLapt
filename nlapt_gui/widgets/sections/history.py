"""历史记录 section - labeled caption snapshots with 回退 / 清空历史.

Bound to ``controller.history`` (:class:`FileHistory`): the CURSOR row is the
applied caption (当前 pill, accent-tinted); every other row - newer or older,
kept within ±5 of the cursor - exposes a 回退 button that jumps the cursor
there and re-applies that caption without deleting anything.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController, TOAST_INFO
from nlapt_gui.widgets.sections.common import CONTENT_MARGINS, ROW_GAP
from nlapt_gui.widgets.tools_panel import resolve_tokens

_LOGGER = get_logger(__name__)

# Exact UI strings.
PILL_CURRENT = "当前"
BUTTON_REVERT = "回退"
BUTTON_CLEAR = "清空历史"
CHARS_LABEL = "{n} 字符"
TOAST_REVERTED = "已回退到 {time}"
TOAST_CLEARED = "已清空历史 (保留当前状态)"

_ROWS_MAX_HEIGHT = 280
_ROW_GAP = 6
_ROW_MARGINS = (9, 6, 9, 6)
_REVERT_HEIGHT = 22
_CLEAR_HEIGHT = 27


class HistorySection(QWidget):
    """Content widget of the 历史记录 collapsible card."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._row_frames: list[QFrame] = []

        self._rows_host = QWidget(self)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(_ROW_GAP)
        self._rows_layout.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_host)
        scroll.setMaximumHeight(_ROWS_MAX_HEIGHT)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tokens = resolve_tokens(controller.settings)
        self.clear_button = QPushButton(BUTTON_CLEAR, self)
        self.clear_button.setObjectName("histClear")
        self.clear_button.setFixedHeight(_CLEAR_HEIGHT)
        self.clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_button.setStyleSheet(
            f"#histClear {{ background: palette(base); border: 1px solid {tokens.bd};"
            f" border-radius: 8px; font-size: 11.5px; color: {tokens.text3}; }}"
            f"#histClear:hover {{ color: {tokens.danger};"
            f" border-color: {tokens.danger}; }}"
        )
        self.clear_button.clicked.connect(self.clear_history)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*CONTENT_MARGINS)
        layout.setSpacing(ROW_GAP)
        layout.addWidget(scroll)
        layout.addWidget(self.clear_button)

        controller.history.changed.connect(self._on_history_changed)
        controller.current_changed.connect(lambda _key: self.refresh())
        controller.dataset_opened.connect(lambda _result: self.refresh())
        self.refresh()

    # -- behavior ---------------------------------------------------------------------
    def row_count(self) -> int:
        return len(self._row_frames)

    def refresh(self) -> None:
        """Rebuild the rows from the current file's history entries."""
        for frame in self._row_frames:
            self._rows_layout.removeWidget(frame)
            frame.deleteLater()
        self._row_frames = []
        key = self._controller.current_key
        if key is None:
            return
        cursor = self._controller.history.current_index(key)
        for index, entry in enumerate(self._controller.history.entries(key)):
            frame = self._build_row(
                index,
                entry.time_label,
                entry.label,
                entry.caption,
                is_current=index == cursor,
            )
            self._row_frames.append(frame)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, frame)

    def revert_to(self, index: int) -> None:
        """Jump the history cursor to ``index`` (either direction)."""
        key = self._controller.current_key
        if key is None:
            return
        entries = self._controller.history.entries(key)
        if not 0 <= index < len(entries):
            _LOGGER.warning("revert index %d out of range for %r", index, key)
            return
        if index == self._controller.history.current_index(key):
            return
        time_label = entries[index].time_label
        try:
            new_text = self._controller.history.revert_caption(key, index)
        except ValidationError:
            _LOGGER.exception("history revert failed for %r", key)
            return
        self._controller.set_caption_no_history(key, new_text)
        self._controller.toast_requested.emit(
            TOAST_REVERTED.format(time=time_label), TOAST_INFO
        )

    def clear_history(self) -> None:
        """清空历史 keeps only the current entry (design behavior)."""
        key = self._controller.current_key
        if key is None:
            return
        self._controller.history.clear_keep_current(key)
        self._controller.toast_requested.emit(TOAST_CLEARED, TOAST_INFO)

    # -- internals ----------------------------------------------------------------------
    def _on_history_changed(self, key: str) -> None:
        if key == (self._controller.current_key or ""):
            self.refresh()

    def _build_row(
        self,
        index: int,
        time_label: str,
        label: str,
        caption: str,
        *,
        is_current: bool,
    ) -> QFrame:
        tokens = resolve_tokens(self._controller.settings)
        frame = QFrame(self._rows_host)
        if is_current:
            # QSS accentSoftBox: accent-soft bg + accent-tinted border.
            frame.setProperty("accentSoftBox", True)
        else:
            frame.setObjectName("histRow")
            frame.setStyleSheet(
                f"#histRow {{ background: palette(window); border: 1px solid {tokens.bd};"
                " border-radius: 8px; }"
            )
        row_layout = QHBoxLayout(frame)
        row_layout.setContentsMargins(*_ROW_MARGINS)
        row_layout.setSpacing(8)
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)
        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(7)
        time_widget = QLabel(time_label, frame)
        time_widget.setProperty("mono", True)
        time_widget.setProperty("muted", True)
        time_widget.setStyleSheet("font-size: 10px; border: none; background: transparent;")
        chars_widget = QLabel(CHARS_LABEL.format(n=len(caption)), frame)
        chars_widget.setProperty("mono", True)
        chars_widget.setProperty("muted", True)
        chars_widget.setStyleSheet("font-size: 10px; border: none; background: transparent;")
        meta_row.addWidget(time_widget)
        meta_row.addWidget(chars_widget)
        meta_row.addStretch(1)
        label_widget = QLabel(label, frame)
        label_widget.setStyleSheet("font-size: 12px; border: none; background: transparent;")
        text_col.addLayout(meta_row)
        text_col.addWidget(label_widget)
        row_layout.addLayout(text_col, 1)
        if is_current:
            pill = QLabel(PILL_CURRENT, frame)
            pill.setProperty("pill", "accentSoft")
            row_layout.addWidget(pill)
        else:
            revert = QPushButton(BUTTON_REVERT, frame)
            revert.setObjectName("histRevert")
            revert.setFixedHeight(_REVERT_HEIGHT)
            revert.setCursor(Qt.CursorShape.PointingHandCursor)
            revert.setStyleSheet(
                f"#histRevert {{ background: palette(base); border: 1px solid {tokens.bd};"
                f" border-radius: 6px; font-size: 10.5px; color: {tokens.text2};"
                " padding: 0 9px; }"
                "#histRevert:hover { border-color: palette(highlight);"
                " color: palette(highlight); }"
            )
            revert.clicked.connect(lambda _=False, i=index: self.revert_to(i))
            row_layout.addWidget(revert)
        return frame
