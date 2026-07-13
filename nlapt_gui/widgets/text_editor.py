"""Free-text (文本) editing mode + the selection action bar.

Live edits update the core caption without a history entry
(``set_caption_no_history``); a labeled ``自由编辑`` history entry is pushed
on focus-out only (prototype behavior). Selecting one or more characters
shows an accent-soft action bar with 替换 / 删除 / 润色, all operating on the
selection range with the prototype's exact comma-tidy rules.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

if TYPE_CHECKING:
    from nlapt_gui.controller import AppController

_LOGGER = get_logger(__name__)

# -- exact UI strings (prototype) --------------------------------------------------
TEXT_PLACEHOLDER = "在此输入标注文本…"
LABEL_FREE_EDIT = "自由编辑"
LABEL_REPLACE_SEL = "替换选中片段"
LABEL_DELETE_SEL = "删除选中片段"
LABEL_TIDY_SEL = "润色选中片段"
SEL_INFO_TEMPLATE = "已选中 {n} 字符"
SEL_REPLACE_PLACEHOLDER = "替换为…"
BTN_REPLACE = "替换"
BTN_DELETE = "删除"
BTN_TIDY = "润色"
TIDY_TOOLTIP = "整理选中片段的空格与逗号"
TOAST_REPLACED_SEL = "已替换选中片段"
TOAST_DELETED_SEL = "已删除选中片段"
TOAST_TIDIED_SEL = "已整理选中片段"

_KIND_OK = "ok"
_KIND_INFO = "info"

# Editor heights from the design (taH: 86px in multi mode, 182px single).
AREA_HEIGHT_SINGLE = 182
AREA_HEIGHT_MULTI = 86

# Comma tidy rules from the prototype's doDeleteSel.
_DOUBLE_COMMA = re.compile(r",\s*,")
_LEADING_COMMA = re.compile(r"^\s*,\s*")
# Tidy joiner from doTidySel.
_TIDY_JOINER = ", "


def delete_selection_text(caption: str, start: int, end: int) -> str:
    """Prototype ``doDeleteSel``: cut range, collapse ',  ,' pairs, strip lead comma."""
    joined = caption[:start] + caption[end:]
    joined = _DOUBLE_COMMA.sub(",", joined)
    return _LEADING_COMMA.sub("", joined)


def tidy_selection_text(caption: str, start: int, end: int) -> str:
    """Prototype ``doTidySel``: split range on ',', trim, drop empties, join ', '."""
    sub = caption[start:end]
    tidied = _TIDY_JOINER.join(
        part.strip() for part in sub.split(",") if part.strip()
    )
    return caption[:start] + tidied + caption[end:]


class CaptionArea(QPlainTextEdit):
    """Plain caption editor that reports focus loss (history push trigger)."""

    focus_lost = Signal()

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802 - Qt override
        super().focusOutEvent(event)
        self.focus_lost.emit()


class TextEditor(QWidget):
    """文本 mode block content: selection action bar + plain text area."""

    def __init__(
        self,
        controller: "AppController",
        key: str,
        *,
        multi: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._key = key
        self._sel: tuple[int, int] | None = None
        self._syncing = False
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        # -- selection action bar (visible only while text is selected) ----------
        self._bar = QFrame(self)
        self._bar.setProperty("accentSoftBox", "true")
        bar_row = QHBoxLayout(self._bar)
        bar_row.setContentsMargins(10, 6, 10, 6)
        bar_row.setSpacing(8)
        self._sel_info = QLabel("", self._bar)
        self._sel_info.setStyleSheet(
            "color: palette(highlight); font-size: 11.5px; font-weight: 600;"
        )
        bar_row.addWidget(self._sel_info)
        self._replace_input = QLineEdit(self._bar)
        self._replace_input.setPlaceholderText(SEL_REPLACE_PLACEHOLDER)
        self._replace_input.setFixedHeight(26)
        bar_row.addWidget(self._replace_input, 1)
        self._replace_btn = QPushButton(BTN_REPLACE, self._bar)
        self._replace_btn.setProperty("variant", "accent")
        self._replace_btn.setFixedHeight(26)
        self._replace_btn.clicked.connect(self.replace_selection)
        bar_row.addWidget(self._replace_btn)
        self._delete_btn = QPushButton(BTN_DELETE, self._bar)
        self._delete_btn.setProperty("variant", "danger-ghost")
        self._delete_btn.setFixedHeight(26)
        self._delete_btn.clicked.connect(self.delete_selection)
        bar_row.addWidget(self._delete_btn)
        self._tidy_btn = QPushButton(BTN_TIDY, self._bar)
        self._tidy_btn.setProperty("variant", "outline")
        self._tidy_btn.setFixedHeight(26)
        self._tidy_btn.setToolTip(TIDY_TOOLTIP)
        self._tidy_btn.clicked.connect(self.tidy_selection)
        bar_row.addWidget(self._tidy_btn)
        self._bar.hide()
        column.addWidget(self._bar)

        # -- caption text area ----------------------------------------------------
        self._area = CaptionArea(self)
        self._area.setProperty("editorField", "true")
        self._area.setPlaceholderText(TEXT_PLACEHOLDER)
        self._area.setFixedHeight(AREA_HEIGHT_MULTI if multi else AREA_HEIGHT_SINGLE)
        self._area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._syncing = True
        self._area.setPlainText(controller.record(key).text)
        self._syncing = False
        self._area.textChanged.connect(self._on_text_changed)
        self._area.selectionChanged.connect(self._on_selection_changed)
        self._area.focus_lost.connect(self._on_focus_lost)
        column.addWidget(self._area)

    # -- public API used by the editor panel / tests -----------------------------------
    @property
    def key(self) -> str:
        return self._key

    @property
    def area(self) -> CaptionArea:
        return self._area

    def selection(self) -> tuple[int, int] | None:
        return self._sel

    def selection_bar_visible(self) -> bool:
        return self._bar.isVisible()

    def set_replacement_text(self, text: str) -> None:
        self._replace_input.setText(text)

    def refresh(self) -> None:
        """Sync the text area after an external caption change."""
        text = self._controller.record(self._key).text
        if self._area.toPlainText() == text:
            return
        self._syncing = True
        self._area.setPlainText(text)
        self._syncing = False
        self._clear_selection_state()

    # -- selection actions (prototype doReplaceSel / doDeleteSel / doTidySel) ------------
    def replace_selection(self) -> None:
        if self._sel is None:
            return
        start, end = self._sel
        caption = self._controller.record(self._key).text
        new_text = caption[:start] + self._replace_input.text() + caption[end:]
        self._controller.set_caption(self._key, new_text, LABEL_REPLACE_SEL)
        self._replace_input.clear()
        self._clear_selection_state()
        self.refresh()
        self._controller.toast_requested.emit(TOAST_REPLACED_SEL, _KIND_OK)

    def delete_selection(self) -> None:
        if self._sel is None:
            return
        start, end = self._sel
        caption = self._controller.record(self._key).text
        new_text = delete_selection_text(caption, start, end)
        self._controller.set_caption(self._key, new_text, LABEL_DELETE_SEL)
        self._clear_selection_state()
        self.refresh()
        self._controller.toast_requested.emit(TOAST_DELETED_SEL, _KIND_OK)

    def tidy_selection(self) -> None:
        if self._sel is None:
            return
        start, end = self._sel
        caption = self._controller.record(self._key).text
        new_text = tidy_selection_text(caption, start, end)
        self._controller.set_caption(self._key, new_text, LABEL_TIDY_SEL)
        self._clear_selection_state()
        self.refresh()
        self._controller.toast_requested.emit(TOAST_TIDIED_SEL, _KIND_INFO)

    # -- internals -------------------------------------------------------------------------
    def _on_text_changed(self) -> None:
        if self._syncing:
            return
        self._controller.set_caption_no_history(self._key, self._area.toPlainText())

    def _on_selection_changed(self) -> None:
        cursor = self._area.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        if start == end:
            self._clear_selection_state()
            return
        self._sel = (start, end)
        self._sel_info.setText(SEL_INFO_TEMPLATE.format(n=end - start))
        self._bar.show()

    def _on_focus_lost(self) -> None:
        """Prototype onTextBlur: push a labeled history entry on blur."""
        self._controller.history.push(
            self._key, LABEL_FREE_EDIT, self._controller.record(self._key).text
        )

    def _clear_selection_state(self) -> None:
        self._sel = None
        self._sel_info.setText("")
        self._bar.hide()
