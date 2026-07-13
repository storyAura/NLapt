"""查找替换 section - live two-level match preview + batch replace.

Match info updates on every input/scope/case/whole-word change and on
controller selection/caption/current changes; the actual replace runs
through ``controller.replace_all`` (core batch engine: snapshot + oplog +
toasts). Matching is substring-level across the whole caption (words
inside sentences included, not limited to comma segments); the 整词 chip
narrows it to word boundaries.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController
from nlapt_gui.widgets.tools_panel import ScopeSelector, repolish

_LOGGER = get_logger(__name__)

# Exact UI strings from the design.
PLACEHOLDER_FIND = "查找内容,如: long hair"
PLACEHOLDER_REPLACE = "替换为 (留空则删除)"
TOGGLE_CASE = "Aa 区分大小写"
TOGGLE_WORD = "W 整词"
TOOLTIP_WORD = "整词匹配:只匹配完整单词,如 hair 不会匹配 hairband"
LABEL_SCOPE = "应用范围"
BUTTON_REPLACE_ALL = "全部替换"
INFO_PROMPT = "输入查找内容以预览匹配"
INFO_MATCHES = "匹配 {total} 处 · {files} 个文件"
INFO_NO_MATCH = "无匹配"

_INPUT_HEIGHT = 29
_BUTTON_HEIGHT = 30
_CONTENT_MARGINS = (13, 2, 13, 13)
_CONTENT_GAP = 8


class FindReplaceSection(QWidget):
    """Content widget of the 查找替换 collapsible card."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller

        self.find_input = QLineEdit(self)
        self.find_input.setPlaceholderText(PLACEHOLDER_FIND)
        self.find_input.setProperty("mono", True)
        self.find_input.setFixedHeight(_INPUT_HEIGHT)
        self.find_input.setStyleSheet("background: palette(window); font-size: 12px;")
        self.replace_input = QLineEdit(self)
        self.replace_input.setPlaceholderText(PLACEHOLDER_REPLACE)
        self.replace_input.setProperty("mono", True)
        self.replace_input.setFixedHeight(_INPUT_HEIGHT)
        self.replace_input.setStyleSheet("background: palette(window); font-size: 12px;")

        self.case_toggle = self._make_chip(TOGGLE_CASE)
        self.case_toggle.toggled.connect(self._on_case_toggled)
        self.word_toggle = self._make_chip(TOGGLE_WORD)
        self.word_toggle.setToolTip(TOOLTIP_WORD)
        self.word_toggle.toggled.connect(self._on_word_toggled)
        self.info_label = QLabel(INFO_PROMPT, self)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Two chips + long match info don't fit one 340px row: chips row,
        # then the info line right-aligned underneath.
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.setSpacing(8)
        toggle_row.addWidget(self.case_toggle)
        toggle_row.addWidget(self.word_toggle)
        toggle_row.addStretch(1)

        scope_label = QLabel(LABEL_SCOPE, self)
        scope_label.setProperty("muted", True)
        scope_label.setStyleSheet("font-size: 10.5px;")
        self.scope = ScopeSelector(controller, self)

        self.apply_button = QPushButton(BUTTON_REPLACE_ALL, self)
        self.apply_button.setProperty("variant", "accent")
        self.apply_button.setFixedHeight(_BUTTON_HEIGHT)
        self.apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_button.clicked.connect(self.apply_replace)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*_CONTENT_MARGINS)
        layout.setSpacing(_CONTENT_GAP)
        layout.addWidget(self.find_input)
        layout.addWidget(self.replace_input)
        layout.addLayout(toggle_row)
        layout.addWidget(self.info_label)
        layout.addWidget(scope_label)
        layout.addWidget(self.scope)
        layout.addWidget(self.apply_button)

        self.find_input.textChanged.connect(lambda _text: self.refresh_info())
        self.scope.changed.connect(lambda _scope: self.refresh_info())
        controller.selection_changed.connect(self.refresh_info)
        controller.caption_changed.connect(lambda _key: self.refresh_info())
        controller.current_changed.connect(lambda _key: self.refresh_info())
        controller.dataset_opened.connect(lambda _result: self.refresh_info())
        self.refresh_info()

    # -- behavior ---------------------------------------------------------------------
    def refresh_info(self) -> None:
        """Recompute the live '匹配 m 处 · k 个文件' preview line."""
        find = self.find_input.text()
        if not find:
            self._set_info(INFO_PROMPT, accent=False)
            return
        total, files = self._controller.count_matches(
            find,
            self.case_toggle.isChecked(),
            self.scope.scope,
            whole_word=self.word_toggle.isChecked(),
        )
        if total > 0:
            self._set_info(INFO_MATCHES.format(total=total, files=files), accent=True)
        else:
            self._set_info(INFO_NO_MATCH, accent=False)

    def apply_replace(self) -> None:
        """Run the batch replace (controller handles validation + toasts)."""
        self._controller.replace_all(
            self.find_input.text(),
            self.replace_input.text(),
            self.case_toggle.isChecked(),
            self.scope.scope,
            whole_word=self.word_toggle.isChecked(),
        )

    # -- internals ----------------------------------------------------------------------
    def _make_chip(self, text: str) -> QPushButton:
        chip = QPushButton(text, self)
        chip.setProperty("toggleChip", True)
        chip.setProperty("chipOn", False)
        chip.setCheckable(True)
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        return chip

    def _on_case_toggled(self, checked: bool) -> None:
        self._sync_chip(self.case_toggle, checked)

    def _on_word_toggled(self, checked: bool) -> None:
        self._sync_chip(self.word_toggle, checked)

    def _sync_chip(self, chip: QPushButton, checked: bool) -> None:
        chip.setProperty("chipOn", checked)
        repolish(chip)
        self.refresh_info()

    def _set_info(self, text: str, *, accent: bool) -> None:
        self.info_label.setText(text)
        if accent:
            # Accent from the live palette (Highlight role = theme accent).
            color = self.palette().color(QPalette.ColorRole.Highlight).name()
            self.info_label.setProperty("muted", False)
            self.info_label.setStyleSheet(f"font-size: 11px; color: {color};")
        else:
            self.info_label.setProperty("muted", True)
            self.info_label.setStyleSheet("font-size: 11px;")
        repolish(self.info_label)
