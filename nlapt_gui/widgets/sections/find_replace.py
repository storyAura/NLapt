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
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController
from nlapt_gui.widgets.sections.common import (
    ScopeRow,
    apply_section_layout,
    make_accent_button,
    make_chip,
    make_input,
    sync_chip,
)
from nlapt_gui.widgets.tools_panel import repolish

_LOGGER = get_logger(__name__)

PLACEHOLDER_FIND = "查找内容,如: long hair"
PLACEHOLDER_REPLACE = "替换为 (留空则删除)"
TOGGLE_CASE = "Aa 区分大小写"
TOGGLE_WORD = "W 整词"
TOOLTIP_WORD = "整词匹配:只匹配完整单词,如 hair 不会匹配 hairband"
BUTTON_REPLACE_ALL = "全部替换"
INFO_PROMPT = "输入查找内容以预览匹配"
INFO_MATCHES = "匹配 {total} 处 · {files} 个文件"
INFO_NO_MATCH = "无匹配"


class FindReplaceSection(QWidget):
    """Content widget of the 查找替换 collapsible card."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller

        self.find_input = make_input(self, PLACEHOLDER_FIND)
        self.replace_input = make_input(self, PLACEHOLDER_REPLACE)

        self.case_toggle = make_chip(self, TOGGLE_CASE)
        self.case_toggle.toggled.connect(self._on_case_toggled)
        self.word_toggle = make_chip(self, TOGGLE_WORD)
        self.word_toggle.setToolTip(TOOLTIP_WORD)
        self.word_toggle.toggled.connect(self._on_word_toggled)
        self.info_label = QLabel(INFO_PROMPT, self)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.setSpacing(8)
        toggle_row.addWidget(self.case_toggle)
        toggle_row.addWidget(self.word_toggle)
        toggle_row.addStretch(1)
        toggle_row.addWidget(self.info_label)

        self.scope_row = ScopeRow(controller, self)
        self.scope = self.scope_row.selector

        self.apply_button = make_accent_button(self, BUTTON_REPLACE_ALL)
        self.apply_button.clicked.connect(self.apply_replace)

        layout = apply_section_layout(self)
        layout.addWidget(self.find_input)
        layout.addWidget(self.replace_input)
        layout.addLayout(toggle_row)
        layout.addWidget(self.scope_row)
        layout.addWidget(self.apply_button)

        self.find_input.textChanged.connect(lambda _text: self.refresh_info())
        self.scope.changed.connect(lambda _scope: self.refresh_info())
        controller.selection_changed.connect(self.refresh_info)
        controller.caption_changed.connect(lambda _key: self.refresh_info())
        controller.current_changed.connect(lambda _key: self.refresh_info())
        controller.dataset_opened.connect(lambda _result: self.refresh_info())
        self.refresh_info()

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

    def _on_case_toggled(self, checked: bool) -> None:
        sync_chip(self.case_toggle, checked)
        self.refresh_info()

    def _on_word_toggled(self, checked: bool) -> None:
        sync_chip(self.word_toggle, checked)
        self.refresh_info()

    def _set_info(self, text: str, *, accent: bool) -> None:
        self.info_label.setText(text)
        if accent:
            color = self.palette().color(QPalette.ColorRole.Highlight).name()
            self.info_label.setProperty("muted", False)
            self.info_label.setStyleSheet(f"font-size: 11px; color: {color};")
        else:
            self.info_label.setProperty("muted", True)
            self.info_label.setStyleSheet("font-size: 11px;")
        repolish(self.info_label)
