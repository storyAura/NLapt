"""前缀 / 后缀 section - batch prefix/suffix with 独立标签 toggle.

Maps straight onto ``controller.apply_prefix_suffix`` which builds the core
``PrefixSuffixSpec`` (joiner ', ' + skip-if-present when 作为独立标签 is on).
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget

from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController, POSITION_PREFIX, POSITION_SUFFIX
from nlapt_gui.widgets.sections.common import (
    ScopeRow,
    apply_section_layout,
    make_accent_button,
    make_chip,
    make_input,
    sync_chip,
)
from nlapt_gui.widgets.tools_panel import SegmentedBar

_LOGGER = get_logger(__name__)

PLACEHOLDER_TEXT = "如: aoba, masterpiece"
POSITION_PREFIX_LABEL = "加到开头"
POSITION_SUFFIX_LABEL = "加到结尾"
TOGGLE_AS_TAG = "作为独立标签 (自动加逗号)"
BUTTON_APPLY = "应用"


class PrefixSuffixSection(QWidget):
    """Content widget of the 前缀 / 后缀 collapsible card."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller

        self.text_input = make_input(self, PLACEHOLDER_TEXT)
        self.position = SegmentedBar(
            (
                (POSITION_PREFIX, POSITION_PREFIX_LABEL),
                (POSITION_SUFFIX, POSITION_SUFFIX_LABEL),
            ),
            current=POSITION_PREFIX,
            parent=self,
        )
        self.as_tag_toggle = make_chip(self, TOGGLE_AS_TAG, checked=True)
        self.as_tag_toggle.toggled.connect(self._on_as_tag_toggled)
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.addWidget(self.as_tag_toggle)
        toggle_row.addStretch(1)

        self.scope_row = ScopeRow(controller, self)
        self.scope = self.scope_row.selector

        self.apply_button = make_accent_button(self, BUTTON_APPLY)
        self.apply_button.clicked.connect(self.apply)

        layout = apply_section_layout(self)
        layout.addWidget(self.text_input)
        layout.addWidget(self.position)
        layout.addLayout(toggle_row)
        layout.addWidget(self.scope_row)
        layout.addWidget(self.apply_button)

    @property
    def as_tag(self) -> bool:
        return self.as_tag_toggle.isChecked()

    def apply(self) -> None:
        """Run the batch prefix/suffix (controller validates + toasts)."""
        self._controller.apply_prefix_suffix(
            self.text_input.text(),
            self.position.current,
            self.as_tag,
            self.scope.scope,
        )

    def _on_as_tag_toggled(self, checked: bool) -> None:
        sync_chip(self.as_tag_toggle, checked)
