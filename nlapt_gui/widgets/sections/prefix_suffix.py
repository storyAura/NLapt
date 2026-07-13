"""前缀 / 后缀 section - batch prefix/suffix with 独立标签 toggle.

Maps straight onto ``controller.apply_prefix_suffix`` which builds the core
``PrefixSuffixSpec`` (joiner ', ' + skip-if-present when 作为独立标签 is on).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController, POSITION_PREFIX, POSITION_SUFFIX
from nlapt_gui.widgets.tools_panel import ScopeSelector, SegmentedBar, repolish

_LOGGER = get_logger(__name__)

# Exact UI strings from the design.
PLACEHOLDER_TEXT = "如: aoba, masterpiece"
POSITION_PREFIX_LABEL = "加到开头"
POSITION_SUFFIX_LABEL = "加到结尾"
TOGGLE_AS_TAG = "作为独立标签 (自动加逗号)"
LABEL_SCOPE = "应用范围"
BUTTON_APPLY = "应用"

_INPUT_HEIGHT = 29
_BUTTON_HEIGHT = 30
_CONTENT_MARGINS = (13, 2, 13, 13)
_CONTENT_GAP = 8


class PrefixSuffixSection(QWidget):
    """Content widget of the 前缀 / 后缀 collapsible card."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller

        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText(PLACEHOLDER_TEXT)
        self.text_input.setProperty("mono", True)
        self.text_input.setFixedHeight(_INPUT_HEIGHT)
        self.text_input.setStyleSheet("background: palette(window); font-size: 12px;")

        self.position = SegmentedBar(
            (
                (POSITION_PREFIX, POSITION_PREFIX_LABEL),
                (POSITION_SUFFIX, POSITION_SUFFIX_LABEL),
            ),
            current=POSITION_PREFIX,
            parent=self,
        )

        # 作为独立标签 defaults ON per the design (psAsTag: true).
        self.as_tag_toggle = QPushButton(TOGGLE_AS_TAG, self)
        self.as_tag_toggle.setProperty("toggleChip", True)
        self.as_tag_toggle.setCheckable(True)
        self.as_tag_toggle.setChecked(True)
        self.as_tag_toggle.setProperty("chipOn", True)
        self.as_tag_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.as_tag_toggle.toggled.connect(self._on_as_tag_toggled)
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.addWidget(self.as_tag_toggle)
        toggle_row.addStretch(1)

        scope_label = QLabel(LABEL_SCOPE, self)
        scope_label.setProperty("muted", True)
        scope_label.setStyleSheet("font-size: 10.5px;")
        self.scope = ScopeSelector(controller, self)

        self.apply_button = QPushButton(BUTTON_APPLY, self)
        self.apply_button.setProperty("variant", "accent")
        self.apply_button.setFixedHeight(_BUTTON_HEIGHT)
        self.apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_button.clicked.connect(self.apply)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*_CONTENT_MARGINS)
        layout.setSpacing(_CONTENT_GAP)
        layout.addWidget(self.text_input)
        layout.addWidget(self.position)
        layout.addLayout(toggle_row)
        layout.addWidget(scope_label)
        layout.addWidget(self.scope)
        layout.addWidget(self.apply_button)

    # -- behavior ---------------------------------------------------------------------
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

    # -- internals ----------------------------------------------------------------------
    def _on_as_tag_toggled(self, checked: bool) -> None:
        self.as_tag_toggle.setProperty("chipOn", checked)
        repolish(self.as_tag_toggle)
