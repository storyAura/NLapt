"""Shared metrics and builders for 修改工具 section cards."""

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

from nlapt_gui.controller import AppController
from nlapt_gui.widgets.tools_panel import ScopeSelector, repolish

CONTENT_MARGINS = (13, 4, 13, 12)
ROW_GAP = 8
INPUT_H = 29
BUTTON_H = 30
LABEL_SCOPE = "应用范围"


def make_input(parent: QWidget, placeholder: str) -> QLineEdit:
    """Mono line edit used by find-replace / prefix-suffix."""
    field = QLineEdit(parent)
    field.setPlaceholderText(placeholder)
    field.setProperty("mono", True)
    field.setFixedHeight(INPUT_H)
    field.setStyleSheet("background: palette(window); font-size: 12px;")
    return field


def make_accent_button(parent: QWidget, text: str) -> QPushButton:
    """Full-width accent action button."""
    button = QPushButton(text, parent)
    button.setProperty("variant", "accent")
    button.setFixedHeight(BUTTON_H)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def make_chip(parent: QWidget, text: str, *, checked: bool = False) -> QPushButton:
    """Checkable toggle chip (Aa / 整词 / 独立标签)."""
    chip = QPushButton(text, parent)
    chip.setProperty("toggleChip", True)
    chip.setCheckable(True)
    chip.setChecked(checked)
    chip.setProperty("chipOn", checked)
    chip.setCursor(Qt.CursorShape.PointingHandCursor)
    return chip


def sync_chip(chip: QPushButton, checked: bool) -> None:
    """Keep the chipOn QSS property in sync after a toggle."""
    chip.setProperty("chipOn", checked)
    repolish(chip)


class ScopeRow(QWidget):
    """应用范围 label + ScopeSelector on one row."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        label = QLabel(LABEL_SCOPE, self)
        label.setProperty("muted", True)
        label.setStyleSheet("font-size: 10.5px;")
        self.selector = ScopeSelector(controller, self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(label)
        layout.addWidget(self.selector, 1)

    @property
    def scope(self) -> str:
        return self.selector.scope

    @property
    def changed(self):  # noqa: ANN201 - Qt signal passthrough
        return self.selector.changed


def apply_section_layout(widget: QWidget) -> QVBoxLayout:
    """Standard vertical layout for a section body."""
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(*CONTENT_MARGINS)
    layout.setSpacing(ROW_GAP)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    return layout
