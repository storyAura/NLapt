"""色彩设置 - a dedicated window for theme + accent color customization.

Spec module 4.2: the fixed accent presets were the only choice before; this
dialog adds a full custom color picker. It lists the five built-in themes
(the same swatch rows as the title-bar popup), the accent presets, and a
自定义色彩 button opening a QColorDialog. Every change applies LIVE through
:class:`ThemeManager` (which also persists it), so the dialog itself only
needs a 恢复默认 and a 完成 button.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QColorDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import ACCENT_OPTIONS, THEMES, ThemeTokens
from nlapt_gui.widgets.dialogs import CenteredDialog

_LOGGER = get_logger(__name__)

WINDOW_TITLE = "色彩设置"
SECTION_THEME = "界面主题"
SECTION_ACCENT = "主题色"
BUTTON_CUSTOM = "自定义色彩…"
BUTTON_RESET = "恢复默认"
BUTTON_DONE = "完成"
CUSTOM_PICKER_TITLE = "选择主题色"
NOTE = "改动会立即生效并自动保存。"

DIALOG_WIDTH = 380
SWATCH_PX = 13
ACCENT_SWATCH_PX = 26
THEME_GRID_COLS = 1


class _ThemeChoiceRow(QFrame):
    """One clickable theme row: swatch dots + name (accent border when active)."""

    picked = Signal(str)

    def __init__(
        self,
        theme_name: str,
        active: bool,
        tokens: ThemeTokens,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.theme_name = theme_name
        base = THEMES[theme_name]
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        border = tokens.accent if active else tokens.bd
        width = "1.5px" if active else "1px"
        self.setStyleSheet(
            f"_ThemeChoiceRow {{ border: {width} solid {border};"
            f" border-radius: 9px; background: {tokens.surface}; }}"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 7, 10, 7)
        row.setSpacing(9)
        for color, bordered in ((base.bg, True), (base.surface, True), (base.accent, False)):
            dot = QLabel(self)
            dot.setFixedSize(SWATCH_PX, SWATCH_PX)
            dot_border = f" border: 1px solid {tokens.bd2};" if bordered else ""
            dot.setStyleSheet(
                f"background: {color}; border-radius: {SWATCH_PX // 2}px;{dot_border}"
            )
            row.addWidget(dot)
        name = QLabel(theme_name, self)
        name.setStyleSheet(f"color: {tokens.text}; background: transparent; border: none;")
        row.addWidget(name, 1)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.picked.emit(self.theme_name)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _AccentSwatch(QPushButton):
    """A round accent color button (accent ring when active)."""

    def __init__(
        self,
        color: str,
        active: bool,
        tokens: ThemeTokens,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.color = color
        self.setFixedSize(ACCENT_SWATCH_PX, ACCENT_SWATCH_PX)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        ring = tokens.text if active else tokens.bd
        width = 2 if active else 1
        self.setStyleSheet(
            f"QPushButton {{ background: {color}; border: {width}px solid {ring};"
            f" border-radius: {ACCENT_SWATCH_PX // 2}px; }}"
        )
        self.setToolTip(color)


class ColorSettingsDialog(CenteredDialog):
    """Theme + accent customization; changes apply live via the ThemeManager."""

    def __init__(
        self, theme_manager: ThemeManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._manager = theme_manager
        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(True)
        self.setMinimumWidth(DIALOG_WIDTH)

        column = QVBoxLayout(self)
        column.setSpacing(8)
        self._theme_title = self._section_label(SECTION_THEME)
        column.addWidget(self._theme_title)
        self._theme_grid_host = QWidget(self)
        self._theme_grid = QGridLayout(self._theme_grid_host)
        self._theme_grid.setContentsMargins(0, 0, 0, 0)
        self._theme_grid.setSpacing(6)
        column.addWidget(self._theme_grid_host)

        self._accent_title = self._section_label(SECTION_ACCENT)
        column.addWidget(self._accent_title)
        self._accent_host = QWidget(self)
        self._accent_row = QHBoxLayout(self._accent_host)
        self._accent_row.setContentsMargins(0, 0, 0, 0)
        self._accent_row.setSpacing(8)
        column.addWidget(self._accent_host)

        self.custom_button = QPushButton(BUTTON_CUSTOM, self)
        self.custom_button.setProperty("variant", "outline")
        self.custom_button.clicked.connect(self.pick_custom_accent)
        self.reset_button = QPushButton(BUTTON_RESET, self)
        self.reset_button.setProperty("variant", "outline")
        self.reset_button.clicked.connect(self.reset_accent)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self.custom_button)
        actions.addWidget(self.reset_button)
        actions.addStretch(1)
        column.addLayout(actions)

        note = QLabel(NOTE, self)
        note.setProperty("muted", True)
        column.addWidget(note)
        column.addStretch(1)

        self.done_button = QPushButton(BUTTON_DONE, self)
        self.done_button.setProperty("variant", "accent")
        self.done_button.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(self.done_button)
        column.addLayout(bottom)

        self._rebuild_choices()
        theme_manager.theme_changed.connect(self._on_theme_changed)

    # -- construction helpers -------------------------------------------------------
    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setStyleSheet("font-weight: 600; font-size: 12px;")
        return label

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _rebuild_choices(self) -> None:
        tokens = self._manager.tokens
        self._clear_layout(self._theme_grid)
        self._theme_rows: list[_ThemeChoiceRow] = []
        for i, name in enumerate(THEMES):
            row = _ThemeChoiceRow(name, name == self._manager.theme_name, tokens, self)
            row.picked.connect(self.apply_theme)
            self._theme_grid.addWidget(row, i // THEME_GRID_COLS, i % THEME_GRID_COLS)
            self._theme_rows.append(row)
        self._clear_layout(self._accent_row)
        self._accent_swatches: list[_AccentSwatch] = []
        current = self._manager.accent.upper()
        options = list(ACCENT_OPTIONS)
        if current not in {option.upper() for option in options}:
            options.append(self._manager.accent)  # show the active custom color
        for color in options:
            swatch = _AccentSwatch(color, color.upper() == current, tokens, self)
            swatch.clicked.connect(lambda _c=False, value=color: self.apply_accent(value))
            self._accent_row.addWidget(swatch)
            self._accent_swatches.append(swatch)
        self._accent_row.addStretch(1)

    # -- live application -------------------------------------------------------------
    def apply_theme(self, name: str) -> None:
        """Switch theme, keeping the current accent."""
        self._manager.apply(name, self._manager.accent)

    def apply_accent(self, color: str) -> None:
        self._manager.apply(self._manager.theme_name, color)

    def reset_accent(self) -> None:
        """Back to the active theme's own accent."""
        base = THEMES[self._manager.theme_name]
        self._manager.apply(self._manager.theme_name, base.accent)

    def pick_custom_accent(self) -> None:
        """Full color picker for an arbitrary accent (spec module 4.2)."""
        chosen = QColorDialog.getColor(
            QColor(self._manager.accent), self, CUSTOM_PICKER_TITLE
        )
        if chosen.isValid():
            self.apply_accent(chosen.name().upper())

    def _on_theme_changed(self, _tokens: ThemeTokens) -> None:
        self._rebuild_choices()
