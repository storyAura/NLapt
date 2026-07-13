"""Right tools panel (修改工具) - four collapsible sections + shared widgets.

Owns the shared segmented-scope control (当前 / 选中 n / 全部 N) used by the
find-replace and prefix-suffix sections, the small accent stroke icons of the
section headers, and the panel assembly itself (header, scrollable section
stack, open-state persistence into ``UISettings.sections``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPaintEvent, QPalette, QPen
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
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.controller import AppController
from nlapt_gui.resources import app_data_dir
from nlapt_gui.settings import UISettings
from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES, ThemeTokens, with_accent
from nlapt_gui.widgets.collapsible import (
    MAX_CONTENT_HEIGHT,
    MIN_CONTENT_HEIGHT,
    CollapsibleSection,
)
from nlapt_gui.widgets.settings_dialog import SettingsDialog

_LOGGER = get_logger(__name__)

# Design metrics.
PANEL_WIDTH = 340
_HEADER_MARGINS = (16, 13, 16, 11)
_STACK_MARGINS = (10, 10, 10, 10)
_STACK_GAP = 10
_SEG_BAR_PADDING = 3
_SEG_BAR_GAP = 3
_SEG_HEIGHT = 24
_ICON_SIZE = 15

# Section ids (persisted keys in UISettings.sections).
SECTION_FIND_REPLACE = "fr"
SECTION_PREFIX_SUFFIX = "ps"
SECTION_TRANSLATE = "tr"
SECTION_HISTORY = "hist"

# Exact UI strings from the design.
PANEL_TITLE = "修改工具"
PANEL_SUBTITLE = "对当前文件或批量范围应用修改"
TITLE_FIND_REPLACE = "查找替换"
TITLE_PREFIX_SUFFIX = "前缀 / 后缀"
TITLE_TRANSLATE = "翻译对照"
TITLE_HISTORY = "历史记录"
SCOPE_CURRENT_LABEL = "当前"
SCOPE_SELECTED_LABEL = "选中 {n}"
SCOPE_ALL_LABEL = "全部 {n}"
HISTORY_COUNT_SUFFIX = "{n} 条"

SCOPE_CURRENT = "current"
SCOPE_SELECTED = "selected"
SCOPE_ALL = "all"

# Per-section user-set content heights persist in a small JSON file this
# widget owns (UISettings is a frozen dataclass that rejects unknown fields,
# so a dedicated file keeps the height feature self-contained).
SECTION_HEIGHTS_FILE = "section_heights.json"


def _section_heights_path() -> Path:
    return app_data_dir() / SECTION_HEIGHTS_FILE


def load_section_heights(path: Path | None = None) -> dict[str, int]:
    """Load persisted per-section content heights; corrupt file -> empty map."""
    target = path if path is not None else _section_heights_path()
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("corrupt section heights %s (%s); ignoring", target, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
            result[key] = max(MIN_CONTENT_HEIGHT, min(MAX_CONTENT_HEIGHT, value))
    return result


def save_section_heights(heights: Mapping[str, int], path: Path | None = None) -> None:
    """Persist per-section content heights atomically as UTF-8 JSON."""
    target = path if path is not None else _section_heights_path()
    atomic_write_text(target, json.dumps(dict(heights), ensure_ascii=False, indent=2))
    _LOGGER.debug("section heights saved to %s", target)


def resolve_tokens(settings: UISettings) -> ThemeTokens:
    """Active theme tokens for widget-level color lookups.

    Derived from the persisted theme/accent names; unknown values fall back
    to the defaults so widgets never crash on stale settings.
    """
    base = THEMES.get(settings.theme, THEMES[DEFAULT_THEME])
    if settings.accent and settings.accent != base.accent:
        try:
            return with_accent(base, settings.accent)
        except ValidationError:
            _LOGGER.warning("invalid accent %r; using theme default", settings.accent)
    return base


def repolish(widget: QWidget) -> None:
    """Re-apply QSS after a dynamic property change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class SegmentedBar(QFrame):
    """Generic design segmented control (surface2 bar + flat active button)."""

    changed = Signal(str)  # option id

    def __init__(
        self,
        options: Sequence[tuple[str, str]],
        *,
        current: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not options:
            raise ValidationError("SegmentedBar requires at least one option")
        self.setProperty("segBar", True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(*(_SEG_BAR_PADDING,) * 4)
        layout.setSpacing(_SEG_BAR_GAP)
        self._buttons: dict[str, QPushButton] = {}
        self._current = current if current is not None else options[0][0]
        for option_id, label in options:
            button = QPushButton(label, self)
            button.setProperty("seg", True)
            button.setFixedHeight(_SEG_HEIGHT)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, oid=option_id: self.set_current(oid))
            layout.addWidget(button, 1)
            self._buttons[option_id] = button
        self._apply_active()

    @property
    def current(self) -> str:
        return self._current

    def set_current(self, option_id: str) -> None:
        if option_id not in self._buttons:
            raise ValidationError(f"unknown segmented option {option_id!r}")
        if option_id == self._current:
            return
        self._current = option_id
        self._apply_active()
        self.changed.emit(option_id)

    def set_label(self, option_id: str, text: str) -> None:
        button = self._buttons.get(option_id)
        if button is not None:
            button.setText(text)

    def _apply_active(self) -> None:
        for option_id, button in self._buttons.items():
            button.setProperty("segActive", option_id == self._current)
            repolish(button)


class ScopeSelector(SegmentedBar):
    """当前 / 选中 n / 全部 N scope control with live controller counts."""

    def __init__(self, controller: AppController, parent: QWidget | None = None) -> None:
        super().__init__(
            (
                (SCOPE_CURRENT, SCOPE_CURRENT_LABEL),
                (SCOPE_SELECTED, SCOPE_SELECTED_LABEL.format(n=0)),
                (SCOPE_ALL, SCOPE_ALL_LABEL.format(n=0)),
            ),
            current=SCOPE_CURRENT,
            parent=parent,
        )
        self._controller = controller
        controller.selection_changed.connect(self.refresh_counts)
        controller.dataset_opened.connect(lambda _result: self.refresh_counts())
        self.refresh_counts()

    @property
    def scope(self) -> str:
        return self.current

    def refresh_counts(self) -> None:
        selected = len(self._controller.selected_keys())
        total = len(self._controller.keys())
        self.set_label(SCOPE_SELECTED, SCOPE_SELECTED_LABEL.format(n=selected))
        self.set_label(SCOPE_ALL, SCOPE_ALL_LABEL.format(n=total))


class SectionIcon(QWidget):
    """15px accent stroke icon approximating the design's section SVGs."""

    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = kind
        self.setFixedSize(_ICON_SIZE, _ICON_SIZE)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Accent from the live palette (ThemeManager sets Highlight = accent).
        pen = QPen(self.palette().color(QPalette.ColorRole.Highlight), 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        size = float(_ICON_SIZE)
        if self._kind == SECTION_FIND_REPLACE:  # two opposing arrows
            painter.drawLine(12, 4, 3, 4)
            painter.drawLine(3, 4, 6, 2)
            painter.drawLine(3, 4, 6, 7)
            painter.drawLine(3, 11, 12, 11)
            painter.drawLine(12, 11, 9, 9)
            painter.drawLine(12, 11, 9, 13)
        elif self._kind == SECTION_PREFIX_SUFFIX:  # bar + right arrow
            painter.drawLine(3, 2, 3, 13)
            painter.drawLine(6, 8, 13, 8)
            painter.drawLine(13, 8, 10, 5)
            painter.drawLine(13, 8, 10, 11)
        elif self._kind == SECTION_TRANSLATE:  # globe
            painter.drawEllipse(2, 2, int(size) - 4, int(size) - 4)
            painter.drawLine(2, int(size / 2), int(size) - 2, int(size / 2))
            painter.drawEllipse(5, 2, int(size) - 10, int(size) - 4)
        else:  # history clock
            painter.drawEllipse(2, 2, int(size) - 4, int(size) - 4)
            painter.drawLine(7, 5, 7, 8)
            painter.drawLine(7, 8, 10, 10)
        painter.end()


class ToolsPanel(QWidget):
    """Right column: panel header + the four collapsible tool sections."""

    def __init__(
        self,
        controller: AppController,
        parent: QWidget | None = None,
        *,
        section_heights_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        # Imported here because the section modules import shared widgets
        # (ScopeSelector, repolish, ...) from this module.
        from nlapt_gui.widgets.sections import (
            FindReplaceSection,
            HistorySection,
            PrefixSuffixSection,
            TranslateSection,
        )

        self._controller = controller
        self._section_heights_path = section_heights_path
        self._section_heights = load_section_heights(section_heights_path)
        self.setProperty("panel", True)
        self.setFixedWidth(PANEL_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())
        divider = QFrame(self)
        divider.setProperty("divider", True)
        divider.setFixedHeight(1)
        outer.addWidget(divider)

        self.find_replace = FindReplaceSection(controller)
        self.prefix_suffix = PrefixSuffixSection(controller)
        self.translate = TranslateSection(controller)
        self.history = HistorySection(controller)
        self.translate.open_settings_requested.connect(self.open_settings_dialog)

        opened = controller.settings.sections
        self._sections: dict[str, CollapsibleSection] = {}
        stack = QWidget(self)
        stack_layout = QVBoxLayout(stack)
        stack_layout.setContentsMargins(*_STACK_MARGINS)
        stack_layout.setSpacing(_STACK_GAP)
        for section_id, title, content in (
            (SECTION_FIND_REPLACE, TITLE_FIND_REPLACE, self.find_replace),
            (SECTION_PREFIX_SUFFIX, TITLE_PREFIX_SUFFIX, self.prefix_suffix),
            (SECTION_TRANSLATE, TITLE_TRANSLATE, self.translate),
            (SECTION_HISTORY, TITLE_HISTORY, self.history),
        ):
            is_open = bool(opened.get(section_id, section_id != SECTION_TRANSLATE))
            card = CollapsibleSection(
                section_id,
                title,
                icon=SectionIcon(section_id),
                open=is_open,
            )
            card.set_content(content)
            card.toggled.connect(self._persist_section_state)
            # Restore any user-set content height before wiring persistence so
            # the restore itself does not trigger a redundant save.
            stored_height = self._section_heights.get(section_id)
            if stored_height is not None:
                card.set_content_height(stored_height)
            card.height_changed.connect(self._persist_section_height)
            stack_layout.addWidget(card)
            self._sections[section_id] = card
        stack_layout.addStretch(1)

        # The 翻译对照 card only issues LLM requests while it is expanded, so
        # mirror its open state onto the section (avoids per-keystroke calls
        # for a collapsed card - the default).
        translate_card = self._sections[SECTION_TRANSLATE]
        self.translate.set_live(translate_card.is_open)
        translate_card.toggled.connect(
            lambda _sid, is_open: self.translate.set_live(is_open)
        )

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(stack)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        controller.history.changed.connect(self._on_history_changed)
        controller.current_changed.connect(lambda _key: self._update_history_count())
        controller.dataset_opened.connect(lambda _result: self._update_history_count())
        self._update_history_count()

    # -- public API -------------------------------------------------------------
    def section(self, section_id: str) -> CollapsibleSection:
        """The collapsible card wrapping one section (fr/ps/tr/hist)."""
        try:
            return self._sections[section_id]
        except KeyError as exc:
            raise ValidationError(f"unknown section id {section_id!r}") from exc

    def open_settings_dialog(self) -> None:
        """Open the modal LLM settings dialog (翻译 guided state link)."""
        dialog = SettingsDialog(self._controller, parent=self.window())
        # After saving a profile the 翻译对照 section must leave its guided
        # unconfigured state without waiting for a file switch.
        dialog.saved.connect(lambda _config: self.translate.refresh())
        dialog.exec()

    def set_busy(self, busy: bool) -> None:
        """Disable batch action buttons while a batch/save is in flight."""
        self.find_replace.apply_button.setEnabled(not busy)
        self.prefix_suffix.apply_button.setEnabled(not busy)

    # -- internals -----------------------------------------------------------------
    def _build_header(self) -> QWidget:
        header = QWidget(self)
        layout = QVBoxLayout(header)
        layout.setContentsMargins(*_HEADER_MARGINS)
        layout.setSpacing(2)
        title = QLabel(PANEL_TITLE, header)
        title.setStyleSheet("font-size: 13px; font-weight: 700;")
        subtitle = QLabel(PANEL_SUBTITLE, header)
        subtitle.setProperty("muted", True)
        subtitle.setStyleSheet("font-size: 11px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _persist_section_state(self, section_id: str, is_open: bool) -> None:
        sections = dict(self._controller.settings.sections)
        sections[section_id] = is_open
        self._controller.update_settings(sections=sections)

    def _persist_section_height(self, section_id: str, height: int) -> None:
        heights = dict(self._section_heights)
        heights[section_id] = height
        self._section_heights = heights
        save_section_heights(heights, self._section_heights_path)

    def _on_history_changed(self, key: str) -> None:
        if key == (self._controller.current_key or ""):
            self._update_history_count()

    def _update_history_count(self) -> None:
        key = self._controller.current_key
        entries = self._controller.history.entries(key) if key else ()
        self._sections[SECTION_HISTORY].set_suffix(
            HISTORY_COUNT_SUFFIX.format(n=len(entries))
        )
