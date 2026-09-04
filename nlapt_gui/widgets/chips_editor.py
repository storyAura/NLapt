"""Chips (胶囊) editing mode + the shared segment-editor machinery.

This module owns two things:

- :class:`SegmentEditorBase` - the edit/insert/commit/reorder state machine
  shared by the chips and sentences modes (one editor instance per caption
  key). All caption writes go through ``AppController.commit_segments`` with
  the exact history labels from the prototype.
- :class:`ChipsEditor` - the flow-layout pill view with inline editing,
  drag reorder (accent drop indicator), ``+ 标签`` insert and empty state.

The floating segment toolbar (owned by ``editor_panel``) drives the active
editor through the ``toolbar_split`` / ``toolbar_insert`` / ``toolbar_delete``
methods implemented here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractAnimation, QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag, QFocusEvent, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui import anim
from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES, ThemeTokens
from nlapt_gui.widgets.flow_layout import FlowLayout

if TYPE_CHECKING:
    from nlapt_gui.controller import AppController

_LOGGER = get_logger(__name__)

# -- exact UI strings (prototype) --------------------------------------------------
LABEL_SPLIT = "分段"
LABEL_REORDER = "拖拽排序"
LABEL_DELETE_EMPTY = "删除分段"  # empty inline commit (prototype _applyEditNow)
INSERT_PLACEHOLDER = "新片段…"
ADD_CHIP_TEXT = "+ 标签"
EMPTY_STATE_TEXT = "暂无内容,点击「+ 标签」添加"
CHIP_EDIT_TOOLTIP = "点击编辑 · 可拖拽排序"
CHIP_DELETE_TOOLTIP = "删除"
TOAST_FINISH_INSERT_FIRST = "先完成插入内容"
TOAST_PUT_CURSOR = "把光标放在要打断的位置"
TOAST_SPLIT_DONE = "已打断为两段"
TOOLBAR_LABEL_EDIT = "第 {n} 段"
TOOLBAR_LABEL_INSERT = "插入新段 · 第 {n} 位"
LABEL_TRANSLATE_REPLACE = "翻译替换"

# toast kinds (mirror controller vocabulary without importing it at runtime).
_KIND_OK = "ok"
_KIND_WARN = "warn"

# History label 「x」 excerpts truncate to 12 chars + ellipsis (prototype _short).
SHORT_LABEL_LIMIT = 12

# Drag payload format for segment reorder.
MIME_SEGMENT = "application/x-nlapt-segment"
# Opacity of the chip/row being dragged (prototype 0.45).
DRAG_SOURCE_OPACITY = 0.45
# Width of the accent drop-indicator bar (prototype 3px box-shadow).
DROP_INDICATOR_PX = 3

# Split-at-cursor comma tidy rules (prototype tbSplit).
_TRAILING_COMMA = re.compile(r"[,，]\s*$")
_LEADING_COMMA = re.compile(r"^[,，]\s*")

# Inline chip editor sizing.
_CHIP_FIELD_MIN_W = 64
_CHIP_FIELD_PAD_W = 34


def short_label(text: str) -> str:
    """Prototype ``_short``: truncate to 12 chars + ``…`` for 「x」 labels."""
    return text[:SHORT_LABEL_LIMIT] + "…" if len(text) > SHORT_LABEL_LIMIT else text


def resolve_tokens(widget: QWidget) -> ThemeTokens:
    """Best-effort active ThemeTokens for custom painting (warn dots etc.).

    Matches the application palette's window color back to a known theme;
    falls back to the default theme when no ThemeManager has run (tests).
    """
    window = widget.palette().color(QPalette.ColorRole.Window).name().upper()
    for tokens in THEMES.values():
        if tokens.bg.upper() == window:
            return tokens
    return THEMES[DEFAULT_THEME]


def split_segment_text(text: str, pos: int) -> tuple[str, str] | None:
    """Split ``text`` at ``pos`` with the prototype's comma-tidy rules.

    Strips a trailing ``[,，]\\s*`` from the left half and a leading one from
    the right half, then trims both. Returns ``None`` when the cursor is at
    an edge or either half ends up empty (caller shows 把光标放在要打断的位置).
    """
    if pos <= 0 or pos >= len(text):
        return None
    left = _TRAILING_COMMA.sub("", text[:pos]).strip()
    right = _LEADING_COMMA.sub("", text[pos:]).strip()
    if not left or not right:
        return None
    return (left, right)


class InlineChipField(QLineEdit):
    """Single-line inline editor: Enter commits, Esc cancels, blur commits."""

    submitted = Signal()
    cancelled = Signal()
    focus_lost = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
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

    def cursor_index(self) -> int:
        return self.cursorPosition()


class SegmentEditorBase(QWidget):
    """Shared select/edit/insert/commit/drag state machine for chips & sents.

    Subclasses implement ``_rebuild`` (view construction), ``_create_field``
    (the inline editor widget) and ``_segment_widget`` (display widget of one
    segment, for toolbar anchoring). Clicking a segment SELECTS it first
    (spec module 2: 单纯选择); clicking the selected segment again opens the
    inline editor. Only one edit OR insert OR selection is active at a time;
    the floating toolbar in ``editor_panel`` reflects
    :meth:`has_focus_target` / :meth:`active_label` and anchors above
    :meth:`active_target_widget`.
    """

    edit_state_changed = Signal()

    def __init__(
        self, controller: "AppController", key: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._key = key
        self._edit_index: int | None = None
        self._insert_pos: int | None = None
        self._selected_index: int | None = None
        self._field: QWidget | None = None
        self._drag_index: int | None = None
        self._reflow_pop_index: int | None = None

    # -- identity / state ------------------------------------------------------------
    @property
    def key(self) -> str:
        return self._key

    def segments(self) -> list[str]:
        return list(self._controller.segments(self._key))

    def edit_index(self) -> int | None:
        return self._edit_index

    def insert_pos(self) -> int | None:
        return self._insert_pos

    def selected_index(self) -> int | None:
        return self._selected_index

    def is_inserting(self) -> bool:
        return self._insert_pos is not None

    def has_active(self) -> bool:
        return self._edit_index is not None or self._insert_pos is not None

    def has_selection(self) -> bool:
        return self._selected_index is not None

    def has_focus_target(self) -> bool:
        """Whether the toolbar should show for this editor (edit/insert/selection)."""
        return self.has_active() or self.has_selection()

    def active_label(self) -> str:
        """Toolbar label: '第 n 段' while editing/selected, '插入新段 · 第 n 位' while inserting."""
        if self._insert_pos is not None:
            return TOOLBAR_LABEL_INSERT.format(n=self._insert_pos + 1)
        if self._edit_index is not None:
            return TOOLBAR_LABEL_EDIT.format(n=self._edit_index + 1)
        if self._selected_index is not None:
            return TOOLBAR_LABEL_EDIT.format(n=self._selected_index + 1)
        return ""

    # -- inline field access (duck-typed line edit / plain text area) ------------------
    def editor_text(self) -> str:
        return self._field.text() if self._field is not None else ""  # type: ignore[attr-defined]

    def set_editor_text(self, text: str) -> None:
        if self._field is not None:
            self._field.setText(text)  # type: ignore[attr-defined]

    def cursor_position(self) -> int | None:
        if self._field is None:
            return None
        return self._field.cursor_index()  # type: ignore[attr-defined]

    def active_text(self) -> str:
        """The text the toolbar operates on: open editor text or selected segment."""
        if self._field is not None:
            return self.editor_text()
        if self._selected_index is not None:
            segs = self.segments()
            if self._selected_index < len(segs):
                return segs[self._selected_index]
        return ""

    def apply_translation(self, text: str) -> None:
        """替换 from the translation preview: into the open editor or the selection."""
        if self.has_active():
            self.set_editor_text(text)
            return
        index = self._selected_index
        if index is None:
            return
        segs = self.segments()
        if index >= len(segs) or segs[index] == text:
            return
        segs[index] = text
        self._commit(segs, LABEL_TRANSLATE_REPLACE)
        self._rebuild()

    def active_target_widget(self) -> QWidget | None:
        """The widget the floating toolbar anchors to (editor field / segment)."""
        if self._field is not None:
            return self._field
        if self._selected_index is not None:
            return self._segment_widget(self._selected_index)
        return None

    # -- refresh (external caption change) ---------------------------------------------
    def refresh(self) -> None:
        """Re-read segments and rebuild the view (called on caption_changed)."""
        changed = False
        count = len(self.segments())
        if self._edit_index is not None and self._edit_index >= count:
            self._edit_index = None
            self._close_field()
            changed = True
        if self._selected_index is not None and self._selected_index >= count:
            self._selected_index = None
            changed = True
        if changed:
            self.edit_state_changed.emit()
        self._rebuild()

    # -- selection lifecycle ----------------------------------------------------------
    def on_segment_clicked(self, index: int) -> None:
        """Select-then-edit click model (spec module 2).

        The first click on a segment only SELECTS it; clicking the selected
        segment again opens the inline editor. A click while another inline
        editor is open commits that edit and selects the clicked segment
        (re-resolved against the index shift the commit may cause).
        """
        if self._edit_index == index:
            return  # already editing this one
        if self.has_active():
            resolved = self._commit_and_resolve(index)
            if resolved is not None:
                self.select_segment(resolved)
            return
        if self._selected_index == index:
            self.start_edit(index)
            return
        self.select_segment(index)

    def select_segment(self, index: int) -> None:
        """Highlight segment ``index`` without opening its editor."""
        if not 0 <= index < len(self.segments()):
            return
        self.commit_active()
        self._edit_index = None
        self._insert_pos = None
        self._selected_index = index
        self._controller.set_current(self._key)
        self._rebuild()
        self.edit_state_changed.emit()

    def clear_selection(self) -> None:
        """Drop the selection highlight (no-op when nothing is selected)."""
        if self._selected_index is None:
            return
        self._selected_index = None
        self._rebuild()
        self.edit_state_changed.emit()

    def _commit_and_resolve(self, index: int) -> int | None:
        """Commit any open inline editor, re-resolving ``index`` for the shift.

        Committing can add or remove a segment and shift every index after it
        (chips take no focus, so there is no blur before the click). Returns
        the shifted index, or None when it fell out of range.
        """
        prior_edit = self._edit_index
        prior_insert = self._insert_pos
        active_text = self.editor_text().strip() if self.has_active() else ""
        will_delete = prior_edit is not None and not active_text
        will_insert = prior_insert is not None and bool(active_text)
        self.commit_active()
        if will_delete and prior_edit is not None and prior_edit < index:
            index -= 1
        elif will_insert and prior_insert is not None and prior_insert <= index:
            index += 1
        if not 0 <= index < len(self.segments()):
            return None
        return index

    # -- edit / insert lifecycle ---------------------------------------------------------
    def start_edit(self, index: int) -> None:
        """Open the inline editor on segment ``index`` (commits any active edit)."""
        segs = self.segments()
        if not 0 <= index < len(segs):
            _LOGGER.warning("start_edit index %s out of range for %r", index, self._key)
            return
        resolved = self._commit_and_resolve(index)
        if resolved is None:
            return
        segs = self.segments()  # commit may have changed them
        self._edit_index = resolved
        self._insert_pos = None
        self._selected_index = None
        self._make_field(segs[resolved], placeholder="")
        self._controller.set_current(self._key)
        self._rebuild()
        self._focus_field(select_all=True)
        self.edit_state_changed.emit()

    def start_insert(self, pos: int) -> None:
        """Open the dashed insert editor at position ``pos``."""
        self.commit_active()
        self._edit_index = None
        self._selected_index = None
        self._insert_pos = max(0, pos)
        self._make_field("", placeholder=INSERT_PLACEHOLDER)
        self._controller.set_current(self._key)
        self._rebuild()
        self._focus_field(select_all=False)
        self.edit_state_changed.emit()

    def start_insert_end(self) -> None:
        self.start_insert(len(self.segments()))

    def commit_active(self) -> None:
        """Apply whichever inline editor is open (blur semantics)."""
        if self._edit_index is not None:
            self._commit_edit()
        elif self._insert_pos is not None:
            self._commit_insert()

    def cancel_active(self) -> None:
        """Close the inline editor without applying (Esc semantics)."""
        if not self.has_active():
            return
        self._edit_index = None
        self._insert_pos = None
        self._close_field()
        self._rebuild()
        self.edit_state_changed.emit()

    def delete_segment(self, index: int) -> None:
        """Delete via the chip × / row delete button: label 删除分段「x」."""
        self.commit_active()
        segs = self.segments()
        if not 0 <= index < len(segs):
            return
        # Keep the selection pointing at the same segment across the shift.
        if self._selected_index is not None:
            if self._selected_index == index:
                self._selected_index = None
                self.edit_state_changed.emit()
            elif self._selected_index > index:
                self._selected_index -= 1
        old = segs.pop(index)
        self._commit(segs, f"删除分段「{short_label(old)}」")
        self._rebuild()

    # -- floating toolbar actions ---------------------------------------------------------
    def toolbar_split(self) -> None:
        """分段: split the active segment at the inline editor's cursor."""
        if self._insert_pos is not None:
            self._toast(TOAST_FINISH_INSERT_FIRST, _KIND_WARN)
            return
        if self._edit_index is None and self._selected_index is not None:
            # Selection only: enter the editor first so a cursor exists.
            self.start_edit(self._selected_index)
            self._toast(TOAST_PUT_CURSOR, _KIND_WARN)
            return
        index = self._edit_index
        if index is None or self._field is None:
            return
        value = self.editor_text()
        pos = self.cursor_position()
        parts = split_segment_text(value, pos) if pos is not None else None
        if parts is None:
            self._toast(TOAST_PUT_CURSOR, _KIND_WARN)
            return
        segs = self.segments()
        if index >= len(segs):
            return
        segs[index : index + 1] = list(parts)
        self._edit_index = None
        self._close_field()
        self._commit(segs, LABEL_SPLIT)
        self._rebuild()
        self.edit_state_changed.emit()
        self._toast(TOAST_SPLIT_DONE, _KIND_OK)

    def toolbar_insert(self) -> None:
        """插入: commit the current edit, then open an insert editor after it."""
        if self._insert_pos is not None:
            return
        index = self._edit_index
        if index is None:
            index = self._selected_index
            if index is None:
                return
            self.start_insert(index + 1)
            return
        self._commit_edit()
        self.start_insert(index + 1)

    def toolbar_delete(self) -> None:
        """删除: delete the active/selected segment, or cancel an insert."""
        if self._insert_pos is not None:
            self.cancel_active()
            return
        if self._edit_index is None:
            index = self._selected_index
            if index is None:
                return
            self.delete_segment(index)
            return
        index = self._edit_index
        segs = self.segments()
        old = segs[index] if index < len(segs) else ""
        if index < len(segs):
            del segs[index]
        self._edit_index = None
        self._close_field()
        self._commit(segs, f"删除分段「{short_label(old)}」")
        self._rebuild()
        self.edit_state_changed.emit()

    # -- drag reorder -----------------------------------------------------------------------
    def reorder(self, from_index: int | None, to_index: int | None) -> None:
        """Prototype ``_reorder``: move a segment, commit label 拖拽排序."""
        if from_index is None or to_index is None:
            return
        segs = self.segments()
        if from_index == to_index or from_index >= len(segs):
            return
        # A reorder invalidates the selection's index mapping — drop it.
        if self._selected_index is not None:
            self._selected_index = None
            self.edit_state_changed.emit()
        item = segs.pop(from_index)
        insert_index = to_index - 1 if to_index > from_index else to_index
        segs.insert(min(insert_index, len(segs)), item)
        self._reflow_pop_index = min(insert_index, len(segs) - 1) if segs else None
        self._commit(segs, LABEL_REORDER)

    # -- internals ------------------------------------------------------------------------
    def _commit_edit(self) -> None:
        index = self._edit_index
        if index is None:
            return
        value = self.editor_text().strip()
        self._edit_index = None
        self._close_field()
        segs = self.segments()
        if index < len(segs):
            if not value:
                del segs[index]
                self._commit(segs, LABEL_DELETE_EMPTY)
            elif segs[index] != value:
                old = segs[index]
                segs[index] = value
                self._commit(segs, f"编辑分段「{short_label(old)}」")
        self._rebuild()
        self.edit_state_changed.emit()

    def _commit_insert(self) -> None:
        pos = self._insert_pos
        if pos is None:
            return
        value = self.editor_text().strip()
        self._insert_pos = None
        self._close_field()
        if value:
            segs = self.segments()
            segs.insert(min(pos, len(segs)), value)
            self._commit(segs, f"插入片段「{short_label(value)}」")
        self._rebuild()
        self.edit_state_changed.emit()

    def _commit(self, segs: list[str], label: str) -> None:
        self._controller.commit_segments(self._key, segs, label)

    def _toast(self, text: str, kind: str) -> None:
        self._controller.toast_requested.emit(text, kind)

    def _make_field(self, initial: str, *, placeholder: str) -> None:
        self._close_field()
        field = self._create_field(initial, placeholder)
        field.submitted.connect(self.commit_active)  # type: ignore[attr-defined]
        field.cancelled.connect(self.cancel_active)  # type: ignore[attr-defined]
        field.focus_lost.connect(self._on_field_blur)  # type: ignore[attr-defined]
        self._field = field

    def _on_field_blur(self) -> None:
        if self.has_active():
            self.commit_active()

    def _close_field(self) -> None:
        if self._field is None:
            return
        field = self._field
        self._field = None
        field.hide()
        field.setParent(None)
        field.deleteLater()

    def _focus_field(self, *, select_all: bool) -> None:
        if self._field is None:
            return
        self._field.setFocus(Qt.FocusReason.OtherFocusReason)
        if select_all:
            self._field.selectAll()  # type: ignore[attr-defined]

    # -- background click clears the selection ------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.childAt(event.position().toPoint()) is None
        ):
            self.clear_selection()
        super().mousePressEvent(event)

    # -- subclass hooks -----------------------------------------------------------------------
    def _create_field(self, initial: str, placeholder: str) -> QWidget:
        raise NotImplementedError

    def _rebuild(self) -> None:
        raise NotImplementedError

    def _segment_widget(self, index: int) -> QWidget | None:
        """Display widget of segment ``index`` (toolbar anchor); None if unknown."""
        return None


class ChipWidget(QFrame):
    """One caption pill: mono text + × delete; click selects (again: edits),
    drag reorders."""

    def __init__(
        self, editor: "ChipsEditor", index: int, text: str, *, selected: bool = False
    ) -> None:
        super().__init__(editor)
        self._editor = editor
        self._index = index
        self._text = text
        self._press_pos: QPoint | None = None
        self._drop_indicator = False
        self.setProperty("chip", "true")
        self.setProperty("chipSelected", "true" if selected else "false")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 5, 7, 5)
        row.setSpacing(7)
        label = QLabel(text, self)
        label.setProperty("mono", "true")
        label.setStyleSheet("font-size: 12.5px;")
        label.setToolTip(CHIP_EDIT_TOOLTIP)
        row.addWidget(label)
        remove = QPushButton("×", self)
        remove.setProperty("variant", "danger-ghost")
        remove.setFixedSize(16, 16)
        remove.setStyleSheet("padding: 0; border-radius: 8px; font-size: 13px;")
        remove.setToolTip(CHIP_DELETE_TOOLTIP)
        remove.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove.clicked.connect(lambda: self._editor.delete_segment(self._index))
        row.addWidget(remove)

    @property
    def index(self) -> int:
        return self._index

    @property
    def chip_text(self) -> str:
        return self._text

    def set_drop_indicator(self, on: bool) -> None:
        if on != self._drop_indicator:
            self._drop_indicator = on
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if self._drop_indicator:
            painter = QPainter(self)
            accent = self.palette().color(QPalette.ColorRole.Highlight)
            painter.fillRect(0, 0, DROP_INDICATOR_PX, self.height(), accent)
            painter.end()

    # click (release without drag) -> select/edit; move past threshold -> drag.
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            # Consume the press: it must not bubble to the editor background
            # handler, which clears the selection on empty-area clicks.
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._press_pos is None:
            return
        distance = (event.position().toPoint() - self._press_pos).manhattanLength()
        if distance >= QApplication.startDragDistance():
            self._press_pos = None
            self._editor.begin_drag(self._index, self)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._press_pos is not None and event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = None
            self._editor.on_segment_clicked(self._index)
        super().mouseReleaseEvent(event)


class ChipsEditor(SegmentEditorBase):
    """胶囊 mode: flow-layout pills with inline edit / insert / drag reorder."""

    def __init__(
        self, controller: "AppController", key: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(controller, key, parent)
        self._flow = FlowLayout(self, h_spacing=8, v_spacing=8)
        self._chips: list[ChipWidget] = []
        self._reflow_anim: QAbstractAnimation | None = None
        self.setAcceptDrops(True)
        self._rebuild()

    # -- view -------------------------------------------------------------------------
    def chips(self) -> tuple[ChipWidget, ...]:
        return tuple(self._chips)

    def _chip_keys(self) -> list[tuple[str, ChipWidget]]:
        seen: dict[str, int] = {}
        keyed: list[tuple[str, ChipWidget]] = []
        for chip in self._chips:
            n = seen.get(chip.chip_text, 0)
            seen[chip.chip_text] = n + 1
            keyed.append((f"{n}\0{chip.chip_text}", chip))
        return keyed

    def _rebuild(self) -> None:
        anim.finish_animation(self._reflow_anim)
        self._reflow_anim = None
        previous = anim.snapshot_named_rects(self._chip_keys())
        while self._flow.count():
            item = self._flow.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is None or widget is self._field:
                continue
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        self._chips = []
        segs = self.segments()
        for i, seg in enumerate(segs):
            if self._edit_index == i and self._field is not None:
                self._flow.addWidget(self._field)
                continue
            chip = ChipWidget(self, i, seg, selected=self._selected_index == i)
            self._chips.append(chip)
            self._flow.addWidget(chip)
        if self._insert_pos is not None and self._field is not None:
            self._flow.insert_widget(min(self._insert_pos, self._flow.count()), self._field)
        if self._insert_pos is None:
            add = QPushButton(ADD_CHIP_TEXT, self)
            add.setProperty("chipAdd", "true")
            add.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            add.clicked.connect(self.start_insert_end)
            self._flow.addWidget(add)
            if not segs and self._edit_index is None:
                empty = QLabel(EMPTY_STATE_TEXT, self)
                empty.setProperty("muted", "true")
                empty.setStyleSheet("font-size: 12px; padding: 6px 2px;")
                self._flow.addWidget(empty)
        layout = self.layout()
        if layout is not None:
            layout.activate()
        pop_index = self._reflow_pop_index
        self._reflow_pop_index = None
        pop = (
            self._chips[pop_index]
            if pop_index is not None and 0 <= pop_index < len(self._chips)
            else None
        )
        self._reflow_anim = anim.flip_reflow(self._chip_keys(), previous, pop=pop)
        if self._reflow_anim is not None:
            self._reflow_anim.finished.connect(lambda: setattr(self, "_reflow_anim", None))

    def _create_field(self, initial: str, placeholder: str) -> QWidget:
        field = InlineChipField(self)
        field.setProperty("chipEditor", "true")
        field.setText(initial)
        if placeholder:
            field.setPlaceholderText(placeholder)
        field.textChanged.connect(lambda _t: self._autosize_field(field))
        self._autosize_field(field)
        return field

    def _autosize_field(self, field: InlineChipField) -> None:
        """Auto width from content (prototype chW: grows while typing)."""
        text = field.text() or field.placeholderText()
        width = field.fontMetrics().horizontalAdvance(text) + _CHIP_FIELD_PAD_W
        field.setFixedWidth(max(_CHIP_FIELD_MIN_W, width))

    # -- drag & drop ---------------------------------------------------------------------
    def begin_drag(self, index: int, chip: ChipWidget) -> None:
        self._drag_index = index
        effect = QGraphicsOpacityEffect(chip)
        effect.setOpacity(DRAG_SOURCE_OPACITY)
        chip.setGraphicsEffect(effect)
        drag = QDrag(chip)
        mime = QMimeData()
        mime.setText(chip.chip_text)
        mime.setData(MIME_SEGMENT, str(index).encode("ascii"))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)
        self._drag_index = None
        self._rebuild()

    def _chip_index_at(self, pos: QPoint) -> int | None:
        for chip in self._chips:
            if chip.geometry().contains(pos):
                return chip.index
        return None

    def _segment_widget(self, index: int) -> QWidget | None:
        for chip in self._chips:
            if chip.index == index:
                return chip
        return None

    def _set_indicator(self, index: int | None) -> None:
        for chip in self._chips:
            chip.set_drop_indicator(chip.index == index)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag_index is not None and event.mimeData().hasFormat(MIME_SEGMENT):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag_index is None:
            return
        self._set_indicator(self._chip_index_at(event.position().toPoint()))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._set_indicator(None)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag_index is None:
            return
        target = self._chip_index_at(event.position().toPoint())
        to_index = target if target is not None else len(self.segments())
        event.acceptProposedAction()
        self._set_indicator(None)
        self.reorder(self._drag_index, to_index)
