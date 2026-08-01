"""标注编辑区 - mode tabs, caption workspace, editor blocks, floating toolbar.

One :class:`EditorBlock` per ``controller.editor_keys()`` (up to 4 in multi
mode, each with a header row: mono name, dirty dot, 编辑中 badge, stats;
clicking the header focuses that file). The tab row also hosts the
:class:`~nlapt_gui.widgets.caption_bar.CaptionBar` (caption-level
翻译 / 重译 / 删除). The floating toolbar anchors directly above the
selected/edited segment (spec module 2) and falls back to the top-center
when no anchor exists; its buttons fire on mouse-press (before the inline
editor would blur).

Translation goes through a duck-typed ``translate_bridge``
(``request(key, text, *, fresh)`` + ``segment_ready`` signal); ``None`` or
an unconfigured bridge produces the guided warn toast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui.widgets.caption_bar import CaptionBar, TranslationPreview
from nlapt_gui.widgets.chips_editor import ChipsEditor, SegmentEditorBase, resolve_tokens
from nlapt_gui.widgets.sents_editor import SentsEditor
from nlapt_gui.widgets.text_editor import TextEditor

if TYPE_CHECKING:
    from nlapt_gui.controller import AppController

_LOGGER = get_logger(__name__)

# -- exact UI strings (design/prototype) --------------------------------------------
MODE_LABELS: tuple[tuple[str, str], ...] = (
    ("chips", "胶囊"),
    ("sents", "分句"),
    ("text", "文本"),
)
MODE_HINTS: dict[str, str] = {
    "chips": "点击选中 · 再次点击编辑 · 拖动胶囊排序 · 悬浮工具栏可分段 / 插入 / 翻译 / 重译 · × 删除 · 按英文逗号分段",
    "sents": "按英文逗号分段 · 拖动手柄排序 · 点击选中 · 再次点击编辑 · 悬浮工具栏可分段 / 插入 / 翻译 / 重译",
    "text": "拖选文字后可替换 / 删除 / 润色选中片段 · 失焦时自动记录历史",
}
BADGE_EDITING = "编辑中"
DIRTY_TOOLTIP = "未保存"
TB_SPLIT = "分段"
TB_INSERT = "插入"
TB_TRANSLATE = "翻译"
TB_RETRANSLATE = "重译"
TB_DELETE = "删除"
TB_SPLIT_TIP = "在光标处打断为两段"
TB_INSERT_TIP = "在此段后插入新片段"
TB_TRANSLATE_TIP = "翻译这一段 (中英互转)"
TB_RETRANSLATE_TIP = "换一种译法"
TB_DELETE_TIP = "删除这一段"
TOAST_TRANSLATE_UNCONFIGURED = "未配置翻译 API — 打开 工具 ▸ 设置"
TOAST_TRANSLATE_FAILED = "翻译失败: {message}"
# Translation preview strings re-exported for compatibility (the class moved
# to caption_bar so both the segment toolbar and the caption workspace share it).
PREVIEW_TITLE = "译文"
PREVIEW_APPLY = "替换"
PREVIEW_CLOSE = "关闭"
PREVIEW_MAX_WIDTH = 560

_KIND_WARN = "warn"

# Overlay geometry: fallback anchor (design: top:44px) and segment-anchored
# offsets (toolbar floats GAP px above its segment, clamped inside the panel).
TOOLBAR_TOP = 44
TOOLBAR_GAP = 6
TOOLBAR_MARGIN = 8
_DOT_SIZE = 7

# Per-widget style snippets (palette roles only - colors stay token-driven).
# objectName that scopes the current-block accent border to the EditorBlock
# frame ONLY. An unscoped rule cascades onto descendant widgets (the chips),
# making every chip look selected in multi mode - so the rule is bound to the
# frame via a type#id selector; child chips keep their own per-chip borders.
_BLOCK_OBJECT_NAME = "editorBlock"
_BLOCK_CURRENT_QSS = (
    f"EditorBlock#{_BLOCK_OBJECT_NAME} {{"
    " background: palette(base); border: 1.5px solid palette(highlight);"
    " border-radius: 11px; }"
)
_TAB_QSS = "font-size: 12.5px; font-weight: 600; padding: 0 15px; border-radius: 8px;"
_TB_BTN_QSS = "font-size: 11.5px; font-weight: 600; padding: 0 10px; border-radius: 7px;"
_TB_LABEL_QSS = "font-size: 10.5px; font-weight: 600; padding: 0 7px;"


class DirtyDot(QWidget):
    """7px warn-colored dot (未保存 marker) painted from ThemeTokens."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(_DOT_SIZE, _DOT_SIZE)
        self.setToolTip(DIRTY_TOOLTIP)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(resolve_tokens(self).warn))
        painter.drawEllipse(self.rect())
        painter.end()


class SegmentToolbar(QFrame):
    """Floating 分段/插入/翻译/重译/删除 toolbar (instant show/hide overlay).

    Every button uses ``pressed`` + ``Qt.NoFocus`` so actions fire on
    mouse-down, before the inline editor could lose focus (prototype used
    ``onMouseDown`` for the same reason). Deliberately no
    QGraphicsOpacityEffect: a permanently-attached graphics effect re-renders
    through an effect buffer while the window resizes and hard-crashes Qt.
    """

    split_requested = Signal()
    insert_requested = Signal()
    translate_requested = Signal()
    retranslate_requested = Signal()
    delete_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("surfaceCard", "true")
        self._target_visible = False
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 3, 4, 3)
        row.setSpacing(2)
        self._label = QLabel("", self)
        self._label.setProperty("muted", "true")
        self._label.setStyleSheet(_TB_LABEL_QSS)
        row.addWidget(self._label)
        self.split_btn = self._button(TB_SPLIT, TB_SPLIT_TIP, self.split_requested)
        row.addWidget(self.split_btn)
        self.insert_btn = self._button(TB_INSERT, TB_INSERT_TIP, self.insert_requested)
        row.addWidget(self.insert_btn)
        row.addWidget(self._divider())
        self.translate_btn = self._button(
            TB_TRANSLATE, TB_TRANSLATE_TIP, self.translate_requested
        )
        row.addWidget(self.translate_btn)
        self.retranslate_btn = self._button(
            TB_RETRANSLATE, TB_RETRANSLATE_TIP, self.retranslate_requested
        )
        row.addWidget(self.retranslate_btn)
        row.addWidget(self._divider())
        self.delete_btn = self._button(
            TB_DELETE, TB_DELETE_TIP, self.delete_requested, danger=True
        )
        row.addWidget(self.delete_btn)
        self.hide()

    def _button(
        self, text: str, tooltip: str, signal: Signal, *, danger: bool = False
    ) -> QPushButton:
        button = QPushButton(text, self)
        button.setProperty("variant", "danger-ghost" if danger else "ghost")
        button.setStyleSheet(_TB_BTN_QSS)
        button.setFixedHeight(25)
        button.setToolTip(tooltip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.pressed.connect(signal.emit)
        return button

    def _divider(self) -> QFrame:
        divider = QFrame(self)
        divider.setProperty("divider", "true")
        divider.setFixedSize(1, 14)
        return divider

    # -- show / hide ------------------------------------------------------------------
    def label_text(self) -> str:
        return self._label.text()

    def is_active(self) -> bool:
        """Whether the toolbar is (fading) visible."""
        return self._target_visible

    def show_with_label(self, label: str) -> None:
        self._label.setText(label)
        self.adjustSize()
        if not self._target_visible:
            self._target_visible = True
            self.show()
            self.raise_()

    def dismiss(self) -> None:
        if not self._target_visible:
            return
        self._target_visible = False
        self.hide()


class _BlockHeader(QWidget):
    """Clickable multi-mode block header (click focuses the block's file)."""

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class EditorBlock(QFrame):
    """One per-file editor card; hosts the mode-specific editor widget."""

    def __init__(
        self,
        controller: "AppController",
        key: str,
        mode: str,
        *,
        multi: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._key = key
        self._multi = multi
        # Scope-target for the current-block accent border (see _BLOCK_CURRENT_QSS).
        self.setObjectName(_BLOCK_OBJECT_NAME)
        self.setProperty("surfaceCard", "true")
        column = QVBoxLayout(self)
        column.setContentsMargins(12, 10, 12, 10)
        column.setSpacing(9)
        self._dot: DirtyDot | None = None
        self._badge: QLabel | None = None
        self._stat: QLabel | None = None
        if multi:
            header = _BlockHeader(self)
            header.clicked.connect(lambda: self._controller.set_current(self._key))
            head_row = QHBoxLayout(header)
            head_row.setContentsMargins(0, 0, 0, 0)
            head_row.setSpacing(8)
            name = QLabel(key.rsplit("/", 1)[-1], header)
            name.setProperty("mono", "true")
            name.setStyleSheet("font-size: 11.5px; font-weight: 600;")
            head_row.addWidget(name)
            self._dot = DirtyDot(header)
            head_row.addWidget(self._dot)
            self._badge = QLabel(BADGE_EDITING, header)
            self._badge.setProperty("pill", "accent")
            head_row.addWidget(self._badge)
            head_row.addStretch(1)
            self._stat = QLabel("", header)
            self._stat.setProperty("mono", "true")
            self._stat.setProperty("muted", "true")
            self._stat.setStyleSheet("font-size: 10px;")
            head_row.addWidget(self._stat)
            column.addWidget(header)
        if mode == "chips":
            self.editor: QWidget = ChipsEditor(controller, key, self)
        elif mode == "sents":
            self.editor = SentsEditor(controller, key, self)
        else:
            self.editor = TextEditor(controller, key, multi=multi, parent=self)
        column.addWidget(self.editor)
        self.refresh_meta()

    @property
    def key(self) -> str:
        return self._key

    def refresh_meta(self) -> None:
        """Update dirty dot / 编辑中 badge / stats / accent border."""
        current = self._controller.current_key == self._key
        if self._multi:
            record = self._controller.record(self._key)
            if self._dot is not None:
                self._dot.setVisible(record.dirty)
            if self._badge is not None:
                self._badge.setVisible(current)
            if self._stat is not None:
                self._stat.setText(self._controller.char_seg_info(self._key))
        self.setStyleSheet(_BLOCK_CURRENT_QSS if self._multi and current else "")


class EditorPanel(QWidget):
    """The whole 标注编辑区: tabs + workspace, blocks, hint, floating toolbar."""

    def __init__(
        self,
        controller: "AppController",
        translate_bridge: object | None = None,
        parent: QWidget | None = None,
        *,
        vision_bridge: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._bridge = translate_bridge
        # (key, source_text, active-editor id, edit_index, insert_pos,
        # selected_index) of an in-flight 翻译/重译 so a slow reply only lands
        # in the exact segment it was requested for (the user may have moved
        # on to another segment).
        self._pending_translate: (
            tuple[str, str, int, int | None, int | None, int | None] | None
        ) = None
        self._blocks: list[EditorBlock] = []
        self._built_sig: tuple[object, ...] | None = None
        self.setProperty("panel", "true")

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        # -- mode tabs + caption workspace + char/seg info ----------------------------
        tab_row = QWidget(self)
        tabs = QHBoxLayout(tab_row)
        tabs.setContentsMargins(14, 10, 14, 8)
        tabs.setSpacing(6)
        self._tab_buttons: dict[str, QPushButton] = {}
        for mode, label in MODE_LABELS:
            button = QPushButton(label, tab_row)
            button.setProperty("seg", "true")
            button.setStyleSheet(_TAB_QSS)
            button.setFixedHeight(29)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _c=False, m=mode: self._controller.set_mode(m))
            self._tab_buttons[mode] = button
            tabs.addWidget(button)
        # 标注工作区: caption-level 翻译 / 重译 / 删除 (spec module 1).
        self.caption_bar = CaptionBar(
            controller,
            translate_bridge=translate_bridge,
            vision_bridge=vision_bridge,
            overlay_host=self,
            overlay_top=TOOLBAR_TOP,
            parent=tab_row,
        )
        tabs.addWidget(self.caption_bar)
        tabs.addStretch(1)
        self._char_info = QLabel("", tab_row)
        self._char_info.setProperty("mono", "true")
        self._char_info.setProperty("muted", "true")
        self._char_info.setStyleSheet("font-size: 11px;")
        tabs.addWidget(self._char_info)
        column.addWidget(tab_row)

        # -- scrollable editor blocks -------------------------------------------------
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._container = QWidget(self._scroll)
        self._container_box = QVBoxLayout(self._container)
        self._container_box.setContentsMargins(14, 2, 14, 8)
        self._container_box.setSpacing(10)
        self._container_box.addStretch(1)
        self._scroll.setWidget(self._container)
        column.addWidget(self._scroll, 1)
        # The toolbar anchors to the active segment — keep it glued while the
        # block list scrolls under it.
        self._scroll.verticalScrollBar().valueChanged.connect(
            lambda _value: self._position_toolbar()
        )

        # -- bottom hint line ------------------------------------------------------------
        self._hint = QLabel("", self)
        self._hint.setProperty("muted", "true")
        self._hint.setStyleSheet("font-size: 11px; padding: 0 16px 10px 16px;")
        column.addWidget(self._hint)

        # -- floating segment toolbar overlay ---------------------------------------------
        self.toolbar = SegmentToolbar(self)
        self.toolbar.split_requested.connect(self._on_toolbar_split)
        self.toolbar.insert_requested.connect(self._on_toolbar_insert)
        self.toolbar.translate_requested.connect(lambda: self._on_translate(fresh=False))
        self.toolbar.retranslate_requested.connect(lambda: self._on_translate(fresh=True))
        self.toolbar.delete_requested.connect(self._on_toolbar_delete)

        # -- floating translation preview (view first, replace on demand) -------------------
        self.translation_preview = TranslationPreview(self)
        self.translation_preview.apply_requested.connect(self._apply_translation)
        self.translation_preview.dismiss_requested.connect(self._dismiss_translation)

        # -- controller wiring ---------------------------------------------------------------
        # dataset_opened may keep the same editor_keys across a rescan while the
        # underlying captions changed, so force a full rebuild (not a meta-only
        # refresh) to drop stale segment widgets.
        controller.dataset_opened.connect(lambda _r: self._rebuild(force=True))
        controller.current_changed.connect(lambda _k: self._rebuild())
        controller.selection_changed.connect(self._rebuild)
        controller.mode_changed.connect(lambda _m: self._rebuild())
        controller.caption_changed.connect(self._on_caption_changed)
        if translate_bridge is not None and hasattr(translate_bridge, "segment_ready"):
            translate_bridge.segment_ready.connect(self._on_segment_ready)
        self._rebuild()

    # -- introspection (blocks/toolbar for the integrator & tests) --------------------------
    def blocks(self) -> tuple[EditorBlock, ...]:
        return tuple(self._blocks)

    def block_for(self, key: str) -> EditorBlock | None:
        for block in self._blocks:
            if block.key == key:
                return block
        return None

    def active_editor(self) -> SegmentEditorBase | None:
        """The segment editor whose inline edit/insert is currently open."""
        for block in self._blocks:
            editor = block.editor
            if isinstance(editor, SegmentEditorBase) and editor.has_active():
                return editor
        return None

    def focus_editor(self) -> SegmentEditorBase | None:
        """The editor the toolbar targets: open edit/insert OR a selection."""
        for block in self._blocks:
            editor = block.editor
            if isinstance(editor, SegmentEditorBase) and editor.has_focus_target():
                return editor
        return None

    def hint_text(self) -> str:
        return self._hint.text()

    def char_info_text(self) -> str:
        return self._char_info.text()

    # -- rebuild ------------------------------------------------------------------------------
    def _rebuild(self, *, force: bool = False) -> None:
        controller = self._controller
        sig: tuple[object, ...] = (
            controller.mode,
            controller.editor_keys(),
            controller.multi_mode(),
        )
        if sig == self._built_sig and not force:
            self._refresh_meta()
            return
        for block in self._blocks:
            block.hide()
            block.setParent(None)
            block.deleteLater()
        self._blocks = []
        multi = controller.multi_mode()
        insert_at = self._container_box.count() - 1  # keep trailing stretch last
        for key in controller.editor_keys():
            block = EditorBlock(
                controller, key, controller.mode, multi=multi, parent=self._container
            )
            editor = block.editor
            if isinstance(editor, SegmentEditorBase):
                editor.edit_state_changed.connect(
                    lambda ed=editor: self._on_editor_state(ed)
                )
            self._container_box.insertWidget(insert_at, block)
            insert_at += 1
            self._blocks.append(block)
        self._built_sig = sig
        self._sync_tabs()
        self._hint.setText(MODE_HINTS[controller.mode])
        self._refresh_meta()
        self._update_toolbar()

    def _refresh_meta(self) -> None:
        for block in self._blocks:
            block.refresh_meta()
        current = self._controller.current_key
        self._char_info.setText(
            self._controller.char_seg_info(current) if current else ""
        )

    def _sync_tabs(self) -> None:
        for mode, button in self._tab_buttons.items():
            active = mode == self._controller.mode
            button.setProperty("segActive", "true" if active else "false")
            button.style().unpolish(button)
            button.style().polish(button)

    def _on_caption_changed(self, key: str) -> None:
        block = self.block_for(key)
        if block is not None:
            block.editor.refresh()  # type: ignore[attr-defined]
            block.refresh_meta()
        if key == self._controller.current_key:
            self._char_info.setText(self._controller.char_seg_info(key))
        self._position_toolbar()

    # -- floating toolbar -----------------------------------------------------------------------
    def _on_editor_state(self, editor: SegmentEditorBase) -> None:
        """One focus target at a time: a new selection clears the others'."""
        if editor.has_focus_target():
            for block in self._blocks:
                other = block.editor
                if (
                    other is not editor
                    and isinstance(other, SegmentEditorBase)
                    and other.has_selection()
                ):
                    other.clear_selection()
        self._update_toolbar()

    def _update_toolbar(self) -> None:
        editor = self.focus_editor()
        # If the toolbar target moved (different editor/segment/selection), a
        # pending translation reply is no longer valid - discard it.
        pending = self._pending_translate
        if pending is not None:
            _key, _src, eid, edit_idx, insert_pos, sel_idx = pending
            if (
                editor is None
                or id(editor) != eid
                or editor.edit_index() != edit_idx
                or editor.insert_pos() != insert_pos
                or editor.selected_index() != sel_idx
            ):
                self._pending_translate = None
                self.translation_preview.dismiss()
        if editor is None:
            self.toolbar.dismiss()
            self.translation_preview.dismiss()
            return
        label = editor.active_label()
        if self._controller.multi_mode():
            name = editor.key.rsplit("/", 1)[-1]
            label = f"{name} · {label}"
        self.toolbar.show_with_label(label)
        self._position_toolbar()
        # Fresh widgets get their real geometry on the next layout pass —
        # re-anchor once the event loop has laid them out.
        QTimer.singleShot(0, self._position_toolbar)

    def _toolbar_anchor(self) -> QWidget | None:
        """The widget the toolbar should float above (field or selected segment)."""
        editor = self.focus_editor()
        if editor is None:
            return None
        target = editor.active_target_widget()
        if target is None or not target.isVisible():
            return None
        return target

    def _position_toolbar(self) -> None:
        if not self.toolbar.is_active():
            return
        self.toolbar.adjustSize()
        target = self._toolbar_anchor()
        if target is not None:
            top_left = target.mapTo(self, QPoint(0, 0))
            x = top_left.x() + (target.width() - self.toolbar.width()) // 2
            y = top_left.y() - self.toolbar.height() - TOOLBAR_GAP
            if y < TOOLBAR_GAP:
                # No room above (segment near the top edge) — flip below it.
                y = top_left.y() + target.height() + TOOLBAR_GAP
            x = max(TOOLBAR_MARGIN, min(x, self.width() - self.toolbar.width() - TOOLBAR_MARGIN))
            y = max(TOOLBAR_GAP, min(y, self.height() - self.toolbar.height() - TOOLBAR_GAP))
        else:
            x = max(0, (self.width() - self.toolbar.width()) // 2)
            y = TOOLBAR_TOP
        self.toolbar.move(x, y)
        self.toolbar.raise_()
        self._position_preview()

    def _position_preview(self) -> None:
        preview = self.translation_preview
        if not preview.is_active():
            return
        # Clamp the card into the panel first (long results scroll inside it,
        # keeping 替换/关闭 clickable), then anchor it under the toolbar.
        preview.fit_within(
            self.width() - 2 * TOOLBAR_MARGIN, self.height() - 2 * TOOLBAR_GAP
        )
        # Below the toolbar, centered on it, clamped inside the panel.
        x = self.toolbar.x() + (self.toolbar.width() - preview.width()) // 2
        x = max(TOOLBAR_MARGIN, min(x, self.width() - preview.width() - TOOLBAR_MARGIN))
        y = self.toolbar.y() + self.toolbar.height() + TOOLBAR_GAP
        y = max(TOOLBAR_GAP, min(y, self.height() - preview.height() - TOOLBAR_GAP))
        preview.move(x, y)
        preview.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._position_toolbar()
        self.caption_bar.reposition_overlay()

    def _on_toolbar_split(self) -> None:
        editor = self.focus_editor()
        if editor is not None:
            editor.toolbar_split()

    def _on_toolbar_insert(self) -> None:
        editor = self.focus_editor()
        if editor is not None:
            editor.toolbar_insert()

    def _on_toolbar_delete(self) -> None:
        editor = self.focus_editor()
        if editor is not None:
            editor.toolbar_delete()

    # -- translation ---------------------------------------------------------------------------
    def _bridge_configured(self) -> bool:
        bridge = self._bridge
        if bridge is None:
            return False
        configured = getattr(bridge, "configured", None)
        if configured is None:
            return True
        try:
            return bool(configured())
        except Exception:
            _LOGGER.exception("translate bridge configured() check failed")
            return False

    def _on_translate(self, *, fresh: bool) -> None:
        editor = self.focus_editor()
        if editor is None:
            return
        if not self._bridge_configured():
            self._controller.toast_requested.emit(TOAST_TRANSLATE_UNCONFIGURED, _KIND_WARN)
            return
        text = editor.active_text().strip()
        if not text:
            return
        self._pending_translate = (
            editor.key,
            text,
            id(editor),
            editor.edit_index(),
            editor.insert_pos(),
            editor.selected_index(),
        )
        self._bridge.request(editor.key, text, fresh=fresh)  # type: ignore[union-attr]

    def _pending_target_editor(self) -> SegmentEditorBase | None:
        """The focus editor IF it still matches the pending request target."""
        pending = self._pending_translate
        if pending is None:
            return None
        _key, _src, eid, edit_idx, insert_pos, sel_idx = pending
        editor = self.focus_editor()
        if (
            editor is None
            or id(editor) != eid
            or editor.edit_index() != edit_idx
            or editor.insert_pos() != insert_pos
            or editor.selected_index() != sel_idx
        ):
            return None
        return editor

    def _on_segment_ready(self, key: str, source: str, result: str, ok: bool) -> None:
        pending = self._pending_translate
        if pending is None or (key, source) != (pending[0], pending[1]):
            return
        # Only surface the reply while its exact request target is still edited.
        if self._pending_target_editor() is None:
            return
        if not ok:
            self._pending_translate = None
            self._controller.toast_requested.emit(
                TOAST_TRANSLATE_FAILED.format(message=result), _KIND_WARN
            )
            return
        # View FIRST: show the translation and let the user decide (替换/关闭).
        self.translation_preview.show_text(result)
        self._position_preview()

    def _apply_translation(self) -> None:
        """替换: write the previewed translation into the editor or selection."""
        editor = self._pending_target_editor()
        text = self.translation_preview.current_text()
        self._pending_translate = None
        self.translation_preview.dismiss()
        if editor is not None and text:
            editor.apply_translation(text)

    def _dismiss_translation(self) -> None:
        """关闭: keep the original text (view-only)."""
        self._pending_translate = None
        self.translation_preview.dismiss()
