"""翻译对照 section - per-segment / whole-text LLM translation + swap.

Driven by :class:`TranslateBridge` (real LLM through the controller's active
profile). Two compare modes (对照方式 segmented bar):

- 分段 (default): one row per comma segment, the swap button replaces that
  segment. Long natural-language captions read badly here — commas cut
  sentences apart and each shard is translated without context.
- 整段: the WHOLE caption is one row translated as a single unit (matching
  the editor's 文本 view); the swap button replaces the entire caption.

中文全部转为英文 follows the same mode: per CJK segment in 分段, one
whole-text translation per file in 整段. When no LLM profile is configured
the section shows a guided state with a 打开设置 link.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal
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

from nlapt_gui.controller import AppController, TOAST_ERR, TOAST_INFO, TOAST_OK, TOAST_WARN
from nlapt_gui.translate_bridge import NOTE_UNCONFIGURED, TranslateBridge, has_cjk
from nlapt_gui.widgets.tools_panel import (
    ScopeSelector,
    SegmentedBar,
    repolish,
    resolve_tokens,
)

_LOGGER = get_logger(__name__)

# Exact UI strings.
BUTTON_ALL_EN = "中文全部转为英文"
BUTTON_ALL_EN_BUSY = "翻译中 {i}/{n}"
LABEL_SCOPE_CAPTION = "应用范围"
# 对照方式: per-segment rows vs the whole caption as one unit.
MODE_SEGMENTS = "segments"
MODE_WHOLE = "whole"
LABEL_MODE_CAPTION = "对照方式"
LABEL_MODE_SEGMENTS = "分段"
LABEL_MODE_WHOLE = "整段"
TIP_MODE_BAR = (
    "分段:按逗号分段逐条对照翻译;整段:整篇标注作为一个整体翻译,"
    "长句描述更通顺,替换时替换整篇"
)
LINK_OPEN_SETTINGS = "打开设置"
HINT_UNCONFIGURED = "未配置翻译 API"
TOOLTIP_SWAP = "用译文替换原文"
PENDING_NOTE = "翻译中…"
TOAST_SWAPPED = "已替换为译文"
TOAST_NO_CJK = "当前文件没有中文内容"
TOAST_SCOPE_NO_CJK = "所选范围内没有中文内容"
TOAST_ALL_EN_OK = "已将 {n} 个中文片段转为英文"
TOAST_ALL_EN_PARTIAL = "已将 {ok} 个中文片段转为英文,{fail} 个失败"
TOAST_ALL_EN_FAILED = "翻译失败,共 {n} 个片段"
LABEL_SWAP = "翻译替换"
LABEL_ALL_EN = "中文全部转为英文"

_ROWS_MAX_HEIGHT = 218
_ROW_GAP = 6
_ROW_MARGINS = (9, 6, 9, 6)
_SWAP_SIZE = 24
_BUTTON_HEIGHT = 29
_CONTENT_MARGINS = (13, 2, 13, 13)
_CONTENT_GAP = 8
# Debounce for live caption edits so typing does not fire one LLM request per
# keystroke; only the settled text is translated.
_DEBOUNCE_MS = 400


@dataclass
class _Row:
    """Mutable per-row view state (widgets + translation result)."""

    source: str
    frame: QFrame
    dst_label: QLabel
    swap_button: QPushButton
    result: str = ""
    ready: bool = False


class TranslateSection(QWidget):
    """Content widget of the 翻译对照 collapsible card."""

    open_settings_requested = Signal()

    def __init__(
        self,
        controller: AppController,
        *,
        bridge: TranslateBridge | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.bridge = bridge if bridge is not None else TranslateBridge(controller, parent=self)
        self._rows: list[_Row] = []
        self._results: dict[str, str] = {}  # source text -> translated text
        self._whole = False  # 对照方式: False=分段 rows, True=整段 caption
        self._batch_active = False
        self._batch_key: str | None = None
        # Scope batch state (指定范围翻译): remaining keys + aggregate counters.
        self._batch_queue: list[str] = []
        self._batch_total = 0
        self._batch_changed = 0
        self._batch_failed = 0
        # Auto-request only when the 翻译对照 card is expanded; gated by the
        # owning ToolsPanel via :meth:`set_live` (default off = collapsed).
        self._live = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self.refresh)

        self._rows_host = QWidget(self)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(_ROW_GAP)
        self._rows_layout.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_host)
        scroll.setMaximumHeight(_ROWS_MAX_HEIGHT)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.hint_row = QWidget(self)
        hint_layout = QHBoxLayout(self.hint_row)
        hint_layout.setContentsMargins(0, 0, 0, 0)
        hint_layout.setSpacing(6)
        hint_label = QLabel(HINT_UNCONFIGURED, self.hint_row)
        hint_label.setProperty("muted", True)
        hint_label.setStyleSheet("font-size: 11px;")
        self.settings_link = QPushButton(LINK_OPEN_SETTINGS, self.hint_row)
        self.settings_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_link.setStyleSheet(
            "background: transparent; border: none; padding: 0;"
            " color: palette(highlight); font-size: 11px; font-weight: 600;"
        )
        self.settings_link.clicked.connect(self.open_settings_requested)
        hint_layout.addWidget(hint_label)
        hint_layout.addWidget(self.settings_link)
        hint_layout.addStretch(1)

        # 对照方式 row: 分段 / 整段 segmented switch above the rows.
        mode_caption = QLabel(LABEL_MODE_CAPTION, self)
        mode_caption.setProperty("muted", True)
        mode_caption.setStyleSheet("font-size: 10.5px;")
        self.mode_bar = SegmentedBar(
            ((MODE_SEGMENTS, LABEL_MODE_SEGMENTS), (MODE_WHOLE, LABEL_MODE_WHOLE)),
            current=MODE_SEGMENTS,
            parent=self,
        )
        self.mode_bar.setToolTip(TIP_MODE_BAR)
        self.mode_bar.changed.connect(self._on_mode_changed)
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(8)
        mode_row.addWidget(mode_caption)
        mode_row.addWidget(self.mode_bar, 1)

        tokens = resolve_tokens(controller.settings)
        self.all_en_button = QPushButton(BUTTON_ALL_EN, self)
        self.all_en_button.setObjectName("allEnButton")
        self.all_en_button.setFixedHeight(_BUTTON_HEIGHT)
        self.all_en_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.all_en_button.setStyleSheet(
            f"#allEnButton {{ background: palette(base); border: 1px solid {tokens.bd};"
            " border-radius: 8px; font-size: 12px; font-weight: 600; }"
            "#allEnButton:hover { border-color: palette(highlight);"
            " color: palette(highlight); }"
            f"#allEnButton:disabled {{ color: {tokens.text3}; }}"
        )
        self.all_en_button.clicked.connect(self.translate_all_to_english)

        # Scope selector (指定范围翻译): 当前 / 选中 n / 全部 N.
        scope_caption = QLabel(LABEL_SCOPE_CAPTION, self)
        scope_caption.setProperty("muted", True)
        scope_caption.setStyleSheet("font-size: 10.5px;")
        self.scope = ScopeSelector(controller, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*_CONTENT_MARGINS)
        layout.setSpacing(_CONTENT_GAP)
        layout.addLayout(mode_row)
        layout.addWidget(scroll)
        layout.addWidget(self.hint_row)
        layout.addWidget(scope_caption)
        layout.addWidget(self.scope)
        layout.addWidget(self.all_en_button)

        self.bridge.segment_ready.connect(self._on_segment_ready)
        self.bridge.all_done.connect(self._on_all_done)
        controller.current_changed.connect(lambda _key: self.refresh())
        controller.caption_changed.connect(self._on_caption_changed)
        controller.dataset_opened.connect(lambda _result: self.refresh())
        self.refresh()

    # -- behavior ---------------------------------------------------------------------
    def rows(self) -> tuple[_Row, ...]:
        """Current row view-states (for tests and the owning panel)."""
        return tuple(self._rows)

    @property
    def whole_mode(self) -> bool:
        """Whether 整段对照 is active (whole caption as one unit)."""
        return self._whole

    def _on_mode_changed(self, option_id: str) -> None:
        self._whole = option_id == MODE_WHOLE
        self.refresh()

    def _sources_for(self, key: str) -> tuple[str, ...]:
        """The compare units of one file under the active 对照方式."""
        if not self._whole:
            return self._controller.segments(key)
        text = self._controller.record(key).text
        return (text,) if text.strip() else ()

    def set_live(self, live: bool) -> None:
        """Enable/disable auto-translation (driven by the card's open state).

        When the 翻译对照 card is collapsed the section is not live and issues no
        LLM requests; expanding it fetches any missing translations lazily.
        """
        if live == self._live:
            return
        self._live = live
        if live:
            self.refresh()

    def refresh(self) -> None:
        """Rebuild rows for the active 对照方式; request dsts when live."""
        self._clear_rows()
        key = self._controller.current_key
        sources = self._sources_for(key) if key else ()
        configured = self.bridge.configured()
        self.hint_row.setVisible(not configured)
        self.all_en_button.setEnabled(configured and bool(sources))
        for index, source in enumerate(sources):
            row = self._build_row(index, source, configured=configured)
            self._rows.append(row)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row.frame)
        # Only reach the LLM when the card is actually visible (self._live);
        # already-cached sources render from _results without a request.
        if configured and key is not None and self._live:
            for source in sources:
                if source not in self._results:
                    self.bridge.request(key, source)

    def swap_segment(self, index: int) -> None:
        """Replace the row's unit (segment, or整段 the whole caption)."""
        key = self._controller.current_key
        if key is None or not 0 <= index < len(self._rows):
            return
        row = self._rows[index]
        if not row.ready or not row.result:
            return
        if self._whole:
            if self._controller.record(key).text != row.source:
                _LOGGER.warning("caption changed under whole swap; ignoring")
                return
            self._controller.set_caption(key, row.result, LABEL_SWAP)
            self._controller.toast_requested.emit(TOAST_SWAPPED, TOAST_INFO)
            return
        segments = list(self._controller.segments(key))
        if index >= len(segments) or segments[index] != row.source:
            _LOGGER.warning("segments changed under swap; ignoring row %d", index)
            return
        segments[index] = row.result
        self._controller.commit_segments(key, segments, LABEL_SWAP)
        self._controller.toast_requested.emit(TOAST_SWAPPED, TOAST_INFO)

    def translate_all_to_english(self) -> None:
        """Translate every CJK unit of the files in the selected scope.

        指定范围翻译: scope 当前/选中/全部 comes from the section's
        :class:`ScopeSelector`; files are processed one by one (the bridge's
        text-hash cache dedupes repeated tags across files) and each file's
        replacements are committed as its batch completes. Under 整段对照
        the unit is the whole caption (one translation per file).
        """
        if self._batch_active or self._batch_queue:
            return  # a scope batch is already running
        scope_keys = self._controller.scope_keys(self.scope.scope)
        keys = [
            key
            for key in scope_keys
            if any(has_cjk(source) for source in self._sources_for(key))
        ]
        if not keys:
            no_cjk = TOAST_NO_CJK if self.scope.scope == "current" else TOAST_SCOPE_NO_CJK
            self._controller.toast_requested.emit(no_cjk, TOAST_INFO)
            return
        self._batch_queue = keys
        self._batch_total = len(keys)
        self._batch_changed = 0
        self._batch_failed = 0
        self.all_en_button.setEnabled(False)
        # Mode switches mid-batch would commit under the wrong unit shape.
        self.mode_bar.setEnabled(False)
        self._start_next_in_scope()

    def _start_next_in_scope(self) -> None:
        key = self._batch_queue[0]
        self._batch_active = True
        self._batch_key = key
        done = self._batch_total - len(self._batch_queue) + 1
        self.all_en_button.setText(
            BUTTON_ALL_EN_BUSY.format(i=done, n=self._batch_total)
        )
        self.bridge.translate_all_cjk(key, self._sources_for(key))

    # -- internals ----------------------------------------------------------------------
    def _on_caption_changed(self, key: str) -> None:
        # Debounce so live typing in 文本 mode does not issue an LLM request per
        # keystroke; the timer fires refresh() once the caption settles.
        if key == (self._controller.current_key or ""):
            self._debounce.start()

    def _on_segment_ready(self, key: str, source: str, result: str, ok: bool) -> None:
        # Cache every successful result (even for a non-current key) so a batch
        # that finishes after the user navigated away can still be committed.
        if ok:
            self._results[source] = result
        if key != (self._controller.current_key or ""):
            return
        for row in self._rows:
            if row.source != source:
                continue
            row.ready = ok
            row.result = result if ok else ""
            self._style_dst(row.dst_label, result if ok else result or NOTE_UNCONFIGURED, ok=ok)
            row.swap_button.setEnabled(ok)

    def _on_all_done(self, key: str, succeeded: int, failed: int) -> None:
        if not self._batch_active or key != self._batch_key:
            return
        self._batch_active = False
        self._batch_key = None
        # Commit against the batch's OWN key, not the current one: the user may
        # have navigated away while the translations were in flight.
        changed = self._commit_results(key)
        self._batch_changed += changed
        self._batch_failed += failed
        if self._batch_queue and self._batch_queue[0] == key:
            self._batch_queue.pop(0)
        if self._batch_queue:
            self._start_next_in_scope()
            return
        # Scope finished: restore the controls and report the aggregate result.
        self.all_en_button.setText(BUTTON_ALL_EN)
        self.all_en_button.setEnabled(True)
        self.mode_bar.setEnabled(True)
        total_changed = self._batch_changed
        total_failed = self._batch_failed
        if total_failed == 0 and total_changed > 0:
            self._controller.toast_requested.emit(
                TOAST_ALL_EN_OK.format(n=total_changed), TOAST_OK
            )
        elif total_changed > 0:
            self._controller.toast_requested.emit(
                TOAST_ALL_EN_PARTIAL.format(ok=total_changed, fail=total_failed),
                TOAST_WARN,
            )
        else:
            self._controller.toast_requested.emit(
                TOAST_ALL_EN_FAILED.format(n=total_failed), TOAST_ERR
            )

    def _commit_results(self, key: str) -> int:
        """Apply cached translations to one file; returns changed-unit count."""
        if self._whole:
            text = self._controller.record(key).text
            if not has_cjk(text):
                return 0
            replacement = self._results.get(text)
            if replacement is None or replacement == text:
                return 0
            self._controller.set_caption(key, replacement, LABEL_ALL_EN)
            return 1
        segments = self._controller.segments(key)
        replaced = [
            self._results.get(segment, segment) if has_cjk(segment) else segment
            for segment in segments
        ]
        changed = sum(1 for old, new in zip(segments, replaced) if old != new)
        if changed:
            self._controller.commit_segments(key, replaced, LABEL_ALL_EN)
        return changed

    def _build_row(self, index: int, segment: str, *, configured: bool) -> _Row:
        tokens = resolve_tokens(self._controller.settings)
        frame = QFrame(self._rows_host)
        frame.setObjectName("trRow")
        frame.setStyleSheet(
            f"#trRow {{ background: palette(window); border: 1px solid {tokens.bd};"
            " border-radius: 8px; }"
        )
        row_layout = QHBoxLayout(frame)
        row_layout.setContentsMargins(*_ROW_MARGINS)
        row_layout.setSpacing(8)
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)
        src_label = QLabel(segment, frame)
        src_label.setStyleSheet("font-size: 12px; border: none; background: transparent;")
        dst_label = QLabel(frame)
        dst_label.setProperty("mono", True)
        text_col.addWidget(src_label)
        text_col.addWidget(dst_label)
        row_layout.addLayout(text_col, 1)
        swap = QPushButton(frame)
        swap.setText("⇄")
        swap.setToolTip(TOOLTIP_SWAP)
        swap.setFixedSize(_SWAP_SIZE, _SWAP_SIZE)
        swap.setCursor(Qt.CursorShape.PointingHandCursor)
        swap.setObjectName("trSwap")
        swap.setStyleSheet(
            f"#trSwap {{ background: palette(base); border: 1px solid {tokens.bd};"
            " border-radius: 6px; padding: 0; }"
            "#trSwap:hover { border-color: palette(highlight); color: palette(highlight); }"
            f"#trSwap:disabled {{ color: {tokens.text3}; }}"
        )
        swap.setEnabled(False)
        swap.clicked.connect(lambda _=False, i=index: self.swap_segment(i))
        row_layout.addWidget(swap)
        row = _Row(source=segment, frame=frame, dst_label=dst_label, swap_button=swap)
        cached = self._results.get(segment)
        if configured and cached:
            row.ready = True
            row.result = cached
            self._style_dst(dst_label, cached, ok=True)
            swap.setEnabled(True)
        elif configured:
            self._style_dst(dst_label, PENDING_NOTE, ok=False)
        else:
            self._style_dst(dst_label, NOTE_UNCONFIGURED, ok=False)
        return row

    def _style_dst(self, label: QLabel, text: str, *, ok: bool) -> None:
        label.setText(text)
        label.setProperty("muted", not ok)
        base = "font-size: 11px; border: none; background: transparent;"
        if ok:
            label.setStyleSheet(base + " color: palette(highlight);")
        else:
            label.setStyleSheet(base)
        repolish(label)

    def _clear_rows(self) -> None:
        for row in self._rows:
            self._rows_layout.removeWidget(row.frame)
            row.frame.deleteLater()
        self._rows = []
