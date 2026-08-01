"""标注工作区 - caption-level 翻译 / LLM 推理 / 本地推理 / 删除 + preview.

Lives in the editor panel's tab row (right of the 胶囊/分句/文本 mode tabs)
and operates on the WHOLE caption of the current file:

- 翻译: translate the caption into 中文 / English / 日本語 (menu). The
  provider (LLM or web translation API) follows 设置 ▸ 翻译服务.
- LLM 推理 / 本地推理: re-infer the image with the cloud vision model or
  the 设置 ▸ 本地推理 model, producing a fresh caption (prompts come from
  设置 ▸ 提示词 — one unified pipeline, with an optional local override).
- 删除: clear the file's caption after a centered confirm dialog.

Results are never written silently: they appear in a floating
:class:`TranslationPreview` (view first) and the user picks 替换 / 关闭.
The preview class lives here (not in editor_panel) so both the segment
toolbar and this bar can use it without a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger
from nlapt.llm.translate import LANG_LABELS, TARGET_LANGS

from nlapt_gui.prompt_store import ENGINE_LLM, ENGINE_LOCAL
from nlapt_gui.widgets.dialogs import ask_confirm

if TYPE_CHECKING:
    from nlapt_gui.controller import AppController

_LOGGER = get_logger(__name__)

# -- exact UI strings ---------------------------------------------------------------
BAR_TRANSLATE = "翻译"
BAR_INFER_LLM = "LLM 推理"
BAR_INFER_LOCAL = "本地推理"
BAR_INFER_BUSY = "推理中…"
BAR_DELETE = "删除"
BAR_TRANSLATE_TIP = "把整篇标注翻译为 中文 / English / 日本語"
BAR_INFER_LLM_TIP = "用在线视觉模型重新推理这张图片,生成新标注"
BAR_INFER_LOCAL_TIP = "用本地模型(设置 ▸ 本地推理)推理这张图片,生成新标注"
BAR_DELETE_TIP = "删除这张图片的全部标注"
TOAST_TRANSLATE_UNCONFIGURED = "未配置翻译 API — 打开 工具 ▸ 设置"
TOAST_VISION_UNCONFIGURED = "未配置视觉模型 — 打开 工具 ▸ 设置"
TOAST_LOCAL_UNCONFIGURED = "本地模型未就绪 — 打开 设置 ▸ 本地推理"
TOAST_EMPTY_CAPTION = "当前图片没有标注内容"
TOAST_TRANSLATE_FAILED = "翻译失败: {message}"
TOAST_REINFER_FAILED = "推理失败: {message}"
TOAST_DELETED = "已删除标注(可用 Ctrl+Z 撤销)"
CONFIRM_DELETE_TITLE = "删除标注"
CONFIRM_DELETE_TEXT = "确定删除「{name}」的全部标注内容?\n删除后可用 Ctrl+Z 撤销。"
LABEL_TRANSLATED = "翻译为{lang}"
LABEL_INFERRED_LLM = "推理(LLM)"
LABEL_INFERRED_LOCAL = "推理(本地)"
PREVIEW_TITLE_TRANSLATE = "{name} · 译文({lang})"
PREVIEW_TITLE_REINFER = "{name} · 推理结果"

# Per-engine UI wiring (button text, history label, unconfigured toast).
_ENGINE_TEXTS: dict[str, str] = {ENGINE_LLM: BAR_INFER_LLM, ENGINE_LOCAL: BAR_INFER_LOCAL}
_ENGINE_LABELS: dict[str, str] = {
    ENGINE_LLM: LABEL_INFERRED_LLM,
    ENGINE_LOCAL: LABEL_INFERRED_LOCAL,
}
_ENGINE_UNCONFIGURED: dict[str, str] = {
    ENGINE_LLM: TOAST_VISION_UNCONFIGURED,
    ENGINE_LOCAL: TOAST_LOCAL_UNCONFIGURED,
}

# Preview overlay strings/geometry (shared with the segment toolbar preview).
PREVIEW_TITLE = "译文"
PREVIEW_APPLY = "替换"
PREVIEW_CLOSE = "关闭"
PREVIEW_MAX_WIDTH = 560
PREVIEW_MIN_WIDTH = 200
# Gap kept between the card and the host's edges when clamping.
PREVIEW_HOST_MARGIN = 8
# The body never shrinks below this, even in a tiny host (still scrollable).
PREVIEW_MIN_BODY_H = 40
_PREVIEW_BTN_H = 24

_KIND_OK = "ok"
_KIND_WARN = "warn"

# Vertical offset of the caption preview inside the overlay host (matches the
# segment toolbar's TOOLBAR_TOP so overlays share one visual anchor line).
DEFAULT_OVERLAY_TOP = 44

_BAR_BTN_QSS = "font-size: 12px; font-weight: 600; padding: 0 11px; border-radius: 7px;"


class TranslationPreview(QFrame):
    """Floating result preview: view first, replace on demand.

    A 翻译/重译 result is shown here instead of being written anywhere. The
    user explicitly picks 替换 (apply) or 关闭 (view-only dismiss). Buttons
    use ``pressed`` + ``Qt.NoFocus`` so they act before an inline editor
    could lose focus.

    The body lives in a scroll area and the whole card is clamped into its
    host via :meth:`fit_within` — a long vision-inference result must never
    push the 替换/关闭 buttons past the host's bottom edge (they would be
    clipped and unclickable).
    """

    apply_requested = Signal()
    dismiss_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("surfaceCard", "true")
        self.setMaximumWidth(PREVIEW_MAX_WIDTH)
        self._text = ""
        column = QVBoxLayout(self)
        column.setContentsMargins(12, 8, 12, 8)
        column.setSpacing(6)
        self._title = QLabel(PREVIEW_TITLE, self)
        self._title.setProperty("muted", "true")
        self._title.setStyleSheet("font-size: 10.5px; font-weight: 600;")
        column.addWidget(self._title)
        self._body = QLabel("", self)
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body.setStyleSheet("font-size: 12.5px; background: transparent;")
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.body_scroll = QScrollArea(self)
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.body_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # Keep the surfaceCard background visible through the scroll area.
        self.body_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        self.body_scroll.viewport().setAutoFillBackground(False)
        self.body_scroll.setWidget(self._body)
        column.addWidget(self.body_scroll, 1)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addStretch(1)
        self.close_btn = QPushButton(PREVIEW_CLOSE, self)
        self.close_btn.setProperty("variant", "ghost")
        self.close_btn.setFixedHeight(24)
        self.close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_btn.pressed.connect(self.dismiss_requested.emit)
        buttons.addWidget(self.close_btn)
        self.apply_btn = QPushButton(PREVIEW_APPLY, self)
        self.apply_btn.setProperty("variant", "accent")
        self.apply_btn.setFixedHeight(24)
        self.apply_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.apply_btn.pressed.connect(self.apply_requested.emit)
        buttons.addWidget(self.apply_btn)
        column.addLayout(buttons)
        self.hide()

    def show_text(self, text: str, *, title: str = PREVIEW_TITLE) -> None:
        self._text = text
        self._title.setText(title)
        self._body.setText(text)
        self.adjustSize()
        self.show()
        self.raise_()

    def fit_within(self, max_width: int, max_height: int) -> None:
        """Clamp the card to the given area; the body scrolls, chrome stays.

        Deterministic manual sizing (no QScrollArea size-hint heuristics):
        the body height comes from the word-wrapped label's heightForWidth
        at the actual inner width, then gets capped by what the host can
        show below the title/buttons chrome.
        """
        width = max(PREVIEW_MIN_WIDTH, min(PREVIEW_MAX_WIDTH, max_width))
        self.setFixedWidth(width)
        margins = self.layout().contentsMargins()
        spacing = self.layout().spacing()
        inner_w = width - margins.left() - margins.right()
        body_h = max(1, self._body.heightForWidth(inner_w))
        chrome = (
            margins.top()
            + margins.bottom()
            + 2 * spacing
            + self._title.sizeHint().height()
            + _PREVIEW_BTN_H
        )
        available_body = max(PREVIEW_MIN_BODY_H, max_height - chrome)
        self.body_scroll.setFixedHeight(min(body_h, available_body))
        self.adjustSize()

    def current_text(self) -> str:
        return self._text

    def is_active(self) -> bool:
        return self.isVisible()

    def dismiss(self) -> None:
        self._text = ""
        self.hide()


class CaptionBar(QWidget):
    """The 翻译 / 重译 / 删除 caption workspace in the editor tab row."""

    def __init__(
        self,
        controller: "AppController",
        *,
        translate_bridge: object | None = None,
        vision_bridge: object | None = None,
        overlay_host: QWidget | None = None,
        overlay_top: int = DEFAULT_OVERLAY_TOP,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._translate_bridge = translate_bridge
        self._vision_bridge = vision_bridge
        self._overlay_top = overlay_top
        # (key, source_text, lang) of the in-flight caption translation.
        self._pending_translate: tuple[str, str, str] | None = None
        # (key, engine) of the in-flight vision inference.
        self._pending_infer: tuple[str, str] | None = None
        # (key, text, label) behind the preview's 替换 button.
        self._preview_payload: tuple[str, str, str] | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        divider = QFrame(self)
        divider.setProperty("divider", "true")
        divider.setFixedSize(1, 16)
        row.addWidget(divider)
        self.translate_btn = self._button(BAR_TRANSLATE + " ▾", BAR_TRANSLATE_TIP)
        self.translate_btn.clicked.connect(self._open_translate_menu)
        row.addWidget(self.translate_btn)
        self.reinfer_btn = self._button(BAR_INFER_LLM, BAR_INFER_LLM_TIP)
        self.reinfer_btn.clicked.connect(lambda: self._on_infer(ENGINE_LLM))
        row.addWidget(self.reinfer_btn)
        self.local_infer_btn = self._button(BAR_INFER_LOCAL, BAR_INFER_LOCAL_TIP)
        self.local_infer_btn.clicked.connect(lambda: self._on_infer(ENGINE_LOCAL))
        row.addWidget(self.local_infer_btn)
        self.delete_btn = self._button(BAR_DELETE, BAR_DELETE_TIP, danger=True)
        self.delete_btn.clicked.connect(self._on_delete)
        row.addWidget(self.delete_btn)

        # Result preview overlay (view first, 替换 on demand).
        host = overlay_host if overlay_host is not None else self
        self.preview = TranslationPreview(host)
        self.preview.apply_requested.connect(self._apply_preview)
        self.preview.dismiss_requested.connect(self._dismiss_preview)

        if translate_bridge is not None and hasattr(translate_bridge, "target_ready"):
            translate_bridge.target_ready.connect(self._on_target_ready)
        if vision_bridge is not None and hasattr(vision_bridge, "caption_ready"):
            vision_bridge.caption_ready.connect(self._on_caption_ready)

        controller.current_changed.connect(lambda _key: self._sync_enabled())
        controller.dataset_opened.connect(lambda _result: self._sync_enabled())
        self._sync_enabled()

    # -- construction ------------------------------------------------------------------
    def _button(self, text: str, tooltip: str, *, danger: bool = False) -> QPushButton:
        button = QPushButton(text, self)
        button.setProperty("variant", "danger-ghost" if danger else "ghost")
        button.setStyleSheet(_BAR_BTN_QSS)
        button.setFixedHeight(29)
        button.setToolTip(tooltip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _sync_enabled(self) -> None:
        has_current = self._controller.current_key is not None
        idle = self._pending_infer is None
        self.translate_btn.setEnabled(has_current)
        self.reinfer_btn.setEnabled(has_current and idle)
        self.local_infer_btn.setEnabled(has_current and idle)
        self.delete_btn.setEnabled(has_current)

    def _infer_button(self, engine: str) -> QPushButton:
        return self.local_infer_btn if engine == ENGINE_LOCAL else self.reinfer_btn

    # -- overlay positioning (driven by the host's resize) ------------------------------
    def reposition_overlay(self) -> None:
        host = self.preview.parentWidget()
        if host is None:
            return
        # Clamp the card into the host so 替换/关闭 always stay clickable —
        # a long inference result scrolls inside the card instead of pushing
        # the buttons past the host's bottom edge.
        self.preview.fit_within(
            host.width() - 2 * PREVIEW_HOST_MARGIN,
            host.height() - self._overlay_top - PREVIEW_HOST_MARGIN,
        )
        x = max(0, (host.width() - self.preview.width()) // 2)
        self.preview.move(x, self._overlay_top)
        self.preview.raise_()

    # -- 翻译 ----------------------------------------------------------------------------
    def _open_translate_menu(self) -> None:
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        for lang in TARGET_LANGS:
            action = menu.addAction(LANG_LABELS[lang])
            action.triggered.connect(
                lambda _checked=False, target=lang: self.request_translate(target)
            )
        menu.popup(self.translate_btn.mapToGlobal(QPoint(0, self.translate_btn.height() + 4)))

    def _bridge_configured(self) -> bool:
        bridge = self._translate_bridge
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

    def request_translate(self, lang: str) -> None:
        """Translate the current caption into ``lang`` (中/英/日)."""
        key = self._controller.current_key
        if key is None:
            return
        if not self._bridge_configured():
            self._toast(TOAST_TRANSLATE_UNCONFIGURED, _KIND_WARN)
            return
        text = self._controller.record(key).text.strip()
        if not text:
            self._toast(TOAST_EMPTY_CAPTION, _KIND_WARN)
            return
        self._pending_translate = (key, text, lang)
        self._translate_bridge.request_to(key, text, lang)  # type: ignore[union-attr]

    def _on_target_ready(
        self, key: str, source: str, lang: str, result: str, ok: bool
    ) -> None:
        pending = self._pending_translate
        if pending is None or (key, source, lang) != pending:
            return
        self._pending_translate = None
        if not ok:
            self._toast(TOAST_TRANSLATE_FAILED.format(message=result), _KIND_WARN)
            return
        name = key.rsplit("/", 1)[-1]
        lang_label = LANG_LABELS.get(lang, lang)
        self._preview_payload = (key, result, LABEL_TRANSLATED.format(lang=lang_label))
        self.preview.show_text(
            result, title=PREVIEW_TITLE_TRANSLATE.format(name=name, lang=lang_label)
        )
        self.reposition_overlay()

    # -- LLM 推理 / 本地推理 --------------------------------------------------------------
    def _vision_configured(self, engine: str) -> bool:
        bridge = self._vision_bridge
        if bridge is None:
            return False
        configured = getattr(bridge, "configured", None)
        if configured is None:
            return True
        try:
            return bool(configured(engine))
        except TypeError:
            # Older duck-typed bridges take no engine argument.
            try:
                return bool(configured())
            except Exception:
                _LOGGER.exception("vision bridge configured() check failed")
                return False
        except Exception:
            _LOGGER.exception("vision bridge configured() check failed")
            return False

    def _on_infer(self, engine: str) -> None:
        key = self._controller.current_key
        if key is None or self._pending_infer is not None:
            return
        if not self._vision_configured(engine):
            self._toast(_ENGINE_UNCONFIGURED[engine], _KIND_WARN)
            return
        self._pending_infer = (key, engine)
        self._infer_button(engine).setText(BAR_INFER_BUSY)
        self._sync_enabled()
        self._vision_bridge.request(key, engine)  # type: ignore[union-attr]

    def _on_caption_ready(self, key: str, result: str, ok: bool) -> None:
        pending = self._pending_infer
        if pending is None or key != pending[0]:
            return
        _key, engine = pending
        self._pending_infer = None
        self._infer_button(engine).setText(_ENGINE_TEXTS[engine])
        self._sync_enabled()
        if not ok:
            self._toast(TOAST_REINFER_FAILED.format(message=result), _KIND_WARN)
            return
        name = key.rsplit("/", 1)[-1]
        self._preview_payload = (key, result, _ENGINE_LABELS[engine])
        self.preview.show_text(result, title=PREVIEW_TITLE_REINFER.format(name=name))
        self.reposition_overlay()

    # -- 删除 ----------------------------------------------------------------------------
    def _on_delete(self) -> None:
        key = self._controller.current_key
        if key is None:
            return
        if not self._controller.record(key).text.strip():
            self._toast(TOAST_EMPTY_CAPTION, _KIND_WARN)
            return
        name = key.rsplit("/", 1)[-1]
        if not ask_confirm(
            self.window(),
            CONFIRM_DELETE_TITLE,
            CONFIRM_DELETE_TEXT.format(name=name),
        ):
            return
        self.delete_caption(key)

    def delete_caption(self, key: str) -> None:
        """Clear the caption (undoable through the labeled history)."""
        self._controller.set_caption(key, "", CONFIRM_DELETE_TITLE)
        self._toast(TOAST_DELETED, _KIND_OK)

    # -- preview actions ---------------------------------------------------------------
    def _apply_preview(self) -> None:
        payload = self._preview_payload
        self._preview_payload = None
        self.preview.dismiss()
        if payload is None:
            return
        key, text, label = payload
        self._controller.set_caption(key, text, label)

    def _dismiss_preview(self) -> None:
        self._preview_payload = None
        self.preview.dismiss()

    def _toast(self, text: str, kind: str) -> None:
        self._controller.toast_requested.emit(text, kind)
