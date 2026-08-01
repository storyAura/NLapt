"""Tests for nlapt_gui.widgets.caption_bar (翻译 / LLM 推理 / 本地推理 / 删除)."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

import nlapt_gui.widgets.caption_bar as caption_bar_module
from nlapt_gui.prompt_store import ENGINE_LLM, ENGINE_LOCAL
from nlapt_gui.widgets.caption_bar import (
    BAR_INFER_BUSY,
    BAR_INFER_LLM,
    BAR_INFER_LOCAL,
    CONFIRM_DELETE_TITLE,
    LABEL_INFERRED_LLM,
    LABEL_INFERRED_LOCAL,
    TOAST_DELETED,
    TOAST_EMPTY_CAPTION,
    TOAST_LOCAL_UNCONFIGURED,
    TOAST_TRANSLATE_UNCONFIGURED,
    TOAST_VISION_UNCONFIGURED,
    CaptionBar,
)

KEY = "0001.png"


class StubTranslateBridge(QObject):
    """Duck-typed TranslateBridge (request_to + target_ready)."""

    target_ready = Signal(str, str, str, str, bool)

    def __init__(self, *, configured: bool = True) -> None:
        super().__init__()
        self._configured = configured
        self.requests: list[tuple[str, str, str]] = []

    def configured(self) -> bool:
        return self._configured

    def request_to(self, key: str, text: str, target_lang: str) -> None:
        self.requests.append((key, text, target_lang))


class StubVisionBridge(QObject):
    """Duck-typed VisionBridge (request(key, engine) + caption_ready)."""

    caption_ready = Signal(str, str, bool)

    def __init__(self, *, configured: bool = True) -> None:
        super().__init__()
        self._configured = configured
        self.requests: list[tuple[str, str]] = []

    def configured(self, engine: str = ENGINE_LLM) -> bool:
        return self._configured

    def request(self, key: str, engine: str = ENGINE_LLM) -> None:
        self.requests.append((key, engine))


def make_bar(
    qtbot,
    controller,
    *,
    translate: StubTranslateBridge | None = None,
    vision: StubVisionBridge | None = None,
) -> CaptionBar:
    bar = CaptionBar(
        controller,
        translate_bridge=translate,
        vision_bridge=vision,
    )
    qtbot.addWidget(bar)
    bar.resize(400, 60)
    bar.show()  # the preview overlay's visibility depends on a visible host
    return bar


class TestTranslate:
    def test_request_translate_sends_caption(self, qtbot, controller) -> None:
        translate = StubTranslateBridge()
        bar = make_bar(qtbot, controller, translate=translate)
        bar.request_translate("ja")
        assert translate.requests == [(KEY, controller.record(KEY).text, "ja")]

    def test_unconfigured_toasts(self, qtbot, controller, toasts) -> None:
        translate = StubTranslateBridge(configured=False)
        bar = make_bar(qtbot, controller, translate=translate)
        bar.request_translate("ja")
        assert (TOAST_TRANSLATE_UNCONFIGURED, "warn") in toasts
        assert not translate.requests

    def test_empty_caption_toasts(self, qtbot, controller, toasts) -> None:
        controller.set_caption(KEY, "", "clear")
        translate = StubTranslateBridge()
        bar = make_bar(qtbot, controller, translate=translate)
        bar.request_translate("zh")
        assert (TOAST_EMPTY_CAPTION, "warn") in toasts
        assert not translate.requests

    def test_result_previews_then_applies(self, qtbot, controller) -> None:
        translate = StubTranslateBridge()
        bar = make_bar(qtbot, controller, translate=translate)
        source = controller.record(KEY).text
        bar.request_translate("ja")
        translate.target_ready.emit(KEY, source, "ja", "猫の絵", True)
        assert bar.preview.is_active()
        assert bar.preview.current_text() == "猫の絵"
        assert controller.record(KEY).text == source  # nothing written yet
        bar.preview.apply_requested.emit()
        assert controller.record(KEY).text == "猫の絵"
        assert controller.history.entries(KEY)[0].label == "翻译为日本語"
        assert not bar.preview.is_active()

    def test_dismiss_keeps_original(self, qtbot, controller) -> None:
        translate = StubTranslateBridge()
        bar = make_bar(qtbot, controller, translate=translate)
        source = controller.record(KEY).text
        bar.request_translate("en")
        translate.target_ready.emit(KEY, source, "en", "translated", True)
        bar.preview.dismiss_requested.emit()
        assert controller.record(KEY).text == source
        assert not bar.preview.is_active()

    def test_failed_result_toasts(self, qtbot, controller, toasts) -> None:
        translate = StubTranslateBridge()
        bar = make_bar(qtbot, controller, translate=translate)
        source = controller.record(KEY).text
        bar.request_translate("zh")
        translate.target_ready.emit(KEY, source, "zh", "boom", False)
        assert any("翻译失败" in text for text, _kind in toasts)
        assert not bar.preview.is_active()

    def test_stale_reply_ignored(self, qtbot, controller) -> None:
        translate = StubTranslateBridge()
        bar = make_bar(qtbot, controller, translate=translate)
        # A reply for a different (key, text, lang) than the pending request.
        translate.target_ready.emit(KEY, "other text", "zh", "x", True)
        assert not bar.preview.is_active()


class TestReinfer:
    def test_llm_request_flow_and_busy_state(self, qtbot, controller) -> None:
        vision = StubVisionBridge()
        bar = make_bar(qtbot, controller, vision=vision)
        bar.reinfer_btn.click()
        assert vision.requests == [(KEY, ENGINE_LLM)]
        assert bar.reinfer_btn.text() == BAR_INFER_BUSY
        assert not bar.reinfer_btn.isEnabled()
        assert not bar.local_infer_btn.isEnabled()  # one inference at a time
        vision.caption_ready.emit(KEY, "new caption", True)
        assert bar.reinfer_btn.text() == BAR_INFER_LLM
        assert bar.reinfer_btn.isEnabled()
        assert bar.local_infer_btn.isEnabled()
        assert bar.preview.is_active()
        bar.preview.apply_requested.emit()
        assert controller.record(KEY).text == "new caption"
        assert controller.history.entries(KEY)[0].label == LABEL_INFERRED_LLM

    def test_local_request_flow(self, qtbot, controller) -> None:
        vision = StubVisionBridge()
        bar = make_bar(qtbot, controller, vision=vision)
        bar.local_infer_btn.click()
        assert vision.requests == [(KEY, ENGINE_LOCAL)]
        assert bar.local_infer_btn.text() == BAR_INFER_BUSY
        assert bar.reinfer_btn.text() == BAR_INFER_LLM  # only the busy engine changes
        vision.caption_ready.emit(KEY, "local caption", True)
        assert bar.local_infer_btn.text() == BAR_INFER_LOCAL
        bar.preview.apply_requested.emit()
        assert controller.record(KEY).text == "local caption"
        assert controller.history.entries(KEY)[0].label == LABEL_INFERRED_LOCAL

    def test_unconfigured_toasts_per_engine(self, qtbot, controller, toasts) -> None:
        vision = StubVisionBridge(configured=False)
        bar = make_bar(qtbot, controller, vision=vision)
        bar.reinfer_btn.click()
        assert (TOAST_VISION_UNCONFIGURED, "warn") in toasts
        bar.local_infer_btn.click()
        assert (TOAST_LOCAL_UNCONFIGURED, "warn") in toasts
        assert not vision.requests

    def test_failure_resets_button(self, qtbot, controller, toasts) -> None:
        vision = StubVisionBridge()
        bar = make_bar(qtbot, controller, vision=vision)
        bar.reinfer_btn.click()
        vision.caption_ready.emit(KEY, "boom", False)
        assert bar.reinfer_btn.text() == BAR_INFER_LLM
        assert any("推理失败" in text for text, _kind in toasts)


class TestPreviewFitsHost:
    """Regression: a long inference result must not clip 替换/关闭 (user report)."""

    HOST_W = 600
    HOST_H = 300
    LONG_TEXT = "A very long natural-language caption sentence. " * 80

    def make_hosted_bar(self, qtbot, controller, vision) -> tuple[CaptionBar, object]:
        from PySide6.QtWidgets import QWidget

        host = QWidget()
        qtbot.addWidget(host)
        host.resize(self.HOST_W, self.HOST_H)
        host.show()
        bar = CaptionBar(controller, vision_bridge=vision, overlay_host=host)
        qtbot.addWidget(bar)
        return bar, host

    def test_long_result_keeps_buttons_inside_host(self, qtbot, controller) -> None:
        vision = StubVisionBridge()
        bar, host = self.make_hosted_bar(qtbot, controller, vision)
        bar.reinfer_btn.click()
        vision.caption_ready.emit(KEY, self.LONG_TEXT, True)
        assert bar.preview.is_active()
        geo = bar.preview.geometry()
        assert geo.bottom() <= host.height()  # buttons stay clickable
        assert geo.right() <= host.width()
        # The body overflow moved INTO the scroll area, not past the host.
        assert bar.preview.body_scroll.verticalScrollBar().maximum() > 0

    def test_long_result_can_still_be_applied(self, qtbot, controller) -> None:
        vision = StubVisionBridge()
        bar, _host = self.make_hosted_bar(qtbot, controller, vision)
        bar.reinfer_btn.click()
        vision.caption_ready.emit(KEY, self.LONG_TEXT, True)
        bar.preview.apply_btn.pressed.emit()
        assert controller.record(KEY).text == self.LONG_TEXT
        assert not bar.preview.is_active()

    def test_short_result_needs_no_scrollbar(self, qtbot, controller) -> None:
        vision = StubVisionBridge()
        bar, _host = self.make_hosted_bar(qtbot, controller, vision)
        bar.reinfer_btn.click()
        vision.caption_ready.emit(KEY, "short caption", True)
        assert bar.preview.body_scroll.verticalScrollBar().maximum() == 0


class TestDelete:
    def test_confirmed_delete_clears_caption(
        self, qtbot, controller, toasts, monkeypatch
    ) -> None:
        monkeypatch.setattr(caption_bar_module, "ask_confirm", lambda *a, **k: True)
        bar = make_bar(qtbot, controller)
        bar.delete_btn.click()
        assert controller.record(KEY).text == ""
        assert controller.history.entries(KEY)[0].label == CONFIRM_DELETE_TITLE
        assert (TOAST_DELETED, "ok") in toasts

    def test_cancelled_delete_keeps_caption(
        self, qtbot, controller, monkeypatch
    ) -> None:
        monkeypatch.setattr(caption_bar_module, "ask_confirm", lambda *a, **k: False)
        bar = make_bar(qtbot, controller)
        original = controller.record(KEY).text
        bar.delete_btn.click()
        assert controller.record(KEY).text == original

    def test_empty_caption_toasts_instead_of_confirm(
        self, qtbot, controller, toasts, monkeypatch
    ) -> None:
        controller.set_caption(KEY, "", "clear")
        called: list[bool] = []
        monkeypatch.setattr(
            caption_bar_module, "ask_confirm", lambda *a, **k: called.append(True) or True
        )
        bar = make_bar(qtbot, controller)
        bar.delete_btn.click()
        assert not called
        assert (TOAST_EMPTY_CAPTION, "warn") in toasts
