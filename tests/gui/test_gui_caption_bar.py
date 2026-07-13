"""Tests for nlapt_gui.widgets.caption_bar (标注工作区 翻译/重译/删除)."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

import nlapt_gui.widgets.caption_bar as caption_bar_module
from nlapt_gui.widgets.caption_bar import (
    BAR_REINFER,
    BAR_REINFER_BUSY,
    CONFIRM_DELETE_TITLE,
    LABEL_REINFERRED,
    TOAST_DELETED,
    TOAST_EMPTY_CAPTION,
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
    """Duck-typed VisionBridge (request + caption_ready)."""

    caption_ready = Signal(str, str, bool)

    def __init__(self, *, configured: bool = True) -> None:
        super().__init__()
        self._configured = configured
        self.requests: list[str] = []

    def configured(self) -> bool:
        return self._configured

    def request(self, key: str) -> None:
        self.requests.append(key)


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
    def test_request_flow_and_busy_state(self, qtbot, controller) -> None:
        vision = StubVisionBridge()
        bar = make_bar(qtbot, controller, vision=vision)
        bar.reinfer_btn.click()
        assert vision.requests == [KEY]
        assert bar.reinfer_btn.text() == BAR_REINFER_BUSY
        assert not bar.reinfer_btn.isEnabled()
        vision.caption_ready.emit(KEY, "new caption", True)
        assert bar.reinfer_btn.text() == BAR_REINFER
        assert bar.reinfer_btn.isEnabled()
        assert bar.preview.is_active()
        bar.preview.apply_requested.emit()
        assert controller.record(KEY).text == "new caption"
        assert controller.history.entries(KEY)[0].label == LABEL_REINFERRED

    def test_unconfigured_toasts(self, qtbot, controller, toasts) -> None:
        vision = StubVisionBridge(configured=False)
        bar = make_bar(qtbot, controller, vision=vision)
        bar.reinfer_btn.click()
        assert (TOAST_VISION_UNCONFIGURED, "warn") in toasts
        assert not vision.requests

    def test_failure_resets_button(self, qtbot, controller, toasts) -> None:
        vision = StubVisionBridge()
        bar = make_bar(qtbot, controller, vision=vision)
        bar.reinfer_btn.click()
        vision.caption_ready.emit(KEY, "boom", False)
        assert bar.reinfer_btn.text() == BAR_REINFER
        assert any("重译失败" in text for text, _kind in toasts)


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
