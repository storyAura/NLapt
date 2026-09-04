"""Tests for the 分层推标 wizard (role / three cards / confirm)."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDialog

from nlapt_gui.layered_prompts import build_scene_prompt, format_card_stats
from nlapt_gui.layered_store import LayeredMemory
from nlapt_gui.prompt_store import ENGINE_LLM
from nlapt_gui.widgets.layered_infer_dialog import (
    PAGE_CARDS,
    PAGE_CONFIRM,
    PAGE_ROLE,
    LayeredInferDialog,
)


class FakeVision(QObject):
    custom_ready = Signal(str, str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[dict[str, object]] = []
        self.custom_calls: list[tuple[str, str, str]] = []
        self._ready = True

    def configured(self, engine: str = ENGINE_LLM) -> bool:  # noqa: ARG002
        return self._ready

    def request_custom(
        self,
        request_id: str,
        key: str,
        engine: str,
        *,
        system: str,  # noqa: ARG002
        user_prompt: str,
    ) -> bool:
        self.custom_calls.append((request_id, key, engine))
        self.custom_ready.emit(request_id, f"CARD {request_id} {user_prompt[-24:]}", True)
        return True

    def request_layered_batch(
        self,
        keys: tuple[str, ...],
        engine: str,
        *,
        card_text: str,
        scene_system: str,
        scene_user: str,
    ) -> bool:
        self.batch_calls.append(
            {
                "keys": keys,
                "engine": engine,
                "card_text": card_text,
                "scene_system": scene_system,
                "scene_user": scene_user,
            }
        )
        return True


class FakeTranslate(QObject):
    target_ready = Signal(str, str, str, str, bool)

    def request_to(self, key: str, text: str, target_lang: str) -> None:
        self.target_ready.emit(key, text, target_lang, f"ZH:{text[:16]}", True)


def _open(
    qtbot,
    controller,
    *,
    vision: FakeVision | None = None,
    florence: bool = False,
    memory: LayeredMemory | None = None,
    keys: tuple[str, ...] = ("0001.png", "0002.png"),
) -> tuple[LayeredInferDialog, FakeVision]:
    bridge = vision if vision is not None else FakeVision()
    dialog = LayeredInferDialog(
        controller,
        bridge,  # type: ignore[arg-type]
        keys,
        translate_bridge=FakeTranslate(),  # type: ignore[arg-type]
        memory=memory if memory is not None else LayeredMemory(name="ema", series="monosaba"),
        florence=florence,
    )
    qtbot.addWidget(dialog)
    return dialog, bridge


class TestWizard:
    def test_memory_prefills_and_generate_fills_cards(
        self, qtbot, controller
    ) -> None:
        dialog, vision = _open(qtbot, controller)
        assert dialog.name_edit.text() == "ema"
        assert dialog.series_edit.text() == "monosaba"
        assert dialog._stack.currentIndex() == PAGE_ROLE
        dialog.next_button.click()
        assert dialog._stack.currentIndex() == PAGE_CARDS
        assert len(vision.custom_calls) == 3
        assert {call[0] for call in vision.custom_calls} == {"card-0", "card-1", "card-2"}
        assert all(call[1] == "0001.png" for call in vision.custom_calls)
        for index, card in enumerate(dialog.cards):
            assert card.english_text().startswith(f"CARD card-{index}")
            assert card.chinese.text().startswith("ZH:")

    def test_confirm_calls_layered_batch(self, qtbot, controller) -> None:
        dialog, vision = _open(qtbot, controller)
        dialog.next_button.click()
        dialog.cards[1].radio.setChecked(True)
        chosen = dialog.cards[1].english_text()
        dialog.next_button.click()
        assert dialog._stack.currentIndex() == PAGE_CONFIRM
        assert dialog.confirm_card.toPlainText() == chosen
        assert "ema" in dialog.confirm_scene.toPlainText()
        dialog.next_button.click()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert len(vision.batch_calls) == 1
        call = vision.batch_calls[0]
        assert call["keys"] == ("0001.png", "0002.png")
        assert call["engine"] == ENGINE_LLM
        assert call["card_text"] == chosen
        assert call["scene_user"] == build_scene_prompt("ema")

    def test_florence_disables_local(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller, florence=True)
        assert not dialog.engine_local.isEnabled()
        assert not dialog.florence_tip.isHidden()
        assert dialog.current_engine() == ENGINE_LLM

    def test_regen_one_slot(self, qtbot, controller) -> None:
        dialog, vision = _open(qtbot, controller)
        dialog.next_button.click()
        vision.custom_calls.clear()
        dialog.cards[2].regen.click()
        assert vision.custom_calls == [("card-2", "0001.png", ENGINE_LLM)]
        assert dialog.cards[2].english_text().startswith("CARD card-2")

    def test_clicking_thumb_changes_reference_key(
        self, qtbot, controller
    ) -> None:
        dialog, vision = _open(qtbot, controller)
        assert dialog.reference_key() == "0001.png"
        cell = dialog.ref_picker.thumb("0002.png")
        assert cell is not None
        qtbot.mouseClick(cell, Qt.MouseButton.LeftButton)
        assert dialog.reference_key() == "0002.png"
        dialog.next_button.click()
        assert all(call[1] == "0002.png" for call in vision.custom_calls)

    def test_card_stats_update_with_english_text(
        self, qtbot, controller
    ) -> None:
        dialog, _vision = _open(qtbot, controller)
        dialog.next_button.click()
        card = dialog.cards[0]
        assert "tokens" in card.stats.text()
        assert "词" in card.stats.text()
        card.set_english("alpha beta gamma")
        assert card.stats.text() == format_card_stats("alpha beta gamma")
        dialog.cards[0].radio.setChecked(True)
        dialog.next_button.click()
        assert dialog.confirm_card_stats.text() == format_card_stats("alpha beta gamma")
        assert "tokens" in dialog.confirm_scene_stats.text()
