"""Tests for the 分层推标 wizard (role / three cards / confirm)."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDialog

from nlapt.core.config import LLMProfile

from nlapt_gui.cha_config import API_MODE_OWN, CHASettings
from nlapt_gui.layered_prompts import (
    WARN_SPLIT_FAILED,
    CardParts,
    build_scene_prompt,
    format_card_stats,
    split_card,
)
from nlapt_gui.layered_store import LayeredMemory
from nlapt_gui.prompt_store import ENGINE_LLM
from nlapt_gui.widgets.layered_infer_cards import CARD_MODEL_FMT, PLACEHOLDER_ZH
from nlapt_gui.widgets.layered_infer_cards import CARDS_SCROLL_HINT
from nlapt_gui.widgets.layered_infer_dialog import (
    ENGINE_LOCAL_TEXT,
    LABEL_BATCH_MODEL_FMT,
    PAGE_CARDS,
    PAGE_CONFIRM,
    PAGE_ROLE,
    STATUS_TRANSLATING,
    TRANSLATE_FAILED_FMT,
    LayeredInferDialog,
)


class FakeVision(QObject):
    custom_ready = Signal(str, str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[dict[str, object]] = []
        self.custom_calls: list[tuple[str, str, str]] = []
        self.custom_profiles: list[LLMProfile | None] = []
        self._ready = True

    def configured(
        self, engine: str = ENGINE_LLM, *, profile: LLMProfile | None = None
    ) -> bool:  # noqa: ARG002
        return self._ready

    def request_custom(
        self,
        request_id: str,
        key: str,
        engine: str,
        *,
        system: str,  # noqa: ARG002
        user_prompt: str,
        profile: LLMProfile | None = None,
    ) -> bool:
        self.custom_calls.append((request_id, key, engine))
        self.custom_profiles.append(profile)
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
        profile: LLMProfile | None = None,
        card_parts: CardParts | None = None,
    ) -> bool:
        self.batch_calls.append(
            {
                "keys": keys,
                "engine": engine,
                "card_text": card_text,
                "scene_system": scene_system,
                "scene_user": scene_user,
                "profile": profile,
                "card_parts": card_parts,
            }
        )
        return True


class FakeTranslate(QObject):
    target_ready = Signal(str, str, str, str, bool)

    def request_to(self, key: str, text: str, target_lang: str) -> None:
        self.target_ready.emit(key, text, target_lang, f"ZH:{text[:16]}", True)


class RecordingTranslate(QObject):
    target_ready = Signal(str, str, str, str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, str, str]] = []

    def request_to(self, key: str, text: str, target_lang: str) -> None:
        self.requests.append((key, text, target_lang))


class FailingTranslate(QObject):
    target_ready = Signal(str, str, str, str, bool)

    def request_to(self, key: str, text: str, target_lang: str) -> None:
        self.target_ready.emit(key, text, target_lang, "Google 翻译返回 HTTP 429", False)


def _open(
    qtbot,
    controller,
    *,
    vision: FakeVision | None = None,
    florence: bool = False,
    memory: LayeredMemory | None = None,
    keys: tuple[str, ...] = ("0001.png", "0002.png"),
    translate: QObject | None = None,
    cha_settings: CHASettings | None = None,
) -> tuple[LayeredInferDialog, FakeVision]:
    bridge = vision if vision is not None else FakeVision()
    dialog = LayeredInferDialog(
        controller,
        bridge,  # type: ignore[arg-type]
        keys,
        translate_bridge=translate or FakeTranslate(),  # type: ignore[arg-type]
        memory=memory if memory is not None else LayeredMemory(name="ema", series="monosaba"),
        florence=florence,
        cha_settings=cha_settings,
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

    def test_last_card_memory_does_not_fill_candidates(
        self, qtbot, controller
    ) -> None:
        dialog, _vision = _open(
            qtbot,
            controller,
            memory=LayeredMemory(
                name="ema", series="monosaba", card_text="OLD CACHED CARD"
            ),
        )
        assert dialog.name_edit.text() == "ema"
        assert dialog.series_edit.text() == "monosaba"
        assert dialog.cards[0].english_text() == ""
        assert dialog.cards[0].chinese.text() == PLACEHOLDER_ZH
        dialog.next_button.click()
        assert "OLD CACHED CARD" not in dialog.cards[0].english_text()

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
        parts = split_card(chosen, "ema")
        assert call["card_parts"] == parts
        assert call["scene_user"] == build_scene_prompt("ema", official_outfit=parts.outfit)

    def test_confirm_page_shows_card_split(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller)
        dialog.next_button.click()
        dialog.cards[0].radio.setChecked(True)
        dialog.cards[0].set_english(
            "ema, black hair and red eyes. She wears a red dress and black boots."
        )
        dialog.next_button.click()
        assert dialog.confirm_appearance.toPlainText() == "ema, black hair and red eyes."
        assert dialog.confirm_outfit.toPlainText() == "She wears a red dress and black boots."
        assert dialog.confirm_split_warning.isHidden()
        assert "She wears a red dress and black boots." in dialog.confirm_scene.toPlainText()

    def test_confirm_page_warns_when_no_outfit_block(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller)
        dialog.next_button.click()
        dialog.cards[0].radio.setChecked(True)
        dialog.cards[0].set_english("ema, black hair and red eyes.")
        dialog.next_button.click()
        assert dialog.confirm_outfit.toPlainText() == ""
        assert not dialog.confirm_split_warning.isHidden()
        assert dialog.confirm_split_warning.text() == WARN_SPLIT_FAILED

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

    def test_translate_failure_shows_message(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller, translate=FailingTranslate())
        dialog.next_button.click()
        expected = TRANSLATE_FAILED_FMT.format(message="Google 翻译返回 HTTP 429")
        for card in dialog.cards:
            assert card.chinese.text() == expected
            assert PLACEHOLDER_ZH not in card.chinese.text()

    def test_zh_queue_is_serial(self, qtbot, controller) -> None:
        translate = RecordingTranslate()
        dialog, _vision = _open(qtbot, controller, translate=translate)
        dialog.next_button.click()
        assert len(translate.requests) == 1
        assert translate.requests[0][0] == "card-0"
        assert dialog.status_label.text() == STATUS_TRANSLATING
        assert dialog.cards[0].chinese.text() == PLACEHOLDER_ZH
        assert dialog.cards[1].chinese.text() == PLACEHOLDER_ZH
        first_en = dialog.cards[0].english_text()
        translate.target_ready.emit("card-0", first_en, "zh", "第一张中文", True)
        assert dialog.cards[0].chinese.text() == "第一张中文"
        assert len(translate.requests) == 2
        assert translate.requests[1][0] == "card-1"
        assert dialog.status_label.text() == STATUS_TRANSLATING
        second_en = dialog.cards[1].english_text()
        translate.target_ready.emit("card-1", second_en, "zh", "第二张中文", True)
        assert len(translate.requests) == 3
        assert translate.requests[2][0] == "card-2"
        third_en = dialog.cards[2].english_text()
        translate.target_ready.emit("card-2", third_en, "zh", "第三张中文", True)
        assert dialog.cards[2].chinese.text() == "第三张中文"
        assert dialog.status_label.text() != STATUS_TRANSLATING

    def test_long_chinese_does_not_grow_dialog(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller)
        dialog.next_button.click()
        hint_before = dialog.cards_scroll.sizeHint()
        assert hint_before.height() == CARDS_SCROLL_HINT.height()
        blob = "对照文字很长。" * 200
        for card in dialog.cards:
            card.set_chinese(blob)
        dialog.adjustSize()
        assert dialog.cards_scroll.sizeHint() == hint_before
        assert dialog.height() < 900
        assert blob[:12] in dialog.cards[0].chinese.text()

    def test_per_card_models_label_and_batch_profile(self, qtbot, controller) -> None:
        settings = CHASettings(
            api_mode=API_MODE_OWN,
            api_type="openai",
            base_url="http://cha.local",
            card_models=("alpha-vis", "beta-vis", "gamma-vis"),
            batch_model="batch-vis",
        )
        dialog, vision = _open(qtbot, controller, cha_settings=settings)
        dialog.next_button.click()
        models = [
            profile.vision_model if profile is not None else None
            for profile in vision.custom_profiles
        ]
        assert models == ["alpha-vis", "beta-vis", "gamma-vis"]
        assert dialog.cards[0].model_label.text() == CARD_MODEL_FMT.format(
            model="alpha-vis"
        )
        assert dialog.cards[1].model_label.text() == CARD_MODEL_FMT.format(
            model="beta-vis"
        )
        dialog.next_button.click()
        assert dialog.confirm_model.text() == LABEL_BATCH_MODEL_FMT.format(
            model="batch-vis"
        )
        dialog.next_button.click()
        profile = vision.batch_calls[0]["profile"]
        assert isinstance(profile, LLMProfile)
        assert profile.vision_model == "batch-vis"

    def test_pool_ref_card_runs_on_its_own_api(self, qtbot, controller) -> None:
        from nlapt.core.config import AppConfig, ModelRef

        main = LLMProfile(
            name="main",
            api_type="openai",
            base_url="http://main",
            models=("main-vis",),
            enabled_models=("main-vis",),
        )
        other = LLMProfile(
            name="other",
            api_type="openai",
            base_url="http://other",
            api_key="sk-other",
            models=("llava",),
            enabled_models=("llava",),
        )
        controller.reload_config(
            AppConfig(
                profiles=(main, other),
                text_target=ModelRef("main", "main-vis"),
                vision_target=ModelRef("main", "main-vis"),
            )
        )
        settings = CHASettings(
            card_models=(ModelRef("other", "llava"), "typed-vis", ModelRef()),
            batch_model=ModelRef("other", "llava"),
        )
        dialog, vision = _open(qtbot, controller, cha_settings=settings)
        dialog.next_button.click()
        profiles = vision.custom_profiles
        assert [p.base_url for p in profiles] == ["http://other", "http://main", "http://main"]
        assert [p.vision_model for p in profiles] == ["llava", "typed-vis", "main-vis"]
        assert profiles[0].api_key == "sk-other"
        # Pool picks on another API are labelled with their provider.
        assert dialog.cards[0].model_label.text() == CARD_MODEL_FMT.format(
            model="other · llava"
        )
        assert dialog.cards[1].model_label.text() == CARD_MODEL_FMT.format(model="typed-vis")
        dialog.next_button.click()
        assert dialog.confirm_model.text() == LABEL_BATCH_MODEL_FMT.format(
            model="other · llava"
        )
        dialog.next_button.click()
        batch_profile = vision.batch_calls[0]["profile"]
        assert isinstance(batch_profile, LLMProfile)
        assert batch_profile.base_url == "http://other"

    def test_local_engine_shows_local_model_label(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller)
        dialog.engine_local.setChecked(True)
        dialog.next_button.click()
        assert dialog.cards[0].model_label.text() == CARD_MODEL_FMT.format(
            model=ENGINE_LOCAL_TEXT
        )
