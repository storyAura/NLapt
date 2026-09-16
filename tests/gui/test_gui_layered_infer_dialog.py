"""Tests for the CHA标注 wizard (role slots / three cards per slot / confirm)."""

from __future__ import annotations

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QDialog

from nlapt.core.config import LLMProfile

from nlapt_gui.cha_config import API_MODE_OWN, CHASettings
from nlapt_gui.controller import FOLDER_ROOT_LABEL
from nlapt_gui.layered_prompts import (
    MAX_CARDS,
    WARN_SPLIT_FAILED,
    CardRoster,
    CharacterCard,
    build_card_prompt,
    build_scene_prompt,
    effective_card_template,
    effective_scene_template,
    format_card_stats,
    split_card,
)
from nlapt_gui.layered_store import LayeredMemory, SlotMemory
from nlapt_gui.prompt_store import ENGINE_LLM
from nlapt_gui.widgets.layered_infer_cards import (
    CARD_MODEL_FMT,
    CARDS_SCROLL_HINT,
    PLACEHOLDER_ZH,
    REF_FOLDER_ALL,
    REF_FOLDER_ALL_FMT,
    REF_FOLDER_FMT,
)
from nlapt_gui.tagger_bridge import (
    REL_DETECTED,
    REL_PARENT,
    CharacterCandidate,
    IdentifyResult,
    TaggerBridge,
)
from nlapt_gui.widgets.layered_infer_dialog import (
    DIALOG_MIN_H,
    DIALOG_MIN_W,
    ENGINE_LOCAL_TEXT,
    LABEL_BATCH_MODEL_FMT,
    PAGE_CARDS,
    PAGE_CONFIRM,
    PAGE_ROLE,
    STATUS_NEED_NAME,
    STATUS_TRANSLATING,
    TRANSLATE_FAILED_FMT,
    LayeredInferDialog,
)
from nlapt_gui.widgets.layered_infer_identify import (
    TOAST_NEED_REF,
    IdentifyCoordinator,
)
from nlapt_gui.widgets.layered_infer_slots import (
    BUTTON_IDENTIFY,
    SLOT_TAB_FMT,
    SLOT_TAB_NAMED_FMT,
    TIP_NEED_TAGGER,
)

EMA_MEMORY = LayeredMemory(slots=(SlotMemory(name="ema", series="monosaba"),))


class FakeVision(QObject):
    custom_ready = Signal(str, str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[dict[str, object]] = []
        self.custom_calls: list[tuple[str, str, str]] = []
        self.custom_user_prompts: list[str] = []
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
        self.custom_user_prompts.append(user_prompt)
        self.custom_profiles.append(profile)
        self.custom_ready.emit(request_id, f"CARD {request_id} {user_prompt[-24:]}", True)
        return True

    def request_layered_batch(
        self,
        keys: tuple[str, ...],
        engine: str,
        *,
        roster: CardRoster,
        scene_system: str,
        scene_user: str,
        profile: LLMProfile | None = None,
    ) -> bool:
        self.batch_calls.append(
            {
                "keys": keys,
                "engine": engine,
                "roster": roster,
                "scene_system": scene_system,
                "scene_user": scene_user,
                "profile": profile,
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


class FakeTaggerBridge(QObject):
    identify_ready = Signal(str, object)
    identify_failed = Signal(str, str)

    def __init__(self, result: IdentifyResult | None = None) -> None:
        super().__init__()
        self.result = result if result is not None else IdentifyResult()
        self.calls: list[tuple[str, object]] = []

    def request_identify(self, request_id: str, image_path: object) -> bool:
        self.calls.append((request_id, image_path))
        self.identify_ready.emit(request_id, self.result)
        return True


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
    tagger_bridge: TaggerBridge | FakeTaggerBridge | None = None,
) -> tuple[LayeredInferDialog, FakeVision]:
    bridge = vision if vision is not None else FakeVision()
    dialog = LayeredInferDialog(
        controller,
        bridge,  # type: ignore[arg-type]
        keys,
        translate_bridge=translate or FakeTranslate(),  # type: ignore[arg-type]
        memory=memory if memory is not None else EMA_MEMORY,
        florence=florence,
        cha_settings=cha_settings,
        tagger_bridge=tagger_bridge,  # type: ignore[arg-type]
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
        assert {call[0] for call in vision.custom_calls} == {"card-0-0", "card-0-1", "card-0-2"}
        assert all(call[1] == "0001.png" for call in vision.custom_calls)
        for index, card in enumerate(dialog.cards):
            assert card.english_text().startswith(f"CARD card-0-{index}")
            assert card.chinese.text().startswith("ZH:")

    def test_last_card_memory_does_not_fill_candidates(
        self, qtbot, controller
    ) -> None:
        dialog, _vision = _open(
            qtbot,
            controller,
            memory=LayeredMemory(
                slots=(SlotMemory(name="ema", series="monosaba", card_text="OLD CACHED CARD"),)
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
        roster = call["roster"]
        assert isinstance(roster, CardRoster)
        assert roster.cards == (CharacterCard.from_text("ema", "monosaba", chosen),)
        assert roster.cards[0].parts == split_card(chosen, "ema")
        assert call["scene_user"] == build_scene_prompt(roster)

    def test_multi_slot_flow_builds_roster(self, qtbot, controller) -> None:
        dialog, vision = _open(qtbot, controller)
        assert len(dialog.slots) == 1
        assert not dialog.remove_slot_button.isEnabled()
        assert dialog.role_tabs.tabText(0) == SLOT_TAB_NAMED_FMT.format(n=1, name="ema")
        dialog.add_slot_button.click()
        assert len(dialog.slots) == 2
        assert dialog.role_tabs.currentIndex() == 1
        assert dialog.role_tabs.tabText(1) == SLOT_TAB_FMT.format(n=2)
        assert dialog.remove_slot_button.isEnabled()
        # Second slot has no name yet: generation is refused and the tab stays.
        dialog.next_button.click()
        assert dialog._stack.currentIndex() == PAGE_ROLE
        assert dialog.status_label.text() == STATUS_NEED_NAME
        dialog.name_edit.setText("rin")
        assert dialog.role_tabs.tabText(1) == SLOT_TAB_NAMED_FMT.format(n=2, name="rin")
        # Pick a distinct reference image for slot 2 only.
        cell = dialog.ref_picker.thumb("0002.png")
        qtbot.mouseClick(cell, Qt.MouseButton.LeftButton)
        assert dialog.reference_key() == "0002.png"
        dialog.role_tabs.setCurrentIndex(0)
        assert dialog.reference_key() == "0001.png"
        dialog.next_button.click()
        assert dialog._stack.currentIndex() == PAGE_CARDS
        assert len(vision.custom_calls) == 6
        by_slot = {}
        for request_id, key, _engine in vision.custom_calls:
            by_slot.setdefault(request_id.split("-")[1], set()).add(key)
        assert by_slot == {"0": {"0001.png"}, "1": {"0002.png"}}
        assert dialog.cards_tabs.count() == 2
        dialog.next_button.click()
        assert dialog._stack.currentIndex() == PAGE_CONFIRM
        assert dialog.confirm_tabs.count() == 2
        scene = dialog.confirm_scene.toPlainText()
        assert "Card 1 — name: ema" in scene and "Card 2 — name: rin" in scene
        dialog.next_button.click()
        roster = vision.batch_calls[0]["roster"]
        assert [card.name for card in roster.cards] == ["ema", "rin"]
        assert roster.cards[1].text.startswith("CARD card-1-")

    def test_slot_limit_and_remove(self, qtbot, controller) -> None:
        dialog, vision = _open(qtbot, controller)
        for _ in range(MAX_CARDS + 2):
            dialog.add_slot_button.click()
        assert len(dialog.slots) == MAX_CARDS
        assert not dialog.add_slot_button.isEnabled()
        assert dialog.role_tabs.count() == MAX_CARDS
        dialog.role_tabs.setCurrentIndex(1)
        dialog.remove_slot_button.click()
        assert len(dialog.slots) == MAX_CARDS - 1
        assert dialog.add_slot_button.isEnabled()
        assert dialog.role_tabs.tabText(2) == SLOT_TAB_FMT.format(n=3)
        assert dialog.cards_tabs.count() == MAX_CARDS - 1
        assert dialog.confirm_tabs.count() == MAX_CARDS - 1
        while len(dialog.slots) > 1:
            dialog.remove_slot_button.click()
        assert len(dialog.slots) == 1
        assert not dialog.remove_slot_button.isEnabled()
        assert dialog.name_edit.text() == "ema"

    def test_memory_with_two_slots_prefills_two_tabs(self, qtbot, controller) -> None:
        memory = LayeredMemory(
            slots=(SlotMemory(name="ema", series="monosaba"), SlotMemory(name="rin"))
        )
        dialog, _vision = _open(qtbot, controller, memory=memory)
        assert len(dialog.slots) == 2
        assert dialog.slots[0].role.name() == "ema"
        assert dialog.slots[1].role.name() == "rin"
        assert dialog.role_tabs.currentIndex() == 0

    def test_removed_slot_reply_is_ignored(self, qtbot, controller) -> None:
        dialog, vision = _open(qtbot, controller)
        dialog.add_slot_button.click()
        dialog.name_edit.setText("rin")
        dialog.next_button.click()
        dialog.back_button.click()
        dialog.role_tabs.setCurrentIndex(1)
        dialog.remove_slot_button.click()
        vision.custom_ready.emit("card-1-0", "late reply", True)
        assert dialog._pending == set()
        assert dialog.cards[0].english_text().startswith("CARD card-0-0")

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
        assert vision.custom_calls == [("card-0-2", "0001.png", ENGINE_LLM)]
        assert dialog.cards[2].english_text().startswith("CARD card-0-2")

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

    def test_wide_two_column_first_page(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller)
        assert dialog.minimumWidth() >= DIALOG_MIN_W
        assert dialog.minimumWidth() > dialog.minimumHeight()
        page = dialog.role_page
        assert page.splitter.count() == 2
        assert page.splitter.widget(1) is page.preview
        # The page-1 controls are reachable under their historical names.
        assert dialog.role_tabs is page.role_tabs
        assert dialog.ref_picker is page.ref_picker
        assert dialog.engine_llm.isChecked()
        # Candidate cards sit side by side on page 2.
        cards = dialog.current_slot.cards_page.cards
        dialog.show()
        dialog.resize(DIALOG_MIN_W, DIALOG_MIN_H)
        dialog._stack.setCurrentIndex(PAGE_CARDS)
        qtbot.waitExposed(dialog)
        assert cards[0].geometry().top() == cards[1].geometry().top()
        assert cards[0].geometry().right() < cards[1].geometry().left()

    def test_reference_strip_lists_whole_dataset(self, qtbot, controller) -> None:
        """The batch is a subset, yet every dataset image is a candidate reference."""
        dialog, vision = _open(qtbot, controller)
        assert dialog.ref_picker.keys() == controller.keys()
        assert dialog.ref_picker.current_folder() == REF_FOLDER_ALL
        assert not dialog.ref_picker.folder_combo.isHidden()
        cell = dialog.ref_picker.thumb("10_concept/0003.png")
        assert cell is not None
        qtbot.mouseClick(cell, Qt.MouseButton.LeftButton)
        assert dialog.reference_key() == "10_concept/0003.png"
        dialog.next_button.click()
        assert vision.custom_calls
        assert all(call[1] == "10_concept/0003.png" for call in vision.custom_calls)
        dialog.cards[0].radio.setChecked(True)
        dialog.next_button.click()
        dialog.next_button.click()
        # The written batch is still the keys the wizard was opened with.
        assert vision.batch_calls[0]["keys"] == ("0001.png", "0002.png")

    def test_folder_filter_narrows_strip(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller)
        picker = dialog.ref_picker
        labels = [picker.folder_combo.itemText(i) for i in range(picker.folder_combo.count())]
        assert labels[0] == REF_FOLDER_ALL_FMT.format(n=len(controller.keys()))
        assert REF_FOLDER_FMT.format(folder="10_concept", n=2) in labels
        picker.set_folder("10_concept")
        assert picker.keys() == ("10_concept/0003.png", "10_concept/0004.png")
        assert picker.thumb("0001.png") is None
        # The hidden reference is still the slot's reference.
        assert dialog.reference_key() == "0001.png"
        picker.set_folder(REF_FOLDER_ALL)
        assert picker.keys() == controller.keys()
        assert picker.selected_key() == "0001.png"
        # Highlight survives the rebuild (accent border is the 2px one).
        assert "2px" in picker.thumb("0001.png").styleSheet()
        assert "1px" in picker.thumb("0002.png").styleSheet()

    def test_switching_slot_widens_filter_to_its_folder(self, qtbot, controller) -> None:
        dialog, _vision = _open(qtbot, controller)
        picker = dialog.ref_picker
        dialog.add_slot()
        qtbot.mouseClick(picker.thumb("10_concept/0003.png"), Qt.MouseButton.LeftButton)
        picker.set_folder(FOLDER_ROOT_LABEL)
        assert picker.thumb("10_concept/0003.png") is None
        dialog.role_tabs.setCurrentIndex(0)
        assert dialog.reference_key() == "0001.png"
        assert picker.selected_key() == "0001.png"
        dialog.role_tabs.setCurrentIndex(1)
        assert dialog.reference_key() == "10_concept/0003.png"
        assert picker.current_folder() == "10_concept"
        assert picker.selected_key() == "10_concept/0003.png"

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
        assert translate.requests[0][0] == "card-0-0"
        assert dialog.status_label.text() == STATUS_TRANSLATING
        assert dialog.cards[0].chinese.text() == PLACEHOLDER_ZH
        assert dialog.cards[1].chinese.text() == PLACEHOLDER_ZH
        first_en = dialog.cards[0].english_text()
        translate.target_ready.emit("card-0-0", first_en, "zh", "第一张中文", True)
        assert dialog.cards[0].chinese.text() == "第一张中文"
        assert len(translate.requests) == 2
        assert translate.requests[1][0] == "card-0-1"
        assert dialog.status_label.text() == STATUS_TRANSLATING
        second_en = dialog.cards[1].english_text()
        translate.target_ready.emit("card-0-1", second_en, "zh", "第二张中文", True)
        assert len(translate.requests) == 3
        assert translate.requests[2][0] == "card-0-2"
        third_en = dialog.cards[2].english_text()
        translate.target_ready.emit("card-0-2", third_en, "zh", "第三张中文", True)
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


class TestIdentifyButton:
    def test_disabled_when_tagger_missing(self, qtbot, controller, monkeypatch) -> None:
        monkeypatch.setattr(
            "nlapt_gui.widgets.layered_infer_identify.is_tagger_ready",
            lambda: False,
        )
        dialog, _ = _open(qtbot, controller)
        button = dialog.slots[0].role.identify_button
        assert button.text() == BUTTON_IDENTIFY
        assert button.isEnabled() is False
        assert button.toolTip() == TIP_NEED_TAGGER

    def test_apply_candidate_fills_name_and_series(
        self, qtbot, controller, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "nlapt_gui.widgets.layered_infer_identify.is_tagger_ready",
            lambda: True,
        )
        result = IdentifyResult(
            candidates=(
                CharacterCandidate(
                    tag="hatsune_miku",
                    display_name="Hatsune Miku",
                    series_tag="vocaloid",
                    series_display="Vocaloid",
                    prob=0.89,
                    relation=REL_DETECTED,
                ),
                CharacterCandidate(
                    tag="hatsune_miku_(append)",
                    display_name="Hatsune Miku",
                    series_tag="vocaloid",
                    series_display="Vocaloid",
                    prob=0.0,
                    relation=REL_PARENT,
                ),
            )
        )
        tagger = FakeTaggerBridge(result)
        popped: list[tuple[object, tuple[CharacterCandidate, ...]]] = []
        monkeypatch.setattr(
            IdentifyCoordinator,
            "_popup_menu",
            lambda self, slot, cands: popped.append((slot, cands)),
        )
        dialog, _ = _open(qtbot, controller, tagger_bridge=tagger)
        slot = dialog.slots[0]
        assert slot.role.identify_button.isEnabled()
        slot.role.series_edit.setText("keep-me")
        slot.role.identify_button.click()
        assert tagger.calls and tagger.calls[0][0] == f"identify-{slot.uid}"
        assert popped and popped[0][1] == result.candidates
        IdentifyCoordinator.apply_candidate(slot, result.candidates[0])
        assert slot.role.name() == "Hatsune Miku"
        assert slot.role.series() == "Vocaloid"
        IdentifyCoordinator.apply_candidate(
            slot,
            CharacterCandidate(
                tag="solo_oc",
                display_name="Solo Oc",
                series_tag="",
                series_display="",
                prob=0.4,
                relation=REL_DETECTED,
            ),
        )
        assert slot.role.name() == "Solo Oc"
        assert slot.role.series() == "Vocaloid"

    def test_identify_without_reference_toasts(
        self, qtbot, controller, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "nlapt_gui.widgets.layered_infer_identify.is_tagger_ready",
            lambda: True,
        )
        dialog, _ = _open(qtbot, controller, tagger_bridge=FakeTaggerBridge())
        toasts: list[tuple[str, str]] = []
        controller.toast_requested.connect(lambda text, kind: toasts.append((text, kind)))
        dialog.slots[0].ref_key = ""
        dialog.slots[0].role.identify_button.click()
        assert (TOAST_NEED_REF, "warn") in toasts

    def test_custom_cha_prompts_used_for_card_and_scene(self, qtbot, controller) -> None:
        card_tpl = "CUSTOM CARD {opening} {name}"
        scene_tpl = "CUSTOM SCENE {ROSTER} NAMES={NAMES}"
        dialog, vision = _open(
            qtbot,
            controller,
            cha_settings=CHASettings(card_prompt=card_tpl, scene_prompt=scene_tpl),
        )
        dialog.next_button.click()
        expected0 = build_card_prompt(
            "ema",
            "monosaba",
            variant=0,
            template=effective_card_template(card_tpl),
        )
        assert vision.custom_user_prompts[0] == expected0
        assert all("CUSTOM CARD ema from monosaba ema" in text for text in vision.custom_user_prompts)
        dialog.next_button.click()
        roster = dialog.roster()
        assert roster is not None
        expected_scene = build_scene_prompt(
            roster, template=effective_scene_template(scene_tpl)
        )
        assert dialog.confirm_scene.toPlainText() == expected_scene
        dialog.next_button.click()
        assert vision.batch_calls[0]["scene_user"] == expected_scene
