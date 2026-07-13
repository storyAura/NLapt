"""Tests for the reworked 设置 dialog (spec module 3): tabs, 统一模型,
并发设置, 获取模型 and the 提示词 / 本地推理 tabs."""

from __future__ import annotations

from typing import Iterator

import pytest

from nlapt.app import NLaptApp
from nlapt.core.config import (
    AppConfig,
    LLMProfile,
    RequestControl,
    load_config,
    save_config,
)
from nlapt.core.errors import LLMRequestError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.controller import AppController
from nlapt_gui.prompt_store import load_vision_prompts
from nlapt_gui.settings import UISettings
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.model_picker import VISION_TAG, ModelPickerDialog
from nlapt_gui.widgets.settings_dialog import (
    TAB_LLM,
    TAB_LOCAL,
    TAB_PROMPTS,
    TAB_TRANSLATE,
    SettingsDialog,
    config_path,
)

API_TYPE = "gui-settings-tabs-mock"
register_client(API_TYPE, lambda profile: MockLLMClient(["ok"]))


@pytest.fixture()
def dlg_controller(qtbot) -> Iterator[AppController]:
    yield AppController(NLaptApp(), settings=UISettings())


def _make_dialog(qtbot, controller: AppController) -> SettingsDialog:
    dialog = SettingsDialog(controller, api_types=(API_TYPE, "openai"))
    qtbot.addWidget(dialog)
    dialog.base_url.setText("http://mock.local")
    dialog.text_model.setText("text-model-1")
    return dialog


class TestTabs:
    def test_four_tabs_in_order(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        labels = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert labels == [TAB_TRANSLATE, TAB_LLM, TAB_PROMPTS, TAB_LOCAL]

    def test_is_a_centered_fading_dialog(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        assert isinstance(dialog, CenteredDialog)

    def test_local_tab_is_placeholder(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        assert not dialog.local_download_button.isEnabled()


class TestUnifiedModel:
    def test_unified_hides_vision_row_and_relabels(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.show()
        dialog.tabs.setCurrentIndex(1)
        assert not dialog.is_unified()
        dialog.unified_check.setChecked(True)
        assert dialog._text_model_label.text() == "模型"
        assert not dialog.vision_model.isVisibleTo(dialog)
        assert "多模态" in dialog.model_hint.text()
        dialog.unified_check.setChecked(False)
        assert dialog._text_model_label.text() == "文本模型"
        assert dialog.vision_model.isVisibleTo(dialog)

    def test_unified_profile_shares_model(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.unified_check.setChecked(True)
        profile = dialog.current_profile()
        assert profile.text_model == profile.vision_model == "text-model-1"

    def test_unified_save_persists_shared_model(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.api_type.setCurrentIndex(0)
        dialog.unified_check.setChecked(True)
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        stored = load_config(config_path())
        profile = stored.profiles[0]
        assert profile.text_model == profile.vision_model == "text-model-1"

    def test_prefill_detects_unified(self, qtbot, dlg_controller) -> None:
        profile = LLMProfile(
            name="default",
            api_type=API_TYPE,
            base_url="http://x",
            text_model="shared",
            vision_model="shared",
        )
        save_config(config_path(), AppConfig(profiles=(profile,), active_profile="default"))
        dialog = SettingsDialog(dlg_controller, api_types=(API_TYPE,))
        qtbot.addWidget(dialog)
        assert dialog.unified_check.isChecked()

    def test_prefill_detects_split(self, qtbot, dlg_controller) -> None:
        profile = LLMProfile(
            name="default",
            api_type=API_TYPE,
            base_url="http://x",
            text_model="t1",
            vision_model="v1",
        )
        save_config(config_path(), AppConfig(profiles=(profile,), active_profile="default"))
        dialog = SettingsDialog(dlg_controller, api_types=(API_TYPE,))
        qtbot.addWidget(dialog)
        assert not dialog.unified_check.isChecked()
        assert dialog.vision_model.text() == "v1"


class TestConcurrency:
    def test_defaults_from_config(self, qtbot, dlg_controller) -> None:
        save_config(
            config_path(),
            AppConfig(request=RequestControl(concurrency=9)),
        )
        dialog = SettingsDialog(dlg_controller, api_types=(API_TYPE,))
        qtbot.addWidget(dialog)
        assert dialog.concurrency_spin.value() == 9

    def test_save_persists_concurrency_and_keeps_other_request_fields(
        self, qtbot, dlg_controller
    ) -> None:
        save_config(
            config_path(),
            AppConfig(request=RequestControl(concurrency=4, timeout=77.0, max_retries=5)),
        )
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.api_type.setCurrentIndex(0)
        dialog.concurrency_spin.setValue(12)
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        stored = load_config(config_path())
        assert stored.request.concurrency == 12
        assert stored.request.timeout == 77.0
        assert stored.request.max_retries == 5


class TestFetchModels:
    def test_fetch_opens_picker_with_models(self, qtbot, dlg_controller, monkeypatch) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        monkeypatch.setattr(
            "nlapt_gui.widgets.settings_dialog.list_models",
            lambda profile: ("model-a", "model-b"),
        )
        captured: list[tuple[str, ...]] = []
        monkeypatch.setattr(
            dialog, "_open_model_picker", lambda models: captured.append(models)
        )
        dialog.fetch_models()
        qtbot.waitUntil(lambda: bool(captured), timeout=2000)
        assert captured[0] == ("model-a", "model-b")
        assert dialog.fetch_models_button.isEnabled()

    def test_fetch_error_reenables_button(self, qtbot, dlg_controller, monkeypatch) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)

        def boom(profile):  # noqa: ANN001
            raise LLMRequestError("endpoint down")

        monkeypatch.setattr("nlapt_gui.widgets.settings_dialog.list_models", boom)
        collected: list[tuple[str, str]] = []
        dlg_controller.toast_requested.connect(
            lambda text, kind: collected.append((text, kind))
        )
        dialog.fetch_models()
        qtbot.waitUntil(
            lambda: any("获取模型失败" in text for text, _ in collected), timeout=2000
        )
        assert dialog.fetch_models_button.isEnabled()

    def test_fetch_without_base_url_warns(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.base_url.setText("")
        collected: list[tuple[str, str]] = []
        dlg_controller.toast_requested.connect(
            lambda text, kind: collected.append((text, kind))
        )
        dialog.fetch_models()
        assert any("Base URL" in text for text, _ in collected)


class TestModelPickerDialog:
    def test_lists_models_with_vision_tag(self, qtbot) -> None:
        picker = ModelPickerDialog(("gpt-4o", "deepseek-r1"))
        qtbot.addWidget(picker)
        labels = [picker.list.item(i).text() for i in range(picker.list.count())]
        assert f"gpt-4o{VISION_TAG}" in labels
        assert "deepseek-r1" in labels

    def test_pick_text_and_vision(self, qtbot) -> None:
        picker = ModelPickerDialog(("gpt-4o", "deepseek-r1"))
        qtbot.addWidget(picker)
        picked: list[tuple[str, str]] = []
        picker.text_model_picked.connect(lambda m: picked.append(("text", m)))
        picker.vision_model_picked.connect(lambda m: picked.append(("vision", m)))
        picker.list.setCurrentRow(0)
        picker.text_button.click()
        picker.list.setCurrentRow(1)
        picker.vision_button.click()
        assert picked == [("text", "gpt-4o"), ("vision", "deepseek-r1")]

    def test_unified_pick_sets_both(self, qtbot) -> None:
        picker = ModelPickerDialog(("gpt-4o",), unified=True)
        qtbot.addWidget(picker)
        picked: list[tuple[str, str]] = []
        picker.text_model_picked.connect(lambda m: picked.append(("text", m)))
        picker.vision_model_picked.connect(lambda m: picked.append(("vision", m)))
        picker.unified_button.click()
        assert picked == [("text", "gpt-4o"), ("vision", "gpt-4o")]


class TestPromptsIntegration:
    def test_save_persists_prompts_tab_state(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.api_type.setCurrentIndex(0)
        dialog.prompts_tab.user_edit.setPlainText("my user prompt")
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        assert load_vision_prompts().user_prompt == "my user prompt"
