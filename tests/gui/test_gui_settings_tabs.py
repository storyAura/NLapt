"""Tests for the tabbed 设置 dialog (spec module 3): tabs, shared model
across both roles, 并发设置, 获取模型 and the 提示词 / 本地推理 tabs."""

from __future__ import annotations

from typing import Iterator

import pytest
from PySide6.QtCore import Qt

from nlapt.app import NLaptApp
from nlapt.core.config import (
    ROLE_TEXT,
    ROLE_VISION,
    AppConfig,
    LLMProfile,
    ModelRef,
    RequestControl,
    load_config,
    save_config,
)
from nlapt.core.errors import LLMRequestError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.api_config import load_app_config, save_app_config
from nlapt_gui.controller import AppController
from nlapt_gui.prompt_store import load_vision_prompts
from nlapt_gui.settings import UISettings
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.settings_dialog import (
    TAB_CHA,
    TAB_COMPARE,
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
    """Dialog with one profile, one enabled model and the text target set."""
    dialog = SettingsDialog(controller, api_types=(API_TYPE, "openai"))
    qtbot.addWidget(dialog)
    tab = dialog.llm_tab
    tab.add_profile()
    tab.name_edit.setText("main")
    tab.base_url.setText("http://mock.local")
    tab.new_model_edit.setText("text-model-1")
    tab.add_model()
    tab.set_target(ROLE_TEXT, ModelRef("main", "text-model-1"))
    return dialog


class TestTabs:
    def test_six_tabs_in_order(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        labels = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert labels == [
            TAB_TRANSLATE, TAB_LLM, TAB_PROMPTS, TAB_LOCAL, TAB_CHA, TAB_COMPARE
        ]

    def test_compare_tab_follows_pool_and_persists(self, qtbot, dlg_controller) -> None:
        from nlapt_gui.compare_config import load_compare_settings

        dialog = _make_dialog(qtbot, dlg_controller)
        tab = dialog.compare_tab
        assert tab.model_list.checked_refs() == ()
        # Add a second model on the LLM tab -> appears in the compare list live.
        dialog.llm_tab.new_model_edit.setText("vl-2")
        dialog.llm_tab.add_model()
        refs = [
            tab.model_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(tab.model_list.count())
        ]
        assert ModelRef("main", "text-model-1") in refs and ModelRef("main", "vl-2") in refs
        tab.model_list.set_checked((ModelRef("main", "text-model-1"), ModelRef("main", "vl-2")))
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        assert load_compare_settings().models == (
            ModelRef("main", "text-model-1"),
            ModelRef("main", "vl-2"),
        )

    def test_is_a_centered_fading_dialog(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        assert isinstance(dialog, CenteredDialog)

    def test_local_tab_is_real(self, qtbot, dlg_controller) -> None:
        from nlapt.local.catalog import all_series

        from nlapt_gui.widgets.local_tab import LocalTab

        dialog = _make_dialog(qtbot, dlg_controller)
        assert isinstance(dialog.local_tab, LocalTab)
        assert dialog.local_tab.tree.topLevelItemCount() == len(all_series())

    def test_save_persists_local_tab_settings(self, qtbot, dlg_controller) -> None:
        from nlapt.local.settings import load_local_settings

        from nlapt_gui.resources import app_data_dir

        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.local_tab.parallel_spin.setValue(9)
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        assert load_local_settings(app_data_dir() / "local_llm.json").parallel == 9


class TestSharedModel:
    """One multimodal model may back both roles: pick the same ref twice."""

    def test_same_ref_for_both_roles_persists(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.llm_tab.set_target(ROLE_VISION, ModelRef("main", "text-model-1"))
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        stored = load_app_config()
        assert stored.text_target == stored.vision_target == ModelRef("main", "text-model-1")
        assert "text-model-1" not in config_path().read_text(encoding="utf-8")
        assert dlg_controller.text_profile().text_model == "text-model-1"
        assert dlg_controller.vision_profile().vision_model == "text-model-1"

    def test_prefill_shared_legacy_profile(self, qtbot, dlg_controller) -> None:
        profile = LLMProfile(
            name="default",
            api_type=API_TYPE,
            base_url="http://x",
            text_model="shared",
            vision_model="shared",
        )
        save_app_config(AppConfig(profiles=(profile,), active_profile="default"))
        dialog = SettingsDialog(dlg_controller, api_types=(API_TYPE,))
        qtbot.addWidget(dialog)
        tab = dialog.llm_tab
        assert tab.text_target() == tab.vision_target() == ModelRef("default", "shared")
        assert tab.model_list.count() == 1

    def test_prefill_split_legacy_profile(self, qtbot, dlg_controller) -> None:
        profile = LLMProfile(
            name="default",
            api_type=API_TYPE,
            base_url="http://x",
            text_model="t1",
            vision_model="v1",
        )
        save_app_config(AppConfig(profiles=(profile,), active_profile="default"))
        dialog = SettingsDialog(dlg_controller, api_types=(API_TYPE,))
        qtbot.addWidget(dialog)
        tab = dialog.llm_tab
        assert tab.text_target() == ModelRef("default", "t1")
        assert tab.vision_target() == ModelRef("default", "v1")
        assert tab.vision_combo.currentData() == ModelRef("default", "v1")


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
        dialog.concurrency_spin.setValue(12)
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        stored = load_config(config_path())
        assert stored.request.concurrency == 12
        assert stored.request.timeout == 77.0
        assert stored.request.max_retries == 5


class TestFetchModels:
    def test_fetch_merges_into_catalog_unchecked(
        self, qtbot, dlg_controller, monkeypatch
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        monkeypatch.setattr(
            "nlapt_gui.widgets.llm_providers_tab.list_models",
            lambda profile: ("model-a", "text-model-1", "model-b"),
        )
        tab = dialog.llm_tab
        tab.fetch_models()
        qtbot.waitUntil(lambda: tab.model_list.count() == 3, timeout=2000)
        profile = tab.current_profile()
        assert profile.models == ("text-model-1", "model-a", "model-b")
        assert profile.enabled_models == ("text-model-1",)  # fetched ids start off
        assert tab.fetch_models_button.isEnabled()

    def test_fetch_error_reenables_button(self, qtbot, dlg_controller, monkeypatch) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)

        def boom(profile):  # noqa: ANN001
            raise LLMRequestError("endpoint down")

        monkeypatch.setattr("nlapt_gui.widgets.llm_providers_tab.list_models", boom)
        collected: list[tuple[str, str]] = []
        dlg_controller.toast_requested.connect(
            lambda text, kind: collected.append((text, kind))
        )
        dialog.llm_tab.fetch_models()
        qtbot.waitUntil(
            lambda: any("获取模型失败" in text for text, _ in collected), timeout=2000
        )
        assert dialog.llm_tab.fetch_models_button.isEnabled()

    def test_fetch_without_base_url_warns(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.llm_tab.base_url.setText("")
        collected: list[tuple[str, str]] = []
        dlg_controller.toast_requested.connect(
            lambda text, kind: collected.append((text, kind))
        )
        dialog.llm_tab.fetch_models()
        assert any("Base URL" in text for text, _ in collected)


class TestPromptsIntegration:
    def test_save_persists_prompts_tab_state(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.prompts_tab.user_edit.setPlainText("my user prompt")
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        assert load_vision_prompts().user_prompt == "my user prompt"
