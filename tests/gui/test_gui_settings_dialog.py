"""Tests for nlapt_gui.widgets.settings_dialog (multi-API LLM tab + save flow)."""

from __future__ import annotations

from typing import Iterator

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit

from nlapt.app import NLaptApp
from nlapt.core.config import ROLE_TEXT, ROLE_VISION, AppConfig, LLMProfile, ModelRef
from nlapt.core.errors import LLMRequestError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.api_config import load_app_config, save_app_config
from nlapt_gui.cha_config import API_MODE_OWN, load_cha_settings
from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings, load_ui_settings
from nlapt_gui.widgets.llm_providers_tab import (
    TOAST_NEED_MODEL,
    TOAST_NEED_PROFILE_URL,
)
from nlapt_gui.widgets.settings_dialog import (
    HINT_DEBUG,
    LABEL_DEBUG,
    TAB_CHA,
    TOAST_NEED_TEXT_MODEL,
    TOAST_SAVED,
    WINDOW_TITLE,
    SettingsDialog,
    config_path,
)
from nlapt_gui.workers import debug_enabled, set_debug

# Unique api_types for this test module.
API_TYPE = "gui-settings-mock"
API_TYPE_FAIL = "gui-settings-mock-fail"

register_client(API_TYPE, lambda profile: MockLLMClient(["ok"]))
register_client(
    API_TYPE_FAIL,
    lambda profile: MockLLMClient(
        ["never"], fail_times=99, failure=LLMRequestError("boom")
    ),
)


@pytest.fixture()
def dlg_controller(qtbot) -> Iterator[AppController]:
    yield AppController(NLaptApp(), settings=UISettings())


@pytest.fixture()
def dlg_toasts(dlg_controller) -> list[tuple[str, str]]:
    collected: list[tuple[str, str]] = []
    dlg_controller.toast_requested.connect(
        lambda text, kind: collected.append((text, kind))
    )
    return collected


def _make_dialog(qtbot, controller: AppController) -> SettingsDialog:
    dialog = SettingsDialog(
        controller, api_types=(API_TYPE, API_TYPE_FAIL, "openai")
    )
    qtbot.addWidget(dialog)
    return dialog


def _fill(dialog: SettingsDialog, *, api_type: str = API_TYPE) -> None:
    """Add one profile named ``default`` with two enabled models and both targets."""
    tab = dialog.llm_tab
    tab.add_profile()
    tab.name_edit.setText("default")
    tab.api_type.setCurrentIndex(tab.api_type.findText(api_type))
    tab.base_url.setText("http://mock.local")
    tab.api_key.setText("sk-test-123456789")
    for model in ("text-model-1", "vision-model-1"):
        tab.new_model_edit.setText(model)
        tab.add_model()
    tab.set_target(ROLE_TEXT, ModelRef("default", "text-model-1"))
    tab.set_target(ROLE_VISION, ModelRef("default", "vision-model-1"))


class TestDialogBasics:
    def test_window_title_and_masked_key(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        assert dialog.windowTitle() == WINDOW_TITLE == "设置"
        assert dialog.llm_tab.api_key.echoMode() == QLineEdit.EchoMode.Password

    def test_draft_config_from_tab(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        config = dialog.draft_config()
        profile = config.profiles[0]
        assert profile.name == "default"
        assert profile.api_type == API_TYPE
        assert profile.base_url == "http://mock.local"
        assert profile.models == ("text-model-1", "vision-model-1")
        assert profile.enabled_models == ("text-model-1", "vision-model-1")
        assert config.text_target == ModelRef("default", "text-model-1")
        assert config.vision_target == ModelRef("default", "vision-model-1")
        assert config.active_profile == ""


class TestValidation:
    def test_save_requires_base_url(self, qtbot, dlg_controller, dlg_toasts) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.llm_tab.add_profile()
        dialog.llm_tab.name_edit.setText("p")
        dialog.save_button.click()
        assert (TOAST_NEED_PROFILE_URL.format(name="p"), "warn") in dlg_toasts
        assert dialog.result() != QDialog.DialogCode.Accepted
        assert not config_path().exists()

    def test_save_requires_text_target_for_llm_provider(
        self, qtbot, dlg_controller, dlg_toasts
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.llm_tab.add_profile()
        dialog.llm_tab.base_url.setText("http://x")
        dialog.save_button.click()
        assert (TOAST_NEED_TEXT_MODEL, "warn") in dlg_toasts
        assert not config_path().exists()


class TestSave:
    def test_save_writes_config_json(self, qtbot, dlg_controller, dlg_toasts) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        assert config_path().exists()
        config = load_app_config()
        assert config.text_target == ModelRef("default", "text-model-1")
        assert config.vision_target == ModelRef("default", "vision-model-1")
        profile = config.profiles[0]
        assert profile.api_type == API_TYPE
        assert profile.base_url == "http://mock.local"
        assert profile.api_key == "sk-test-123456789"
        assert profile.enabled_models == ("text-model-1", "vision-model-1")
        assert "sk-test-123456789" not in config_path().read_text(encoding="utf-8")
        assert (TOAST_SAVED, "ok") in dlg_toasts
        assert dialog.result() == QDialog.DialogCode.Accepted

    def test_save_writes_cha_annotation(self, qtbot, dlg_controller, dlg_toasts) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        assert dialog.tabs.tabText(dialog.tabs.indexOf(dialog.cha_tab)) == TAB_CHA
        dialog.cha_tab.sync_check.setChecked(False)
        dialog.cha_tab.base_url.setText("http://cha.local")
        dialog.cha_tab.card_combos[0].setCurrentText("card-a")
        # Card 2 picks a pool model from the LLM tab's provider group.
        combo = dialog.cha_tab.card_combos[1]
        pool_index = next(
            i for i in range(combo.count())
            if combo.itemData(i) == ModelRef("default", "vision-model-1")
        )
        combo.setCurrentIndex(pool_index)
        dialog.cha_tab.batch_combo.setCurrentText("batch-b")
        dialog.save_button.click()
        loaded = load_cha_settings()
        assert loaded.api_mode == API_MODE_OWN
        assert loaded.base_url == "http://cha.local"
        assert loaded.card_models[0] == ModelRef("", "card-a")
        assert loaded.card_models[1] == ModelRef("default", "vision-model-1")
        assert loaded.batch_model == ModelRef("", "batch-b")
        assert (TOAST_SAVED, "ok") in dlg_toasts

    def test_cha_combos_follow_enabled_pool_models(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        combo = dialog.cha_tab.card_combos[0]
        items = [combo.itemText(i) for i in range(combo.count())]
        assert items == [
            "",
            "── default ──",
            "default · text-model-1",
            "default · vision-model-1",
        ]
        # Switching a model off on the LLM tab drops it from the CHA dropdown.
        dialog.llm_tab.model_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        items = [combo.itemText(i) for i in range(combo.count())]
        assert items == ["", "── default ──", "default · vision-model-1"]

    def test_save_reloads_controller_translator(self, qtbot, dlg_controller) -> None:
        assert dlg_controller.make_translator_or_none() is None
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        dialog.save_button.click()
        # The cached "unconfigured" translator is invalidated and rebuilt
        # from the freshly saved text target.
        assert dlg_controller.make_translator_or_none() is not None
        assert dlg_controller.vision_profile().vision_model == "vision-model-1"

    def test_save_keeps_other_profiles_and_settings(
        self, qtbot, dlg_controller
    ) -> None:
        other = LLMProfile(
            name="alt",
            api_type="openai",
            base_url="http://alt",
            models=("alt-m",),
            enabled_models=("alt-m",),
        )
        save_app_config(
            AppConfig(
                profiles=(other,),
                text_target=ModelRef("alt", "alt-m"),
                snapshot_retention=7,
            ),
        )
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        dialog.save_button.click()
        config = load_app_config()
        names = [p.name for p in config.profiles]
        assert names == ["alt", "default"]
        assert config.text_target == ModelRef("default", "text-model-1")
        assert config.snapshot_retention == 7

    def test_prefill_from_existing_config(self, qtbot, dlg_controller) -> None:
        existing = LLMProfile(
            name="default",
            api_type=API_TYPE,
            base_url="http://old.local",
            api_key="sk-old",
            text_model="old-model",
        )
        save_app_config(
            AppConfig(profiles=(existing,), active_profile="default"),
        )
        dialog = _make_dialog(qtbot, dlg_controller)
        tab = dialog.llm_tab
        assert tab.api_type.currentText() == API_TYPE
        assert tab.base_url.text() == "http://old.local"
        assert tab.api_key.text() == "sk-old"
        # Legacy single-profile config is upgraded: the old text model is the
        # only catalog entry, switched on, and already the text target.
        assert tab.model_list.count() == 1
        assert tab.text_target() == ModelRef("default", "old-model")
        assert tab.text_combo.currentData() == ModelRef("default", "old-model")


class TestConnectionProbe:
    def test_test_connection_success_toast(
        self, qtbot, dlg_controller, dlg_toasts
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        dialog.llm_tab.test_button.click()
        # 测速: the success toast reports the measured round-trip seconds.
        qtbot.waitUntil(
            lambda: any(
                text.startswith("连接成功(") and text.endswith("秒)") and kind == "ok"
                for text, kind in dlg_toasts
            ),
            timeout=2000,
        )
        assert dialog.llm_tab.test_button.isEnabled()

    def test_test_connection_failure_toast(
        self, qtbot, dlg_controller, dlg_toasts
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog, api_type=API_TYPE_FAIL)
        dialog.llm_tab.test_button.click()
        qtbot.waitUntil(
            lambda: any(
                text.startswith("连接失败: ") and kind == "err"
                for text, kind in dlg_toasts
            ),
            timeout=2000,
        )
        assert dialog.llm_tab.test_button.isEnabled()

    def test_test_connection_requires_a_model(
        self, qtbot, dlg_controller, dlg_toasts
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.llm_tab.add_profile()
        dialog.llm_tab.base_url.setText("http://x")
        dialog.llm_tab.test_button.click()
        assert (TOAST_NEED_MODEL, "warn") in dlg_toasts


class TestDebugMode:
    def test_checkbox_defaults_off(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        assert dialog.debug_check.text() == LABEL_DEBUG
        assert not dialog.debug_check.isChecked()
        assert any(
            isinstance(child, QLabel) and child.text() == HINT_DEBUG
            for child in dialog.findChildren(QLabel)
        )

    def test_save_persists_debug_and_gates_workers(
        self, qtbot, dlg_controller
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        dialog.debug_check.setChecked(True)
        try:
            dialog.save_button.click()
            assert dlg_controller.settings.debug is True
            assert load_ui_settings().debug is True
            assert debug_enabled() is True
        finally:
            set_debug(False)
