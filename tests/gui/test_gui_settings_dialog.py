"""Tests for nlapt_gui.widgets.settings_dialog (minimal LLM profile dialog)."""

from __future__ import annotations

from typing import Iterator

import pytest
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, LLMProfile
from nlapt.core.errors import LLMRequestError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.api_config import load_app_config, save_app_config
from nlapt_gui.cha_config import API_MODE_OWN, load_cha_settings
from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings, load_ui_settings
from nlapt_gui.widgets.settings_dialog import (
    HINT_DEBUG,
    LABEL_DEBUG,
    TAB_CHA,
    TOAST_NEED_BASE_URL,
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
    index = dialog.api_type.findText(api_type)
    dialog.api_type.setCurrentIndex(index)
    dialog.base_url.setText("http://mock.local")
    dialog.api_key.setText("sk-test-123456789")
    dialog.text_model.setText("text-model-1")
    dialog.vision_model.setText("vision-model-1")


class TestDialogBasics:
    def test_window_title_and_masked_key(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        assert dialog.windowTitle() == WINDOW_TITLE == "设置"
        assert dialog.api_key.echoMode() == QLineEdit.EchoMode.Password

    def test_current_profile_from_fields(self, qtbot, dlg_controller) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        profile = dialog.current_profile()
        assert profile.name == "default"
        assert profile.api_type == API_TYPE
        assert profile.base_url == "http://mock.local"
        assert profile.text_model == "text-model-1"
        assert profile.vision_model == "vision-model-1"


class TestValidation:
    def test_save_requires_base_url(self, qtbot, dlg_controller, dlg_toasts) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.text_model.setText("m")
        dialog.save_button.click()
        assert (TOAST_NEED_BASE_URL, "warn") in dlg_toasts
        assert dialog.result() != QDialog.DialogCode.Accepted
        assert not config_path().exists()

    def test_save_requires_text_model(self, qtbot, dlg_controller, dlg_toasts) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.base_url.setText("http://x")
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
        assert config.active_profile == "default"
        profile = config.profiles[0]
        assert profile.api_type == API_TYPE
        assert profile.base_url == "http://mock.local"
        assert profile.api_key == "sk-test-123456789"
        assert profile.text_model == "text-model-1"
        assert profile.vision_model == "vision-model-1"
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
        dialog.cha_tab.batch_combo.setCurrentText("batch-b")
        dialog.save_button.click()
        loaded = load_cha_settings()
        assert loaded.api_mode == API_MODE_OWN
        assert loaded.base_url == "http://cha.local"
        assert loaded.card_models[0] == "card-a"
        assert loaded.batch_model == "batch-b"
        assert (TOAST_SAVED, "ok") in dlg_toasts

    def test_save_reloads_controller_translator(self, qtbot, dlg_controller) -> None:
        assert dlg_controller.make_translator_or_none() is None
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        dialog.save_button.click()
        # The cached "unconfigured" translator is invalidated and rebuilt
        # from the freshly saved profile.
        assert dlg_controller.make_translator_or_none() is not None

    def test_save_keeps_other_profiles_and_settings(
        self, qtbot, dlg_controller
    ) -> None:
        other = LLMProfile(name="alt", api_type="openai", base_url="http://alt")
        save_app_config(
            AppConfig(profiles=(other,), active_profile="alt", snapshot_retention=7),
        )
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        dialog.save_button.click()
        config = load_app_config()
        names = [p.name for p in config.profiles]
        assert names == ["default", "alt"]
        assert config.active_profile == "default"
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
        assert dialog.api_type.currentText() == API_TYPE
        assert dialog.base_url.text() == "http://old.local"
        assert dialog.api_key.text() == "sk-old"
        assert dialog.text_model.text() == "old-model"


class TestConnectionProbe:
    def test_test_connection_success_toast(
        self, qtbot, dlg_controller, dlg_toasts
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog)
        dialog.test_button.click()
        # 测速: the success toast reports the measured round-trip seconds.
        qtbot.waitUntil(
            lambda: any(
                text.startswith("连接成功(") and text.endswith("秒)") and kind == "ok"
                for text, kind in dlg_toasts
            ),
            timeout=2000,
        )
        assert dialog.test_button.isEnabled()

    def test_test_connection_failure_toast(
        self, qtbot, dlg_controller, dlg_toasts
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        _fill(dialog, api_type=API_TYPE_FAIL)
        dialog.test_button.click()
        qtbot.waitUntil(
            lambda: any(
                text.startswith("连接失败: ") and kind == "err"
                for text, kind in dlg_toasts
            ),
            timeout=2000,
        )
        assert dialog.test_button.isEnabled()

    def test_test_connection_requires_fields(
        self, qtbot, dlg_controller, dlg_toasts
    ) -> None:
        dialog = _make_dialog(qtbot, dlg_controller)
        dialog.test_button.click()
        assert (TOAST_NEED_BASE_URL, "warn") in dlg_toasts


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
