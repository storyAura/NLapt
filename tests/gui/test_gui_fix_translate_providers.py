"""Fix #5 tests: built-in web translation providers wired through the GUI.

Covers TranslationConfig persistence, TranslateBridge provider selection
(configured() reflecting the chosen provider, web translation via
httpx.MockTransport), and SettingsDialog provider persistence -- all offscreen,
no real network.
"""

from __future__ import annotations

from typing import Iterator

import httpx
import pytest

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, LLMProfile, RequestControl
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings
from nlapt_gui.translate_bridge import TranslateBridge
from nlapt_gui.translate_config import (
    DEFAULT_PROVIDER,
    TranslationConfig,
    load_translation_config,
    save_translation_config,
    translation_config_path,
)
from nlapt_gui.widgets.settings_dialog import (
    TOAST_SAVED,
    TOAST_TR_NEED_KEY,
    SettingsDialog,
    config_path,
)

# Unique api_type for this module (registry is process-global).
API_TYPE = "gui-fix-providers-mock"
register_client(API_TYPE, lambda profile: MockLLMClient(["llm-out"]))


def _google_transport(text: str = "a girl") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[[text, "少女", None, None]], None, "zh-CN"])

    return httpx.MockTransport(handler)


def _configured_controller() -> AppController:
    profile = LLMProfile(
        name="default", api_type=API_TYPE, base_url="http://mock", text_model="m1"
    )
    config = AppConfig(
        profiles=(profile,),
        active_profile="default",
        request=RequestControl(max_retries=0),
    )
    return AppController(NLaptApp(config=config), settings=UISettings())


@pytest.fixture()
def unconfigured_controller(qtbot) -> Iterator[AppController]:
    yield AppController(NLaptApp(), settings=UISettings())


# -- TranslationConfig -------------------------------------------------------
class TestTranslationConfig:
    def test_defaults_to_llm(self) -> None:
        assert TranslationConfig().provider == DEFAULT_PROVIDER == "llm"

    def test_round_trip(self) -> None:
        original = TranslationConfig(
            provider="baidu",
            baidu_appid="app-1",
            baidu_key="secret-1",
            deepl_key="dk",
        )
        save_translation_config(original)
        assert translation_config_path().exists()
        loaded = load_translation_config()
        assert loaded == original

    def test_missing_file_returns_defaults(self) -> None:
        assert not translation_config_path().exists()
        assert load_translation_config() == TranslationConfig()

    def test_unknown_provider_falls_back(self, tmp_path) -> None:
        path = tmp_path / "t.json"
        path.write_text('{"provider": "bogus"}', encoding="utf-8")
        assert load_translation_config(path).provider == DEFAULT_PROVIDER

    def test_corrupt_file_returns_defaults(self, tmp_path) -> None:
        path = tmp_path / "t.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_translation_config(path) == TranslationConfig()

    def test_credentials_mapping(self) -> None:
        config = TranslationConfig(baidu_appid="a", baidu_key="b", deepl_key="c")
        assert config.credentials() == {
            "baidu_appid": "a",
            "baidu_key": "b",
            "deepl_key": "c",
        }


# -- TranslateBridge provider selection --------------------------------------
class TestBridgeConfigured:
    def test_llm_provider_needs_profile(self, qtbot, unconfigured_controller) -> None:
        bridge = TranslateBridge(
            unconfigured_controller, config=TranslationConfig(provider="llm")
        )
        assert not bridge.configured()

    def test_llm_provider_with_profile(self, qtbot) -> None:
        bridge = TranslateBridge(
            _configured_controller(), config=TranslationConfig(provider="llm")
        )
        assert bridge.configured()

    def test_google_always_configured(self, qtbot, unconfigured_controller) -> None:
        bridge = TranslateBridge(
            unconfigured_controller, config=TranslationConfig(provider="google")
        )
        assert bridge.configured()

    def test_baidu_requires_keys(self, qtbot, unconfigured_controller) -> None:
        no_keys = TranslateBridge(
            unconfigured_controller, config=TranslationConfig(provider="baidu")
        )
        assert not no_keys.configured()
        with_keys = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(
                provider="baidu", baidu_appid="a", baidu_key="b"
            ),
        )
        assert with_keys.configured()

    def test_deepl_requires_key(self, qtbot, unconfigured_controller) -> None:
        no_key = TranslateBridge(
            unconfigured_controller, config=TranslationConfig(provider="deepl")
        )
        assert not no_key.configured()
        with_key = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(provider="deepl", deepl_key="k"),
        )
        assert with_key.configured()


class TestBridgeWebRequest:
    def test_google_translation_via_mock_transport(
        self, qtbot, unconfigured_controller
    ) -> None:
        bridge = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(provider="google"),
            transport=_google_transport("a girl"),
        )
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "少女")
        assert blocker.args == ["0001.png", "少女", "a girl", True]

    def test_google_result_is_cached(self, qtbot, unconfigured_controller) -> None:
        bridge = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(provider="google"),
            transport=_google_transport("a girl"),
        )
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000):
            bridge.request("0001.png", "少女")
        # A second identical request is served from the text-hash cache; the
        # transport (which would 500 the second time) is never hit again.
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0002.png", "少女")
        assert blocker.args == ["0002.png", "少女", "a girl", True]

    def test_google_transient_failure_is_retried(
        self, qtbot, unconfigured_controller
    ) -> None:
        """Web providers get spec-8 retries (they have none internally)."""
        calls: list[int] = []

        def flaky(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(500, text="boom")
            return httpx.Response(
                200, json=[[["a girl", "少女", None, None]], None, "zh-CN"]
            )

        sleeps: list[float] = []
        bridge = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(provider="google"),
            transport=httpx.MockTransport(flaky),
            retry_sleep=sleeps.append,
        )
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "少女")
        assert blocker.args[2] == "a girl"
        assert blocker.args[3] is True
        assert len(calls) == 2  # failed once, then retried
        assert sleeps  # backoff waited through the injected sleep only

    def test_baidu_unconfigured_emits_note(
        self, qtbot, unconfigured_controller
    ) -> None:
        from nlapt_gui.translate_bridge import NOTE_UNCONFIGURED

        bridge = TranslateBridge(
            unconfigured_controller, config=TranslationConfig(provider="baidu")
        )
        with qtbot.waitSignal(bridge.segment_ready, timeout=1000) as blocker:
            bridge.request("0001.png", "少女")
        assert blocker.args[3] is False
        assert blocker.args[2] == NOTE_UNCONFIGURED

    def test_default_bridge_reads_disk_config(
        self, qtbot, unconfigured_controller
    ) -> None:
        # No injected config -> reads translate.json fresh; default is llm and
        # this controller has no profile, so it is unconfigured.
        bridge = TranslateBridge(unconfigured_controller)
        assert not bridge.configured()
        # Switching the on-disk selection to google is picked up without any
        # explicit reload wiring.
        save_translation_config(TranslationConfig(provider="google"))
        assert bridge.configured()


# -- SettingsDialog provider persistence -------------------------------------
class TestSettingsDialogProviders:
    def _dialog(self, qtbot, controller) -> SettingsDialog:
        dialog = SettingsDialog(controller, api_types=(API_TYPE, "openai"))
        qtbot.addWidget(dialog)
        return dialog

    def _toasts(self, controller) -> list[tuple[str, str]]:
        collected: list[tuple[str, str]] = []
        controller.toast_requested.connect(
            lambda text, kind: collected.append((text, kind))
        )
        return collected

    def test_default_provider_is_llm(self, qtbot, unconfigured_controller) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        assert dialog.selected_provider() == "llm"

    def test_registration_note_updates_on_change(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        index = dialog.provider.findData("baidu")
        dialog.provider.setCurrentIndex(index)
        assert "fanyi-api.baidu.com" in dialog.registration_note.text()
        assert dialog.baidu_appid.isVisibleTo(dialog)

    def test_save_google_persists_without_llm_fields(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        toasts = self._toasts(unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("google"))
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        assert (TOAST_SAVED, "ok") in toasts
        assert load_translation_config().provider == "google"
        # Google needs no LLM profile, so config.json is left untouched.
        assert not config_path().exists()

    def test_save_baidu_without_keys_warns(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        toasts = self._toasts(unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("baidu"))
        dialog.save_button.click()
        assert (TOAST_TR_NEED_KEY, "warn") in toasts
        assert not translation_config_path().exists()

    def test_save_baidu_with_keys_persists(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("baidu"))
        dialog.baidu_appid.setText("app-1")
        dialog.baidu_key.setText("secret-1")
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        loaded = load_translation_config()
        assert loaded.provider == "baidu"
        assert loaded.baidu_appid == "app-1"
        assert loaded.baidu_key == "secret-1"

    def test_prefill_from_saved_translation_config(
        self, qtbot, unconfigured_controller
    ) -> None:
        save_translation_config(
            TranslationConfig(provider="deepl", deepl_key="dk-123")
        )
        dialog = self._dialog(qtbot, unconfigured_controller)
        assert dialog.selected_provider() == "deepl"
        assert dialog.deepl_key.text() == "dk-123"
        assert dialog.deepl_key.isVisibleTo(dialog)
