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
from nlapt_gui.api_config import api_config_path
from nlapt_gui.translate_config import (
    DEFAULT_PROVIDER,
    SUGGESTED_FALLBACK_ORDER,
    TranslationConfig,
    load_translation_config,
    normalize_fallback_order,
    provider_chain,
    save_translation_config,
    translation_config_path,
)
from nlapt_gui.widgets.settings_dialog import (
    TOAST_SAVED,
    TOAST_TR_NEED_CUSTOM,
    TOAST_TR_NEED_DEEPLX,
    TOAST_TR_NEED_KEY,
    TOAST_TR_NEED_LOCAL_MT,
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
            "deeplx_url": "",
            "deeplx_token": "",
            "custom_base_url": "",
            "custom_api_key": "",
            "custom_model": "",
        }

    def test_normalize_drops_primary_unknowns_and_dupes(self) -> None:
        assert normalize_fallback_order(
            "deeplx",
            ("google", "deeplx", "nope", "google", "baidu", "local_mt"),
        ) == ("google", "baidu", "local_mt")

    def test_provider_chain_puts_primary_first(self) -> None:
        config = TranslationConfig(
            provider="deeplx", fallback_order=("google", "baidu", "local_mt")
        )
        assert provider_chain(config) == (
            "deeplx",
            "google",
            "baidu",
            "local_mt",
        )

    def test_post_init_strips_primary_from_fallbacks(self) -> None:
        config = TranslationConfig(
            provider="google", fallback_order=("google", "baidu")
        )
        assert config.fallback_order == ("baidu",)

    def test_fallback_order_round_trip(self) -> None:
        original = TranslationConfig(
            provider="deeplx",
            deeplx_url="http://127.0.0.1:1188",
            fallback_order=("google", "baidu", "local_mt"),
        )
        save_translation_config(original)
        loaded = load_translation_config()
        assert loaded.fallback_order == ("google", "baidu", "local_mt")
        assert loaded.provider == "deeplx"

    def test_missing_fallback_order_is_empty(self, tmp_path) -> None:
        path = tmp_path / "t.json"
        path.write_text('{"provider": "google"}', encoding="utf-8")
        assert load_translation_config(path).fallback_order == ()

    def test_invalid_fallback_order_is_empty(self, tmp_path) -> None:
        path = tmp_path / "t.json"
        path.write_text(
            '{"provider": "google", "fallback_order": "google"}', encoding="utf-8"
        )
        assert load_translation_config(path).fallback_order == ()

    def test_deeplx_secrets_live_in_documents_not_appdata(self) -> None:
        original = TranslationConfig(
            provider="deeplx",
            deeplx_url="https://api.deeplx.org/secret-path/translate",
            deeplx_token="tok-secret",
        )
        save_translation_config(original)
        loaded = load_translation_config()
        assert loaded == original
        appdata_text = translation_config_path().read_text(encoding="utf-8")
        assert "secret-path" not in appdata_text
        assert "tok-secret" not in appdata_text
        docs_text = api_config_path().read_text(encoding="utf-8")
        assert "secret-path" in docs_text
        assert "tok-secret" in docs_text

    def test_migrates_legacy_deeplx_from_appdata(self, tmp_path) -> None:
        path = tmp_path / "legacy.json"
        path.write_text(
            '{"provider": "deeplx", "deeplx_url": "http://old", "deeplx_token": "old-tok"}',
            encoding="utf-8",
        )
        loaded = load_translation_config(path)
        assert loaded.deeplx_url == "http://old"
        assert loaded.deeplx_token == "old-tok"
        save_translation_config(loaded, path)
        saved = path.read_text(encoding="utf-8")
        assert "deeplx_url" not in saved
        assert "old-tok" not in saved
        assert "http://old" in api_config_path().read_text(encoding="utf-8")

    def test_custom_api_lives_in_documents_not_appdata(self) -> None:
        original = TranslationConfig(
            provider="custom",
            custom_base_url="https://api.example.com/v1",
            custom_api_key="sk-secret",
            custom_model="mini",
        )
        save_translation_config(original)
        loaded = load_translation_config()
        assert loaded == original
        appdata_text = translation_config_path().read_text(encoding="utf-8")
        assert "sk-secret" not in appdata_text
        assert "api.example.com" not in appdata_text
        docs_text = api_config_path().read_text(encoding="utf-8")
        assert "sk-secret" in docs_text
        assert "mini" in docs_text


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

    def test_custom_requires_url_and_model(self, qtbot, unconfigured_controller) -> None:
        missing = TranslateBridge(
            unconfigured_controller, config=TranslationConfig(provider="custom")
        )
        assert not missing.configured()
        ready = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(
                provider="custom",
                custom_base_url="https://api.example.com/v1",
                custom_model="mini",
            ),
        )
        assert ready.configured()

    def test_deeplx_requires_url(self, qtbot, unconfigured_controller) -> None:
        missing = TranslateBridge(
            unconfigured_controller, config=TranslationConfig(provider="deeplx")
        )
        assert not missing.configured()
        ready = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(
                provider="deeplx", deeplx_url="http://127.0.0.1:1188"
            ),
        )
        assert ready.configured()


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

    def test_configured_when_only_fallback_is_ready(
        self, unconfigured_controller
    ) -> None:
        bridge = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(provider="llm", fallback_order=("google",)),
        )
        assert bridge.configured()

    def test_deeplx_error_falls_back_to_google(
        self, qtbot, unconfigured_controller
    ) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if "googleapis" in str(request.url):
                return httpx.Response(
                    200, json=[[["a girl", "少女", None, None]], None, "zh-CN"]
                )
            return httpx.Response(400, json={"message": "nope"})

        bridge = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(
                provider="deeplx",
                deeplx_url="http://deeplx.test",
                fallback_order=("google",),
            ),
            transport=httpx.MockTransport(handler),
            retry_sleep=lambda _delay: None,
        )
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "少女")
        assert blocker.args[2] == "a girl"
        assert blocker.args[3] is True
        assert any("deeplx.test" in url for url in seen)
        assert any("googleapis" in url for url in seen)

    def test_request_to_falls_back_to_google(
        self, qtbot, unconfigured_controller
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "googleapis" in str(request.url):
                return httpx.Response(
                    200, json=[[["hello", "你好", None, None]], None, "en"]
                )
            return httpx.Response(400, json={"message": "nope"})

        bridge = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(
                provider="deeplx",
                deeplx_url="http://deeplx.test",
                fallback_order=("google",),
            ),
            transport=httpx.MockTransport(handler),
            retry_sleep=lambda _delay: None,
        )
        with qtbot.waitSignal(bridge.target_ready, timeout=2000) as blocker:
            bridge.request_to("0001.png", "你好", "en")
        assert blocker.args[3] == "hello"
        assert blocker.args[4] is True

    def test_all_providers_fail_emits_last_error(
        self, qtbot, unconfigured_controller
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"message": "nope"})

        bridge = TranslateBridge(
            unconfigured_controller,
            config=TranslationConfig(
                provider="deeplx",
                deeplx_url="http://deeplx.test",
                fallback_order=("google",),
            ),
            transport=httpx.MockTransport(handler),
            retry_sleep=lambda _delay: None,
        )
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "少女")
        assert blocker.args[3] is False
        assert "Google" in blocker.args[2]


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

    def test_save_custom_without_fields_warns(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        toasts = self._toasts(unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("custom"))
        dialog.save_button.click()
        assert (TOAST_TR_NEED_CUSTOM, "warn") in toasts
        assert not translation_config_path().exists()

    def test_save_custom_writes_documents_file(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("custom"))
        dialog.custom_base_url.setText("https://api.example.com/v1")
        dialog.custom_api_key.setText("sk-dialog")
        dialog.custom_model.setText("mini")
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        loaded = load_translation_config()
        assert loaded.provider == "custom"
        assert loaded.custom_base_url == "https://api.example.com/v1"
        assert loaded.custom_api_key == "sk-dialog"
        assert loaded.custom_model == "mini"
        assert "sk-dialog" not in translation_config_path().read_text(encoding="utf-8")
        assert "sk-dialog" in api_config_path().read_text(encoding="utf-8")

    def test_save_deeplx_without_url_warns(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        toasts = self._toasts(unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("deeplx"))
        dialog.save_button.click()
        assert (TOAST_TR_NEED_DEEPLX, "warn") in toasts
        assert not translation_config_path().exists()

    def test_save_deeplx_persists_url_and_token(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("deeplx"))
        dialog.deeplx_url.setText("http://127.0.0.1:1188")
        dialog.deeplx_token.setText("tok-1")
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        loaded = load_translation_config()
        assert loaded.provider == "deeplx"
        assert loaded.deeplx_url == "http://127.0.0.1:1188"
        assert loaded.deeplx_token == "tok-1"
        assert "tok-1" not in translation_config_path().read_text(encoding="utf-8")
        assert "tok-1" in api_config_path().read_text(encoding="utf-8")

    def test_save_local_mt_without_download_warns(
        self, qtbot, unconfigured_controller, tmp_path
    ) -> None:
        from nlapt_gui.local_bridge import LocalBridge

        LocalBridge().update_settings(models_dir=str(tmp_path))
        dialog = self._dialog(qtbot, unconfigured_controller)
        toasts = self._toasts(unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("local_mt"))
        dialog.save_button.click()
        assert (TOAST_TR_NEED_LOCAL_MT, "warn") in toasts
        assert not translation_config_path().exists()

    def test_save_local_mt_persists_tier(
        self, qtbot, unconfigured_controller, tmp_path, monkeypatch
    ) -> None:
        from nlapt.local.mt_catalog import TIER_FAST, find_mt_model, mt_model_path
        from nlapt_gui.local_bridge import LocalBridge

        LocalBridge().update_settings(models_dir=str(tmp_path))
        dest = mt_model_path(tmp_path, find_mt_model(TIER_FAST))
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"gguf")
        monkeypatch.setattr(
            "nlapt_gui.widgets.settings_dialog.is_tier_downloaded",
            lambda *_a, **_k: True,
        )
        dialog = self._dialog(qtbot, unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("local_mt"))
        dialog.local_mt_tier.setCurrentIndex(dialog.local_mt_tier.findData(TIER_FAST))
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        loaded = load_translation_config()
        assert loaded.provider == "local_mt"
        assert loaded.local_mt_tier == TIER_FAST

    def test_save_persists_fallback_order(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        dialog.provider.setCurrentIndex(dialog.provider.findData("deeplx"))
        dialog.deeplx_url.setText("http://127.0.0.1:1188")
        dialog.fallback_editor.apply_suggested()
        assert dialog.fallback_editor.order() == SUGGESTED_FALLBACK_ORDER
        with qtbot.waitSignal(dialog.saved, timeout=1000):
            dialog.save_button.click()
        loaded = load_translation_config()
        assert loaded.provider == "deeplx"
        assert loaded.fallback_order == SUGGESTED_FALLBACK_ORDER

    def test_prefill_restores_fallback_order(
        self, qtbot, unconfigured_controller
    ) -> None:
        save_translation_config(
            TranslationConfig(
                provider="deeplx",
                deeplx_url="http://127.0.0.1:1188",
                fallback_order=("google", "baidu"),
            )
        )
        dialog = self._dialog(qtbot, unconfigured_controller)
        assert dialog.selected_provider() == "deeplx"
        assert dialog.fallback_editor.order() == ("google", "baidu")

    def test_changing_primary_drops_it_from_fallbacks(
        self, qtbot, unconfigured_controller
    ) -> None:
        dialog = self._dialog(qtbot, unconfigured_controller)
        dialog.fallback_editor.set_order(("google", "baidu"))
        dialog.provider.setCurrentIndex(dialog.provider.findData("google"))
        assert dialog.fallback_editor.order() == ("baidu",)
