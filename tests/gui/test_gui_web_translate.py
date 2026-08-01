"""Tests for nlapt.llm.web_translate (Google/Baidu/DeepL providers).

All HTTP is driven through ``httpx.MockTransport`` -- no real network. The
providers are the core, GUI-independent building blocks selected by the
translation bridge.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from nlapt.core.errors import LLMConfigError, LLMRequestError
from nlapt.llm.translate import Direction
from nlapt.llm.web_translate import (
    PROVIDER_BAIDU,
    PROVIDER_DEEPL,
    PROVIDER_GOOGLE,
    PROVIDER_LLM,
    REGISTRATION_INFO,
    BaiduProvider,
    DeepLProvider,
    GoogleFreeProvider,
    create_provider,
)


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# -- Google ------------------------------------------------------------------
class TestGoogleFreeProvider:
    def test_parses_translated_text(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            body = [[["a girl", "少女", None, None]], None, "zh-CN"]
            return httpx.Response(200, json=body)

        provider = GoogleFreeProvider(transport=_transport(handler))
        result = provider.translate("少女", Direction.ZH_TO_EN)
        assert result == "a girl"
        assert "translate_a/single" in str(captured[0].url)
        assert "tl=en" in str(captured[0].url)

    def test_concatenates_multiple_segments(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = [[["Hello, ", "x"], ["world", "y"]], None, "en"]
            return httpx.Response(200, json=body)

        provider = GoogleFreeProvider(transport=_transport(handler))
        assert provider.translate("你好世界", Direction.ZH_TO_EN) == "Hello, world"

    def test_http_error_raises_request_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limited")

        provider = GoogleFreeProvider(transport=_transport(handler))
        with pytest.raises(LLMRequestError):
            provider.translate("少女", Direction.ZH_TO_EN)

    def test_malformed_body_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": True})

        provider = GoogleFreeProvider(transport=_transport(handler))
        with pytest.raises(LLMRequestError):
            provider.translate("少女", Direction.ZH_TO_EN)

    def test_empty_text_raises(self) -> None:
        provider = GoogleFreeProvider(transport=_transport(lambda r: httpx.Response(200)))
        with pytest.raises(LLMRequestError):
            provider.translate("   ", Direction.ZH_TO_EN)


# -- Baidu -------------------------------------------------------------------
class TestBaiduProvider:
    def test_unconfigured_raises_config_error(self) -> None:
        with pytest.raises(LLMConfigError):
            BaiduProvider("", "")
        with pytest.raises(LLMConfigError):
            BaiduProvider("appid", "")

    def test_translates_with_mock_keys(self) -> None:
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(dict(httpx.QueryParams(request.content.decode())))
            body = {"from": "zh", "to": "en", "trans_result": [{"src": "少女", "dst": "a girl"}]}
            return httpx.Response(200, json=body)

        provider = BaiduProvider("app-1", "secret-1", transport=_transport(handler))
        assert provider.translate("少女", Direction.ZH_TO_EN) == "a girl"
        sent = captured[0]
        assert sent["appid"] == "app-1"
        assert sent["q"] == "少女"
        assert sent["sign"]  # md5 signature present

    def test_error_code_body_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"error_code": "54001", "error_msg": "Invalid Sign"}
            )

        provider = BaiduProvider("app", "key", transport=_transport(handler))
        with pytest.raises(LLMRequestError) as exc:
            provider.translate("少女", Direction.ZH_TO_EN)
        assert "54001" in exc.value.message


# -- DeepL -------------------------------------------------------------------
class TestDeepLProvider:
    def test_unconfigured_raises_config_error(self) -> None:
        with pytest.raises(LLMConfigError):
            DeepLProvider("")

    def test_translates_with_mock_key(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            body = {"translations": [{"detected_source_language": "ZH", "text": "a girl"}]}
            return httpx.Response(200, json=body)

        provider = DeepLProvider("key-abc", transport=_transport(handler))
        assert provider.translate("少女", Direction.ZH_TO_EN) == "a girl"
        assert captured[0].headers["Authorization"] == "DeepL-Auth-Key key-abc"
        params = httpx.QueryParams(captured[0].content.decode())
        assert params["target_lang"] == "EN-US"

    def test_target_lang_for_en_to_zh(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"translations": [{"text": "你好"}]})

        provider = DeepLProvider("key", transport=_transport(handler))
        assert provider.translate("hello", Direction.EN_TO_ZH) == "你好"
        params = httpx.QueryParams(captured[0].content.decode())
        assert params["target_lang"] == "ZH"

    def test_http_error_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="forbidden")

        provider = DeepLProvider("key", transport=_transport(handler))
        with pytest.raises(LLMRequestError):
            provider.translate("hello", Direction.EN_TO_ZH)


# -- registry + registration info --------------------------------------------
class TestRegistry:
    def test_create_google_needs_no_keys(self) -> None:
        provider = create_provider(PROVIDER_GOOGLE, {})
        assert isinstance(provider, GoogleFreeProvider)

    def test_create_baidu_missing_keys_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            create_provider(PROVIDER_BAIDU, {})

    def test_create_baidu_with_keys(self) -> None:
        provider = create_provider(
            PROVIDER_BAIDU, {"baidu_appid": "a", "baidu_key": "b"}
        )
        assert isinstance(provider, BaiduProvider)

    def test_create_deepl_with_key(self) -> None:
        provider = create_provider(PROVIDER_DEEPL, {"deepl_key": "k"})
        assert isinstance(provider, DeepLProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            create_provider("nope", {})

    def test_registration_info_covers_all_providers(self) -> None:
        for provider_id in (PROVIDER_LLM, PROVIDER_GOOGLE, PROVIDER_BAIDU, PROVIDER_DEEPL):
            info = REGISTRATION_INFO[provider_id]
            assert isinstance(info["needs_key"], bool)
            assert isinstance(info["note_zh"], str) and info["note_zh"]

    def test_baidu_deepl_notes_mention_registration_urls(self) -> None:
        assert "fanyi-api.baidu.com" in REGISTRATION_INFO[PROVIDER_BAIDU]["note_zh"]
        assert "deepl.com" in REGISTRATION_INFO[PROVIDER_DEEPL]["note_zh"]
        # Google is free / no registration; the note must say so.
        assert "无需注册" in REGISTRATION_INFO[PROVIDER_GOOGLE]["note_zh"]
        assert REGISTRATION_INFO[PROVIDER_GOOGLE]["needs_key"] is False
        assert REGISTRATION_INFO[PROVIDER_BAIDU]["needs_key"] is True
        assert REGISTRATION_INFO[PROVIDER_DEEPL]["needs_key"] is True
