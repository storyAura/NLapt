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
from nlapt.llm.retry import MinIntervalLimiter
from nlapt.llm.web_translate import (
    MSG_DEEPLX_BUSY,
    PROVIDER_BAIDU,
    PROVIDER_CUSTOM,
    PROVIDER_DEEPL,
    PROVIDER_DEEPLX,
    PROVIDER_GOOGLE,
    PROVIDER_LLM,
    PROVIDER_LOCAL_MT,
    REGISTRATION_INFO,
    BaiduProvider,
    CustomOpenAIProvider,
    DeepLProvider,
    DeepLXProvider,
    GoogleFreeProvider,
    create_provider,
    display_deeplx_url,
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
            return httpx.Response(500, text="upstream")

        provider = GoogleFreeProvider(transport=_transport(handler))
        with pytest.raises(LLMRequestError, match="500"):
            provider.translate("少女", Direction.ZH_TO_EN)

    def test_http_429_retries_then_raises(self) -> None:
        seen: list[httpx.Request] = []
        slept: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(429, text="rate limited")

        provider = GoogleFreeProvider(transport=_transport(handler), sleep=slept.append)
        with pytest.raises(LLMRequestError, match="429"):
            provider.translate("少女", Direction.ZH_TO_EN)
        assert len(seen) == 3
        assert slept == [0.5, 1.0]

    def test_http_429_then_success(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, text="rate limited")
            body = [[["a girl", "少女", None, None]], None, "zh-CN"]
            return httpx.Response(200, json=body)

        provider = GoogleFreeProvider(
            transport=_transport(handler), sleep=lambda _delay: None
        )
        assert provider.translate("少女", Direction.ZH_TO_EN) == "a girl"
        assert calls["n"] == 3

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


class TestDeepLXProvider:
    def test_requires_url(self) -> None:
        with pytest.raises(LLMConfigError):
            DeepLXProvider("")

    def test_posts_translate_json(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={"code": 200, "data": "你好世界", "id": 1, "method": "Free"},
            )

        provider = DeepLXProvider(
            "http://127.0.0.1:1188", transport=_transport(handler)
        )
        assert provider.translate("Hello World", Direction.EN_TO_ZH) == "你好世界"
        assert str(captured[0].url).endswith("/translate")
        body = captured[0].content
        assert b'"target_lang"' in body and b"ZH" in body
        assert b'"source_lang"' in body and b"auto" in body

    def test_keeps_full_translate_url_and_sends_bearer(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"code": 200, "data": "cat"})

        provider = DeepLXProvider(
            "https://deeplx.example/translate",
            "secret-token",
            transport=_transport(handler),
        )
        assert provider.translate_to("猫", "en") == "cat"
        assert str(captured[0].url) == "https://deeplx.example/translate"
        assert captured[0].headers["authorization"] == "Bearer secret-token"

    def test_nonzero_code_retries_then_raises(self) -> None:
        seen: list[httpx.Request] = []
        slept: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"code": 429, "message": "busy"})

        provider = DeepLXProvider(
            "http://x", transport=_transport(handler), sleep=slept.append
        )
        with pytest.raises(LLMRequestError, match=MSG_DEEPLX_BUSY) as exc:
            provider.translate("hi", Direction.EN_TO_ZH)
        assert getattr(exc.value, "status_code", None) == 429
        assert "teapot" not in exc.value.message
        assert len(seen) == 4
        assert slept == [1.0, 2.0, 4.0]

    def test_http_418_then_success(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(
                    418, json={"code": 418, "message": "I'm a teapot"}
                )
            return httpx.Response(200, json={"code": 200, "data": "你好"})

        provider = DeepLXProvider(
            "http://x",
            transport=_transport(handler),
            sleep=lambda _delay: None,
        )
        assert provider.translate("hi", Direction.EN_TO_ZH) == "你好"
        assert calls["n"] == 3

    def test_http_418_exhausted_hides_teapot(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                418, json={"code": 418, "message": "I'm a teapot"}
            )

        provider = DeepLXProvider(
            "http://x", transport=_transport(handler), sleep=lambda _delay: None
        )
        with pytest.raises(LLMRequestError, match=MSG_DEEPLX_BUSY) as exc:
            provider.translate("hi", Direction.EN_TO_ZH)
        assert getattr(exc.value, "status_code", None) == 418
        assert "teapot" not in exc.value.message.lower()
        assert "418" not in exc.value.message

    def test_http_429_then_success(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, json={"code": 429, "message": "too many requests"})
            return httpx.Response(200, json={"code": 200, "data": "你好"})

        provider = DeepLXProvider(
            "http://x",
            transport=_transport(handler),
            sleep=lambda _delay: None,
        )
        assert provider.translate("hi", Direction.EN_TO_ZH) == "你好"
        assert calls["n"] == 3

    def test_error_message_redacts_key_in_url(self) -> None:
        secret = "super-secret-token-path"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not-json")

        provider = DeepLXProvider(
            f"https://api.deeplx.org/{secret}/translate",
            transport=_transport(handler),
        )
        with pytest.raises(LLMRequestError) as exc:
            provider.translate("hi", Direction.EN_TO_ZH)
        assert secret not in exc.value.message
        assert "api.deeplx.org" in exc.value.message
        assert "/…/translate" in exc.value.message

    def test_display_url_strips_path_and_query_secrets(self) -> None:
        assert display_deeplx_url(
            "https://api.deeplx.org/abc123/translate?token=hidden"
        ) == "https://api.deeplx.org/…/translate"
        assert display_deeplx_url("http://127.0.0.1:1188/translate") == (
            "http://127.0.0.1:1188/translate"
        )

    def test_shared_limiter_spaces_successive_calls(self) -> None:
        slept: list[float] = []

        class _Clock:
            def __init__(self) -> None:
                self.now = 0.0

            def __call__(self) -> float:
                return self.now

        clock = _Clock()
        limiter = MinIntervalLimiter(1.0, clock=clock, sleep=slept.append)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"code": 200, "data": "x"})

        first = DeepLXProvider(
            "http://x", transport=_transport(handler), limiter=limiter
        )
        second = DeepLXProvider(
            "http://x", transport=_transport(handler), limiter=limiter
        )
        first.translate("a", Direction.EN_TO_ZH)
        second.translate("b", Direction.EN_TO_ZH)
        assert slept == pytest.approx([1.0])

    def test_create_provider_wires_url(self) -> None:
        provider = create_provider(
            PROVIDER_DEEPLX, {"deeplx_url": "http://127.0.0.1:1188"}
        )
        assert isinstance(provider, DeepLXProvider)


class TestCustomOpenAIProvider:
    def test_requires_url_and_model(self) -> None:
        with pytest.raises(LLMConfigError):
            CustomOpenAIProvider("", "sk", "")
        with pytest.raises(LLMConfigError):
            CustomOpenAIProvider("https://api.example.com/v1", "sk", "")

    def test_translates_via_chat_completions(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            body = {
                "choices": [{"message": {"content": "a girl, solo"}}]
            }
            return httpx.Response(200, json=body)

        provider = CustomOpenAIProvider(
            "https://api.example.com/v1",
            "sk-test",
            "mini",
            transport=_transport(handler),
        )
        assert provider.translate("少女, 单人", Direction.ZH_TO_EN) == "a girl, solo"
        assert str(captured[0].url).endswith("/chat/completions")
        assert captured[0].headers["authorization"] == "Bearer sk-test"

    def test_full_endpoint_url_is_kept(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "猫"}}]}
            )

        provider = CustomOpenAIProvider(
            "https://api.example.com/v1/chat/completions",
            "",
            "mini",
            transport=_transport(handler),
        )
        assert provider.translate_to("cat", "zh") == "猫"
        assert seen[0] == "https://api.example.com/v1/chat/completions"

    def test_create_provider_wires_custom_creds(self) -> None:
        provider = create_provider(
            PROVIDER_CUSTOM,
            {
                "custom_base_url": "https://api.example.com/v1",
                "custom_api_key": "k",
                "custom_model": "m",
            },
        )
        assert isinstance(provider, CustomOpenAIProvider)


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
        for provider_id in (
            PROVIDER_LLM,
            PROVIDER_GOOGLE,
            PROVIDER_BAIDU,
            PROVIDER_DEEPL,
            PROVIDER_DEEPLX,
            PROVIDER_LOCAL_MT,
            PROVIDER_CUSTOM,
        ):
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
        assert REGISTRATION_INFO[PROVIDER_CUSTOM]["needs_key"] is True
        assert "文档" in REGISTRATION_INFO[PROVIDER_CUSTOM]["note_zh"]
        assert REGISTRATION_INFO[PROVIDER_DEEPLX]["needs_key"] is True
        assert "deeplx.owo.network" in REGISTRATION_INFO[PROVIDER_DEEPLX]["note_zh"]
