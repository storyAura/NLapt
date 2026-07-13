"""Tests for target-language translation (中/英/日, spec module 1).

Covers ``Translator.translate_to`` (LLM path, MockLLMClient) and the
``translate_to`` methods of the built-in web providers (httpx.MockTransport,
no real network).
"""

from __future__ import annotations

import json

import httpx
import pytest

from nlapt.core.config import LLMProfile
from nlapt.core.errors import LLMConfigError, ValidationError
from nlapt.llm.mock import MockLLMClient
from nlapt.llm.templates import DEFAULT_TEMPLATES
from nlapt.llm.translate import (
    LANG_EN,
    LANG_JA,
    LANG_LABELS,
    LANG_ZH,
    TARGET_LANGS,
    TEMPLATE_KEY_BY_TARGET,
    Translator,
)
from nlapt.llm.web_translate import (
    BaiduProvider,
    DeepLProvider,
    GoogleFreeProvider,
)

PROFILE = LLMProfile(
    name="p", api_type="openai", base_url="http://mock", text_model="m1"
)


class TestVocabulary:
    def test_target_langs(self) -> None:
        assert TARGET_LANGS == ("zh", "en", "ja")
        assert set(LANG_LABELS) == set(TARGET_LANGS)
        assert LANG_LABELS[LANG_ZH] == "中文"
        assert LANG_LABELS[LANG_EN] == "English"
        assert LANG_LABELS[LANG_JA] == "日本語"

    def test_templates_exist_for_every_target(self) -> None:
        for lang in TARGET_LANGS:
            assert TEMPLATE_KEY_BY_TARGET[lang] in DEFAULT_TEMPLATES


class TestTranslatorTranslateTo:
    def test_translates_with_target_template(self) -> None:
        client = MockLLMClient(["猫と犬"])
        translator = Translator(client, PROFILE)
        entry = translator.translate_to("k", "cat and dog", LANG_JA)
        assert entry.translated == "猫と犬"
        prompt = client.requests[0].messages[0].text
        assert "Japanese" in prompt
        assert "cat and dog" in prompt

    def test_each_language_uses_its_template(self) -> None:
        for lang, keyword in ((LANG_ZH, "Chinese"), (LANG_EN, "English"), (LANG_JA, "Japanese")):
            client = MockLLMClient(["out"])
            Translator(client, PROFILE).translate_to("k", "text", lang)
            assert keyword in client.requests[0].messages[0].text

    def test_invalid_target_raises(self) -> None:
        translator = Translator(MockLLMClient(["x"]), PROFILE)
        with pytest.raises(ValidationError):
            translator.translate_to("k", "text", "fr")

    def test_empty_text_raises(self) -> None:
        translator = Translator(MockLLMClient(["x"]), PROFILE)
        with pytest.raises(ValidationError):
            translator.translate_to("k", "   ", LANG_ZH)

    def test_missing_text_model_raises(self) -> None:
        profile = LLMProfile(name="p", api_type="openai", base_url="http://mock")
        translator = Translator(MockLLMClient(["x"]), profile)
        with pytest.raises(LLMConfigError):
            translator.translate_to("k", "text", LANG_ZH)


def _capture_transport(response: httpx.Response) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response

    return httpx.MockTransport(handler), seen


class TestGoogleTranslateTo:
    def test_maps_target_lang(self) -> None:
        transport, seen = _capture_transport(
            httpx.Response(200, json=[[["ねこ", "cat", None, None]]])
        )
        provider = GoogleFreeProvider(transport=transport)
        assert provider.translate_to("cat", "ja") == "ねこ"
        params = dict(seen[0].url.params)
        assert params["tl"] == "ja"
        assert params["sl"] == "auto"

    def test_zh_maps_to_zh_cn(self) -> None:
        transport, seen = _capture_transport(
            httpx.Response(200, json=[[["猫", "cat", None, None]]])
        )
        GoogleFreeProvider(transport=transport).translate_to("cat", "zh")
        assert dict(seen[0].url.params)["tl"] == "zh-CN"

    def test_unknown_lang_raises(self) -> None:
        provider = GoogleFreeProvider()
        with pytest.raises(LLMConfigError):
            provider.translate_to("cat", "fr")


class TestBaiduTranslateTo:
    def test_auto_source_and_jp_code(self) -> None:
        transport, seen = _capture_transport(
            httpx.Response(200, json={"trans_result": [{"src": "cat", "dst": "ねこ"}]})
        )
        provider = BaiduProvider("appid", "key", transport=transport)
        assert provider.translate_to("cat", "ja") == "ねこ"
        body = dict(
            pair.split("=", 1)
            for pair in seen[0].content.decode("utf-8").split("&")
        )
        assert body["from"] == "auto"
        assert body["to"] == "jp"  # Baidu's Japanese code differs from ISO


class TestDeepLTranslateTo:
    def test_target_lang_payload(self) -> None:
        transport, seen = _capture_transport(
            httpx.Response(200, json={"translations": [{"text": "ねこ"}]})
        )
        provider = DeepLProvider("key", transport=transport)
        assert provider.translate_to("cat", "ja") == "ねこ"
        body = seen[0].content.decode("utf-8")
        assert "target_lang=JA" in body

    def test_direction_api_still_works(self) -> None:
        # The Direction-based segment path must be untouched by the refactor.
        from nlapt.llm.translate import Direction

        transport, seen = _capture_transport(
            httpx.Response(200, json={"translations": [{"text": "猫"}]})
        )
        provider = DeepLProvider("key", transport=transport)
        assert provider.translate(json.dumps(["cat"]), Direction.EN_TO_ZH) == "猫"
        assert "target_lang=ZH" in seen[0].content.decode("utf-8")
