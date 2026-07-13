"""Tests for nlapt.llm.translate: direction detection, cache, Translator."""

from __future__ import annotations

import pytest

from nlapt.core.config import LLMProfile
from nlapt.core.errors import LLMConfigError, LLMOutputError, ValidationError
from nlapt.llm.cleaning import OUTPUT_CONSTRAINT
from nlapt.llm.mock import MockLLMClient
from nlapt.llm.translate import (
    CachedTranslation,
    Direction,
    TranslationCache,
    Translator,
    detect_direction,
)

FIXED_TIME = 1_700_000_000.0


def make_profile(**overrides: object) -> LLMProfile:
    defaults: dict = {
        "name": "p",
        "api_type": "openai",
        "base_url": "https://api.test/v1",
        "text_model": "text-model",
        "system_prompt": "You are a translator.",
    }
    defaults.update(overrides)
    return LLMProfile(**defaults)


class TestDetectDirection:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("a red fox in the snow", Direction.EN_TO_ZH),
            ("你好世界", Direction.ZH_TO_EN),
            ("", Direction.EN_TO_ZH),
            ("   \n\t", Direction.EN_TO_ZH),
            # Mixed: 2 CJK of 7 non-space chars -> ratio ~0.29 < 0.3.
            ("hello你好", Direction.EN_TO_ZH),
            # Mixed: 3 CJK of 5 non-space chars -> ratio 0.6.
            ("hi 你好吗", Direction.ZH_TO_EN),
            # Boundary: exactly 3 CJK of 10 -> ratio 0.3 (>= threshold).
            ("abcdefg你好吗", Direction.ZH_TO_EN),
            # Whitespace is ignored in the ratio.
            ("你 好 世 界", Direction.ZH_TO_EN),
            # Mostly English sentence with one CJK word.
            ("a girl wearing a 和服 standing in a garden", Direction.EN_TO_ZH),
        ],
    )
    def test_detection(self, text: str, expected: Direction) -> None:
        assert detect_direction(text) == expected

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValidationError):
            detect_direction(None)  # type: ignore[arg-type]


class TestTranslationCache:
    def test_get_missing_returns_none(self) -> None:
        cache = TranslationCache()
        assert cache.get("k") is None

    def test_put_and_get(self) -> None:
        cache = TranslationCache()
        entry = CachedTranslation(
            source_text="hello",
            translated="你好",
            direction=Direction.EN_TO_ZH,
            created_at=FIXED_TIME,
        )
        cache.put("k", entry)
        assert cache.get("k") is entry

    def test_is_stale_missing(self) -> None:
        cache = TranslationCache()
        assert cache.is_stale("k", "hello") is True

    def test_is_stale_on_source_change(self) -> None:
        cache = TranslationCache()
        entry = CachedTranslation(
            source_text="hello",
            translated="你好",
            direction=Direction.EN_TO_ZH,
            created_at=FIXED_TIME,
        )
        cache.put("k", entry)
        assert cache.is_stale("k", "hello") is False
        assert cache.is_stale("k", "hello world") is True

    def test_put_validation(self) -> None:
        cache = TranslationCache()
        entry = CachedTranslation(
            source_text="s",
            translated="t",
            direction=Direction.EN_TO_ZH,
            created_at=0.0,
        )
        with pytest.raises(ValidationError):
            cache.put("", entry)
        with pytest.raises(ValidationError):
            cache.put("k", "not-an-entry")  # type: ignore[arg-type]


class TestTranslator:
    def test_translate_uses_text_model_and_cleans_output(self) -> None:
        client = MockLLMClient(['"一只红色的狐狸"'])
        translator = Translator(
            client, make_profile(), cache=TranslationCache(), clock=lambda: FIXED_TIME
        )
        entry = translator.translate("img1.png", "a red fox")

        assert entry.translated == "一只红色的狐狸"  # quotes cleaned
        assert entry.direction == Direction.EN_TO_ZH
        assert entry.source_text == "a red fox"
        assert entry.created_at == FIXED_TIME

        request = client.requests[0]
        assert request.model == "text-model"
        assert request.system == "You are a translator."
        prompt = request.messages[0].text
        assert "a red fox" in prompt
        assert OUTPUT_CONSTRAINT in prompt

    def test_fresh_cache_hit_skips_llm_call(self) -> None:
        client = MockLLMClient(["译文"])
        cache = TranslationCache()
        translator = Translator(
            client, make_profile(), cache=cache, clock=lambda: FIXED_TIME
        )
        first = translator.translate("k", "a red fox")
        second = translator.translate("k", "a red fox")
        assert second is first
        assert client.calls == 1

    def test_source_change_invalidates_cache(self) -> None:
        client = MockLLMClient(["译文一", "译文二"])
        cache = TranslationCache()
        translator = Translator(
            client, make_profile(), cache=cache, clock=lambda: FIXED_TIME
        )
        translator.translate("k", "a red fox")
        assert cache.is_stale("k", "a blue fox") is True
        entry = translator.translate("k", "a blue fox")
        assert entry.translated == "译文二"
        assert client.calls == 2
        assert cache.get("k").source_text == "a blue fox"

    def test_direction_override_bypasses_matching_cache(self) -> None:
        client = MockLLMClient(["first", "second"])
        cache = TranslationCache()
        translator = Translator(
            client, make_profile(), cache=cache, clock=lambda: FIXED_TIME
        )
        translator.translate("k", "hello", Direction.EN_TO_ZH)
        # Same text but explicit opposite direction -> new LLM call.
        entry = translator.translate("k", "hello", Direction.ZH_TO_EN)
        assert entry.direction == Direction.ZH_TO_EN
        assert client.calls == 2

    def test_explicit_direction_selects_template(self) -> None:
        client = MockLLMClient(["result"])
        templates = {
            "translate_en_zh": "EN2ZH: {caption}",
            "translate_zh_en": "ZH2EN: {caption}",
        }
        translator = Translator(client, make_profile(), templates=templates)
        translator.translate("k", "hello", Direction.ZH_TO_EN)
        assert client.requests[0].messages[0].text.startswith("ZH2EN: hello")

    def test_auto_detection_selects_chinese_template(self) -> None:
        client = MockLLMClient(["result"])
        templates = {
            "translate_en_zh": "EN2ZH: {caption}",
            "translate_zh_en": "ZH2EN: {caption}",
        }
        translator = Translator(client, make_profile(), templates=templates)
        translator.translate("k", "你好世界")
        assert client.requests[0].messages[0].text.startswith("ZH2EN: 你好世界")

    def test_works_without_cache(self) -> None:
        client = MockLLMClient(["译文"])
        translator = Translator(client, make_profile())
        entry = translator.translate("k", "a red fox")
        assert entry.translated == "译文"

    def test_empty_text_raises(self) -> None:
        translator = Translator(MockLLMClient(["x"]), make_profile())
        with pytest.raises(ValidationError):
            translator.translate("k", "   ")

    def test_empty_key_raises(self) -> None:
        translator = Translator(MockLLMClient(["x"]), make_profile())
        with pytest.raises(ValidationError):
            translator.translate("", "hello")

    def test_missing_text_model_raises(self) -> None:
        translator = Translator(MockLLMClient(["x"]), make_profile(text_model=""))
        with pytest.raises(LLMConfigError):
            translator.translate("k", "hello")

    def test_empty_llm_output_raises_output_error(self) -> None:
        translator = Translator(MockLLMClient(['""']), make_profile())
        with pytest.raises(LLMOutputError):
            translator.translate("k", "hello")

    def test_invalid_client_raises(self) -> None:
        with pytest.raises(ValidationError):
            Translator("not-a-client", make_profile())  # type: ignore[arg-type]
