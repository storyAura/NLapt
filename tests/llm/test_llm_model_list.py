"""Tests for nlapt.llm.model_list (设置 ▸ 获取模型)."""

from __future__ import annotations

import httpx
import pytest

from nlapt.core.config import LLMProfile
from nlapt.core.errors import LLMConfigError, LLMRequestError, ValidationError
from nlapt.llm.model_list import list_models, looks_vision_capable


def _profile(api_type: str, *, api_key: str = "") -> LLMProfile:
    return LLMProfile(
        name="p", api_type=api_type, base_url="http://mock/v1", api_key=api_key
    )


def _transport(payload: object, *, status: int = 200) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler), seen


class TestOpenAI:
    def test_lists_sorted_unique_ids(self) -> None:
        transport, seen = _transport(
            {"data": [{"id": "gpt-b"}, {"id": "gpt-a"}, {"id": "gpt-b"}]}
        )
        models = list_models(_profile("openai", api_key="sk-1"), transport=transport)
        assert models == ("gpt-a", "gpt-b")
        assert seen[0].url.path.endswith("/models")
        assert seen[0].headers["Authorization"] == "Bearer sk-1"

    def test_no_key_sends_no_auth_header(self) -> None:
        transport, seen = _transport({"data": []})
        list_models(_profile("openai"), transport=transport)
        assert "Authorization" not in seen[0].headers

    def test_http_error_raises(self) -> None:
        transport, _seen = _transport({"error": "x"}, status=500)
        with pytest.raises(LLMRequestError):
            list_models(_profile("openai"), transport=transport)

    def test_malformed_payload_raises(self) -> None:
        transport, _seen = _transport({"unexpected": True})
        with pytest.raises(LLMRequestError):
            list_models(_profile("openai"), transport=transport)


class TestAnthropic:
    def test_lists_with_api_key_header(self) -> None:
        transport, seen = _transport({"data": [{"id": "claude-fable-5"}]})
        models = list_models(
            _profile("anthropic", api_key="ak-1"), transport=transport
        )
        assert models == ("claude-fable-5",)
        assert seen[0].url.path.endswith("/v1/models")
        assert seen[0].headers["x-api-key"] == "ak-1"
        assert "anthropic-version" in seen[0].headers


class TestOllama:
    def test_lists_tag_names(self) -> None:
        transport, seen = _transport(
            {"models": [{"name": "llava:13b"}, {"name": "qwen2.5"}]}
        )
        models = list_models(_profile("ollama"), transport=transport)
        assert models == ("llava:13b", "qwen2.5")
        assert seen[0].url.path.endswith("/api/tags")


class TestValidation:
    def test_missing_base_url_raises(self) -> None:
        profile = LLMProfile(name="p", api_type="openai", base_url="")
        with pytest.raises(LLMConfigError):
            list_models(profile)

    def test_unknown_api_type_raises(self) -> None:
        with pytest.raises(LLMConfigError):
            list_models(_profile("weird"))

    def test_non_profile_raises(self) -> None:
        with pytest.raises(ValidationError):
            list_models("nope")  # type: ignore[arg-type]


class TestVisionHeuristic:
    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-4o-mini",
            "claude-sonnet-5",
            "gemini-2.5-flash",
            "qwen2.5-vl-7b",
            "llava:13b",
            "glm-4v-plus",
            "some-vision-model",
        ],
    )
    def test_vision_names(self, model_id: str) -> None:
        assert looks_vision_capable(model_id)

    @pytest.mark.parametrize("model_id", ["deepseek-r1", "text-embedding-3", "qwen2.5"])
    def test_text_names(self, model_id: str) -> None:
        assert not looks_vision_capable(model_id)
