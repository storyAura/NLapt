"""Tests for the httpx-based OpenAI/Anthropic/Ollama clients (MockTransport)."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx
import pytest

from nlapt.core.config import LLMProfile
from nlapt.core.errors import LLMRequestError, LLMTimeoutError
from nlapt.llm.anthropic_client import AnthropicClient
from nlapt.llm.base import LLMMessage, LLMRequest, create_client
from nlapt.llm.ollama_client import OllamaClient
from nlapt.llm.openai_client import OpenAIClient

API_KEY = "sk-secret-key-abcdef123456"
PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepixels"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


def make_profile(api_type: str, base_url: str) -> LLMProfile:
    return LLMProfile(
        name="p1",
        api_type=api_type,
        base_url=base_url,
        api_key=API_KEY,
        text_model="text-model",
        vision_model="vision-model",
    )


def capture_transport(
    captured: list[httpx.Request],
    *,
    status_code: int = 200,
    json_body: Any = None,
    text_body: str | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if text_body is not None:
            return httpx.Response(status_code, text=text_body)
        return httpx.Response(status_code, json=json_body)

    return httpx.MockTransport(handler)


def timeout_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated timeout")

    return httpx.MockTransport(handler)


def basic_request(**overrides: Any) -> LLMRequest:
    defaults: dict[str, Any] = {
        "messages": (LLMMessage(role="user", text="describe"),),
        "model": "text-model",
        "system": "be brief",
        "temperature": 0.3,
        "max_tokens": 128,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)


OPENAI_OK = {"choices": [{"message": {"content": "a red fox"}}]}
ANTHROPIC_OK = {"content": [{"type": "text", "text": "a red fox"}]}
OLLAMA_OK = {"message": {"content": "a red fox"}}


class TestOpenAIClient:
    def test_payload_shape_and_headers(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("openai", "https://api.test/v1")
        client = OpenAIClient(profile, transport=capture_transport(captured, json_body=OPENAI_OK))
        response = client.complete(basic_request())

        assert response.text == "a red fox"
        assert response.model == "text-model"
        request = captured[0]
        assert str(request.url) == "https://api.test/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        payload = json.loads(request.content)
        assert payload["model"] == "text-model"
        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 128
        assert payload["messages"][0] == {"role": "system", "content": "be brief"}
        assert payload["messages"][1] == {"role": "user", "content": "describe"}

    def test_image_sent_as_data_url(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("openai", "https://api.test/v1")
        client = OpenAIClient(profile, transport=capture_transport(captured, json_body=OPENAI_OK))
        message = LLMMessage(role="user", text="describe", images=(PNG_BYTES,))
        client.complete(basic_request(messages=(message,), model="vision-model"))

        payload = json.loads(captured[0].content)
        content = payload["messages"][1]["content"]
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image_url"
        url = content[1]["image_url"]["url"]
        assert url == f"data:image/png;base64,{PNG_B64}"

    def test_http_error_raises_with_status(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("openai", "https://api.test/v1")
        client = OpenAIClient(
            profile,
            transport=capture_transport(
                captured, status_code=500, json_body={"error": "boom"}
            ),
        )
        with pytest.raises(LLMRequestError) as excinfo:
            client.complete(basic_request())
        assert "500" in str(excinfo.value)
        assert API_KEY not in str(excinfo.value)

    def test_timeout_raises_timeout_error_without_key(self) -> None:
        profile = make_profile("openai", "https://api.test/v1")
        client = OpenAIClient(profile, transport=timeout_transport())
        with pytest.raises(LLMTimeoutError) as excinfo:
            client.complete(basic_request())
        assert API_KEY not in str(excinfo.value)

    def test_malformed_response_raises(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("openai", "https://api.test/v1")
        client = OpenAIClient(
            profile, transport=capture_transport(captured, json_body={"nope": 1})
        )
        with pytest.raises(LLMRequestError):
            client.complete(basic_request())

    def test_non_json_response_raises(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("openai", "https://api.test/v1")
        client = OpenAIClient(
            profile, transport=capture_transport(captured, text_body="<html>oops</html>")
        )
        with pytest.raises(LLMRequestError):
            client.complete(basic_request())

    def test_no_auth_header_without_key(self) -> None:
        captured: list[httpx.Request] = []
        profile = LLMProfile(name="p", api_type="openai", base_url="https://api.test/v1")
        client = OpenAIClient(profile, transport=capture_transport(captured, json_body=OPENAI_OK))
        client.complete(basic_request())
        assert "Authorization" not in captured[0].headers

    def test_api_key_never_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("openai", "https://api.test/v1")
        client = OpenAIClient(profile, transport=capture_transport(captured, json_body=OPENAI_OK))
        with caplog.at_level(logging.DEBUG, logger="nlapt"):
            client.complete(basic_request())
        assert API_KEY not in caplog.text


class TestAnthropicClient:
    def test_payload_shape_and_headers(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("anthropic", "https://api.test")
        client = AnthropicClient(
            profile, transport=capture_transport(captured, json_body=ANTHROPIC_OK)
        )
        response = client.complete(basic_request())

        assert response.text == "a red fox"
        request = captured[0]
        assert str(request.url) == "https://api.test/v1/messages"
        assert request.headers["x-api-key"] == API_KEY
        assert request.headers["anthropic-version"] == "2023-06-01"
        payload = json.loads(request.content)
        assert payload["model"] == "text-model"
        assert payload["system"] == "be brief"
        assert payload["max_tokens"] == 128
        assert payload["messages"] == [{"role": "user", "content": "describe"}]

    def test_image_sent_as_base64_source_block(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("anthropic", "https://api.test")
        client = AnthropicClient(
            profile, transport=capture_transport(captured, json_body=ANTHROPIC_OK)
        )
        message = LLMMessage(role="user", text="describe", images=(PNG_BYTES,))
        client.complete(basic_request(messages=(message,), model="vision-model"))

        payload = json.loads(captured[0].content)
        content = payload["messages"][0]["content"]
        assert content[0] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": PNG_B64,
            },
        }
        assert content[1] == {"type": "text", "text": "describe"}

    def test_http_error_and_key_safety(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("anthropic", "https://api.test")
        client = AnthropicClient(
            profile,
            transport=capture_transport(
                captured, status_code=401, json_body={"error": "unauthorized"}
            ),
        )
        with pytest.raises(LLMRequestError) as excinfo:
            client.complete(basic_request())
        assert "401" in str(excinfo.value)
        assert API_KEY not in str(excinfo.value)

    def test_timeout(self) -> None:
        profile = make_profile("anthropic", "https://api.test")
        client = AnthropicClient(profile, transport=timeout_transport())
        with pytest.raises(LLMTimeoutError):
            client.complete(basic_request())

    def test_missing_content_blocks_raise(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("anthropic", "https://api.test")
        client = AnthropicClient(
            profile, transport=capture_transport(captured, json_body={"content": []})
        )
        with pytest.raises(LLMRequestError):
            client.complete(basic_request())


class TestOllamaClient:
    def test_payload_shape(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("ollama", "http://localhost:11434")
        client = OllamaClient(
            profile, transport=capture_transport(captured, json_body=OLLAMA_OK)
        )
        response = client.complete(basic_request())

        assert response.text == "a red fox"
        request = captured[0]
        assert str(request.url) == "http://localhost:11434/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "text-model"
        assert payload["stream"] is False
        assert payload["options"] == {"temperature": 0.3, "num_predict": 128}
        assert payload["messages"][0] == {"role": "system", "content": "be brief"}
        assert payload["messages"][1] == {"role": "user", "content": "describe"}

    def test_images_sent_as_base64_list(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("ollama", "http://localhost:11434")
        client = OllamaClient(
            profile, transport=capture_transport(captured, json_body=OLLAMA_OK)
        )
        message = LLMMessage(role="user", text="describe", images=(PNG_BYTES,))
        client.complete(basic_request(messages=(message,)))

        payload = json.loads(captured[0].content)
        assert payload["messages"][1]["images"] == [PNG_B64]

    def test_http_error(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("ollama", "http://localhost:11434")
        client = OllamaClient(
            profile,
            transport=capture_transport(captured, status_code=404, json_body={}),
        )
        with pytest.raises(LLMRequestError) as excinfo:
            client.complete(basic_request())
        assert "404" in str(excinfo.value)

    def test_timeout(self) -> None:
        profile = make_profile("ollama", "http://localhost:11434")
        client = OllamaClient(profile, transport=timeout_transport())
        with pytest.raises(LLMTimeoutError):
            client.complete(basic_request())

    def test_malformed_response(self) -> None:
        captured: list[httpx.Request] = []
        profile = make_profile("ollama", "http://localhost:11434")
        client = OllamaClient(
            profile, transport=capture_transport(captured, json_body={"message": {}})
        )
        with pytest.raises(LLMRequestError):
            client.complete(basic_request())


class TestCreateClientIntegration:
    @pytest.mark.parametrize(
        ("api_type", "expected_class"),
        [
            ("openai", OpenAIClient),
            ("anthropic", AnthropicClient),
            ("ollama", OllamaClient),
        ],
    )
    def test_create_client_builds_expected_type(
        self, api_type: str, expected_class: type
    ) -> None:
        client = create_client(make_profile(api_type, "https://api.test"))
        assert isinstance(client, expected_class)
