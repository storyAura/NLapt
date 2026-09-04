"""Tests for nlapt.llm.base: models, registry, lazy httpx dependency."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import nlapt.llm  # noqa: F401  (registers built-in clients)
from nlapt.core.config import LLMProfile
from nlapt.core.errors import LLMConfigError, ValidationError
from nlapt.llm.base import (
    CLIENT_REGISTRY,
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    create_client,
    detect_image_media_type,
    join_url,
    register_client,
    require_httpx,
)
from nlapt.llm.mock import MockLLMClient

PNG_BYTES = b"\x89PNG\r\n\x1a\nrest"
JPEG_BYTES = b"\xff\xd8\xff\xe0rest"


def make_profile(**overrides: object) -> LLMProfile:
    defaults: dict = {
        "name": "test",
        "api_type": "openai",
        "base_url": "https://api.test/v1",
        "api_key": "sk-secret-key-123456",
        "text_model": "gpt-test",
        "vision_model": "gpt-test-vision",
    }
    defaults.update(overrides)
    return LLMProfile(**defaults)


class TestLLMMessage:
    def test_valid_roles(self) -> None:
        for role in ("user", "assistant"):
            message = LLMMessage(role=role, text="hi")
            assert message.role == role

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMMessage(role="system", text="hi")

    def test_non_string_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMMessage(role="user", text=123)  # type: ignore[arg-type]

    def test_images_coerced_to_tuple(self) -> None:
        message = LLMMessage(role="user", text="hi", images=[b"a", b"b"])
        assert message.images == (b"a", b"b")
        assert isinstance(message.images, tuple)

    def test_non_bytes_image_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMMessage(role="user", text="hi", images=("not-bytes",))  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        message = LLMMessage(role="user", text="hi")
        with pytest.raises(AttributeError):
            message.text = "other"  # type: ignore[misc]


class TestLLMRequest:
    def test_defaults(self) -> None:
        request = LLMRequest(
            messages=(LLMMessage(role="user", text="hi"),), model="m"
        )
        assert request.system == ""
        assert request.temperature == 0.7
        assert request.max_tokens == 1024
        assert request.timeout == 60.0

    def test_empty_messages_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(messages=(), model="m")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(messages=(LLMMessage(role="user", text="x"),), model="")

    def test_bad_timeout_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=(LLMMessage(role="user", text="x"),), model="m", timeout=0
            )

    def test_non_message_entry_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(messages=("hello",), model="m")  # type: ignore[arg-type]


class TestRegistry:
    def test_builtin_types_registered(self) -> None:
        for api_type in ("openai", "anthropic", "ollama"):
            assert api_type in CLIENT_REGISTRY

    def test_create_client_unknown_type(self) -> None:
        with pytest.raises(LLMConfigError):
            create_client(make_profile(api_type="does-not-exist"))

    def test_create_client_missing_base_url(self) -> None:
        with pytest.raises(LLMConfigError):
            create_client(make_profile(base_url=""))

    def test_create_client_missing_api_type(self) -> None:
        with pytest.raises(LLMConfigError):
            create_client(make_profile(api_type=""))

    def test_create_client_non_profile(self) -> None:
        with pytest.raises(ValidationError):
            create_client("nope")  # type: ignore[arg-type]

    def test_register_and_create_custom_type(self) -> None:
        created: list[LLMProfile] = []

        class DummyClient(LLMClient):
            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(text="dummy", model=request.model)

        def factory(profile: LLMProfile) -> LLMClient:
            created.append(profile)
            return DummyClient()

        register_client("custom-test-type", factory)
        try:
            profile = make_profile(api_type="custom-test-type")
            client = create_client(profile)
            assert isinstance(client, DummyClient)
            assert created == [profile]
        finally:
            CLIENT_REGISTRY.pop("custom-test-type", None)

    def test_register_invalid_inputs(self) -> None:
        with pytest.raises(ValidationError):
            register_client("", lambda profile: None)  # type: ignore[arg-type, return-value]
        with pytest.raises(ValidationError):
            register_client("x-type", "not-callable")  # type: ignore[arg-type]


class TestLazyHttpx:
    def test_client_modules_have_no_top_level_httpx_import(self) -> None:
        package_dir = Path(nlapt.llm.__file__).parent
        for module_name in ("openai_client.py", "anthropic_client.py", "ollama_client.py", "base.py"):
            tree = ast.parse((package_dir / module_name).read_text(encoding="utf-8"))
            for node in tree.body:  # top level only
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                    assert "httpx" not in names, f"{module_name} imports httpx eagerly"
                if isinstance(node, ast.ImportFrom):
                    assert node.module != "httpx", f"{module_name} imports httpx eagerly"

    def test_require_httpx_missing_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "httpx", None)
        with pytest.raises(LLMConfigError) as excinfo:
            require_httpx()
        assert "httpx" in str(excinfo.value)


class TestConnectionTest:
    def test_default_test_connection_sends_minimal_request(self) -> None:
        client = MockLLMClient(["pong"])
        assert client.test_connection("model-x") is True
        assert len(client.requests) == 1
        assert client.requests[0].model == "model-x"

    def test_budget_survives_thinking_models(self) -> None:
        """Regression: an 8-token cap made 测试连接 always fail on thinking
        models (the entire budget went to hidden reasoning, content came back
        empty). Keep enough room for the reasoning plus the visible reply."""
        from nlapt.llm.base import CONNECTION_TEST_MAX_TOKENS

        client = MockLLMClient(["pong"])
        client.test_connection("model-x")
        assert client.requests[0].max_tokens == CONNECTION_TEST_MAX_TOKENS
        assert CONNECTION_TEST_MAX_TOKENS >= 512

    def test_empty_model_raises(self) -> None:
        client = MockLLMClient(["pong"])
        with pytest.raises(LLMConfigError):
            client.test_connection("")


class TestHelpers:
    def test_detect_image_media_type(self) -> None:
        assert detect_image_media_type(PNG_BYTES) == "image/png"
        assert detect_image_media_type(JPEG_BYTES) == "image/jpeg"
        assert detect_image_media_type(b"RIFF1234WEBPxxxx") == "image/webp"
        assert detect_image_media_type(b"unknown") == "image/jpeg"

    def test_join_url_strips_trailing_slash(self) -> None:
        assert join_url("https://a/v1/", "/chat") == "https://a/v1/chat"
        assert join_url("https://a/v1", "/chat") == "https://a/v1/chat"
        with pytest.raises(LLMConfigError):
            join_url("  ", "/chat")


class TestMockClient:
    def test_scripted_responses_in_order_then_repeat_last(self) -> None:
        client = MockLLMClient(["one", "two"])
        request = LLMRequest(messages=(LLMMessage(role="user", text="x"),), model="m")
        assert client.complete(request).text == "one"
        assert client.complete(request).text == "two"
        assert client.complete(request).text == "two"

    def test_single_string_is_one_response(self) -> None:
        client = MockLLMClient("hello world")
        request = LLMRequest(messages=(LLMMessage(role="user", text="x"),), model="m")
        assert client.complete(request).text == "hello world"

    def test_callable_provider(self) -> None:
        client = MockLLMClient(lambda request: f"echo:{request.model}")
        request = LLMRequest(messages=(LLMMessage(role="user", text="x"),), model="m2")
        assert client.complete(request).text == "echo:m2"

    def test_fail_times_then_success(self) -> None:
        from nlapt.core.errors import LLMRequestError

        client = MockLLMClient(["ok"], fail_times=2)
        request = LLMRequest(messages=(LLMMessage(role="user", text="x"),), model="m")
        for _ in range(2):
            with pytest.raises(LLMRequestError):
                client.complete(request)
        assert client.complete(request).text == "ok"
        assert client.calls == 3

    def test_empty_responses_raises(self) -> None:
        with pytest.raises(ValidationError):
            MockLLMClient([])

    def test_negative_fail_times_raises(self) -> None:
        with pytest.raises(ValidationError):
            MockLLMClient(["x"], fail_times=-1)
