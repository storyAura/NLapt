"""Facade factories wiring the active profile + RequestControl into LLM services."""

from __future__ import annotations

import pytest

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, LLMProfile, RequestControl
from nlapt.core.errors import LLMConfigError
from nlapt.llm.base import LLMRequest, register_client
from nlapt.llm.mock import MockLLMClient
from nlapt.llm.rewrite import RewriteSpec, RewriteType
from nlapt.llm.translate import Translator

API_TYPE = "test-app-factories"
TIMEOUT = 42.5


@pytest.fixture(scope="module", autouse=True)
def _register_mock_api_type() -> None:
    captured: list[LLMRequest] = []

    def factory(profile: LLMProfile) -> MockLLMClient:
        return MockLLMClient(lambda request: captured.append(request) or "mock output")

    register_client(API_TYPE, factory)


def _make_config(*, vision_model: str = "", active: str = "main") -> AppConfig:
    profile = LLMProfile(
        name="main",
        api_type=API_TYPE,
        base_url="http://localhost:9",
        text_model="text-model",
        vision_model=vision_model,
    )
    return AppConfig(
        profiles=(profile,),
        active_profile=active,
        request=RequestControl(timeout=TIMEOUT),
        custom_templates={"translate_en_zh": "XYZTEMPLATE {caption}"},
    )


def _captured_request(client: MockLLMClient) -> LLMRequest:
    # MockLLMClient records every request; the provider closure above also
    # sees them, but reading through the client keeps the test self-contained.
    return client.requests[-1]


class TestMakeRewriteService:
    def test_requires_active_profile(self) -> None:
        app = NLaptApp(config=AppConfig())
        with pytest.raises(LLMConfigError):
            app.make_rewrite_service()

    def test_request_control_reaches_the_client(self) -> None:
        app = NLaptApp(config=_make_config())
        service = app.make_rewrite_service(trigger="trig")
        result = service.run_one(
            RewriteSpec(type=RewriteType.POLISH),
            key="a.png",
            caption="1girl, smile",
            filename="a.png",
        )
        assert result.result == "mock output"
        request = _captured_request(service._text_client)  # noqa: SLF001
        assert request.timeout == TIMEOUT
        assert request.model == "text-model"

    def test_vision_client_only_when_vision_model_configured(self) -> None:
        spec = RewriteSpec(type=RewriteType.POLISH, use_vision=True)
        text_only = NLaptApp(config=_make_config()).make_rewrite_service()
        with pytest.raises(LLMConfigError):
            text_only.run_one(spec, key="a.png", caption="c", filename="a.png", image=b"x")

        with_vision = NLaptApp(config=_make_config(vision_model="vis-model"))
        service = with_vision.make_rewrite_service()
        service.run_one(spec, key="a.png", caption="c", filename="a.png", image=b"x")
        request = _captured_request(service._vision_client)  # noqa: SLF001
        assert request.model == "vis-model"
        assert request.timeout == TIMEOUT


class TestMakeTranslator:
    def test_requires_active_profile(self) -> None:
        app = NLaptApp(config=AppConfig(profiles=(), active_profile="missing"))
        with pytest.raises(LLMConfigError):
            app.make_translator()

    def test_custom_template_and_timeout_applied(self) -> None:
        app = NLaptApp(config=_make_config())
        translator = app.make_translator()
        assert isinstance(translator, Translator)
        entry = translator.translate("a.png", "an english caption")
        assert entry.translated == "mock output"
        request = _captured_request(translator._client)  # noqa: SLF001
        assert request.timeout == TIMEOUT
        assert "XYZTEMPLATE" in request.messages[0].text
        assert "an english caption" in request.messages[0].text
