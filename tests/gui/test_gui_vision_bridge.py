"""Tests for nlapt_gui.vision_bridge (标注工作区 ▸ 重译 async captioning)."""

from __future__ import annotations

from typing import Iterator

import pytest

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, LLMProfile, RequestControl
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.controller import AppController
from nlapt_gui.prompt_store import DEFAULT_USER_PROMPT, VisionPrompts
from nlapt_gui.settings import UISettings
from nlapt_gui.vision_bridge import NOTE_VISION_UNCONFIGURED, VisionBridge

# Unique api_type for this module (registry is process-global).
API_TYPE = "gui-vision-mock"

_RESPONSES: list[str] = ["a fresh caption"]


class _Recorder:
    last: MockLLMClient | None = None


def _factory(profile: LLMProfile) -> MockLLMClient:
    client = MockLLMClient(list(_RESPONSES))
    _Recorder.last = client
    return client


register_client(API_TYPE, _factory)


def _app(*, vision_model: str) -> NLaptApp:
    profile = LLMProfile(
        name="default",
        api_type=API_TYPE,
        base_url="http://mock",
        text_model="m-text",
        vision_model=vision_model,
        system_prompt="profile-system",
    )
    config = AppConfig(
        profiles=(profile,),
        active_profile="default",
        request=RequestControl(max_retries=0),
    )
    return NLaptApp(config=config)


@pytest.fixture()
def vision_controller(qtbot, demo_dataset) -> Iterator[AppController]:
    _RESPONSES[:] = ["a fresh caption"]
    _Recorder.last = None
    ctrl = AppController(_app(vision_model="m-vision"), settings=UISettings())
    with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
        ctrl.open_dataset(demo_dataset)
    yield ctrl


class TestConfigured:
    def test_configured_with_vision_model(self, vision_controller) -> None:
        assert VisionBridge(vision_controller).configured()

    def test_unconfigured_without_vision_model(self, qtbot, demo_dataset) -> None:
        ctrl = AppController(_app(vision_model=""), settings=UISettings())
        with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
            ctrl.open_dataset(demo_dataset)
        assert not VisionBridge(ctrl).configured()

    def test_unconfigured_without_profile(self, qtbot) -> None:
        ctrl = AppController(NLaptApp(), settings=UISettings())
        assert not VisionBridge(ctrl).configured()


class TestRequest:
    def test_emits_caption(self, qtbot, vision_controller) -> None:
        bridge = VisionBridge(vision_controller)
        with qtbot.waitSignal(bridge.caption_ready, timeout=3000) as blocker:
            bridge.request("0001.png")
        assert blocker.args == ["0001.png", "a fresh caption", True]
        request = _Recorder.last.requests[0]
        assert request.model == "m-vision"
        assert request.messages[0].images  # image bytes attached
        assert request.messages[0].text == DEFAULT_USER_PROMPT
        assert request.system == "profile-system"

    def test_uses_custom_prompts(self, qtbot, vision_controller) -> None:
        prompts = VisionPrompts(
            active="动漫", prompts={"动漫": "custom-system"}, user_prompt="custom-user"
        )
        bridge = VisionBridge(vision_controller, prompts=prompts)
        with qtbot.waitSignal(bridge.caption_ready, timeout=3000):
            bridge.request("0001.png")
        request = _Recorder.last.requests[0]
        assert request.system == "custom-system"
        assert request.messages[0].text == "custom-user"

    def test_unconfigured_emits_note(self, qtbot) -> None:
        ctrl = AppController(NLaptApp(), settings=UISettings())
        bridge = VisionBridge(ctrl)
        with qtbot.waitSignal(bridge.caption_ready, timeout=1000) as blocker:
            bridge.request("0001.png")
        assert blocker.args == ["0001.png", NOTE_VISION_UNCONFIGURED, False]

    def test_unknown_key_emits_error(self, qtbot, vision_controller) -> None:
        bridge = VisionBridge(vision_controller)
        with qtbot.waitSignal(bridge.caption_ready, timeout=1000) as blocker:
            bridge.request("missing.png")
        assert blocker.args[0] == "missing.png"
        assert blocker.args[2] is False
