"""Tests for nlapt_gui.vision_bridge (LLM / 本地 async captioning + batches)."""

from __future__ import annotations

from typing import Iterator

import pytest

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, LLMProfile, RequestControl
from nlapt.core.errors import LocalInferenceError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.controller import AppController
from nlapt_gui.layered_prompts import CardParts
from nlapt_gui.prompt_store import (
    DEFAULT_USER_PROMPT,
    ENGINE_LLM,
    ENGINE_LOCAL,
    VisionPrompts,
)
from nlapt_gui.settings import UISettings
from nlapt_gui.vision_bridge import NOTE_VISION_UNCONFIGURED, VisionBridge

# Unique api_type for this module (registry is process-global).
API_TYPE = "gui-vision-mock"

_RESPONSES: list[str] = ["a fresh caption"]
# First N complete() calls fail per client (retry tests); reset per fixture.
_FAIL_TIMES: list[int] = [0]


class _Recorder:
    last: MockLLMClient | None = None


def _factory(profile: LLMProfile) -> MockLLMClient:
    client = MockLLMClient(list(_RESPONSES), fail_times=_FAIL_TIMES[0])
    _Recorder.last = client
    return client


register_client(API_TYPE, _factory)


def _app(*, vision_model: str, max_retries: int = 0) -> NLaptApp:
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
        request=RequestControl(max_retries=max_retries),
    )
    return NLaptApp(config=config)


@pytest.fixture(autouse=True)
def _isolate_idle_stopper(monkeypatch) -> None:
    """Keep the process-wide idle stopper (real 30s timers) out of this module.

    Tests that assert on stopper behavior patch get_idle_stopper again with
    their own instance; everything else silently uses this inert one.
    """
    stopper, _timers = make_stopper(FakeServerManager())
    monkeypatch.setattr(
        "nlapt_gui.vision_bridge.get_idle_stopper", lambda: stopper
    )


@pytest.fixture()
def vision_controller(qtbot, demo_dataset) -> Iterator[AppController]:
    _RESPONSES[:] = ["a fresh caption"]
    _FAIL_TIMES[0] = 0
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


class TestInterceptionHandling:
    """Intercepted/refused replies fail visibly; transient errors retry."""

    def test_refusal_reply_fails_instead_of_writing(
        self, qtbot, vision_controller
    ) -> None:
        _RESPONSES[:] = ["I'm sorry, but I can't describe this image."]
        bridge = VisionBridge(vision_controller)
        with qtbot.waitSignal(bridge.caption_ready, timeout=3000) as blocker:
            bridge.request("0001.png")
        assert blocker.args[0] == "0001.png"
        assert blocker.args[2] is False
        assert "拦截" in blocker.args[1]

    def test_transient_failure_retried_with_injected_sleep(
        self, qtbot, demo_dataset
    ) -> None:
        _RESPONSES[:] = ["a fresh caption"]
        _FAIL_TIMES[0] = 1  # first complete() raises LLMRequestError
        _Recorder.last = None
        ctrl = AppController(
            _app(vision_model="m-vision", max_retries=1), settings=UISettings()
        )
        with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
            ctrl.open_dataset(demo_dataset)
        sleeps: list[float] = []
        captioner = ctrl.make_vision_captioner_or_none(retry_sleep=sleeps.append)
        assert captioner is not None
        result = captioner(ctrl.image_path("0001.png"), "sys", "user")
        assert result == "a fresh caption"
        assert sleeps  # backoff waited through the injected sleep only
        assert _Recorder.last is not None
        assert _Recorder.last.calls == 2  # failed once, then retried


LOCAL_PROMPTS = VisionPrompts(
    active="动漫",
    prompts={"动漫": "shared-sys"},
    user_prompt="shared-user",
    local_unified=False,
    local_system="loc-sys",
    local_user_prompt="loc-user",
)


class TestLocalEngine:
    def test_local_request_uses_factory_and_local_prompts(
        self, qtbot, vision_controller
    ) -> None:
        calls: list[tuple[str, str]] = []

        def captioner(path, system, user):  # noqa: ANN001
            calls.append((system, user))
            return "local caption"

        bridge = VisionBridge(
            vision_controller,
            prompts=LOCAL_PROMPTS,
            local_captioner_factory=lambda: captioner,
        )
        with qtbot.waitSignal(bridge.caption_ready, timeout=3000) as blocker:
            bridge.request("0001.png", ENGINE_LOCAL)
        assert blocker.args == ["0001.png", "local caption", True]
        assert calls == [("loc-sys", "loc-user")]

    def test_local_unified_prompts_follow_shared_set(
        self, qtbot, vision_controller
    ) -> None:
        prompts = LOCAL_PROMPTS.with_changes(local_unified=True)
        calls: list[tuple[str, str]] = []

        def captioner(path, system, user):  # noqa: ANN001
            calls.append((system, user))
            return "x"

        bridge = VisionBridge(
            vision_controller,
            prompts=prompts,
            local_captioner_factory=lambda: captioner,
        )
        with qtbot.waitSignal(bridge.caption_ready, timeout=3000):
            bridge.request("0001.png", ENGINE_LOCAL)
        assert calls == [("shared-sys", "shared-user")]

    def test_local_factory_error_is_emitted(self, qtbot, vision_controller) -> None:
        def broken():
            raise LocalInferenceError("本地模型尚未下载完成")

        bridge = VisionBridge(vision_controller, local_captioner_factory=broken)
        with qtbot.waitSignal(bridge.caption_ready, timeout=1000) as blocker:
            bridge.request("0001.png", ENGINE_LOCAL)
        assert blocker.args[2] is False
        assert "尚未下载" in blocker.args[1]

    def test_local_unconfigured_by_default(self, qtbot, vision_controller) -> None:
        # Fresh isolated data dir: no local model selected -> not configured.
        assert not VisionBridge(vision_controller).configured(ENGINE_LOCAL)
        assert VisionBridge(vision_controller).configured(ENGINE_LLM)


class FakeServerManager:
    def __init__(self, *, running: bool = False) -> None:
        self.running = running
        self.stop_calls = 0

    def is_running(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


class FakeTimer:
    """Records the armed delay; fires only when the test says so."""

    def __init__(self, delay: float, fn) -> None:  # noqa: ANN001
        self.delay = delay
        self.fn = fn
        self.daemon = False
        self.cancelled = False

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.fn()


def make_stopper(manager: FakeServerManager):
    """(IdleServerStopper with fake timers, list of armed timers)."""
    from nlapt_gui.local_bridge import IdleServerStopper

    timers: list[FakeTimer] = []

    def factory(delay: float, fn):  # noqa: ANN001
        timer = FakeTimer(delay, fn)
        timers.append(timer)
        return timer

    return IdleServerStopper(manager, timer_factory=factory), timers


class TestRequestBatch:
    def test_local_batch_arms_idle_stop_instead_of_immediate_unload(
        self, qtbot, vision_controller, monkeypatch
    ) -> None:
        from nlapt_gui.local_bridge import IDLE_STOP_DELAY_SECONDS

        manager = FakeServerManager(running=False)
        stopper, timers = make_stopper(manager)
        monkeypatch.setattr(
            "nlapt_gui.vision_bridge.get_idle_stopper", lambda: stopper
        )
        bridge = VisionBridge(
            vision_controller,
            prompts=VisionPrompts(),
            local_captioner_factory=lambda: (
                lambda path, system, user: f"batch caption {path.name}"
            ),
        )
        keys = ("0001.png", "0002.png")
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_batch(keys, ENGINE_LOCAL)
        for key in keys:
            assert vision_controller.record(key).text.startswith("batch caption")
            assert vision_controller.history.entries(key)[0].label == "推标(本地模型)"
        # 推理完先不卸载: nothing stopped yet, the 30s idle timer is armed.
        qtbot.waitUntil(lambda: stopper.pending(), timeout=2000)
        assert manager.stop_calls == 0
        assert timers[-1].delay == IDLE_STOP_DELAY_SECONDS
        # Only the idle window elapsing actually unloads the model.
        timers[-1].fire()
        assert manager.stop_calls == 1

    def test_new_batch_within_idle_window_keeps_model_loaded(
        self, qtbot, vision_controller, monkeypatch
    ) -> None:
        manager = FakeServerManager(running=False)
        stopper, timers = make_stopper(manager)
        monkeypatch.setattr(
            "nlapt_gui.vision_bridge.get_idle_stopper", lambda: stopper
        )
        bridge = VisionBridge(
            vision_controller,
            prompts=VisionPrompts(),
            local_captioner_factory=lambda: (lambda path, system, user: "cap"),
        )
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_batch(("0001.png",), ENGINE_LOCAL)
        qtbot.waitUntil(lambda: stopper.pending(), timeout=2000)
        first_timer = timers[-1]
        # 少数推理需要反复: the next run cancels the pending stop.
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_batch(("0002.png",), ENGINE_LOCAL)
        assert first_timer.cancelled
        first_timer.fire()  # a late fire of the cancelled timer is inert
        assert manager.stop_calls == 0

    def test_local_batch_keeps_a_prestarted_server(
        self, qtbot, vision_controller, monkeypatch
    ) -> None:
        manager = FakeServerManager(running=True)
        stopper, timers = make_stopper(manager)
        monkeypatch.setattr(
            "nlapt_gui.vision_bridge.get_idle_stopper", lambda: stopper
        )
        bridge = VisionBridge(
            vision_controller,
            prompts=VisionPrompts(),
            local_captioner_factory=lambda: (lambda path, system, user: "cap"),
        )
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_batch(("0001.png",), ENGINE_LOCAL)
        qtbot.wait(50)  # give a wrong stop/arm a chance to fire
        assert manager.stop_calls == 0
        assert not timers  # user-started server: no idle stop is ever armed

    def test_unready_engine_toasts_and_refuses(
        self, qtbot, vision_controller
    ) -> None:
        toasts: list[tuple[str, str]] = []
        vision_controller.toast_requested.connect(
            lambda text, kind: toasts.append((text, kind))
        )

        def broken():
            raise LocalInferenceError("未选择本地模型")

        bridge = VisionBridge(vision_controller, local_captioner_factory=broken)
        assert not bridge.request_batch(("0001.png",), ENGINE_LOCAL)
        assert any("未选择本地模型" in text for text, _ in toasts)


class TestRequestCustom:
    def test_three_concurrent_callbacks(self, qtbot, vision_controller) -> None:
        _RESPONSES[:] = ["card-a", "card-b", "card-c"]
        bridge = VisionBridge(vision_controller)
        got: list[tuple[str, str, bool]] = []
        bridge.custom_ready.connect(lambda rid, text, ok: got.append((rid, text, ok)))
        for index in range(3):
            assert bridge.request_custom(
                f"card-{index}",
                "0001.png",
                ENGINE_LLM,
                system="sys",
                user_prompt=f"user-{index}",
            )
        qtbot.waitUntil(lambda: len(got) == 3, timeout=4000)
        assert {item[0] for item in got} == {"card-0", "card-1", "card-2"}
        assert all(ok for _rid, _text, ok in got)
        assert all(text.startswith("card-") for _rid, text, _ok in got)

    def test_unknown_key_fails_immediately(self, qtbot, vision_controller) -> None:
        bridge = VisionBridge(vision_controller)
        with qtbot.waitSignal(bridge.custom_ready, timeout=1000) as blocker:
            assert not bridge.request_custom(
                "card-0", "missing.png", ENGINE_LLM, system="", user_prompt="x"
            )
        assert blocker.args[0] == "card-0"
        assert blocker.args[2] is False

    def test_profile_override_reaches_captioner(self, qtbot, vision_controller) -> None:
        override = LLMProfile(
            name="cha",
            api_type=API_TYPE,
            base_url="http://mock",
            text_model="m-text",
            vision_model="card-vis",
        )
        bridge = VisionBridge(vision_controller)
        with qtbot.waitSignal(bridge.custom_ready, timeout=3000):
            assert bridge.request_custom(
                "card-0",
                "0001.png",
                ENGINE_LLM,
                system="sys",
                user_prompt="user",
                profile=override,
            )
        assert _Recorder.last is not None
        assert _Recorder.last.requests[0].model == "card-vis"
        assert bridge.configured(profile=override)


class TestRequestLayeredBatch:
    def test_writes_card_blank_scene_and_history(
        self, qtbot, vision_controller
    ) -> None:
        _RESPONSES[:] = ["scene paragraph"]
        bridge = VisionBridge(vision_controller)
        keys = ("0001.png", "0002.png")
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_layered_batch(
                keys,
                ENGINE_LLM,
                card_text="locked card",
                scene_system="scene-sys",
                scene_user="scene-user",
            )
        for key in keys:
            assert vision_controller.record(key).text == "locked card\n\nscene paragraph"
            assert vision_controller.history.entries(key)[0].label == "CHA标注"
        request = _Recorder.last.requests[0]
        assert request.system == "scene-sys"
        assert request.messages[0].text == "scene-user"

    def test_same_outfit_header_keeps_card(self, qtbot, vision_controller) -> None:
        _RESPONSES[:] = ["OUTFIT: SAME\n\nscene paragraph"]
        bridge = VisionBridge(vision_controller)
        parts = CardParts("ema, black hair.", "She wears a red dress.")
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_layered_batch(
                ("0001.png",),
                ENGINE_LLM,
                card_text="ema, black hair. She wears a red dress.",
                scene_system="s",
                scene_user="u",
                card_parts=parts,
            )
        text = vision_controller.record("0001.png").text
        assert text == "ema, black hair. She wears a red dress.\n\nscene paragraph"

    def test_rewritten_outfit_replaces_clothing_block(
        self, qtbot, vision_controller
    ) -> None:
        _RESPONSES[:] = ["OUTFIT: She wears a blue swimsuit.\n\nscene paragraph"]
        bridge = VisionBridge(vision_controller)
        parts = CardParts("ema, black hair.", "She wears a red dress.")
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_layered_batch(
                ("0001.png",),
                ENGINE_LLM,
                card_text="ema, black hair. She wears a red dress.",
                scene_system="s",
                scene_user="u",
                card_parts=parts,
            )
        text = vision_controller.record("0001.png").text
        assert text == "ema, black hair. She wears a blue swimsuit.\n\nscene paragraph"

    def test_rewrite_without_card_parts_keeps_card(
        self, qtbot, vision_controller
    ) -> None:
        _RESPONSES[:] = ["OUTFIT: She wears a blue swimsuit.\n\nscene paragraph"]
        bridge = VisionBridge(vision_controller)
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_layered_batch(
                ("0001.png",),
                ENGINE_LLM,
                card_text="locked card",
                scene_system="s",
                scene_user="u",
            )
        assert vision_controller.record("0001.png").text == "locked card\n\nscene paragraph"

    def test_local_captioner_assembles(self, qtbot, vision_controller) -> None:
        bridge = VisionBridge(
            vision_controller,
            local_captioner_factory=lambda: (
                lambda path, system, user: f"{system}|{user}|{path.name}"
            ),
        )
        with qtbot.waitSignal(vision_controller.batch_finished, timeout=4000):
            assert bridge.request_layered_batch(
                ("0001.png",),
                ENGINE_LOCAL,
                card_text="CARD",
                scene_system="SS",
                scene_user="SU",
            )
        assert vision_controller.record("0001.png").text == "CARD\n\nSS|SU|0001.png"


class TestFlorenceGuard:
    def test_empty_family_is_not_florence(self) -> None:
        from nlapt_gui.vision_bridge import local_engine_is_florence

        assert not local_engine_is_florence()

    def test_florence_family_is_detected(self) -> None:
        from nlapt.local.settings import LocalSettings, save_local_settings

        from nlapt_gui.local_bridge import local_settings_path
        from nlapt_gui.vision_bridge import local_engine_is_florence

        save_local_settings(
            local_settings_path(),
            LocalSettings(family_id="florence2-promptgen-v2"),
        )
        assert local_engine_is_florence()
