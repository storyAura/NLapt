"""Tests for nlapt_gui.translate_bridge (async LLM segment translation)."""

from __future__ import annotations

import threading
from typing import Iterator

import pytest
from PySide6.QtCore import QThreadPool

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, LLMProfile, RequestControl
from nlapt.core.errors import LLMRequestError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings
from nlapt_gui.translate_bridge import (
    NOTE_UNCONFIGURED,
    TRANSLATE_POOL_MAX_THREADS,
    TranslateBridge,
    get_translate_pool,
    has_cjk,
)

# Unique api_type for this test module (registry is process-global).
API_TYPE = "gui-bridge-mock"

_RESPONSES: list[str] = ["translated"]
_FAIL_TIMES: list[int] = [0]


class _Recorder:
    last: MockLLMClient | None = None


def _factory(profile: LLMProfile) -> MockLLMClient:
    client = MockLLMClient(
        list(_RESPONSES),
        fail_times=_FAIL_TIMES[0],
        failure=LLMRequestError("mock network down"),
    )
    _Recorder.last = client
    return client


register_client(API_TYPE, _factory)


def _configured_app() -> NLaptApp:
    profile = LLMProfile(
        name="default", api_type=API_TYPE, base_url="http://mock", text_model="m1"
    )
    config = AppConfig(
        profiles=(profile,),
        active_profile="default",
        request=RequestControl(max_retries=0),
    )
    return NLaptApp(config=config)


@pytest.fixture()
def bridge(qtbot) -> Iterator[TranslateBridge]:
    _RESPONSES[:] = ["translated"]
    _FAIL_TIMES[0] = 0
    _Recorder.last = None
    controller = AppController(_configured_app(), settings=UISettings())
    yield TranslateBridge(controller)


@pytest.fixture()
def unconfigured_bridge(qtbot) -> Iterator[TranslateBridge]:
    controller = AppController(NLaptApp(), settings=UISettings())
    yield TranslateBridge(controller)


class TestHasCjk:
    def test_detects_han_characters(self) -> None:
        assert has_cjk("少女站在樱花树下。")
        assert has_cjk("long hair 少女")
        assert not has_cjk("long hair")
        assert not has_cjk("")


class TestDedicatedPool:
    def test_default_pool_is_not_global(self, bridge: TranslateBridge) -> None:
        assert bridge._pool is get_translate_pool()
        assert bridge._pool is not QThreadPool.globalInstance()
        assert get_translate_pool().maxThreadCount() == TRANSLATE_POOL_MAX_THREADS


class TestInvalidatePending:
    def test_drops_in_flight_and_queued(
        self, qtbot, monkeypatch
    ) -> None:
        controller = AppController(_configured_app(), settings=UISettings())
        pool = QThreadPool()
        pool.setMaxThreadCount(1)
        bridge = TranslateBridge(controller, pool=pool)
        started = threading.Event()
        release = threading.Event()
        seen: list[tuple[object, ...]] = []
        bridge.segment_ready.connect(lambda *args: seen.append(args))

        def slow(text: str, direction: object) -> str:
            started.set()
            assert release.wait(timeout=3)
            return "译"

        monkeypatch.setattr(bridge, "_resolve_translate_fn", lambda: slow)
        bridge.request("a.png", "hello")
        qtbot.waitUntil(started.is_set, timeout=2000)
        bridge.request("b.png", "world")
        bridge.invalidate_pending()
        release.set()
        qtbot.waitUntil(lambda: pool.activeThreadCount() == 0, timeout=2000)
        assert seen == []

    def test_dataset_opened_bumps_generation(self) -> None:
        controller = AppController(_configured_app(), settings=UISettings())
        bridge = TranslateBridge(controller)
        assert bridge._generation == 0
        controller.dataset_opened.emit(object())
        assert bridge._generation == 1


class TestConfigured:
    def test_configured_true_with_profile(self, bridge: TranslateBridge) -> None:
        assert bridge.configured()

    def test_configured_false_without_profile(
        self, unconfigured_bridge: TranslateBridge
    ) -> None:
        assert not unconfigured_bridge.configured()

    def test_request_unconfigured_emits_note(
        self, qtbot, unconfigured_bridge: TranslateBridge
    ) -> None:
        with qtbot.waitSignal(
            unconfigured_bridge.segment_ready, timeout=1000
        ) as blocker:
            unconfigured_bridge.request("0001.png", "long hair")
        assert blocker.args == ["0001.png", "long hair", NOTE_UNCONFIGURED, False]


class TestRequest:
    def test_request_emits_translation(self, qtbot, bridge: TranslateBridge) -> None:
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "long hair")
        assert blocker.args == ["0001.png", "long hair", "translated", True]
        assert _Recorder.last is not None
        assert _Recorder.last.calls == 1

    def test_identical_text_shares_cache(self, qtbot, bridge: TranslateBridge) -> None:
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000):
            bridge.request("0001.png", "long hair")
        # Second request (even for another file) is served from the cache.
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0002.png", "long hair")
        assert blocker.args == ["0002.png", "long hair", "translated", True]
        assert _Recorder.last.calls == 1

    def test_fresh_bypasses_cache(self, qtbot, bridge: TranslateBridge) -> None:
        _RESPONSES[:] = ["first", "second"]
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "long hair")
        assert blocker.args[2] == "first"
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "long hair", fresh=True)
        assert blocker.args[2] == "second"
        assert _Recorder.last.calls == 2
        # The fresh alternative replaces the shared cache entry.
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "long hair")
        assert blocker.args[2] == "second"
        assert _Recorder.last.calls == 2

    def test_error_emits_not_ok(self, qtbot, bridge: TranslateBridge) -> None:
        _FAIL_TIMES[0] = 99
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000) as blocker:
            bridge.request("0001.png", "long hair")
        assert blocker.args[0] == "0001.png"
        assert blocker.args[3] is False
        assert "mock network down" in blocker.args[2]

    def test_empty_text_emits_not_ok(self, qtbot, bridge: TranslateBridge) -> None:
        with qtbot.waitSignal(bridge.segment_ready, timeout=1000) as blocker:
            bridge.request("0001.png", "   ")
        assert blocker.args[3] is False
        assert _Recorder.last is None or _Recorder.last.calls == 0

    def test_direction_detected_per_segment(self, qtbot, bridge: TranslateBridge) -> None:
        with qtbot.waitSignal(bridge.segment_ready, timeout=2000):
            bridge.request("0001.png", "少女站在樱花树下。")
        prompt = _Recorder.last.requests[0].messages[0].text
        assert "少女站在樱花树下。" in prompt


class TestRequestTo:
    """Caption-level target-language translation (中/英/日, spec module 1)."""

    def test_translates_to_target_language(self, qtbot, bridge: TranslateBridge) -> None:
        with qtbot.waitSignal(bridge.target_ready, timeout=2000) as blocker:
            bridge.request_to("0001.png", "a cat", "ja")
        assert blocker.args == ["0001.png", "a cat", "ja", "translated", True]
        prompt = _Recorder.last.requests[0].messages[0].text
        assert "Japanese" in prompt
        assert "a cat" in prompt

    def test_results_cached_per_text_and_language(
        self, qtbot, bridge: TranslateBridge
    ) -> None:
        with qtbot.waitSignal(bridge.target_ready, timeout=2000):
            bridge.request_to("0001.png", "a cat", "ja")
        with qtbot.waitSignal(bridge.target_ready, timeout=2000):
            bridge.request_to("0002.png", "a cat", "ja")
        assert _Recorder.last.calls == 1  # second reply came from the cache
        # A different target language is a distinct cache entry.
        with qtbot.waitSignal(bridge.target_ready, timeout=2000):
            bridge.request_to("0001.png", "a cat", "zh")
        assert _Recorder.last.calls == 2

    def test_unconfigured_emits_note(self, qtbot, unconfigured_bridge) -> None:
        with qtbot.waitSignal(unconfigured_bridge.target_ready, timeout=1000) as blocker:
            unconfigured_bridge.request_to("0001.png", "a cat", "zh")
        assert blocker.args == ["0001.png", "a cat", "zh", NOTE_UNCONFIGURED, False]

    def test_empty_text_emits_not_ok(self, qtbot, bridge: TranslateBridge) -> None:
        with qtbot.waitSignal(bridge.target_ready, timeout=1000) as blocker:
            bridge.request_to("0001.png", "   ", "zh")
        assert blocker.args[4] is False

    def test_error_emits_not_ok(self, qtbot, bridge: TranslateBridge) -> None:
        _FAIL_TIMES[0] = 99
        with qtbot.waitSignal(bridge.target_ready, timeout=2000) as blocker:
            bridge.request_to("0001.png", "a cat", "en")
        assert blocker.args[4] is False
        assert "mock network down" in blocker.args[3]


class TestTranslateAllCjk:
    def test_translates_only_cjk_segments(self, qtbot, bridge: TranslateBridge) -> None:
        ready: list[tuple[str, str, str, bool]] = []
        bridge.segment_ready.connect(lambda *args: ready.append(args))
        with qtbot.waitSignal(bridge.all_done, timeout=2000) as blocker:
            bridge.translate_all_cjk(
                "0001.png", ("long hair", "少女", "solo", "樱花树")
            )
        assert blocker.args == ["0001.png", 2, 0]
        sources = {args[1] for args in ready}
        assert sources == {"少女", "樱花树"}
        assert _Recorder.last.calls == 2

    def test_no_cjk_emits_all_done_zero(self, qtbot, bridge: TranslateBridge) -> None:
        with qtbot.waitSignal(bridge.all_done, timeout=1000) as blocker:
            bridge.translate_all_cjk("0001.png", ("long hair", "solo"))
        assert blocker.args == ["0001.png", 0, 0]
        assert _Recorder.last is None or _Recorder.last.calls == 0

    def test_counts_failures(self, qtbot, bridge: TranslateBridge) -> None:
        _FAIL_TIMES[0] = 99
        with qtbot.waitSignal(bridge.all_done, timeout=2000) as blocker:
            bridge.translate_all_cjk("0001.png", ("少女", "樱花树"))
        assert blocker.args == ["0001.png", 0, 2]
