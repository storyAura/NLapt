"""Tests for the Hy-MT2 local translation provider and download helpers."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from nlapt.core.errors import LLMConfigError, LLMRequestError, ValidationError
from nlapt.llm.translate import Direction
from nlapt.local.mt_catalog import TIER_BALANCED, find_mt_model, hymt_prompt, mt_model_path
from nlapt.local.server import LocalServerManager

from nlapt_gui.local_bridge import LocalBridge
from nlapt_gui.mt_bridge import (
    HYMT_MAX_TOKENS,
    HYMT_REPEAT_PENALTY,
    HYMT_TEMPERATURE,
    HYMT_TOP_K,
    HYMT_TOP_P,
    MT_CONTEXT_LENGTH,
    LocalMTProvider,
    delete_mt_model,
    ensure_mt_server,
    get_mt_server_manager,
    is_tier_downloaded,
    mt_server_port,
    reset_mt_singletons_for_tests,
)
from nlapt_gui.translate_bridge import TranslateBridge
from nlapt_gui.translate_config import TranslationConfig


class _QuietStopper:
    def note_request(self) -> None:
        return None

    def note_finished(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _isolate_mt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reset_mt_singletons_for_tests(
        manager=LocalServerManager(
            popen=lambda _args: _FakeProc(),
            health_check=lambda _url: True,
            sleep=lambda _s: None,
        )
    )
    monkeypatch.setattr(
        "nlapt_gui.mt_bridge.ensure_runtime",
        lambda *_a, **_k: tmp_path / "llama-server.exe",
    )
    (tmp_path / "llama-server.exe").write_bytes(b"exe")
    bridge = LocalBridge()
    bridge.update_settings(models_dir=str(tmp_path), gpu_layers=0, server_path=str(tmp_path / "llama-server.exe"))


class _FakeProc:
    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


def _touch_tier(tmp_path: Path, tier: str = TIER_BALANCED) -> Path:
    dest = mt_model_path(tmp_path, find_mt_model(tier))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"gguf")
    return dest


class TestLocalMTProvider:
    def test_unknown_tier_raises(self) -> None:
        with pytest.raises(ValidationError):
            LocalMTProvider("nope", retry_sleep=lambda _delay: None)

    def test_translates_via_official_prompt(self, tmp_path: Path) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "一件红裙"}, "finish_reason": "stop"}]},
            )

        provider = LocalMTProvider(
            TIER_BALANCED,
            models_dir=tmp_path,
            retry_sleep=lambda _delay: None,
            transport=httpx.MockTransport(handler),
            ensure=lambda: "http://127.0.0.1:18435/v1",
            idle_stopper=_QuietStopper(),
        )
        assert provider.translate("a red dress", Direction.EN_TO_ZH) == "一件红裙"
        assert str(captured[0].url) == "http://127.0.0.1:18435/v1/chat/completions"
        payload = json.loads(captured[0].content)
        assert payload["temperature"] == HYMT_TEMPERATURE
        assert payload["top_p"] == HYMT_TOP_P
        assert payload["max_tokens"] == HYMT_MAX_TOKENS == 4096
        assert payload["top_k"] == HYMT_TOP_K == 20
        assert payload["repeat_penalty"] == HYMT_REPEAT_PENALTY == 1.05
        assert payload["min_p"] == 0.0
        assert payload["stream"] is False
        assert payload["model"] == find_mt_model(TIER_BALANCED).filename
        assert payload["messages"] == [
            {"role": "user", "content": hymt_prompt("a red dress", "zh")}
        ]
        assert "prompt" not in payload

    def test_translate_to_japanese(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "コート"}, "finish_reason": "stop"}]}
            )

        provider = LocalMTProvider(
            TIER_BALANCED,
            transport=httpx.MockTransport(handler),
            retry_sleep=lambda _delay: None,
            ensure=lambda: "http://127.0.0.1:9/v1",
            idle_stopper=_QuietStopper(),
        )
        assert provider.translate_to("coat", "ja") == "コート"

    def test_empty_text_raises(self) -> None:
        provider = LocalMTProvider(
            TIER_BALANCED,
            ensure=lambda: "http://x/v1",
            retry_sleep=lambda _delay: None,
            idle_stopper=_QuietStopper(),
        )
        with pytest.raises(LLMRequestError):
            provider.translate("   ", Direction.EN_TO_ZH)

    def test_multi_paragraph_caption_is_translated_per_paragraph(self, tmp_path: Path) -> None:
        # Hy-MT2 treats everything before the last blank line as untranslated
        # context, so a two-paragraph caption must become two requests.
        replies = {
            "Hsin has white hair.": "希恩有白发。",
            "Hsin sits on a red chair.": "希恩坐在红椅上。",
        }
        prompts: list[str] = []
        ensure_calls: list[int] = []
        events: list[str] = []

        class Stopper:
            def note_request(self) -> None:
                events.append("request")

            def note_finished(self) -> None:
                events.append("finished")

        def ensure() -> str:
            ensure_calls.append(1)
            return "http://127.0.0.1:18435/v1"

        def handler(request: httpx.Request) -> httpx.Response:
            prompt = json.loads(request.content)["messages"][0]["content"]
            prompts.append(prompt)
            source = prompt.rsplit("\n\n", 1)[-1]
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": replies[source]}, "finish_reason": "stop"}]},
            )

        provider = LocalMTProvider(
            TIER_BALANCED,
            models_dir=tmp_path,
            retry_sleep=lambda _delay: None,
            transport=httpx.MockTransport(handler),
            ensure=ensure,
            idle_stopper=Stopper(),
        )
        result = provider.translate_to(
            "Hsin has white hair.\n\n\nHsin sits on a red chair.\n", "zh"
        )
        assert result == "希恩有白发。\n\n希恩坐在红椅上。"
        assert prompts == [
            hymt_prompt("Hsin has white hair.", "zh"),
            hymt_prompt("Hsin sits on a red chair.", "zh"),
        ]
        assert ensure_calls == [1]
        assert events == ["request", "finished"]

    def test_incomplete_paragraph_fails_whole_translation(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "第一段"}, "finish_reason": "stop"}]},
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "第二"}, "finish_reason": "length"}]},
            )

        provider = LocalMTProvider(
            TIER_BALANCED,
            models_dir=tmp_path,
            retry_sleep=lambda _delay: None,
            transport=httpx.MockTransport(handler),
            ensure=lambda: "http://127.0.0.1:18435/v1",
            idle_stopper=_QuietStopper(),
        )
        with pytest.raises(LLMRequestError, match="finish_reason='length'"):
            provider.translate_to("first paragraph\n\nsecond paragraph", "zh")

    @pytest.mark.parametrize("reply", [
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"text": "legacy completion", "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": None, "reasoning_content": "thinking"}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": "partial"}}]},
    ])
    def test_invalid_completion_raises(self, reply: dict[str, object]) -> None:
        events: list[str] = []

        class Stopper:
            def note_request(self) -> None:
                events.append("request")

            def note_finished(self) -> None:
                events.append("finished")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=reply)

        provider = LocalMTProvider(
            TIER_BALANCED,
            transport=httpx.MockTransport(handler),
            retry_sleep=lambda _delay: None,
            ensure=lambda: "http://x/v1",
            idle_stopper=Stopper(),
        )
        with pytest.raises(LLMRequestError):
            provider.translate("hi", Direction.EN_TO_ZH)
        assert events == ["request", "finished"]

    @pytest.mark.parametrize("failures", [1, 3])
    def test_request_retries_then_returns_or_raises(self, failures: int) -> None:
        requests: list[httpx.Request] = []
        delays: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) <= failures:
                return httpx.Response(503, text=f"server busy {len(requests)}")
            return httpx.Response(200, json={"choices": [
                {"message": {"content": "你好"}, "finish_reason": "stop"}
            ]})

        provider = LocalMTProvider(
            TIER_BALANCED,
            transport=httpx.MockTransport(handler),
            retry_sleep=delays.append,
            ensure=lambda: "http://127.0.0.1:9/v1",
            idle_stopper=_QuietStopper(),
        )
        if failures == 1:
            assert provider.translate("hi", Direction.EN_TO_ZH) == "你好"
            assert delays == [1.0]
            assert len(requests) == 2
        else:
            with pytest.raises(LLMRequestError, match="HTTP 503: server busy 3"):
                provider.translate("hi", Direction.EN_TO_ZH)
            assert delays == [1.0, 2.0]
            assert len(requests) == 3


class TestMTServerHelpers:
    def test_is_tier_downloaded(self, tmp_path: Path, monkeypatch) -> None:
        assert not is_tier_downloaded(TIER_BALANCED, models_dir=tmp_path)
        _touch_tier(tmp_path)
        monkeypatch.setattr(
            "nlapt_gui.mt_bridge.is_mt_downloaded", lambda *_a, **_k: True
        )
        assert is_tier_downloaded(TIER_BALANCED, models_dir=tmp_path)

    def test_ensure_requires_download(self, tmp_path: Path) -> None:
        with pytest.raises(LLMConfigError):
            ensure_mt_server(TIER_BALANCED, models_dir=tmp_path)

    def test_ensure_reuses_manager(self, tmp_path: Path, monkeypatch) -> None:
        _touch_tier(tmp_path)
        monkeypatch.setattr(
            "nlapt_gui.mt_bridge.is_mt_downloaded", lambda *_a, **_k: True
        )
        url = ensure_mt_server(TIER_BALANCED, models_dir=tmp_path)
        assert url.startswith("http://127.0.0.1:")
        assert url.endswith("/v1")
        spec = get_mt_server_manager().current_spec
        assert spec is not None
        assert spec.context_length == MT_CONTEXT_LENGTH
        assert spec.parallel == 1

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        dest = _touch_tier(tmp_path)
        delete_mt_model(TIER_BALANCED, models_dir=tmp_path)
        assert not dest.exists()

    def test_mt_port_is_caption_plus_one(self) -> None:
        from nlapt.local.settings import LocalSettings

        assert mt_server_port(LocalSettings(port=18434)) == 18435
        assert mt_server_port(LocalSettings(port=65535)) == 65534


class TestBridgeConfigured:
    def test_local_mt_needs_downloaded_file(
        self, qtbot, tmp_path: Path, monkeypatch
    ) -> None:
        from nlapt.app import NLaptApp
        from nlapt_gui.controller import AppController
        from nlapt_gui.settings import UISettings

        controller = AppController(NLaptApp(), settings=UISettings())
        missing = TranslateBridge(
            controller, config=TranslationConfig(provider="local_mt")
        )
        assert not missing.configured()
        _touch_tier(tmp_path)
        monkeypatch.setattr(
            "nlapt_gui.translate_bridge.is_tier_downloaded", lambda *_a, **_k: True
        )
        ready = TranslateBridge(
            controller,
            config=TranslationConfig(provider="local_mt", local_mt_tier=TIER_BALANCED),
        )
        assert ready.configured()
