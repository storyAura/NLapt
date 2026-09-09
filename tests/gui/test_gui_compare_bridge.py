"""Tests for nlapt_gui.compare_bridge (多对比推标 fan-out runner)."""

from __future__ import annotations

import threading

from nlapt.core.config import AppConfig, LLMProfile, ModelRef
from nlapt.core.errors import LLMRequestError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.compare_bridge import (
    ERROR_NO_KEYS,
    ERROR_RUNNING,
    ERROR_STALE_MODEL,
    ERROR_TOO_FEW,
    CompareRunner,
)
from nlapt_gui.model_targets import choice_label

API_ALPHA = "gui-compare-alpha"
API_BETA = "gui-compare-beta"
API_FAIL = "gui-compare-fail"

K1 = "0001.png"
K2 = "0002.png"


class _Gauge:
    """Counts concurrently running captioner calls across worker threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.release = threading.Event()

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


GAUGE = _Gauge()


def _alpha_factory(profile: LLMProfile) -> MockLLMClient:
    def reply(request):  # noqa: ANN001
        GAUGE.enter()
        GAUGE.release.wait(timeout=2.0)
        GAUGE.leave()
        return f"alpha:{request.model}"

    return MockLLMClient(reply)


register_client(API_ALPHA, _alpha_factory)
register_client(API_BETA, lambda profile: MockLLMClient(lambda r: f"beta:{r.model}"))
register_client(
    API_FAIL,
    lambda profile: MockLLMClient(["never"], fail_times=99, failure=LLMRequestError("boom")),
)

ALPHA = LLMProfile(
    name="alpha", api_type=API_ALPHA, base_url="http://alpha", models=("a1",), enabled_models=("a1",)
)
BETA = LLMProfile(
    name="beta", api_type=API_BETA, base_url="http://beta", models=("b1", "b-off"), enabled_models=("b1",)
)
FAIL = LLMProfile(
    name="fail", api_type=API_FAIL, base_url="http://fail", models=("f1",), enabled_models=("f1",)
)
REFS = (ModelRef("alpha", "a1"), ModelRef("beta", "b1"))


def _configure(controller, *profiles: LLMProfile) -> None:
    controller.reload_config(AppConfig(profiles=profiles or (ALPHA, BETA)))


def _collect(runner: CompareRunner):
    items: list[tuple[str, ModelRef, str, bool]] = []
    progress: list[tuple[int, int]] = []
    finished: list[bool] = []
    runner.item_ready.connect(lambda k, r, t, ok: items.append((k, r, t, ok)))
    runner.progress.connect(lambda d, t: progress.append((d, t)))
    runner.finished.connect(lambda: finished.append(True))
    return items, progress, finished


class TestStartErrors:
    def test_validation_messages(self, qtbot, controller) -> None:
        _configure(controller)
        runner = CompareRunner(controller)
        assert runner.start((), REFS, system="", user_prompt="p") == ERROR_NO_KEYS
        assert runner.start((K1,), REFS[:1], system="", user_prompt="p") == ERROR_TOO_FEW.format(n=2)
        stale = ModelRef("beta", "b-off")
        assert runner.start((K1,), (REFS[0], stale), system="", user_prompt="p") == (
            ERROR_STALE_MODEL.format(label=choice_label(stale))
        )
        assert not runner.running()


class TestRun:
    def test_every_key_gets_one_result_per_model(self, qtbot, controller) -> None:
        _configure(controller)
        GAUGE.release.set()
        runner = CompareRunner(controller)
        items, progress, finished = _collect(runner)
        assert runner.start((K1, K2), REFS, system="sys", user_prompt="p", concurrency=4) is None
        assert runner.running()
        assert runner.start((K1,), REFS, system="", user_prompt="p") == ERROR_RUNNING
        qtbot.waitUntil(lambda: bool(finished), timeout=4000)
        assert progress[0] == (0, 4) and progress[-1] == (4, 4)
        got = {(k, r): (t, ok) for k, r, t, ok in items}
        assert got[(K1, REFS[0])] == ("alpha:a1", True)
        assert got[(K2, REFS[1])] == ("beta:b1", True)
        assert len(got) == 4
        assert not runner.running()

    def test_failures_are_reported_not_raised(self, qtbot, controller) -> None:
        _configure(controller, ALPHA, FAIL)
        GAUGE.release.set()
        runner = CompareRunner(controller)
        items, _progress, finished = _collect(runner)
        refs = (ModelRef("alpha", "a1"), ModelRef("fail", "f1"))
        assert runner.start((K1,), refs, system="", user_prompt="p") is None
        qtbot.waitUntil(lambda: bool(finished), timeout=4000)
        failed = [(t, ok) for _k, r, t, ok in items if r == refs[1]]
        assert failed and failed[0][1] is False and "boom" in failed[0][0]

    def test_concurrency_bound_is_respected(self, qtbot, controller) -> None:
        _configure(controller)
        GAUGE.release.clear()
        GAUGE.peak = 0
        runner = CompareRunner(controller)
        _items, progress, finished = _collect(runner)
        alpha_only_keys = (K1, K2, "10_concept/0003.png", "10_concept/0004.png")
        try:
            assert runner.start(alpha_only_keys, REFS, system="", user_prompt="p", concurrency=2) is None
            # Alpha jobs block until released; at most 2 jobs may be in flight overall.
            qtbot.waitUntil(lambda: GAUGE.active >= 1, timeout=2000)
            qtbot.wait(100)
            assert GAUGE.peak <= 2
        finally:
            GAUGE.release.set()
        qtbot.waitUntil(lambda: bool(finished), timeout=6000)
        assert progress[-1] == (8, 8)
        assert GAUGE.peak <= 2

    def test_cancel_stops_dispatch_and_finishes(self, qtbot, controller) -> None:
        _configure(controller)
        GAUGE.release.clear()
        runner = CompareRunner(controller)
        items, _progress, finished = _collect(runner)
        try:
            assert runner.start((K1, K2), REFS, system="", user_prompt="p", concurrency=1) is None
            qtbot.waitUntil(lambda: GAUGE.active >= 1, timeout=2000)
            runner.cancel()
            assert not finished  # one job still in flight
        finally:
            GAUGE.release.set()
        qtbot.waitUntil(lambda: bool(finished), timeout=4000)
        # The in-flight job's result is dropped and nothing else was dispatched.
        assert items == []
        assert not runner.running()
        # A cancelled runner can start again.
        assert runner.start((K1,), REFS, system="", user_prompt="p") is None
