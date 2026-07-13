"""Regression tests for FIXPLAN Group 2 (items 2.1-2.6).

Covers: RequestControl wiring into the rewrite/translate request paths (2.1),
whole-string quote stripping (2.2), forced DRAFT on accept-for-edit (2.3),
boundary-aware skip_if_present (2.4), paused-engine future draining (2.5),
and checkpoint-file self-healing (2.6).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from nlapt.batch.checkpoint import (
    CHECKPOINT_FILE_NAME,
    CORRUPT_FILE_SUFFIX,
    CheckpointStore,
    make_checkpoint_id,
)
from nlapt.batch.engine import BatchEngine
from nlapt.batch.progress import BatchController, BatchItemResult, BatchStatus
from nlapt.captions.store import CaptionStore, PendingSuggestion
from nlapt.core.config import LLMProfile, RequestControl
from nlapt.core.errors import LLMRequestError, ValidationError
from nlapt.core.events import (
    EVT_CAPTION_CHANGED,
    EVT_STATE_CHANGED,
    Event,
    EventBus,
)
from nlapt.core.states import CaptionState
from nlapt.llm.base import DEFAULT_TIMEOUT_SECONDS
from nlapt.llm.cleaning import clean_llm_output
from nlapt.llm.mock import MockLLMClient
from nlapt.llm.retry import MinIntervalLimiter
from nlapt.llm.rewrite import RewriteService, RewriteSpec, RewriteType
from nlapt.llm.translate import Translator
from nlapt.ops.prefix_suffix import PrefixSuffixOperation, PrefixSuffixSpec, make_trigger_add_op
from nlapt.storage.session import SESSION_DIR_NAME
from nlapt.workflow.suggestions import accept_for_edit, accept_suggestion

JOIN_TIMEOUT = 5.0
TINY_WAIT = 0.05
CONFIGURED_TIMEOUT = 300.0
BACKOFF_FIRST_DELAY = 1.0  # RetryPolicy default: base_delay * multiplier**0
MIN_INTERVAL = 1.0
FIXED_CLOCK_TIME = 100.0


def make_profile(**overrides: object) -> LLMProfile:
    defaults: dict = {
        "name": "p",
        "api_type": "openai",
        "base_url": "https://api.test/v1",
        "text_model": "text-model",
        "vision_model": "vision-model",
        "system_prompt": "sys",
    }
    defaults.update(overrides)
    return LLMProfile(**defaults)


def make_rewrite_service(client: MockLLMClient, **kwargs: object) -> RewriteService:
    return RewriteService(
        text_client=client, vision_client=None, profile=make_profile(), **kwargs
    )


def run_polish(service: RewriteService) -> None:
    service.run_one(
        RewriteSpec(type=RewriteType.POLISH), key="a.png", caption="a cat", filename="a.png"
    )


# -- 2.1 RequestControl wiring (findings [7], [10], [14]) --------------------------


class TestRequestControlWiring:
    def test_rewrite_configured_timeout_reaches_client(self) -> None:
        client = MockLLMClient(["ok"])
        service = make_rewrite_service(
            client, request=RequestControl(timeout=CONFIGURED_TIMEOUT)
        )
        run_polish(service)
        assert client.requests[0].timeout == CONFIGURED_TIMEOUT

    def test_translate_configured_timeout_reaches_client(self) -> None:
        client = MockLLMClient(["好"])
        translator = Translator(
            client, make_profile(), request=RequestControl(timeout=CONFIGURED_TIMEOUT)
        )
        translator.translate("a.png", "a cat")
        assert client.requests[0].timeout == CONFIGURED_TIMEOUT

    def test_rewrite_without_request_control_keeps_default_timeout(self) -> None:
        client = MockLLMClient(["ok"])
        run_polish(make_rewrite_service(client))
        assert client.requests[0].timeout == DEFAULT_TIMEOUT_SECONDS

    def test_rewrite_retries_transient_failure_with_backoff(self) -> None:
        sleeps: list[float] = []
        client = MockLLMClient(["ok"], fail_times=1)
        service = make_rewrite_service(
            client, request=RequestControl(max_retries=2), retry_sleep=sleeps.append
        )
        result = service.run_one(
            RewriteSpec(type=RewriteType.POLISH),
            key="a.png",
            caption="a cat",
            filename="a.png",
        )
        assert result.result == "ok"
        assert client.calls == 2  # one failure + one retried success
        assert sleeps == [BACKOFF_FIRST_DELAY]

    def test_translate_retries_transient_failure_with_backoff(self) -> None:
        sleeps: list[float] = []
        client = MockLLMClient(["好"], fail_times=1)
        translator = Translator(
            client,
            make_profile(),
            request=RequestControl(max_retries=2),
            retry_sleep=sleeps.append,
        )
        entry = translator.translate("a.png", "a cat")
        assert entry.translated == "好"
        assert client.calls == 2
        assert sleeps == [BACKOFF_FIRST_DELAY]

    def test_rewrite_without_request_control_does_not_retry(self) -> None:
        client = MockLLMClient(["ok"], fail_times=1)
        service = make_rewrite_service(client)
        with pytest.raises(LLMRequestError):
            run_polish(service)
        assert client.calls == 1  # legacy behavior preserved

    def test_shared_limiter_consulted_once_per_request(self) -> None:
        sleeps: list[float] = []
        limiter = MinIntervalLimiter(
            MIN_INTERVAL, clock=lambda: FIXED_CLOCK_TIME, sleep=sleeps.append
        )
        client = MockLLMClient(["ok"])
        service = make_rewrite_service(
            client,
            request=RequestControl(min_interval=MIN_INTERVAL),
            limiter=limiter,
        )
        run_polish(service)  # first request: slot available, no wait
        run_polish(service)  # second request: paced by min_interval
        assert client.calls == 2
        assert sleeps == [MIN_INTERVAL]

    def test_exhausted_retries_reraise_last_error(self) -> None:
        sleeps: list[float] = []
        client = MockLLMClient(["ok"], fail_times=3)
        service = make_rewrite_service(
            client, request=RequestControl(max_retries=2), retry_sleep=sleeps.append
        )
        with pytest.raises(LLMRequestError):
            run_polish(service)
        assert client.calls == 3  # 1 attempt + 2 retries
        assert len(sleeps) == 2

    def test_invalid_request_control_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_rewrite_service(MockLLMClient(["ok"]), request="nope")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Translator(MockLLMClient(["ok"]), make_profile(), request="nope")  # type: ignore[arg-type]


# -- 2.2 whole-string quote stripping (finding [3]) --------------------------------


class TestQuoteStripping:
    def test_two_quoted_phrases_stay_intact(self) -> None:
        raw = '"blue eyes" and "red hair"'
        assert clean_llm_output(raw) == raw

    def test_whole_caption_quotes_stripped(self) -> None:
        assert clean_llm_output('"whole caption"') == "whole caption"

    def test_fullwidth_wrapper_quotes_stripped(self) -> None:
        assert clean_llm_output("“中文引号”") == "中文引号"

    def test_alternating_quoted_words_stay_intact(self) -> None:
        raw = '"a" or "b"'
        assert clean_llm_output(raw) == raw

    def test_nested_distinct_pairs_still_unwrapped(self) -> None:
        assert clean_llm_output("\"'a red fox'\"") == "a red fox"


# -- 2.3 accept_for_edit forces DRAFT (finding [5]) ---------------------------------


PENDING = PendingSuggestion(text="ai text", source="rewrite:polish", created_at=1.0)
KEY = "img/a.jpg"


class TestAcceptForEditForcesDraft:
    def test_accept_for_edit_yields_draft_without_revert_on_edit(self) -> None:
        bus = EventBus()
        store = CaptionStore(bus, revert_confirmed_on_edit=False)
        store.load(KEY, "human text")
        store.confirm(KEY)
        store.set_pending(KEY, PENDING)

        record = accept_for_edit(store, KEY)

        assert record.state is CaptionState.DRAFT  # never CONFIRMED AI text
        assert record.text == PENDING.text
        assert record.dirty is True
        assert record.pending is None

    def test_accept_suggestion_keeps_contract_compliant_transition(self) -> None:
        bus = EventBus()
        store = CaptionStore(bus, revert_confirmed_on_edit=False)
        store.load(KEY, "human text")
        store.confirm(KEY)
        store.set_pending(KEY, PENDING)
        # Plain accept keeps the normal set_text transition (spec-compliant).
        record = accept_suggestion(store, KEY)
        assert record.state is CaptionState.CONFIRMED

    def test_forced_state_publishes_change_and_state_events(self) -> None:
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(None, events.append)
        store = CaptionStore(bus)
        store.load(KEY, "body")
        store.confirm(KEY)

        store.set_text_forced_state(KEY, "new body", CaptionState.DRAFT)

        names = [event.name for event in events]
        assert EVT_CAPTION_CHANGED in names
        assert names.count(EVT_STATE_CHANGED) == 2  # confirm + forced change

    def test_forced_state_without_state_change_skips_state_event(self) -> None:
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(None, events.append)
        store = CaptionStore(bus)
        store.load(KEY, "body")  # DRAFT

        store.set_text_forced_state(KEY, "new body", CaptionState.DRAFT)

        names = [event.name for event in events]
        assert EVT_CAPTION_CHANGED in names
        assert EVT_STATE_CHANGED not in names

    def test_forced_state_rejects_empty_text_for_non_unlabeled(self) -> None:
        store = CaptionStore(EventBus())
        store.load(KEY, "body")
        with pytest.raises(ValidationError):
            store.set_text_forced_state(KEY, "   ", CaptionState.DRAFT)
        with pytest.raises(ValidationError):
            store.set_text_forced_state(KEY, "x", "draft")  # type: ignore[arg-type]


# -- 2.4 boundary-aware skip_if_present (findings [9], [15]) -----------------------


class TestSkipIfPresentBoundary:
    def test_trigger_added_to_longer_word_sharing_prefix(self) -> None:
        op = make_trigger_add_op("cat")
        assert op.apply("caterpillar on a leaf") == "cat, caterpillar on a leaf"

    def test_trigger_not_readded_when_exactly_present(self) -> None:
        op = make_trigger_add_op("minahamu")
        assert op.apply("minahamu, 1girl, solo") == "minahamu, 1girl, solo"
        assert op.apply("minahamu") == "minahamu"  # bare trigger as whole caption

    def test_trigger_added_when_prefix_is_substring_of_first_tag(self) -> None:
        op = make_trigger_add_op("minahamu")
        assert op.apply("minahamuko, 1girl") == "minahamu, minahamuko, 1girl"

    def test_suffix_added_despite_shared_trailing_substring(self) -> None:
        op = PrefixSuffixOperation(PrefixSuffixSpec(suffix="cat"))
        assert op.apply("a bobcat") == "a bobcat, cat"

    def test_suffix_skipped_only_with_joiner_boundary(self) -> None:
        op = PrefixSuffixOperation(PrefixSuffixSpec(suffix="masterpiece"))
        assert (
            op.apply("1girl, this piece is a masterpiece")
            == "1girl, this piece is a masterpiece, masterpiece"
        )
        assert op.apply("1girl, masterpiece") == "1girl, masterpiece"
        assert op.apply("masterpiece") == "masterpiece"  # bare suffix as whole caption

    def test_empty_joiner_keeps_raw_semantics(self) -> None:
        prefix_op = PrefixSuffixOperation(PrefixSuffixSpec(prefix="cat", joiner=""))
        assert prefix_op.apply("caterpillar") == "caterpillar"  # user chose raw mode
        suffix_op = PrefixSuffixOperation(PrefixSuffixSpec(suffix="cat", joiner=""))
        assert suffix_op.apply("bobcat") == "bobcat"

    def test_preview_matches_apply_for_boundary_cases(self) -> None:
        op = make_trigger_add_op("cat")
        previews = op.preview("caterpillar on a leaf")
        assert len(previews) == 1
        assert previews[0].replacement == "cat, "


# -- 2.5 paused engine keeps draining finished futures (finding [16]) ---------------


class TestPausedEngineDrainsFutures:
    def test_progress_and_checkpoint_marks_fire_while_paused(self, tmp_path: Path) -> None:
        keys = ("k1", "k2", "k3")
        started = {key: threading.Event() for key in keys}
        gates = {key: threading.Event() for key in keys}
        progress_keys: list[str] = []
        two_collected = threading.Event()
        lock = threading.Lock()

        def worker(key: str) -> BatchItemResult:
            started[key].set()
            if not gates[key].wait(JOIN_TIMEOUT):
                raise TimeoutError(f"gate for {key} never opened")
            return BatchItemResult(key=key, ok=True)

        def on_progress(done: int, total: int, key: str) -> None:
            with lock:
                progress_keys.append(key)
                if len(progress_keys) >= 2:
                    two_collected.set()

        checkpoints = CheckpointStore(tmp_path)
        cid = make_checkpoint_id("rewrite", keys, "params")
        controller = BatchController()
        engine = BatchEngine(snapshots=None, bus=EventBus(), checkpoints=checkpoints)
        box: dict[str, object] = {}

        def target() -> None:
            box["report"] = engine.run(
                operation="rewrite",
                keys=keys,
                worker=worker,
                concurrency=2,
                controller=controller,
                on_progress=on_progress,
                take_snapshot=False,
                checkpoint_id=cid,
            )

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        assert started["k1"].wait(JOIN_TIMEOUT)
        assert started["k2"].wait(JOIN_TIMEOUT)

        controller.pause()
        gates["k1"].set()
        gates["k2"].set()

        # THE FIX: finished in-flight items are collected while still paused.
        assert two_collected.wait(JOIN_TIMEOUT), (
            "paused engine did not drain finished in-flight futures"
        )
        assert set(progress_keys[:2]) == {"k1", "k2"}
        assert checkpoints.completed(cid) == frozenset({"k1", "k2"})
        # Pause still blocks NEW dispatches.
        assert not started["k3"].wait(TINY_WAIT), "paused engine dispatched a new item"

        controller.resume()
        assert started["k3"].wait(JOIN_TIMEOUT), "resume did not restart dispatch"
        gates["k3"].set()
        thread.join(JOIN_TIMEOUT)
        assert not thread.is_alive()

        report = box["report"]
        assert report.status is BatchStatus.COMPLETED  # type: ignore[union-attr]
        assert report.succeeded == 3  # type: ignore[union-attr]


# -- 2.6 corrupt checkpoints.json self-heals (finding [2]) --------------------------


class TestCorruptCheckpointSelfHeal:
    @pytest.mark.parametrize(
        "content", ["{ this is not json", "[]", '{"cid": "a.txt"}', '{"cid": [1]}']
    )
    def test_corrupt_file_starts_empty_and_is_moved_aside(
        self, tmp_path: Path, content: str
    ) -> None:
        path = tmp_path / SESSION_DIR_NAME / CHECKPOINT_FILE_NAME
        path.parent.mkdir(parents=True)
        path.write_text(content, encoding="utf-8")

        store = CheckpointStore(tmp_path)  # must not raise

        assert store.has("cid") is False
        corrupt = path.with_name(path.name + CORRUPT_FILE_SUFFIX)
        assert corrupt.read_text(encoding="utf-8") == content
        assert not path.exists()

    def test_store_usable_after_self_heal(self, tmp_path: Path) -> None:
        path = tmp_path / SESSION_DIR_NAME / CHECKPOINT_FILE_NAME
        path.parent.mkdir(parents=True)
        path.write_text("{ broken", encoding="utf-8")

        store = CheckpointStore(tmp_path)
        store.mark("cid", "a.txt")

        assert store.completed("cid") == frozenset({"a.txt"})
        assert json.loads(path.read_text(encoding="utf-8")) == {"cid": ["a.txt"]}

    def test_missing_file_still_means_empty_store(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path)
        assert store.completed("cid") == frozenset()
