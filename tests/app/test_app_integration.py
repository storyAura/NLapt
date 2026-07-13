"""End-to-end acceptance flows (NLapt.md section 14, P0/P1).

P0: open folder -> edit -> batch find/replace (preview, snapshot) -> confirm
-> files on disk correct -> simulate mistake -> rollback from oplog -> disk
restored.

P1: batch AI rewrite via MockLLMClient -> pending suggestions (body
untouched) -> accept / reject / accept-for-edit -> accepted text persisted ->
next-pending navigation; plus session crash-recovery on a second NLaptApp.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.app import NLaptApp, ScopeType
from nlapt.batch.progress import BatchStatus
from nlapt.core.config import LLMProfile
from nlapt.core.errors import ValidationError
from nlapt.core.events import EVT_BATCH_FINISHED, EVT_SNAPSHOT_CREATED, Event
from nlapt.core.states import CaptionState
from nlapt.llm.base import LLMRequest
from nlapt.llm.mock import MockLLMClient
from nlapt.llm.rewrite import RewriteService, RewriteSpec, RewriteType
from nlapt.ops.find_replace import FindReplaceOperation, FindReplaceSpec
from nlapt.workflow.diff import DiffOp

from tests.app.conftest import CAPTION_A, CAPTION_B

UTF8 = "utf-8"

POLISHED = {
    "smiling": "A woman smiling softly.",
    "hat": "A woman wearing a hat.",
    "cat": "A cat resting on a chair.",
}


def _read(root: Path, name: str) -> str:
    return (root / name).read_text(encoding=UTF8)


def _mock_service() -> tuple[MockLLMClient, RewriteService]:
    """Deterministic mock rewrite service keyed off the caption in the prompt."""

    def provider(request: LLMRequest) -> str:
        prompt = request.messages[0].text
        for marker, reply in POLISHED.items():
            if marker in prompt:
                return reply
        return "A generic caption."

    client = MockLLMClient(provider)
    profile = LLMProfile(
        name="test", api_type="openai", base_url="http://localhost", text_model="test-model"
    )
    return client, RewriteService(text_client=client, vision_client=None, profile=profile)


class TestP0Loop:
    """P0 acceptance: open -> edit -> batch replace -> confirm -> rollback."""

    def test_full_p0_flow(self, dataset_root: Path) -> None:
        app = NLaptApp()
        events: list[Event] = []
        app.bus.subscribe(EVT_SNAPSHOT_CREATED, events.append)
        app.bus.subscribe(EVT_BATCH_FINISHED, events.append)

        # 1. Open folder.
        result = app.open_dataset(dataset_root)
        assert [f.key for f in result.images] == ["a.png", "b.png", "c.png"]

        # 2. Edit the unlabeled file.
        app.edit("c.png", "a cat sitting")
        assert app.caption("c.png").state is CaptionState.DRAFT

        # 3. Batch find/replace across ALL scope with preview.
        op = FindReplaceOperation(FindReplaceSpec(find="girl", replace="woman"))
        keys = app.resolve_scope(ScopeType.ALL)
        previews = app.preview_operation(op, keys)
        assert set(previews) == {"a.png", "b.png"}

        report = app.apply_operation(op, keys, description='replace "girl" -> "woman"')
        assert report.status is BatchStatus.COMPLETED
        assert report.failed == 0
        assert report.snapshot is not None  # automatic pre-execution snapshot
        assert (dataset_root / ".backups").is_dir()
        assert any(evt.name == EVT_SNAPSHOT_CREATED for evt in events)
        assert any(evt.name == EVT_BATCH_FINISHED for evt in events)

        # 4. Files on disk are correct (auto-saved by the batch).
        assert _read(dataset_root, "a.txt") == "a woman, smiling"
        assert _read(dataset_root, "b.txt") == "a woman with hat"

        # 5. Confirm jumps to the next unconfirmed file.
        assert app.confirm("a.png") == "b.png"
        assert app.caption("a.png").state is CaptionState.CONFIRMED

        # 6. Simulate a mistake: destructive batch replace.
        bad = FindReplaceOperation(FindReplaceSpec(find="woman", replace="X"))
        bad_report = app.apply_operation(bad, keys, description="bad replace")
        assert bad_report.succeeded == 3
        assert _read(dataset_root, "a.txt") == "a X, smiling"

        # 7. Roll back from the operation log; disk and memory are restored.
        bad_record = app.oplog.records()[0]
        assert bad_record.description == "bad replace"
        restore = app.rollback_operation(bad_record.op_id)
        assert set(restore.restored_files) == {"a.txt", "b.txt"}
        assert _read(dataset_root, "a.txt") == "a woman, smiling"
        assert _read(dataset_root, "b.txt") == "a woman with hat"
        assert app.caption("a.png").text == "a woman, smiling"
        assert app.caption("b.png").text == "a woman with hat"
        # The rollback itself is logged and could be rolled back in turn.
        assert app.oplog.records()[0].kind == "rollback"


class TestP1Loop:
    """P1 acceptance: AI rewrite batch -> Diff review (accept/reject) flow."""

    def test_full_p1_flow(self, dataset_root: Path) -> None:
        app = NLaptApp()
        app.open_dataset(dataset_root)
        app.edit("c.png", "a cat")
        app.save("c.png")

        _, service = _mock_service()
        spec = RewriteSpec(type=RewriteType.POLISH)
        keys = app.resolve_scope(ScopeType.ALL)

        # 1. Batch AI rewrite: results land as pending suggestions.
        report = app.run_rewrite_batch(spec, keys, service=service)
        assert report.status is BatchStatus.COMPLETED
        assert report.succeeded == 3
        for key, original in (("a.png", CAPTION_A), ("b.png", CAPTION_B), ("c.png", "a cat")):
            record = app.caption(key)
            assert record.text == original  # body untouched
            assert record.pending is not None
            assert record.pending.source == "rewrite:polish"

        # 2. Word-level diff is available for review.
        segments = app.suggestion_diff("a.png")
        assert any(segment.op is DiffOp.INSERT for segment in segments)

        # 3. Accept one: persisted to disk, navigation to next pending.
        next_key = app.accept_suggestion("a.png")
        assert next_key == "b.png"
        assert _read(dataset_root, "a.txt") == POLISHED["smiling"]
        assert app.caption("a.png").pending is None

        # 4. Reject one: body and disk untouched.
        next_key = app.reject_suggestion("b.png")
        assert next_key == "c.png"
        assert app.caption("b.png").text == CAPTION_B
        assert _read(dataset_root, "b.txt") == CAPTION_B

        # 5. Accept-for-edit the last one: DRAFT body, no auto-save.
        record = app.accept_for_edit("c.png")
        assert record.state is CaptionState.DRAFT
        assert record.dirty
        assert record.pending is None
        assert _read(dataset_root, "c.txt") == "a cat"

        # 6. No pending suggestions remain.
        assert app.caption("c.png").pending is None
        with pytest.raises(ValidationError):
            app.suggestion_diff("c.png")

    def test_rewrite_failures_are_collected(self, dataset_root: Path) -> None:
        app = NLaptApp()
        app.open_dataset(dataset_root)
        app.edit("c.png", "a cat")

        client = MockLLMClient(["ok caption"], fail_times=1)
        profile = LLMProfile(
            name="t", api_type="openai", base_url="http://localhost", text_model="m"
        )
        service = RewriteService(text_client=client, vision_client=None, profile=profile)
        report = app.run_rewrite_batch(
            RewriteSpec(type=RewriteType.POLISH), ("a.png",), service=service
        )
        assert report.failed == 1
        assert report.failed_keys == ("a.png",)
        assert app.caption("a.png").pending is None


class TestSessionCrashRecovery:
    """Spec 12 reliability: drafts + pending suggestions survive a crash."""

    def test_unsaved_state_survives_second_app(self, dataset_root: Path) -> None:
        app = NLaptApp()
        app.open_dataset(dataset_root)
        app.edit("a.png", "unsaved draft text")
        app.set_suggestion("b.png", "a lady with hat", "rewrite:polish")
        app.edit("c.png", "a cat")
        app.confirm("c.png")
        app.save_session()  # periodic auto-save; the process then "crashes"

        recovered = NLaptApp()
        recovered.open_dataset(dataset_root)
        # Draft restored in memory, not on disk.
        assert recovered.caption("a.png").text == "unsaved draft text"
        assert recovered.caption("a.png").dirty
        assert _read(dataset_root, "a.txt") == CAPTION_A
        # Pending suggestion restored with its source.
        pending = recovered.caption("b.png").pending
        assert pending is not None
        assert pending.text == "a lady with hat"
        assert pending.source == "rewrite:polish"
        # Confirmed state restored (confirm auto-saved the text to disk).
        assert recovered.caption("c.png").state is CaptionState.CONFIRMED
        assert _read(dataset_root, "c.txt") == "a cat"

    def test_clean_close_then_reopen(self, dataset_root: Path) -> None:
        app = NLaptApp()
        app.open_dataset(dataset_root)
        app.edit("a.png", "final text")
        app.confirm("a.png")
        app.close()

        reopened = NLaptApp()
        reopened.open_dataset(dataset_root)
        assert reopened.caption("a.png").text == "final text"
        assert reopened.caption("a.png").state is CaptionState.CONFIRMED
        assert not reopened.caption("a.png").dirty
