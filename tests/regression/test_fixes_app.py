"""Regression tests for FIXPLAN Group 3 — app facade (items 3.1-3.4).

3.1 duplicate-stem images must not alias one txt as two editable records;
3.2 pending suggestions must be at least as durable as checkpoint marks;
3.3 needs_conversion must be surfaced and convertible through the facade;
3.4 AI rewrite batches must be roll-back-able (clearing their suggestions).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.app import AI_BATCH_KIND, NLaptApp, ScopeType, SuggestionRollbackResult
from nlapt.batch.progress import BatchController, BatchStatus
from nlapt.core.config import AppConfig, LLMProfile, RequestControl
from nlapt.core.errors import ValidationError
from nlapt.core.events import EVT_ENCODING_ISSUES, EVT_TXT_CONFLICT, Event
from nlapt.history.oplog import ROLLBACK_KIND
from nlapt.llm.base import LLMRequest
from nlapt.llm.mock import MockLLMClient
from nlapt.llm.rewrite import RewriteService, RewriteSpec, RewriteType
from nlapt.ops.find_replace import FindReplaceOperation, FindReplaceSpec
from nlapt.storage.paths import dataset_state_dir
from nlapt.storage.session import SESSION_FILE_NAME

UTF8 = "utf-8"
STUB_IMAGE_BYTES = b"\x89"
GBK_CAPTION = "一只猫，坐在窗台上"
GBK_ENCODING = "gb18030"


def _write_pair(root: Path, stem: str, caption: str | None, *, ext: str = ".png") -> None:
    (root / f"{stem}{ext}").write_bytes(STUB_IMAGE_BYTES)
    if caption is not None:
        (root / f"{stem}.txt").write_text(caption, encoding=UTF8)


def _standard_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_pair(root, "a", "a girl, smiling")
    _write_pair(root, "b", "a girl with hat")
    return root


def _mock_service(client: MockLLMClient) -> RewriteService:
    profile = LLMProfile(
        name="test", api_type="openai", base_url="http://localhost", text_model="test-model"
    )
    return RewriteService(text_client=client, vision_client=None, profile=profile)


def _echo_provider(request: LLMRequest) -> str:
    return f"polished::{hash(request.messages[0].text) & 0xFFFF}"


class TestFix31TxtConflicts:
    """3.1 — same-stem images resolving to one txt yield ONE editable record."""

    def _conflict_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "cat.jpg").write_bytes(STUB_IMAGE_BYTES)
        (root / "cat.png").write_bytes(STUB_IMAGE_BYTES)
        (root / "cat.txt").write_text("original caption", encoding=UTF8)
        _write_pair(root, "dog", "a dog")
        return root

    def test_only_canonical_key_is_editable(self, tmp_path: Path) -> None:
        root = self._conflict_root(tmp_path)
        app = NLaptApp()
        events: list[Event] = []
        app.bus.subscribe(EVT_TXT_CONFLICT, events.append)
        result = app.open_dataset(root)

        # The scanner still reports both images (kohya pairing stays as-is)...
        assert {f.key for f in result.images} >= {"cat.jpg", "cat.png"}
        # ...but only the first key in natural order is an editable record.
        assert app.caption("cat.jpg").text == "original caption"
        with pytest.raises(KeyError):
            app.caption("cat.png")
        assert app.resolve_scope(ScopeType.ALL) == ("cat.jpg", "dog.png")

        assert app.txt_conflicts() == {"cat.jpg": ("cat.png",)}
        assert len(events) == 1
        assert events[0].payload["key"] == "cat.jpg"
        assert events[0].payload["excluded"] == ("cat.png",)
        assert events[0].payload["txt"] == "cat.txt"

    def test_conflict_free_dataset_reports_nothing(self, tmp_path: Path) -> None:
        root = _standard_dataset(tmp_path)
        app = NLaptApp()
        events: list[Event] = []
        app.bus.subscribe(EVT_TXT_CONFLICT, events.append)
        app.open_dataset(root)
        assert app.txt_conflicts() == {}
        assert events == []

    def test_batch_apply_and_rollback_stay_consistent(self, tmp_path: Path) -> None:
        root = self._conflict_root(tmp_path)
        app = NLaptApp()
        app.open_dataset(root)
        keys = app.resolve_scope(ScopeType.ALL)

        op = FindReplaceOperation(FindReplaceSpec(find="original", replace="changed"))
        report = app.apply_operation(op, keys, description="replace original")
        assert report.status is BatchStatus.COMPLETED
        assert (root / "cat.txt").read_text(encoding=UTF8) == "changed caption"
        # The shared txt was processed exactly once (no double-apply hazard).
        record = app.oplog.records()[0]
        assert record.affected_keys == ("cat.txt",)

        result = app.rollback_operation(record.op_id)
        assert "cat.txt" in result.restored_files
        assert (root / "cat.txt").read_text(encoding=UTF8) == "original caption"
        # The editable record was reloaded from disk (no stale alias remains).
        assert app.caption("cat.jpg").text == "original caption"
        assert not app.caption("cat.jpg").dirty
        # Re-saving the canonical record must NOT silently undo the rollback.
        app.save("cat.jpg")
        assert (root / "cat.txt").read_text(encoding=UTF8) == "original caption"

    def test_same_stem_without_txt_still_deduplicated(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "cat.jpg").write_bytes(STUB_IMAGE_BYTES)
        (root / "cat.png").write_bytes(STUB_IMAGE_BYTES)  # no cat.txt at all
        app = NLaptApp()
        app.open_dataset(root)
        # Both images would WRITE the same cat.txt on save -> one editable record.
        assert app.txt_conflicts() == {"cat.jpg": ("cat.png",)}
        assert app.resolve_scope(ScopeType.ALL) == ("cat.jpg",)


class TestFix32SuggestionDurability:
    """3.2 — suggestions are persisted no later than their checkpoint marks."""

    def test_crash_after_batch_restores_pending_and_resume_skips(
        self, tmp_path: Path
    ) -> None:
        root = _standard_dataset(tmp_path)
        config = AppConfig(request=RequestControl(concurrency=1))
        spec = RewriteSpec(type=RewriteType.POLISH)
        keys = ("a.png", "b.png")

        app1 = NLaptApp(config=config)
        app1.open_dataset(root)
        controller = BatchController()

        def cancel_after_first(done: int, total: int, key: str) -> None:
            controller.cancel()

        client1 = MockLLMClient(_echo_provider)
        report1 = app1.run_rewrite_batch(
            spec, keys, service=_mock_service(client1),
            controller=controller, on_progress=cancel_after_first,
        )
        assert report1.status is BatchStatus.CANCELLED
        assert report1.succeeded == 1
        completed_key = report1.results[0].key
        assert completed_key == "a.png"
        expected_text = app1.caption(completed_key).pending.text  # type: ignore[union-attr]
        # The session was written WITHOUT close() — the process now "crashes".
        assert (dataset_state_dir(root) / SESSION_FILE_NAME).exists()
        assert not (root / ".nlapt").exists()
        del app1

        app2 = NLaptApp(config=config)
        app2.open_dataset(root)
        # Every completed key has its pending suggestion restored.
        restored = app2.caption(completed_key).pending
        assert restored is not None
        assert restored.text == expected_text

        # A resumed batch (same checkpoint id) skips exactly the completed keys.
        client2 = MockLLMClient(_echo_provider)
        report2 = app2.run_rewrite_batch(spec, keys, service=_mock_service(client2))
        assert report2.status is BatchStatus.COMPLETED
        assert client2.calls == 1
        assert tuple(r.key for r in report2.results) == ("b.png",)
        assert app2.caption("b.png").pending is not None
        # No key lost its suggestion across the crash + resume.
        assert app2.caption("a.png").pending is not None

    def test_session_saved_when_batch_completes(self, tmp_path: Path) -> None:
        root = _standard_dataset(tmp_path)
        app = NLaptApp()
        app.open_dataset(root)
        client = MockLLMClient(_echo_provider)
        report = app.run_rewrite_batch(
            RewriteSpec(type=RewriteType.POLISH), ("a.png",), service=_mock_service(client)
        )
        assert report.status is BatchStatus.COMPLETED
        # Durable without close(): a fresh app sees the pending suggestion.
        recovered = NLaptApp()
        recovered.open_dataset(root)
        assert recovered.caption("a.png").pending is not None


class TestFix33EncodingConversion:
    """3.3 — needs_conversion is surfaced and convertible via the facade."""

    def _gbk_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "g.png").write_bytes(STUB_IMAGE_BYTES)
        (root / "g.txt").write_bytes(GBK_CAPTION.encode(GBK_ENCODING))
        _write_pair(root, "u", "plain utf-8 caption")
        return root

    def test_gbk_file_is_reported(self, tmp_path: Path) -> None:
        root = self._gbk_root(tmp_path)
        app = NLaptApp()
        events: list[Event] = []
        app.bus.subscribe(EVT_ENCODING_ISSUES, events.append)
        app.open_dataset(root)

        assert app.keys_needing_conversion() == {"g.png": GBK_ENCODING}
        assert app.caption("g.png").text == GBK_CAPTION
        assert len(events) == 1
        assert events[0].payload["keys"] == ("g.png",)
        assert events[0].payload["encodings"] == {"g.png": GBK_ENCODING}

    def test_convert_to_utf8_rewrites_file_and_clears_mapping(
        self, tmp_path: Path
    ) -> None:
        root = self._gbk_root(tmp_path)
        app = NLaptApp()
        app.open_dataset(root)

        converted = app.convert_to_utf8(["g.png"])
        assert converted == ("g.png",)
        raw = (root / "g.txt").read_bytes()
        assert raw.decode(UTF8) == GBK_CAPTION  # valid UTF-8 now
        assert not raw.startswith(b"\xef\xbb\xbf")  # no BOM
        assert app.keys_needing_conversion() == {}
        # Idempotent: converting again is a no-op.
        assert app.convert_to_utf8(["g.png"]) == ()

    def test_convert_skips_clean_files_and_rejects_unknown_keys(
        self, tmp_path: Path
    ) -> None:
        root = self._gbk_root(tmp_path)
        app = NLaptApp()
        app.open_dataset(root)
        assert app.convert_to_utf8(["u.png"]) == ()
        with pytest.raises(ValidationError):
            app.convert_to_utf8(["ghost.png"])
        with pytest.raises(ValidationError):
            app.convert_to_utf8("g.png")  # a bare string is not a key sequence

    def test_utf8_dataset_publishes_no_encoding_event(self, tmp_path: Path) -> None:
        root = _standard_dataset(tmp_path)
        app = NLaptApp()
        events: list[Event] = []
        app.bus.subscribe(EVT_ENCODING_ISSUES, events.append)
        app.open_dataset(root)
        assert app.keys_needing_conversion() == {}
        assert events == []


class TestFix34AiBatchRollback:
    """3.4 — AI rewrite batches can be rolled back via the operation log."""

    def test_rollback_clears_all_pending_suggestions(self, tmp_path: Path) -> None:
        root = _standard_dataset(tmp_path)
        app = NLaptApp()
        app.open_dataset(root)
        client = MockLLMClient(_echo_provider)
        keys = ("a.png", "b.png")
        report = app.run_rewrite_batch(
            RewriteSpec(type=RewriteType.POLISH), keys, service=_mock_service(client)
        )
        assert report.succeeded == 2

        record = app.oplog.records()[0]
        assert record.kind == AI_BATCH_KIND
        assert record.snapshot is None
        assert set(record.affected_keys) == set(keys)

        result = app.rollback_operation(record.op_id)
        assert isinstance(result, SuggestionRollbackResult)
        assert set(result.restored_files) == set(keys)
        assert result.pre_restore_snapshot is None
        for key in keys:
            assert app.caption(key).pending is None
        # The rollback is logged as its own record.
        rollback_record = app.oplog.records()[0]
        assert rollback_record.kind == ROLLBACK_KIND
        assert set(rollback_record.affected_keys) == set(keys)
        # And the cleared state is durable (session saved by the rollback).
        recovered = NLaptApp()
        recovered.open_dataset(root)
        assert recovered.caption("a.png").pending is None
        assert recovered.caption("b.png").pending is None

    def test_rollback_skips_already_reviewed_keys(self, tmp_path: Path) -> None:
        root = _standard_dataset(tmp_path)
        app = NLaptApp()
        app.open_dataset(root)
        client = MockLLMClient(_echo_provider)
        keys = ("a.png", "b.png")
        app.run_rewrite_batch(
            RewriteSpec(type=RewriteType.POLISH), keys, service=_mock_service(client)
        )
        record = app.oplog.records()[0]
        app.reject_suggestion("a.png")  # user already handled this one

        result = app.rollback_operation(record.op_id)
        assert result.restored_files == ("b.png",)
        assert app.caption("b.png").pending is None

    def test_snapshot_backed_rollback_path_unchanged(self, tmp_path: Path) -> None:
        root = _standard_dataset(tmp_path)
        app = NLaptApp()
        app.open_dataset(root)
        op = FindReplaceOperation(FindReplaceSpec(find="girl", replace="woman"))
        app.apply_operation(op, ("a.png",), description="replace girl")
        record = app.oplog.records()[0]
        assert record.kind == "batch"

        result = app.rollback_operation(record.op_id)
        assert not isinstance(result, SuggestionRollbackResult)
        assert result.pre_restore_snapshot is not None
        assert (root / "a.txt").read_text(encoding=UTF8) == "a girl, smiling"
