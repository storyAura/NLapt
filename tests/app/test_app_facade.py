"""Unit tests for the NLaptApp facade (contract section "nlapt.app")."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.app import NLaptApp, ScopeType
from nlapt.batch.progress import BatchStatus
from nlapt.core.errors import OperationError, ValidationError
from nlapt.core.events import EVT_FILE_SAVED, Event
from nlapt.core.states import CaptionState
from nlapt.ops.find_replace import FindReplaceOperation, FindReplaceSpec
from nlapt.workflow.diff import DiffOp

from nlapt.storage.paths import dataset_state_dir
from nlapt.storage.session import SESSION_FILE_NAME

from tests.app.conftest import CAPTION_A, CAPTION_B


@pytest.fixture
def app(dataset_root: Path) -> NLaptApp:
    application = NLaptApp()
    application.open_dataset(dataset_root)
    return application


def _replace_op(find: str, replace: str) -> FindReplaceOperation:
    return FindReplaceOperation(FindReplaceSpec(find=find, replace=replace))


class TestOpenDataset:
    def test_loads_captions_in_natural_order(self, app: NLaptApp) -> None:
        assert app.caption("a.png").text == CAPTION_A
        assert app.caption("b.png").text == CAPTION_B
        assert app.caption("c.png").text == ""

    def test_initial_states(self, app: NLaptApp) -> None:
        assert app.caption("a.png").state is CaptionState.DRAFT
        assert app.caption("c.png").state is CaptionState.UNLABELED

    def test_scan_result_keys(self, dataset_root: Path) -> None:
        result = NLaptApp().open_dataset(dataset_root)
        assert [f.key for f in result.images] == ["a.png", "b.png", "c.png"]

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            NLaptApp().open_dataset(tmp_path / "nope")

    def test_methods_require_open_dataset(self) -> None:
        bare = NLaptApp()
        with pytest.raises(ValidationError):
            bare.caption("a.png")
        with pytest.raises(ValidationError):
            bare.search("girl")
        with pytest.raises(ValidationError):
            bare.save_session()


class TestEditing:
    def test_edit_marks_draft_and_dirty(self, app: NLaptApp) -> None:
        record = app.edit("c.png", "a cat")
        assert record.state is CaptionState.DRAFT
        assert record.dirty

    def test_edit_updates_index(self, app: NLaptApp) -> None:
        app.edit("c.png", "a red panda")
        assert "c.png" in app.search("panda")

    def test_undo_redo_roundtrip(self, app: NLaptApp) -> None:
        app.edit("a.png", "edited text")
        assert app.undo("a.png") == CAPTION_A
        assert app.caption("a.png").text == CAPTION_A
        assert app.redo("a.png") == "edited text"
        assert app.caption("a.png").text == "edited text"

    def test_undo_nothing_returns_none(self, app: NLaptApp) -> None:
        assert app.undo("a.png") is None
        assert app.redo("a.png") is None

    def test_undo_refreshes_index(self, app: NLaptApp) -> None:
        app.edit("a.png", "zebra stripes")
        app.undo("a.png")
        assert "a.png" not in app.search("zebra")
        assert "a.png" in app.search("girl")

    def test_token_count_positive(self, app: NLaptApp) -> None:
        assert app.token_count("a.png") > 0
        assert app.token_count("c.png") == 0


class TestSaving:
    def test_save_writes_disk_and_publishes(self, app: NLaptApp, dataset_root: Path) -> None:
        events: list[Event] = []
        app.bus.subscribe(EVT_FILE_SAVED, events.append)
        app.edit("a.png", "new caption")
        app.save("a.png")
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == "new caption"
        assert not app.caption("a.png").dirty
        assert events and events[-1].payload["key"] == "a.png"

    def test_save_all_dirty(self, app: NLaptApp, dataset_root: Path) -> None:
        app.edit("a.png", "one")
        app.edit("c.png", "two")
        saved = app.save_all_dirty()
        assert set(saved) == {"a.png", "c.png"}
        assert (dataset_root / "c.txt").read_text(encoding="utf-8") == "two"

    def test_export_dataset_zips_images_and_txts(
        self, app: NLaptApp, dataset_root: Path, tmp_path: Path
    ) -> None:
        import zipfile

        dest = tmp_path / "set.zip"
        count = app.export_dataset(dest)
        with zipfile.ZipFile(dest) as archive:
            names = set(archive.namelist())
        assert names == {"a.png", "a.txt", "b.png", "b.txt", "c.png"}
        assert count == 5

    def test_confirm_saves_and_returns_next_unconfirmed(
        self, app: NLaptApp, dataset_root: Path
    ) -> None:
        app.edit("a.png", "confirmed caption")
        next_key = app.confirm("a.png")
        assert next_key == "b.png"
        assert app.caption("a.png").state is CaptionState.CONFIRMED
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == "confirmed caption"

    def test_confirm_empty_caption_raises(self, app: NLaptApp) -> None:
        with pytest.raises(ValidationError):
            app.confirm("c.png")

    def test_close_saves_dirty_and_session(self, app: NLaptApp, dataset_root: Path) -> None:
        app.edit("a.png", "before close")
        app.close()
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == "before close"
        assert (dataset_state_dir(dataset_root) / SESSION_FILE_NAME).exists()
        assert not (dataset_root / ".nlapt").exists()

    def test_open_migrates_legacy_dataset_dirs(self, tmp_path: Path) -> None:
        from tests.app.conftest import make_dataset

        root = make_dataset(tmp_path / "legacy")
        (root / ".backups").mkdir()
        (root / ".backups" / "old.zip").write_bytes(b"zip")
        (root / ".nlapt").mkdir()
        (root / ".nlapt" / SESSION_FILE_NAME).write_text("{}", encoding="utf-8")
        NLaptApp().open_dataset(root)
        state = dataset_state_dir(root)
        assert (state / "backups" / "old.zip").read_bytes() == b"zip"
        assert not (root / ".backups").exists()
        assert not (root / ".nlapt").exists()


class TestSearchAndScope:
    def test_search_caption_and_filename(self, app: NLaptApp) -> None:
        assert app.search("girl") == ("a.png", "b.png")
        assert "a.png" in app.search("a.png")

    def test_search_negation(self, app: NLaptApp) -> None:
        assert app.search("girl -hat") == ("a.png",)

    def test_blank_query_returns_all(self, app: NLaptApp) -> None:
        assert app.search("") == ("a.png", "b.png", "c.png")

    def test_resolve_scope_all(self, app: NLaptApp) -> None:
        assert app.resolve_scope(ScopeType.ALL) == ("a.png", "b.png", "c.png")

    def test_resolve_scope_current(self, app: NLaptApp) -> None:
        assert app.resolve_scope(ScopeType.CURRENT, current="b.png") == ("b.png",)

    def test_resolve_scope_current_requires_key(self, app: NLaptApp) -> None:
        with pytest.raises(ValidationError):
            app.resolve_scope(ScopeType.CURRENT)

    def test_resolve_scope_selected_and_filtered(self, app: NLaptApp) -> None:
        assert app.resolve_scope(ScopeType.SELECTED, selected=["c.png"]) == ("c.png",)
        assert app.resolve_scope(ScopeType.FILTERED, filtered=["a.png", "b.png"]) == (
            "a.png",
            "b.png",
        )

    def test_resolve_scope_unknown_key_raises(self, app: NLaptApp) -> None:
        with pytest.raises(ValidationError):
            app.resolve_scope(ScopeType.SELECTED, selected=["ghost.png"])


class TestBatchOperations:
    def test_preview_lists_only_matching_files(self, app: NLaptApp) -> None:
        previews = app.preview_operation(_replace_op("hat", "cap"), app.resolve_scope(ScopeType.ALL))
        assert set(previews) == {"b.png"}
        assert previews["b.png"][0].matched == "hat"

    def test_apply_operation_saves_and_logs(self, app: NLaptApp, dataset_root: Path) -> None:
        report = app.apply_operation(
            _replace_op("girl", "woman"),
            app.resolve_scope(ScopeType.ALL),
            description="replace girl with woman",
        )
        assert report.status is BatchStatus.COMPLETED
        assert report.succeeded == 3
        assert report.snapshot is not None
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == "a woman, smiling"
        record = app.oplog.records()[0]
        assert record.kind == "batch"
        assert set(record.affected_keys) == {"a.txt", "b.txt"}
        assert record.snapshot is not None

    def test_apply_operation_requires_description(self, app: NLaptApp) -> None:
        with pytest.raises(ValidationError):
            app.apply_operation(_replace_op("a", "b"), ("a.png",), description="  ")

    def test_rollback_unknown_op_raises(self, app: NLaptApp) -> None:
        with pytest.raises(OperationError):
            app.rollback_operation(999)


class TestSuggestions:
    def test_set_suggestion_keeps_body(self, app: NLaptApp) -> None:
        record = app.set_suggestion("a.png", "a lady, smiling", "rewrite:polish")
        assert record.text == CAPTION_A
        assert record.pending is not None
        assert record.pending.source == "rewrite:polish"

    def test_set_suggestion_requires_source(self, app: NLaptApp) -> None:
        with pytest.raises(ValidationError):
            app.set_suggestion("a.png", "text", "")

    def test_suggestion_diff_marks_changes(self, app: NLaptApp) -> None:
        app.set_suggestion("a.png", "a lady, smiling", "rewrite:polish")
        segments = app.suggestion_diff("a.png")
        assert any(segment.op is DiffOp.INSERT for segment in segments)
        assert any(segment.op is DiffOp.DELETE for segment in segments)

    def test_suggestion_diff_without_pending_raises(self, app: NLaptApp) -> None:
        with pytest.raises(ValidationError):
            app.suggestion_diff("a.png")

    def test_accept_saves_and_returns_next_pending(
        self, app: NLaptApp, dataset_root: Path
    ) -> None:
        app.set_suggestion("a.png", "a lady, smiling", "rewrite:polish")
        app.set_suggestion("b.png", "a lady with hat", "rewrite:polish")
        next_key = app.accept_suggestion("a.png")
        assert next_key == "b.png"
        assert app.caption("a.png").pending is None
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == "a lady, smiling"

    def test_reject_keeps_body_untouched(self, app: NLaptApp, dataset_root: Path) -> None:
        app.set_suggestion("b.png", "discarded", "rewrite:polish")
        assert app.reject_suggestion("b.png") is None
        assert app.caption("b.png").text == CAPTION_B
        assert (dataset_root / "b.txt").read_text(encoding="utf-8") == CAPTION_B

    def test_accept_for_edit_yields_draft(self, app: NLaptApp, dataset_root: Path) -> None:
        app.set_suggestion("a.png", "editable suggestion", "rewrite:polish")
        record = app.accept_for_edit("a.png")
        assert record.state is CaptionState.DRAFT
        assert record.dirty
        assert record.pending is None
        # Not saved yet: disk still has the old caption.
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == CAPTION_A

    def test_accept_without_pending_raises(self, app: NLaptApp) -> None:
        with pytest.raises(OperationError):
            app.accept_suggestion("a.png")
