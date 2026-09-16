"""Tests for NLaptApp.run_caption_batch (推标: destructive vision batch)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.app import BATCH_KIND, EMPTY_CAPTION_ERROR, SKIPPED_DETAIL, NLaptApp
from nlapt.core.errors import ValidationError


def open_app(dataset_root: Path) -> NLaptApp:
    app = NLaptApp()
    app.open_dataset(dataset_root)
    return app


class TestRunCaptionBatch:
    def test_writes_saves_snapshots_and_logs(self, dataset_root: Path) -> None:
        app = open_app(dataset_root)
        report = app.run_caption_batch(
            ("a.png", "b.png"),
            lambda key, path: f"new {key}",
            description="推标测试",
            engine="llm",
            concurrency=2,
        )
        assert report.succeeded == 2 and report.failed == 0
        assert report.snapshot is not None  # destructive batch => safety snapshot
        for key in ("a.png", "b.png"):
            record = app.caption(key)
            assert record.text == f"new {key}"
            assert not record.dirty  # saved to disk per item
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == "new a.png"
        logged = app.oplog.records()[-1]
        assert logged.kind == BATCH_KIND
        assert logged.description == "推标测试"
        assert set(logged.affected_keys) == {"a.txt", "b.txt"}

    def test_rollback_restores_previous_captions(self, dataset_root: Path) -> None:
        app = open_app(dataset_root)
        original = app.caption("a.png").text
        app.run_caption_batch(
            ("a.png",), lambda key, path: "overwritten", description="推标"
        )
        op_id = app.oplog.records()[-1].op_id
        app.rollback_operation(op_id)
        assert app.caption("a.png").text == original
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == original

    def test_empty_result_fails_item_and_keeps_caption(
        self, dataset_root: Path
    ) -> None:
        app = open_app(dataset_root)
        original = app.caption("a.png").text

        def caption(key: str, path: Path) -> str:
            return "   " if key == "a.png" else "fine"

        report = app.run_caption_batch(
            ("a.png", "b.png"), caption, description="推标"
        )
        assert report.failed == 1
        assert report.failed_keys == ("a.png",)
        failed = next(item for item in report.results if item.key == "a.png")
        assert failed.error == EMPTY_CAPTION_ERROR
        assert app.caption("a.png").text == original
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == original

    def test_none_result_skips_item_without_failing(self, dataset_root: Path) -> None:
        app = open_app(dataset_root)
        original = app.caption("a.png").text

        def caption(key: str, path: Path) -> str | None:
            return None if key == "a.png" else "fine"

        report = app.run_caption_batch(
            ("a.png", "b.png"), caption, description="推标"
        )
        assert report.succeeded == 2 and report.failed == 0
        skipped = next(item for item in report.results if item.key == "a.png")
        assert skipped.ok and skipped.detail == SKIPPED_DETAIL
        assert app.caption("a.png").text == original
        assert (dataset_root / "a.txt").read_text(encoding="utf-8") == original
        assert app.oplog.records()[-1].affected_keys == ("b.txt",)

    def test_unchanged_text_is_ok_but_not_logged_as_changed(
        self, dataset_root: Path
    ) -> None:
        app = open_app(dataset_root)
        same = app.caption("a.png").text
        report = app.run_caption_batch(
            ("a.png",), lambda key, path: same, description="推标"
        )
        assert report.succeeded == 1
        assert app.oplog.records()[-1].affected_keys == ()

    def test_worker_receives_the_image_path(self, dataset_root: Path) -> None:
        app = open_app(dataset_root)
        seen: list[Path] = []

        def caption(key: str, path: Path) -> str:
            seen.append(path)
            return "with path"

        app.run_caption_batch(("a.png",), caption, description="推标")
        assert seen == [dataset_root / "a.png"]

    def test_validates_inputs(self, dataset_root: Path) -> None:
        app = open_app(dataset_root)
        with pytest.raises(ValidationError):
            app.run_caption_batch(("a.png",), "not-callable", description="d")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            app.run_caption_batch(("a.png",), lambda k, p: "x", description="  ")
        with pytest.raises(ValidationError):
            app.run_caption_batch(("ghost.png",), lambda k, p: "x", description="d")
