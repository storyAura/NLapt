"""Tests for nlapt.storage.session (spec 12 reliability)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlapt.core.errors import SessionError, ValidationError
from nlapt.storage.session import (
    SESSION_DIR_NAME,
    SESSION_FILE_NAME,
    SessionSnapshot,
    SessionStore,
)

SAVED_AT = 1_752_000_000.5


def make_snapshot(**overrides: object) -> SessionSnapshot:
    values: dict[str, object] = {
        "drafts": {"img1.png": "a draft caption, 中文内容"},
        "pending": {"img2.png": "suggested text"},
        "pending_sources": {"img2.png": "rewrite:polish"},
        "states": {"img1.png": "draft", "img2.png": "confirmed"},
        "saved_at": SAVED_AT,
    }
    values.update(overrides)
    return SessionSnapshot(**values)  # type: ignore[arg-type]


def session_file(root: Path) -> Path:
    return root / SESSION_DIR_NAME / SESSION_FILE_NAME


class TestInit:
    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            SessionStore(tmp_path / "nope")

    def test_file_root_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(ValidationError):
            SessionStore(target)


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        snapshot = make_snapshot()
        store.save(snapshot)
        loaded = store.load()
        assert loaded is not None
        assert dict(loaded.drafts) == dict(snapshot.drafts)
        assert dict(loaded.pending) == dict(snapshot.pending)
        assert dict(loaded.pending_sources) == dict(snapshot.pending_sources)
        assert dict(loaded.states) == dict(snapshot.states)
        assert loaded.saved_at == SAVED_AT

    def test_save_creates_session_dir_and_utf8_json(self, tmp_path: Path) -> None:
        SessionStore(tmp_path).save(make_snapshot())
        path = session_file(tmp_path)
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["drafts"]["img1.png"] == "a draft caption, 中文内容"
        assert payload["saved_at"] == SAVED_AT

    def test_state_dir_writes_session_outside_dataset(self, tmp_path: Path) -> None:
        dest = tmp_path / "state"
        SessionStore(tmp_path, state_dir=dest).save(make_snapshot())
        assert (dest / SESSION_FILE_NAME).is_file()
        assert not session_file(tmp_path).exists()

    def test_save_overwrites_previous(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.save(make_snapshot())
        store.save(make_snapshot(drafts={"other.png": "new"}, saved_at=99.0))
        loaded = store.load()
        assert loaded is not None
        assert dict(loaded.drafts) == {"other.png": "new"}
        assert loaded.saved_at == 99.0

    def test_empty_mappings_round_trip(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.save(
            make_snapshot(drafts={}, pending={}, pending_sources={}, states={})
        )
        loaded = store.load()
        assert loaded is not None
        assert dict(loaded.drafts) == {}

    def test_loaded_mappings_are_immutable(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.save(make_snapshot())
        loaded = store.load()
        assert loaded is not None
        with pytest.raises(TypeError):
            loaded.drafts["img1.png"] = "mutated"  # type: ignore[index]

    def test_save_validates_mapping_values(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        with pytest.raises(ValidationError):
            store.save(make_snapshot(drafts={"k": 123}))
        with pytest.raises(ValidationError):
            store.save(make_snapshot(saved_at="not a number"))
        with pytest.raises(ValidationError):
            store.save(make_snapshot(pending="not a mapping"))


class TestLoadFailures:
    def test_load_absent_returns_none(self, tmp_path: Path) -> None:
        assert SessionStore(tmp_path).load() is None

    def test_corrupt_json_raises(self, tmp_path: Path) -> None:
        path = session_file(tmp_path)
        path.parent.mkdir()
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SessionError):
            SessionStore(tmp_path).load()

    def test_non_object_top_level_raises(self, tmp_path: Path) -> None:
        path = session_file(tmp_path)
        path.parent.mkdir()
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SessionError):
            SessionStore(tmp_path).load()

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        path = session_file(tmp_path)
        path.parent.mkdir()
        path.write_text(
            json.dumps({"drafts": {}, "pending": {}, "states": {}, "saved_at": 1.0}),
            encoding="utf-8",
        )
        with pytest.raises(SessionError):
            SessionStore(tmp_path).load()

    def test_non_string_values_raise(self, tmp_path: Path) -> None:
        path = session_file(tmp_path)
        path.parent.mkdir()
        payload = {
            "drafts": {"k": 5},
            "pending": {},
            "pending_sources": {},
            "states": {},
            "saved_at": 1.0,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SessionError):
            SessionStore(tmp_path).load()

    def test_bad_saved_at_raises(self, tmp_path: Path) -> None:
        path = session_file(tmp_path)
        path.parent.mkdir()
        payload = {
            "drafts": {},
            "pending": {},
            "pending_sources": {},
            "states": {},
            "saved_at": "yesterday",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SessionError):
            SessionStore(tmp_path).load()

    def test_invalid_utf8_raises(self, tmp_path: Path) -> None:
        path = session_file(tmp_path)
        path.parent.mkdir()
        path.write_bytes(b"\xff\xfe\x00broken")
        with pytest.raises(SessionError):
            SessionStore(tmp_path).load()


class TestClear:
    def test_clear_removes_file(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.save(make_snapshot())
        assert session_file(tmp_path).exists()
        store.clear()
        assert not session_file(tmp_path).exists()
        assert store.load() is None

    def test_clear_when_absent_is_noop(self, tmp_path: Path) -> None:
        SessionStore(tmp_path).clear()  # must not raise
