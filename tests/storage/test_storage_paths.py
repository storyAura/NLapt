"""Tests for nlapt.storage.paths (per-user / per-dataset state)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.storage.paths import (
    DATASETS_DIR_NAME,
    DOCUMENTS_FOLDER_NAME,
    ENV_DATA_DIR,
    ENV_DOCUMENTS_DIR,
    LEGACY_BACKUP_DIR_NAME,
    LEGACY_SESSION_DIR_NAME,
    STATE_BACKUPS_DIR_NAME,
    WINDOWS_DIR_NAME,
    app_state_dir,
    dataset_state_dir,
    dataset_state_key,
    migrate_legacy_dataset_state,
    user_documents_app_dir,
    user_documents_dir,
)
from nlapt.storage.session import SESSION_FILE_NAME


class TestAppStateDir:
    def test_honours_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "override-home"
        monkeypatch.setenv(ENV_DATA_DIR, str(target))
        assert app_state_dir() == target
        assert target.is_dir()


class TestUserDocumentsDir:
    def test_honours_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "docs-home"
        monkeypatch.setenv(ENV_DOCUMENTS_DIR, str(target))
        monkeypatch.setattr(
            "nlapt.storage.paths.windows_known_documents_dir",
            lambda: tmp_path / "known-should-lose",
        )
        assert user_documents_dir() == target
        app_dir = user_documents_app_dir()
        assert app_dir == target / WINDOWS_DIR_NAME
        assert app_dir.is_dir()

    def test_known_folder_used_when_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_DOCUMENTS_DIR, raising=False)
        known = tmp_path / "known-docs"
        monkeypatch.setattr(
            "nlapt.storage.paths.windows_known_documents_dir", lambda: known
        )
        assert user_documents_dir() == known

    def test_falls_back_to_home_documents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_DOCUMENTS_DIR, raising=False)
        monkeypatch.setattr(
            "nlapt.storage.paths.windows_known_documents_dir", lambda: None
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert user_documents_dir() == tmp_path / DOCUMENTS_FOLDER_NAME


class TestDatasetStateDir:
    def test_key_is_stable_and_normcased(self, tmp_path: Path) -> None:
        root = tmp_path / "set"
        root.mkdir()
        first = dataset_state_key(root)
        second = dataset_state_key(Path(str(root)))
        assert first == second
        assert len(first) == 40

    def test_dir_lives_under_app_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        monkeypatch.setenv(ENV_DATA_DIR, str(home))
        root = tmp_path / "dataset"
        root.mkdir()
        state = dataset_state_dir(root)
        assert state == home / DATASETS_DIR_NAME / dataset_state_key(root)
        assert state.is_dir()


class TestMigrateLegacy:
    def test_moves_backups_and_session_then_removes_legacy(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        legacy_backups = root / LEGACY_BACKUP_DIR_NAME
        legacy_backups.mkdir()
        (legacy_backups / "2026-01-01_1200_op.zip").write_bytes(b"zip")
        legacy_session = root / LEGACY_SESSION_DIR_NAME
        legacy_session.mkdir()
        (legacy_session / SESSION_FILE_NAME).write_text("{}", encoding="utf-8")
        (legacy_session / "checkpoints.json").write_text("{}", encoding="utf-8")

        dest = tmp_path / "state"
        migrate_legacy_dataset_state(root, dest)

        assert (dest / STATE_BACKUPS_DIR_NAME / "2026-01-01_1200_op.zip").read_bytes() == b"zip"
        assert (dest / SESSION_FILE_NAME).read_text(encoding="utf-8") == "{}"
        assert (dest / "checkpoints.json").read_text(encoding="utf-8") == "{}"
        assert not legacy_backups.exists()
        assert not legacy_session.exists()

    def test_existing_dest_wins_and_legacy_is_still_removed(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        legacy_session = root / LEGACY_SESSION_DIR_NAME
        legacy_session.mkdir()
        (legacy_session / SESSION_FILE_NAME).write_text("old", encoding="utf-8")
        dest = tmp_path / "state"
        dest.mkdir()
        (dest / SESSION_FILE_NAME).write_text("new", encoding="utf-8")

        migrate_legacy_dataset_state(root, dest)

        assert (dest / SESSION_FILE_NAME).read_text(encoding="utf-8") == "new"
        assert not legacy_session.exists()

    def test_missing_legacy_dirs_are_a_noop(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        dest = tmp_path / "state"
        migrate_legacy_dataset_state(root, dest)
        assert dest.is_dir()
        assert list(dest.iterdir()) == []
