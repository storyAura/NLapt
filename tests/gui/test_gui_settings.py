"""Tests for nlapt_gui.settings (UISettings JSON round-trip)."""

from __future__ import annotations

import json
from pathlib import Path

from nlapt_gui.settings import UISettings, load_ui_settings, save_ui_settings


class TestDefaults:
    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        settings = load_ui_settings(tmp_path / "nope.json")
        assert settings == UISettings()

    def test_default_values_match_contract(self) -> None:
        settings = UISettings()
        assert settings.theme == "雾灰"
        assert settings.accent == "#0E9384"
        assert settings.thumb_min == 96
        assert settings.view_mode == "mid"
        assert settings.editor_h == 330
        assert dict(settings.sections) == {"fr": True, "ps": False, "tr": False, "hist": True}
        assert settings.last_root == ""
        assert dict(settings.folder_open) == {}


class TestRoundTrip:
    def test_save_then_load_is_identity(self, tmp_path: Path) -> None:
        path = tmp_path / "ui.json"
        original = UISettings(
            theme="墨黑",
            accent="#4F63E7",
            thumb_min=120,
            view_mode="big",
            editor_h=400,
            folder_open={"根目录": False, "10_concept": True},
            last_root="D:/data/set1",
            sections={"fr": False, "ps": True, "tr": True, "hist": False},
        )
        save_ui_settings(original, path)
        loaded = load_ui_settings(path)
        assert loaded == original

    def test_default_path_used_when_omitted(self) -> None:
        # NLAPT_DATA_DIR is isolated per test by conftest.
        save_ui_settings(UISettings(theme="石墨"))
        assert load_ui_settings().theme == "石墨"

    def test_file_is_valid_utf8_json(self, tmp_path: Path) -> None:
        path = tmp_path / "ui.json"
        save_ui_settings(UISettings(last_root="D:/训练集"), path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["last_root"] == "D:/训练集"


class TestCorruption:
    def test_corrupt_json_falls_back_to_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "ui.json"
        path.write_text("{not valid json!!", encoding="utf-8")
        assert load_ui_settings(path) == UISettings()

    def test_non_object_json_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "ui.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_ui_settings(path) == UISettings()

    def test_invalid_fields_fall_back_individually(self, tmp_path: Path) -> None:
        path = tmp_path / "ui.json"
        payload = {
            "theme": "石墨",
            "accent": 12345,               # wrong type
            "thumb_min": "huge",           # wrong type
            "view_mode": "gigantic",       # not an allowed mode
            "editor_h": 99999,             # clamped to range
            "folder_open": "nope",         # wrong type
            "sections": {"fr": 0},
            "last_root": None,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_ui_settings(path)
        assert loaded.theme == "石墨"
        assert loaded.accent == UISettings().accent
        assert loaded.thumb_min == UISettings().thumb_min
        assert loaded.view_mode == "mid"
        assert loaded.editor_h == 620  # clamped to EDITOR_H_RANGE max
        assert dict(loaded.folder_open) == {}
        assert loaded.last_root == ""
        # sections merge over defaults
        assert dict(loaded.sections) == {"fr": False, "ps": False, "tr": False, "hist": True}

    def test_thumb_min_clamped_to_range(self, tmp_path: Path) -> None:
        path = tmp_path / "ui.json"
        path.write_text(json.dumps({"thumb_min": 10}), encoding="utf-8")
        assert load_ui_settings(path).thumb_min == 72
