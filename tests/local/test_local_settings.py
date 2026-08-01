"""Tests for nlapt.local.settings: roundtrip, clamping, corruption handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.errors import StorageError
from nlapt.local.settings import (
    CONTEXT_RANGE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_PARALLEL,
    DEFAULT_PORT,
    PARALLEL_RANGE,
    PORT_RANGE,
    LocalSettings,
    load_local_settings,
    save_local_settings,
)


def settings_file(tmp_path: Path) -> Path:
    return tmp_path / "local_llm.json"


class TestDefaults:
    def test_missing_file_gives_defaults(self, tmp_path: Path) -> None:
        settings = load_local_settings(settings_file(tmp_path))
        assert settings == LocalSettings()
        assert settings.port == DEFAULT_PORT
        assert settings.parallel == DEFAULT_PARALLEL
        assert settings.context_length == DEFAULT_CONTEXT_LENGTH

    def test_corrupt_json_gives_defaults(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        target.write_text("{not json", encoding="utf-8")
        assert load_local_settings(target) == LocalSettings()

    def test_non_object_gives_defaults(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        target.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_local_settings(target) == LocalSettings()


class TestRoundtrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        original = LocalSettings(
            models_dir=str(tmp_path / "models"),
            extra_dirs=(str(tmp_path / "shared-a"), str(tmp_path / "shared-b")),
            server_path=str(tmp_path / "llama-server.exe"),
            port=8080,
            context_length=8192,
            gpu_layers=20,
            threads=8,
            parallel=4,
            family_id="gemma4-12b",
            quant_label="Q4_K_M",
        )
        save_local_settings(target, original)
        assert load_local_settings(target) == original

    def test_extra_dirs_sanitized_on_load(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        target.write_text(
            '{"extra_dirs": ["D:/ok", "", "   ", 5, null]}', encoding="utf-8"
        )
        assert load_local_settings(target).extra_dirs == ("D:/ok",)

    def test_extra_dirs_non_list_ignored(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        target.write_text('{"extra_dirs": "not-a-list"}', encoding="utf-8")
        assert load_local_settings(target).extra_dirs == ()

    def test_florence_task_roundtrip(self, tmp_path: Path) -> None:
        from nlapt.local.florence import TASK_ANALYZE

        target = settings_file(tmp_path)
        save_local_settings(target, LocalSettings(florence_task=TASK_ANALYZE))
        assert load_local_settings(target).florence_task == TASK_ANALYZE

    def test_florence_task_invalid_falls_back_to_default(
        self, tmp_path: Path
    ) -> None:
        from nlapt.local.florence import DEFAULT_FLORENCE_TASK

        target = settings_file(tmp_path)
        target.write_text('{"florence_task": "<NOPE>"}', encoding="utf-8")
        assert load_local_settings(target).florence_task == DEFAULT_FLORENCE_TASK

    def test_prompt_preset_roundtrip(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        save_local_settings(target, LocalSettings(prompt_preset="Danbooru tag list"))
        assert load_local_settings(target).prompt_preset == "Danbooru tag list"

    def test_prompt_preset_defaults_to_empty(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        target.write_text("{}", encoding="utf-8")
        assert load_local_settings(target).prompt_preset == ""

    def test_save_rejects_wrong_type(self, tmp_path: Path) -> None:
        with pytest.raises(StorageError):
            save_local_settings(settings_file(tmp_path), {"port": 1})  # type: ignore[arg-type]

    def test_with_changes_returns_new_object(self) -> None:
        base = LocalSettings()
        changed = base.with_changes(parallel=8)
        assert changed.parallel == 8
        assert base.parallel == DEFAULT_PARALLEL
        assert changed is not base


class TestClamping:
    def test_out_of_range_values_clamped(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        target.write_text(
            '{"port": 80, "context_length": 999999999, "parallel": 99,'
            ' "threads": -3, "gpu_layers": -50}',
            encoding="utf-8",
        )
        settings = load_local_settings(target)
        assert settings.port == PORT_RANGE[0]
        assert settings.context_length == CONTEXT_RANGE[1]
        assert settings.parallel == PARALLEL_RANGE[1]
        assert settings.threads == 0
        assert settings.gpu_layers == -1

    def test_non_numeric_values_fall_back(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        target.write_text(
            '{"port": "abc", "parallel": null, "context_length": {}}',
            encoding="utf-8",
        )
        settings = load_local_settings(target)
        assert settings.port == DEFAULT_PORT
        assert settings.parallel == DEFAULT_PARALLEL
        assert settings.context_length == DEFAULT_CONTEXT_LENGTH

    def test_string_fields_coerced(self, tmp_path: Path) -> None:
        target = settings_file(tmp_path)
        target.write_text('{"family_id": 123, "models_dir": 0}', encoding="utf-8")
        settings = load_local_settings(target)
        assert settings.family_id == "123"
        assert settings.models_dir == "0"
