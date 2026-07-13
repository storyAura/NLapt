"""Tests for nlapt_gui.prompt_store (custom system/user prompts, 设置 ▸ 提示词)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.errors import StorageError

from nlapt_gui.prompt_store import (
    DEFAULT_PROMPT_NAME,
    DEFAULT_USER_PROMPT,
    VisionPrompts,
    load_vision_prompts,
    save_vision_prompts,
    vision_prompts_path,
)


class TestModel:
    def test_defaults(self) -> None:
        prompts = VisionPrompts()
        assert prompts.active == DEFAULT_PROMPT_NAME == "默认"
        assert prompts.system_text() == ""  # 默认留空 (user supplies later)
        assert prompts.user_prompt == ""
        assert prompts.effective_user_prompt() == DEFAULT_USER_PROMPT

    def test_names_lists_default_first(self) -> None:
        prompts = VisionPrompts(prompts={"动漫": "sys-a", "写实": "sys-b"})
        assert prompts.names() == (DEFAULT_PROMPT_NAME, "动漫", "写实")

    def test_system_text_of_active_custom(self) -> None:
        prompts = VisionPrompts(active="动漫", prompts={"动漫": "sys-a"})
        assert prompts.system_text() == "sys-a"

    def test_effective_user_prompt_prefers_custom(self) -> None:
        prompts = VisionPrompts(user_prompt="describe briefly")
        assert prompts.effective_user_prompt() == "describe briefly"

    def test_with_changes_is_immutable_update(self) -> None:
        original = VisionPrompts()
        changed = original.with_changes(user_prompt="x")
        assert original.user_prompt == ""
        assert changed.user_prompt == "x"


class TestPersistence:
    def test_round_trip(self) -> None:
        original = VisionPrompts(
            active="动漫", prompts={"动漫": "sys-a"}, user_prompt="up"
        )
        save_vision_prompts(original)
        assert vision_prompts_path().exists()
        loaded = load_vision_prompts()
        assert loaded == original

    def test_missing_file_gives_defaults(self) -> None:
        assert load_vision_prompts() == VisionPrompts()

    def test_corrupt_file_gives_defaults(self, tmp_path: Path) -> None:
        target = tmp_path / "vision_prompts.json"
        target.write_text("{not json", encoding="utf-8")
        assert load_vision_prompts(target) == VisionPrompts()

    def test_unknown_active_falls_back_to_default(self, tmp_path: Path) -> None:
        target = tmp_path / "vision_prompts.json"
        target.write_text(
            '{"active": "missing", "prompts": {}, "user_prompt": ""}',
            encoding="utf-8",
        )
        assert load_vision_prompts(target).active == DEFAULT_PROMPT_NAME

    def test_default_name_never_stored_as_custom(self, tmp_path: Path) -> None:
        target = tmp_path / "vision_prompts.json"
        target.write_text(
            '{"active": "默认", "prompts": {"默认": "sneaky"}, "user_prompt": ""}',
            encoding="utf-8",
        )
        loaded = load_vision_prompts(target)
        assert "默认" not in loaded.prompts
        assert loaded.system_text() == ""

    def test_save_rejects_wrong_type(self) -> None:
        with pytest.raises(StorageError):
            save_vision_prompts({"active": "x"})  # type: ignore[arg-type]
