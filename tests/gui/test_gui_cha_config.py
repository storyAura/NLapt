"""Tests for nlapt_gui.cha_config (CHA标注 API + per-scheme models)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.config import LLMProfile
from nlapt.core.errors import StorageError

from nlapt_gui.api_config import api_config_path
from nlapt_gui.cha_config import (
    API_MODE_OWN,
    API_MODE_SYNC,
    CHASettings,
    cha_settings_path,
    load_cha_settings,
    resolve_base_profile,
    resolve_batch_profile,
    resolve_card_profile,
    save_cha_settings,
)

ACTIVE = LLMProfile(
    name="default",
    api_type="openai",
    base_url="http://main.local",
    api_key="sk-main",
    text_model="text-main",
    vision_model="vision-main",
    temperature=0.4,
    max_tokens=512,
    system_prompt="sys",
)


class TestRoundTrip:
    def test_save_keeps_key_out_of_appdata(self) -> None:
        settings = CHASettings(
            api_mode=API_MODE_OWN,
            api_type="openai",
            base_url="http://cha.local",
            api_key="sk-secret-cha",
            card_models=("m0", "m1", "m2"),
            batch_model="mbatch",
        )
        save_cha_settings(settings)
        loaded = load_cha_settings()
        assert loaded == settings
        public = cha_settings_path().read_text(encoding="utf-8")
        assert "sk-secret-cha" not in public
        assert "http://cha.local" not in public
        unified = api_config_path().read_text(encoding="utf-8")
        assert "sk-secret-cha" in unified
        assert "http://cha.local" in unified

    def test_missing_file_is_sync_defaults(self) -> None:
        assert load_cha_settings() == CHASettings()

    def test_corrupt_file_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("not-json", encoding="utf-8")
        assert load_cha_settings(path) == CHASettings()

    def test_unknown_mode_and_short_models_normalize(self) -> None:
        settings = CHASettings(api_mode="nope", card_models=("only",))
        assert settings.api_mode == API_MODE_SYNC
        assert settings.card_models == ("only", "", "")

    def test_save_rejects_wrong_type(self) -> None:
        with pytest.raises(StorageError):
            save_cha_settings("nope")  # type: ignore[arg-type]


class TestResolve:
    def test_sync_uses_active_profile(self) -> None:
        settings = CHASettings()
        assert resolve_base_profile(settings, ACTIVE) is ACTIVE
        card = resolve_card_profile(settings, ACTIVE, 0)
        assert card is not None
        assert card.vision_model == "vision-main"
        batch = resolve_batch_profile(settings, ACTIVE)
        assert batch is not None
        assert batch.vision_model == "vision-main"

    def test_sync_without_active_is_none(self) -> None:
        settings = CHASettings()
        assert resolve_base_profile(settings, None) is None
        assert resolve_card_profile(settings, None, 0) is None
        assert resolve_batch_profile(settings, None) is None

    def test_own_uses_independent_endpoint(self) -> None:
        settings = CHASettings(
            api_mode=API_MODE_OWN,
            api_type="ollama",
            base_url="http://cha.local",
            api_key="sk-cha",
            card_models=("c0", "", "c2"),
            batch_model="c-batch",
        )
        base = resolve_base_profile(settings, ACTIVE)
        assert base is not None
        assert base.name == "cha"
        assert base.api_type == "ollama"
        assert base.base_url == "http://cha.local"
        assert base.api_key == "sk-cha"
        assert base.vision_model == "vision-main"
        assert base.temperature == ACTIVE.temperature
        card0 = resolve_card_profile(settings, ACTIVE, 0)
        card1 = resolve_card_profile(settings, ACTIVE, 1)
        batch = resolve_batch_profile(settings, ACTIVE)
        assert card0 is not None and card0.vision_model == "c0"
        assert card1 is not None and card1.vision_model == "vision-main"
        assert batch is not None and batch.vision_model == "c-batch"

    def test_own_without_base_url_is_none(self) -> None:
        settings = CHASettings(api_mode=API_MODE_OWN, card_models=("c0", "", ""))
        assert resolve_base_profile(settings, ACTIVE) is None
        assert resolve_card_profile(settings, ACTIVE, 0) is None

    def test_own_without_active_uses_card_override(self) -> None:
        settings = CHASettings(
            api_mode=API_MODE_OWN,
            base_url="http://cha.local",
            card_models=("solo", "", ""),
        )
        card = resolve_card_profile(settings, None, 0)
        assert card is not None
        assert card.vision_model == "solo"
        assert resolve_card_profile(settings, None, 1) is None
        assert resolve_batch_profile(settings, None) is None

    def test_empty_models_without_vision_is_none(self) -> None:
        bare = LLMProfile(name="default", api_type="openai", base_url="http://x")
        settings = CHASettings()
        assert resolve_card_profile(settings, bare, 0) is None
        assert resolve_batch_profile(settings, bare) is None
