"""Tests for nlapt_gui.model_targets (pool view records + target switching)."""

from __future__ import annotations

import pytest

from nlapt.core.config import AppConfig, LLMProfile, ModelRef
from nlapt.core.errors import ValidationError

from nlapt_gui.api_config import load_api_config
from nlapt_gui.model_targets import (
    LABEL_ROLE_TEXT,
    LABEL_ROLE_VISION,
    LABEL_STALE,
    LABEL_UNSET,
    ROLE_TEXT,
    ROLE_VISION,
    choice_label,
    current_target,
    find_choice,
    grouped_pool_choices,
    persist_targets,
    pool_choices,
    role_label,
    switch_target,
    target_display,
)


def _config() -> AppConfig:
    alpha = LLMProfile(
        name="alpha",
        api_type="openai",
        base_url="http://a",
        models=("gpt-plain", "gpt-4o", "off"),
        enabled_models=("gpt-plain", "gpt-4o"),
    )
    beta = LLMProfile(
        name="beta",
        api_type="ollama",
        base_url="http://b",
        models=("llava",),
        enabled_models=("llava",),
    )
    return AppConfig(
        profiles=(alpha, beta),
        active_profile="alpha",
        text_target=ModelRef("alpha", "gpt-plain"),
        vision_target=ModelRef("beta", "llava"),
    )


class TestPoolChoices:
    def test_choices_follow_pool_order_with_labels_and_vision_hint(self) -> None:
        choices = pool_choices(_config())
        assert [c.ref for c in choices] == [
            ModelRef("alpha", "gpt-plain"),
            ModelRef("alpha", "gpt-4o"),
            ModelRef("beta", "llava"),
        ]
        assert choices[0].label == "alpha · gpt-plain"
        assert choices[0].vision_hint is False
        assert choices[1].vision_hint is True
        assert choices[2].vision_hint is True
        assert pool_choices(AppConfig()) == ()

    def test_grouped_choices_follow_profiles_and_skip_empty(self) -> None:
        cfg = _config()
        silent = LLMProfile(name="silent", api_type="openai", base_url="http://s", models=("x",))
        cfg = AppConfig(profiles=(*cfg.profiles, silent))
        groups = grouped_pool_choices(cfg)
        assert [name for name, _ in groups] == ["alpha", "beta"]
        assert [c.ref for c in groups[0][1]] == [
            ModelRef("alpha", "gpt-plain"),
            ModelRef("alpha", "gpt-4o"),
        ]
        assert groups[1][1][0].label == "beta · llava"
        assert grouped_pool_choices(AppConfig()) == ()

    def test_find_choice_and_labels(self) -> None:
        choices = pool_choices(_config())
        assert find_choice(choices, ModelRef("beta", "llava")) is choices[2]
        assert find_choice(choices, ModelRef("beta", "nope")) is None
        assert choice_label(ModelRef()) == LABEL_UNSET
        assert choice_label(ModelRef("p", "m")) == "p · m"
        assert role_label(ROLE_TEXT) == LABEL_ROLE_TEXT
        assert role_label(ROLE_VISION) == LABEL_ROLE_VISION
        with pytest.raises(ValidationError):
            role_label("audio")

    def test_target_display_states(self) -> None:
        cfg = _config()
        assert current_target(cfg, ROLE_TEXT) == ModelRef("alpha", "gpt-plain")
        assert target_display(cfg, ROLE_TEXT) == "gpt-plain"
        assert target_display(cfg, ROLE_VISION) == "llava"
        unset = switch_target(cfg, ROLE_VISION, ModelRef())
        assert target_display(unset, ROLE_VISION) == LABEL_UNSET
        stale = AppConfig(profiles=cfg.profiles, text_target=ModelRef("alpha", "off"))
        assert target_display(stale, ROLE_TEXT) == LABEL_STALE


class TestSwitchTarget:
    def test_switch_sets_role_and_clears_legacy_active(self) -> None:
        cfg = _config()
        switched = switch_target(cfg, ROLE_TEXT, ModelRef("alpha", "gpt-4o"))
        assert switched.text_target == ModelRef("alpha", "gpt-4o")
        assert switched.vision_target == cfg.vision_target
        assert switched.active_profile == ""
        assert cfg.text_target == ModelRef("alpha", "gpt-plain")  # input untouched

    def test_switch_rejects_stale_refs_and_bad_roles(self) -> None:
        cfg = _config()
        with pytest.raises(ValidationError):
            switch_target(cfg, ROLE_TEXT, ModelRef("alpha", "off"))
        with pytest.raises(ValidationError):
            switch_target(cfg, ROLE_TEXT, ModelRef("ghost", "gpt-4o"))
        with pytest.raises(ValidationError):
            switch_target(cfg, "audio", ModelRef("alpha", "gpt-4o"))
        with pytest.raises(ValidationError):
            switch_target(cfg, ROLE_TEXT, "alpha/gpt-4o")  # type: ignore[arg-type]

    def test_persist_targets_writes_api_json(self) -> None:
        cfg = switch_target(_config(), ROLE_VISION, ModelRef("alpha", "gpt-4o"))
        persist_targets(cfg)
        saved = load_api_config()
        assert saved.vision_target == ModelRef("alpha", "gpt-4o")
        assert saved.text_target == ModelRef("alpha", "gpt-plain")
        assert [p.name for p in saved.profiles] == ["alpha", "beta"]
        assert saved.active_profile == ""
