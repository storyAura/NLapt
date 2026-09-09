"""Tests for nlapt.core.config (models, persistence, masking)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from nlapt.core.config import (
    ROLE_TEXT,
    ROLE_VISION,
    AppConfig,
    LLMProfile,
    ModelRef,
    RequestControl,
    bind_model_ref,
    enabled_model_refs,
    find_profile,
    get_active_profile,
    load_config,
    mask_secret,
    masked_config_dict,
    profile_from_dict,
    profiles_from_list,
    resolve_model_ref,
    resolve_text_profile,
    resolve_vision_profile,
    save_config,
    upgrade_legacy_targets,
)
from nlapt.core.errors import StorageError, ValidationError

SECRET = "sk-secret123456789"


def _profile(name: str = "main", api_type: str = "openai") -> LLMProfile:
    return LLMProfile(
        name=name,
        api_type=api_type,
        base_url="http://localhost:9999/v1",
        api_key=SECRET,
        text_model="gpt-x",
        vision_model="gpt-x-vision",
        temperature=0.5,
        max_tokens=512,
        system_prompt="you caption images",
    )


def _config() -> AppConfig:
    return AppConfig(
        profiles=(_profile(),),
        active_profile="main",
        request=RequestControl(concurrency=2, min_interval=0.1, timeout=30.0, max_retries=1),
        image_max_edge=768,
        revert_confirmed_on_edit=False,
        snapshot_retention=5,
        prompt_orphan_cleanup=True,
        custom_templates={"mystyle": "rewrite {caption}"},
        trigger_presets=("minahamu",),
    )


# ---- dataclasses -----------------------------------------------------------


def test_config_dataclasses_are_frozen() -> None:
    cfg = _config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.active_profile = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.profiles[0].api_key = "y"  # type: ignore[misc]


def test_app_config_defends_against_shared_mutable_input() -> None:
    templates = {"a": "b"}
    cfg = AppConfig(custom_templates=templates, profiles=[], trigger_presets=["t"])
    templates["a"] = "changed"
    assert cfg.custom_templates["a"] == "b"
    assert cfg.profiles == ()
    assert cfg.trigger_presets == ("t",)


def test_profile_from_dict_and_profiles_from_list() -> None:
    parsed = profile_from_dict(
        {
            "name": "p",
            "api_type": "openai",
            "base_url": "http://x",
            "api_key": "k",
        },
        0,
    )
    assert parsed.name == "p"
    assert parsed.api_key == "k"
    batch = profiles_from_list(
        [{"name": "a", "api_type": "openai", "base_url": "u"}]
    )
    assert len(batch) == 1
    assert batch[0].name == "a"
    with pytest.raises(ValidationError):
        profiles_from_list("nope")


def test_defaults_match_contract() -> None:
    cfg = AppConfig()
    assert cfg.request == RequestControl()
    assert cfg.request.concurrency == 4
    assert cfg.request.max_retries == 2
    assert cfg.image_max_edge == 1024
    assert cfg.revert_confirmed_on_edit is True
    assert cfg.snapshot_retention == 20
    assert cfg.prompt_orphan_cleanup is False


# ---- load / save -----------------------------------------------------------


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "absent.json") == AppConfig()


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    original = _config()
    save_config(path, original)
    assert load_config(path) == original
    # saved file is valid UTF-8 JSON without BOM
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert json.loads(raw.decode("utf-8"))["active_profile"] == "main"


def test_load_corrupt_json_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{not json!", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_non_object_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('["a", "b"]', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_wrong_field_types_raise(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"profiles": "nope"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)
    path.write_text(
        json.dumps({"profiles": [{"name": "p", "api_type": "openai", "base_url": "u",
                                  "max_tokens": True}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_unknown_api_type_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"profiles": [{"name": "p", "api_type": "weird-api", "base_url": "u"}],
                    "active_profile": "p"}),
        encoding="utf-8",
    )
    cfg = load_config(path)  # must not raise at config layer
    assert cfg.profiles[0].api_type == "weird-api"


def test_load_ignores_unknown_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"future_option": 42}), encoding="utf-8")
    assert load_config(path) == AppConfig()


def test_save_to_missing_directory_raises_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        save_config(tmp_path / "no" / "dir" / "config.json", AppConfig())


# ---- active profile --------------------------------------------------------


def test_get_active_profile() -> None:
    cfg = _config()
    active = get_active_profile(cfg)
    assert active is not None and active.name == "main"
    assert get_active_profile(AppConfig()) is None
    assert get_active_profile(dataclasses.replace(cfg, active_profile="ghost")) is None


# ---- model pool --------------------------------------------------------------


def _pool_config() -> AppConfig:
    alpha = LLMProfile(
        name="alpha",
        api_type="openai",
        base_url="http://a/v1",
        models=("a-text", "a-vision", "a-off"),
        enabled_models=("a-text", "a-vision"),
    )
    beta = LLMProfile(
        name="beta",
        api_type="anthropic",
        base_url="http://b",
        models=("b-vision",),
        enabled_models=("b-vision",),
    )
    return AppConfig(
        profiles=(alpha, beta),
        text_target=ModelRef("alpha", "a-text"),
        vision_target=ModelRef("beta", "b-vision"),
    )


def test_profile_lists_are_tuples_and_parse_round_trip(tmp_path: Path) -> None:
    profile = LLMProfile(
        name="p", api_type="openai", base_url="u", models=["m1"], enabled_models=["m1"]
    )
    assert profile.models == ("m1",)
    assert profile.enabled_models == ("m1",)
    path = tmp_path / "config.json"
    save_config(path, _pool_config())
    loaded = load_config(path)
    assert loaded == _pool_config()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["text_target"] == {"profile": "alpha", "model": "a-text"}
    assert raw["profiles"][0]["enabled_models"] == ["a-text", "a-vision"]


def test_parse_rejects_bad_model_lists_and_targets(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        profile_from_dict({"name": "p", "api_type": "openai", "base_url": "u", "models": "m"})
    with pytest.raises(ValidationError):
        profile_from_dict(
            {"name": "p", "api_type": "openai", "base_url": "u", "enabled_models": [1]}
        )
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_target": "alpha/a"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_model_ref_is_set() -> None:
    assert not ModelRef().is_set()
    assert not ModelRef(profile="a").is_set()
    assert ModelRef("a", "m").is_set()


def test_enabled_model_refs_keeps_profile_then_model_order() -> None:
    assert enabled_model_refs(_pool_config()) == (
        ModelRef("alpha", "a-text"),
        ModelRef("alpha", "a-vision"),
        ModelRef("beta", "b-vision"),
    )
    assert enabled_model_refs(AppConfig()) == ()


def test_find_profile_and_resolve_model_ref() -> None:
    cfg = _pool_config()
    assert find_profile(cfg, "beta") is cfg.profiles[1]
    assert find_profile(cfg, "") is None
    assert find_profile(cfg, "ghost") is None
    assert resolve_model_ref(cfg, ModelRef("alpha", "a-text")) is cfg.profiles[0]
    assert resolve_model_ref(cfg, ModelRef()) is None
    assert resolve_model_ref(cfg, ModelRef("ghost", "a-text")) is None
    # Known on the endpoint but switched off -> stale, not silently used.
    assert resolve_model_ref(cfg, ModelRef("alpha", "a-off")) is None


def test_bind_model_ref_binds_role_field_and_rejects_stale() -> None:
    cfg = _pool_config()
    vision = bind_model_ref(cfg, ModelRef("alpha", "a-vision"), ROLE_VISION)
    assert vision is not None and vision.vision_model == "a-vision"
    assert vision.text_model == ""  # only the requested role is bound
    text = bind_model_ref(cfg, ModelRef("beta", "b-vision"), ROLE_TEXT)
    assert text is not None and text.text_model == "b-vision" and text.base_url == "http://b"
    assert bind_model_ref(cfg, ModelRef("alpha", "a-off"), ROLE_VISION) is None
    assert bind_model_ref(cfg, ModelRef(), ROLE_VISION) is None
    with pytest.raises(ValidationError):
        bind_model_ref(cfg, ModelRef("alpha", "a-text"), "audio")


def test_resolve_role_profiles_bind_the_chosen_model() -> None:
    cfg = _pool_config()
    text = resolve_text_profile(cfg)
    vision = resolve_vision_profile(cfg)
    assert text is not None and text.name == "alpha" and text.text_model == "a-text"
    assert vision is not None and vision.name == "beta" and vision.vision_model == "b-vision"
    assert vision.base_url == "http://b"
    stale = dataclasses.replace(cfg, vision_target=ModelRef("alpha", "a-off"))
    assert resolve_vision_profile(stale) is None
    assert resolve_text_profile(dataclasses.replace(cfg, text_target=ModelRef())) is None
    with pytest.raises(ValidationError):
        resolve_text_profile("nope")  # type: ignore[arg-type]


def test_resolve_role_falls_back_to_legacy_active_profile() -> None:
    legacy = _config()  # active_profile="main", no targets, no enabled_models
    text = resolve_text_profile(legacy)
    vision = resolve_vision_profile(legacy)
    assert text is not None and text.text_model == "gpt-x"
    assert vision is not None and vision.vision_model == "gpt-x-vision"
    no_vision = AppConfig(
        profiles=(dataclasses.replace(_profile(), vision_model=""),), active_profile="main"
    )
    assert resolve_text_profile(no_vision) is not None
    assert resolve_vision_profile(no_vision) is None
    assert resolve_text_profile(AppConfig()) is None


def test_upgrade_legacy_targets_derives_pool_state_and_is_idempotent() -> None:
    legacy = _config()
    upgraded = upgrade_legacy_targets(legacy)
    profile = upgraded.profiles[0]
    assert profile.enabled_models == ("gpt-x", "gpt-x-vision")
    assert profile.models == ("gpt-x", "gpt-x-vision")
    assert upgraded.text_target == ModelRef("main", "gpt-x")
    assert upgraded.vision_target == ModelRef("main", "gpt-x-vision")
    assert upgrade_legacy_targets(upgraded) is upgraded
    # Already-pooled configs are returned untouched.
    pooled = _pool_config()
    assert upgrade_legacy_targets(pooled) is pooled
    # A profile that explicitly enabled nothing but whose default model is
    # missing from the catalog gets the catalog completed, nothing else.
    partial = AppConfig(
        profiles=(
            LLMProfile(
                name="p", api_type="openai", base_url="u", enabled_models=("m",), models=()
            ),
        )
    )
    assert upgrade_legacy_targets(partial).profiles[0].models == ("m",)
    # A catalog with everything switched off stays off (deliberate user choice).
    all_off = AppConfig(
        profiles=(
            LLMProfile(
                name="p", api_type="openai", base_url="u", text_model="m", models=("m",)
            ),
        )
    )
    assert upgrade_legacy_targets(all_off) is all_off
    with pytest.raises(ValidationError):
        upgrade_legacy_targets(None)  # type: ignore[arg-type]


def test_upgrade_legacy_targets_keeps_explicit_targets() -> None:
    cfg = dataclasses.replace(
        _config(), text_target=ModelRef("main", "gpt-x-vision")
    )
    upgraded = upgrade_legacy_targets(cfg)
    assert upgraded.text_target == ModelRef("main", "gpt-x-vision")
    assert upgraded.vision_target == ModelRef("main", "gpt-x-vision")


# ---- masking ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ""),
        ("a", "***"),
        ("short", "***"),
        ("abcdefg", "***"),            # 7 chars: fully masked
        ("abcdefgh", "***"),           # 8 chars: partial reveal would leak 7/8
        ("abcdefghi", "***"),          # 9 chars: fully masked
        ("abcdefghij", "***"),         # 10 chars: fully masked
        ("abcdefghijk", "***"),        # 11 chars: fully masked
        ("abcdefghijkl", "abc***ijkl"),  # 12 chars: minimum partial-reveal length
        (SECRET, "sk-***6789"),
    ],
)
def test_mask_secret(value: str, expected: str) -> None:
    assert mask_secret(value) == expected


def test_mask_secret_hides_at_least_half_of_short_secrets() -> None:
    # Boundary guarantee: below the partial-reveal threshold the mask reveals
    # NOTHING, so at least half of any such secret always remains hidden.
    for length in range(1, 12):
        assert mask_secret("x" * length) == "***"


def test_mask_secret_rejects_non_string() -> None:
    with pytest.raises(ValidationError):
        mask_secret(None)  # type: ignore[arg-type]


def test_masked_config_dict_never_contains_raw_key() -> None:
    data = masked_config_dict(_config())
    dumped = json.dumps(data)
    assert SECRET not in dumped
    assert data["profiles"][0]["api_key"] == "sk-***6789"
    # everything else survives untouched
    assert data["profiles"][0]["text_model"] == "gpt-x"
    assert data["custom_templates"] == {"mystyle": "rewrite {caption}"}
