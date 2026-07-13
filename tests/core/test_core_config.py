"""Tests for nlapt.core.config (models, persistence, masking)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from nlapt.core.config import (
    AppConfig,
    LLMProfile,
    RequestControl,
    get_active_profile,
    load_config,
    mask_secret,
    masked_config_dict,
    save_config,
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
