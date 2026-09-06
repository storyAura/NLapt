"""Tests for nlapt_gui.api_config (unified Documents/NLapt/api.json)."""

from __future__ import annotations

import json

import pytest

from nlapt.core.config import AppConfig, LLMProfile, RequestControl, load_config, save_config
from nlapt.core.errors import StorageError

from nlapt_gui.api_config import (
    API_FILE_NAME,
    CHAApi,
    ApiConfig,
    TranslateCredentials,
    api_config_path,
    load_api_config,
    load_app_config,
    migrate_legacy_api_files,
    save_api_config,
    save_app_config,
    update_api_config,
)
from nlapt_gui.resources import app_data_dir, config_path
from nlapt_gui.translate_config import translation_config_path


def _profile(name: str = "default", key: str = "sk-main") -> LLMProfile:
    return LLMProfile(
        name=name,
        api_type="openai",
        base_url="http://llm.local",
        api_key=key,
        text_model="text-1",
        vision_model="vision-1",
    )


class TestRoundTrip:
    def test_save_then_load(self) -> None:
        original = ApiConfig(
            profiles=(_profile(),),
            active_profile="default",
            translate=TranslateCredentials(baidu_appid="app", baidu_key="bk"),
            cha=CHAApi(base_url="http://cha.local", api_key="sk-cha"),
        )
        save_api_config(original)
        assert api_config_path().name == API_FILE_NAME
        assert load_api_config() == original

    def test_update_replaces_only_one_section(self) -> None:
        save_api_config(
            ApiConfig(
                profiles=(_profile(),),
                active_profile="default",
                translate=TranslateCredentials(deepl_key="keep-me"),
            )
        )
        update_api_config(cha=CHAApi(base_url="http://new", api_key="sk-new"))
        loaded = load_api_config()
        assert loaded.profiles[0].api_key == "sk-main"
        assert loaded.translate.deepl_key == "keep-me"
        assert loaded.cha.base_url == "http://new"
        assert loaded.cha.api_key == "sk-new"

    def test_corrupt_file_returns_defaults(self) -> None:
        target = api_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{not json", encoding="utf-8")
        assert load_api_config() == ApiConfig()

    def test_save_rejects_wrong_type(self) -> None:
        with pytest.raises(StorageError):
            save_api_config("nope")  # type: ignore[arg-type]


class TestAppConfigSplit:
    def test_save_app_config_strips_profiles_from_appdata(self) -> None:
        config = AppConfig(
            profiles=(_profile(),),
            active_profile="default",
            request=RequestControl(concurrency=8),
            snapshot_retention=7,
        )
        save_app_config(config)
        loaded = load_app_config()
        assert loaded.profiles[0].api_key == "sk-main"
        assert loaded.active_profile == "default"
        assert loaded.request.concurrency == 8
        assert loaded.snapshot_retention == 7
        on_disk = load_config(config_path())
        assert on_disk.profiles == ()
        assert on_disk.active_profile == ""
        assert "sk-main" not in config_path().read_text(encoding="utf-8")
        assert "sk-main" in api_config_path().read_text(encoding="utf-8")

    def test_missing_api_json_falls_back_to_legacy_config(self) -> None:
        save_config(
            config_path(),
            AppConfig(profiles=(_profile("alt", "sk-legacy"),), active_profile="alt"),
        )
        assert not api_config_path().exists()
        loaded = load_api_config()
        assert loaded.active_profile == "alt"
        assert loaded.profiles[0].api_key == "sk-legacy"


class TestMigrate:
    def test_merges_scattered_files_then_is_idempotent(self) -> None:
        save_config(
            config_path(),
            AppConfig(profiles=(_profile(),), active_profile="default"),
        )
        translation_config_path().write_text(
            json.dumps(
                {
                    "provider": "baidu",
                    "baidu_appid": "app-1",
                    "baidu_key": "secret-baidu",
                    "deepl_key": "dk",
                }
            ),
            encoding="utf-8",
        )
        docs = api_config_path().parent
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "translate_api.json").write_text(
            json.dumps(
                {
                    "base_url": "https://custom.example/v1",
                    "api_key": "sk-custom",
                    "model": "mini",
                    "deeplx_url": "http://dx",
                    "deeplx_token": "tok",
                }
            ),
            encoding="utf-8",
        )
        (app_data_dir() / "cha_annotation.json").write_text(
            json.dumps(
                {
                    "api_mode": "own",
                    "api_type": "ollama",
                    "base_url": "http://cha.local",
                    "card_models": ["a", "b", "c"],
                    "batch_model": "batch",
                }
            ),
            encoding="utf-8",
        )
        (docs / "cha_api.json").write_text(
            json.dumps({"api_key": "sk-cha"}), encoding="utf-8"
        )

        assert migrate_legacy_api_files() is True
        loaded = load_api_config()
        assert loaded.profiles[0].api_key == "sk-main"
        assert loaded.translate.baidu_key == "secret-baidu"
        assert loaded.translate.custom_api_key == "sk-custom"
        assert loaded.translate.deeplx_url == "http://dx"
        assert loaded.cha.api_key == "sk-cha"
        assert loaded.cha.base_url == "http://cha.local"

        appdata_cfg = config_path().read_text(encoding="utf-8")
        assert "sk-main" not in appdata_cfg
        translate_text = translation_config_path().read_text(encoding="utf-8")
        assert "secret-baidu" not in translate_text
        cha_text = (app_data_dir() / "cha_annotation.json").read_text(encoding="utf-8")
        assert "http://cha.local" not in cha_text
        assert not (docs / "translate_api.json").exists()
        assert not (docs / "cha_api.json").exists()

        assert migrate_legacy_api_files() is False
        assert load_api_config().translate.baidu_key == "secret-baidu"

    def test_no_legacy_content_is_noop(self) -> None:
        assert migrate_legacy_api_files() is False
        assert not api_config_path().exists()
