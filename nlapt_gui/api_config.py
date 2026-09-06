"""Unified API configuration in ``Documents/NLapt/api.json``.

LLM profiles, translation credentials, and the CHA own-API endpoint live
in one file. ``%APPDATA%/NLapt/config.json`` keeps non-interface AppConfig
fields; ``translate.json`` / ``cha_annotation.json`` keep provider choice
and CHA public options. ``NLAPT_DOCUMENTS_DIR`` isolates the Documents
root in tests.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from nlapt.core.config import (
    AppConfig,
    LLMProfile,
    load_config,
    profiles_from_list,
    save_config,
)
from nlapt.core.errors import StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text
from nlapt.storage.paths import WINDOWS_DIR_NAME, user_documents_app_dir, user_documents_dir

from nlapt_gui.resources import app_data_dir, config_path

_LOGGER = get_logger(__name__)

API_FILE_NAME = "api.json"
API_FORMAT_VERSION = 1
DEFAULT_CHA_API_TYPE = "openai"

# Legacy locations read during fallback / migrate (do not import those modules).
LEGACY_TRANSLATE_PUBLIC = "translate.json"
LEGACY_TRANSLATE_SECRET = "translate_api.json"
LEGACY_CHA_PUBLIC = "cha_annotation.json"
LEGACY_CHA_SECRET = "cha_api.json"


@dataclass(frozen=True)
class TranslateCredentials:
    """Translation-channel secrets and endpoints."""

    baidu_appid: str = ""
    baidu_key: str = ""
    deepl_key: str = ""
    deeplx_url: str = ""
    deeplx_token: str = ""
    custom_base_url: str = ""
    custom_api_key: str = ""
    custom_model: str = ""


@dataclass(frozen=True)
class CHAApi:
    """CHA标注 own-API endpoint (used when ``api_mode=own``)."""

    api_type: str = DEFAULT_CHA_API_TYPE
    base_url: str = ""
    api_key: str = ""


@dataclass(frozen=True)
class ApiConfig:
    """Immutable union of every interface the GUI can call."""

    profiles: tuple[LLMProfile, ...] = ()
    active_profile: str = ""
    translate: TranslateCredentials = TranslateCredentials()
    cha: CHAApi = CHAApi()

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", tuple(self.profiles))

    def with_changes(self, **changes: object) -> "ApiConfig":
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]


def api_config_path() -> Path:
    """``<Documents>/NLapt/api.json`` — does not create the directory."""
    return user_documents_dir() / WINDOWS_DIR_NAME / API_FILE_NAME


def load_api_config() -> ApiConfig:
    """Load ``api.json``; missing file falls back to legacy locations."""
    target = api_config_path()
    if not target.exists():
        return _legacy_api_config()
    raw = _read_json_object(target)
    if raw is None:
        _LOGGER.warning("api config %s missing or corrupt; using defaults", target)
        return ApiConfig()
    return _api_from_dict(raw)


def save_api_config(config: ApiConfig) -> None:
    """Persist ``api.json`` atomically. Raises StorageError on IO failure."""
    if not isinstance(config, ApiConfig):
        raise StorageError(f"expected ApiConfig, got {type(config).__name__}")
    user_documents_app_dir()
    payload = {
        "version": API_FORMAT_VERSION,
        "llm": {
            "active_profile": config.active_profile,
            "profiles": [dataclasses.asdict(profile) for profile in config.profiles],
        },
        "translate": dataclasses.asdict(config.translate),
        "cha": dataclasses.asdict(config.cha),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(api_config_path(), text)
    _LOGGER.info("api config saved (profiles=%s)", len(config.profiles))


def update_api_config(
    *,
    profiles: tuple[LLMProfile, ...] | None = None,
    active_profile: str | None = None,
    translate: TranslateCredentials | None = None,
    cha: CHAApi | None = None,
) -> ApiConfig:
    """Read-modify-write ``api.json``, replacing only the provided sections."""
    current = load_api_config()
    changes: dict[str, object] = {}
    if profiles is not None:
        changes["profiles"] = profiles
    if active_profile is not None:
        changes["active_profile"] = active_profile
    if translate is not None:
        changes["translate"] = translate
    if cha is not None:
        changes["cha"] = cha
    updated = current.with_changes(**changes) if changes else current
    save_api_config(updated)
    return updated


def load_app_config() -> AppConfig:
    """AppConfig from AppData, with profiles overlaid from ``api.json``."""
    base = load_config(config_path())
    api = load_api_config()
    return replace(base, profiles=api.profiles, active_profile=api.active_profile)


def save_app_config(config: AppConfig) -> None:
    """Write profiles to ``api.json``; write the rest to ``config.json``."""
    if not isinstance(config, AppConfig):
        raise StorageError(f"expected AppConfig, got {type(config).__name__}")
    update_api_config(profiles=config.profiles, active_profile=config.active_profile)
    stripped = replace(config, profiles=(), active_profile="")
    save_config(config_path(), stripped)


def migrate_legacy_api_files() -> bool:
    """Copy scattered secrets into ``api.json`` once, then strip leftovers.

    Returns True when a migration write happened. Failures are logged and
    never raised — startup must not block on this.
    """
    if api_config_path().exists():
        return False
    legacy = _legacy_api_config()
    if not _has_content(legacy):
        return False
    try:
        save_api_config(legacy)
        _strip_legacy_files()
    except (OSError, StorageError, ValidationError) as exc:
        _LOGGER.warning("could not migrate legacy API files: %s", exc)
        return False
    _LOGGER.info("migrated API settings into %s", api_config_path())
    return True


def _api_from_dict(raw: dict[str, Any]) -> ApiConfig:
    llm = raw.get("llm")
    llm_obj = llm if isinstance(llm, dict) else {}
    profiles_raw = llm_obj.get("profiles", [])
    try:
        profiles = profiles_from_list(profiles_raw) if isinstance(profiles_raw, list) else ()
    except ValidationError:
        _LOGGER.warning("api.json llm.profiles is invalid; using empty list")
        profiles = ()
    active = llm_obj.get("active_profile", "")
    translate_raw = raw.get("translate")
    cha_raw = raw.get("cha")
    return ApiConfig(
        profiles=profiles,
        active_profile=active if isinstance(active, str) else "",
        translate=_translate_from_dict(translate_raw if isinstance(translate_raw, dict) else {}),
        cha=_cha_from_dict(cha_raw if isinstance(cha_raw, dict) else {}),
    )


def _translate_from_dict(raw: dict[str, Any]) -> TranslateCredentials:
    return TranslateCredentials(
        baidu_appid=str(raw.get("baidu_appid", "")),
        baidu_key=str(raw.get("baidu_key", "")),
        deepl_key=str(raw.get("deepl_key", "")),
        deeplx_url=str(raw.get("deeplx_url", "")),
        deeplx_token=str(raw.get("deeplx_token", "")),
        custom_base_url=str(raw.get("custom_base_url", raw.get("base_url", ""))),
        custom_api_key=str(raw.get("custom_api_key", raw.get("api_key", ""))),
        custom_model=str(raw.get("custom_model", raw.get("model", ""))),
    )


def _cha_from_dict(raw: dict[str, Any]) -> CHAApi:
    api_type = str(raw.get("api_type", "")).strip() or DEFAULT_CHA_API_TYPE
    return CHAApi(
        api_type=api_type,
        base_url=str(raw.get("base_url", "")).strip(),
        api_key=str(raw.get("api_key", "")).strip(),
    )


def _legacy_api_config() -> ApiConfig:
    """Assemble an ApiConfig from the pre-unification files (read-only)."""
    profiles: tuple[LLMProfile, ...] = ()
    active = ""
    try:
        cfg = load_config(config_path())
        profiles = cfg.profiles
        active = cfg.active_profile
    except (StorageError, ValidationError) as exc:
        _LOGGER.warning("could not read legacy config.json: %s", exc)

    public = _read_json_object(app_data_dir() / LEGACY_TRANSLATE_PUBLIC) or {}
    secret = _read_json_object(_documents_file(LEGACY_TRANSLATE_SECRET)) or {}
    translate = TranslateCredentials(
        baidu_appid=str(public.get("baidu_appid", "")),
        baidu_key=str(public.get("baidu_key", "")),
        deepl_key=str(public.get("deepl_key", "")),
        deeplx_url=str(secret.get("deeplx_url", "") or public.get("deeplx_url", "")),
        deeplx_token=str(secret.get("deeplx_token", "") or public.get("deeplx_token", "")),
        custom_base_url=str(secret.get("base_url", "")),
        custom_api_key=str(secret.get("api_key", "")),
        custom_model=str(secret.get("model", "")),
    )

    cha_pub = _read_json_object(app_data_dir() / LEGACY_CHA_PUBLIC) or {}
    cha_sec = _read_json_object(_documents_file(LEGACY_CHA_SECRET)) or {}
    cha = CHAApi(
        api_type=str(cha_pub.get("api_type", "")).strip() or DEFAULT_CHA_API_TYPE,
        base_url=str(cha_pub.get("base_url", "")).strip(),
        api_key=str(cha_sec.get("api_key", "")).strip(),
    )
    return ApiConfig(
        profiles=profiles, active_profile=active, translate=translate, cha=cha
    )


def _has_content(config: ApiConfig) -> bool:
    if config.profiles or config.active_profile:
        return True
    if any(dataclasses.asdict(config.translate).values()):
        return True
    cha = config.cha
    return bool(cha.base_url or cha.api_key or (cha.api_type and cha.api_type != DEFAULT_CHA_API_TYPE))


def _strip_legacy_files() -> None:
    """Rewrite AppData files without secrets; delete old Documents secrets."""
    try:
        cfg = load_config(config_path())
    except (StorageError, ValidationError):
        cfg = None
    if cfg is not None and (cfg.profiles or cfg.active_profile):
        save_config(config_path(), replace(cfg, profiles=(), active_profile=""))

    translate_path = app_data_dir() / LEGACY_TRANSLATE_PUBLIC
    public = _read_json_object(translate_path)
    if public is not None:
        cleaned = {
            "fallback_order": public.get("fallback_order", []),
            "local_mt_tier": public.get("local_mt_tier", ""),
            "provider": public.get("provider", ""),
        }
        atomic_write_text(
            translate_path,
            json.dumps(cleaned, ensure_ascii=False, indent=2, sort_keys=True),
        )

    cha_path = app_data_dir() / LEGACY_CHA_PUBLIC
    cha_pub = _read_json_object(cha_path)
    if cha_pub is not None:
        cleaned_cha = {
            "api_mode": cha_pub.get("api_mode", "sync"),
            "batch_model": cha_pub.get("batch_model", ""),
            "card_models": cha_pub.get("card_models", ["", "", ""]),
        }
        atomic_write_text(
            cha_path,
            json.dumps(cleaned_cha, ensure_ascii=False, indent=2, sort_keys=True),
        )

    for name in (LEGACY_TRANSLATE_SECRET, LEGACY_CHA_SECRET):
        leftover = _documents_file(name)
        if leftover.exists():
            try:
                leftover.unlink()
            except OSError as exc:
                _LOGGER.warning("could not remove legacy %s: %s", leftover, exc)


def _documents_file(name: str) -> Path:
    return user_documents_dir() / WINDOWS_DIR_NAME / name


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("could not read %s (%s)", path, exc)
        return None
    if not isinstance(raw, dict):
        _LOGGER.warning("%s is not a JSON object", path)
        return None
    return raw
