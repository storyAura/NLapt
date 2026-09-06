"""Persisted selection of the active translation provider + its credentials.

The GUI lets the user pick which translation service backs 翻译对照 and the
editor's inline 翻译/重译 actions: the LLM translator (default, preserving the
original behavior) or one of the built-in web providers (Google / Baidu /
DeepL / DeepLX / custom OpenAI-compatible / local Hy-MT2). Provider choice
lives in ``app_data_dir()/translate.json``. All channel credentials live in
``Documents/NLapt/api.json`` (see :mod:`nlapt_gui.api_config`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.llm.web_translate import (
    PROVIDER_BAIDU,
    PROVIDER_CUSTOM,
    PROVIDER_DEEPL,
    PROVIDER_DEEPLX,
    PROVIDER_GOOGLE,
    PROVIDER_LLM,
    PROVIDER_LOCAL_MT,
)
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.api_config import TranslateCredentials, load_api_config, update_api_config
from nlapt_gui.resources import app_data_dir

_LOGGER = get_logger(__name__)

CONFIG_FILE_NAME = "translate.json"
# Default preserves the original behavior: translate via the configured LLM.
DEFAULT_PROVIDER = PROVIDER_LLM
DEFAULT_LOCAL_MT_TIER = "balanced"
# All selectable provider ids, in display order (llm first = default).
KNOWN_PROVIDERS: tuple[str, ...] = (
    PROVIDER_LLM,
    PROVIDER_GOOGLE,
    PROVIDER_BAIDU,
    PROVIDER_DEEPL,
    PROVIDER_DEEPLX,
    PROVIDER_LOCAL_MT,
    PROVIDER_CUSTOM,
)
# Suggested 备选顺序 (DeepLX → Google → 百度 → 本地); primary is excluded.
SUGGESTED_FALLBACK_ORDER: tuple[str, ...] = (
    PROVIDER_GOOGLE,
    PROVIDER_BAIDU,
    PROVIDER_LOCAL_MT,
)
# Human-readable dropdown / list labels (UI stays Chinese).
PROVIDER_LABELS: dict[str, str] = {
    PROVIDER_LLM: "大模型 (LLM)",
    PROVIDER_GOOGLE: "Google 翻译 (免费)",
    PROVIDER_BAIDU: "百度翻译",
    PROVIDER_DEEPL: "DeepL",
    PROVIDER_DEEPLX: "DeepLX (免费)",
    PROVIDER_LOCAL_MT: "本地模型 (Hy-MT2)",
    PROVIDER_CUSTOM: "自定义 API",
}


@dataclass(frozen=True)
class TranslationConfig:
    """Immutable translation-provider selection + credentials."""

    provider: str = DEFAULT_PROVIDER
    baidu_appid: str = ""
    baidu_key: str = ""
    deepl_key: str = ""
    deeplx_url: str = ""
    deeplx_token: str = ""
    custom_base_url: str = ""
    custom_api_key: str = ""
    custom_model: str = ""
    local_mt_tier: str = DEFAULT_LOCAL_MT_TIER
    fallback_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fallback_order",
            normalize_fallback_order(self.provider, self.fallback_order),
        )

    def credentials(self) -> dict[str, str]:
        """Credential mapping consumed by ``web_translate.create_provider``."""
        return {
            "baidu_appid": self.baidu_appid,
            "baidu_key": self.baidu_key,
            "deepl_key": self.deepl_key,
            "deeplx_url": self.deeplx_url,
            "deeplx_token": self.deeplx_token,
            "custom_base_url": self.custom_base_url,
            "custom_api_key": self.custom_api_key,
            "custom_model": self.custom_model,
        }

    def with_changes(self, **changes: object) -> "TranslationConfig":
        """Return a copy with the given fields replaced (immutable update)."""
        return replace(self, **changes)


def normalize_fallback_order(
    provider: str, fallbacks: Sequence[str]
) -> tuple[str, ...]:
    """Drop the primary, unknowns, and duplicates; keep first-seen order."""
    seen = {provider}
    ordered: list[str] = []
    for item in fallbacks:
        if item not in KNOWN_PROVIDERS or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def provider_chain(config: TranslationConfig) -> tuple[str, ...]:
    """Preferred provider followed by configured fallbacks."""
    return (config.provider, *config.fallback_order)


def translation_config_path() -> Path:
    """Location of the persisted translation config (provider + fallback)."""
    return app_data_dir() / CONFIG_FILE_NAME


def load_translation_config(path: Path | None = None) -> TranslationConfig:
    """Load the config, falling back to defaults on missing/corrupt files."""
    target = path if path is not None else translation_config_path()
    provider = DEFAULT_PROVIDER
    local_mt_tier = DEFAULT_LOCAL_MT_TIER
    fallback_raw: object = None
    legacy_baidu_appid = ""
    legacy_baidu_key = ""
    legacy_deepl_key = ""
    legacy_deeplx_url = ""
    legacy_deeplx_token = ""
    if target.exists():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _LOGGER.warning("could not read %s (%s); using defaults", target, exc)
            raw = None
        if raw is None:
            pass
        elif not isinstance(raw, dict):
            _LOGGER.warning(
                "translate config %s is not an object; using defaults", target
            )
        else:
            candidate = raw.get("provider", DEFAULT_PROVIDER)
            if not isinstance(candidate, str) or candidate not in KNOWN_PROVIDERS:
                _LOGGER.warning(
                    "unknown provider %r in %s; using default", candidate, target
                )
            else:
                provider = candidate
            legacy_baidu_appid = str(raw.get("baidu_appid", ""))
            legacy_baidu_key = str(raw.get("baidu_key", ""))
            legacy_deepl_key = str(raw.get("deepl_key", ""))
            legacy_deeplx_url = str(raw.get("deeplx_url", ""))
            legacy_deeplx_token = str(raw.get("deeplx_token", ""))
            tier = str(raw.get("local_mt_tier", DEFAULT_LOCAL_MT_TIER)).strip()
            if tier:
                local_mt_tier = tier
            fallback_raw = raw.get("fallback_order")
    creds = load_api_config().translate
    return TranslationConfig(
        provider=provider,
        baidu_appid=creds.baidu_appid or legacy_baidu_appid,
        baidu_key=creds.baidu_key or legacy_baidu_key,
        deepl_key=creds.deepl_key or legacy_deepl_key,
        deeplx_url=creds.deeplx_url or legacy_deeplx_url,
        deeplx_token=creds.deeplx_token or legacy_deeplx_token,
        custom_base_url=creds.custom_base_url,
        custom_api_key=creds.custom_api_key,
        custom_model=creds.custom_model,
        local_mt_tier=local_mt_tier,
        fallback_order=_parse_fallback_order(fallback_raw, provider),
    )


def save_translation_config(
    config: TranslationConfig, path: Path | None = None
) -> None:
    """Persist provider choice in AppData; credentials in ``api.json``."""
    if not isinstance(config, TranslationConfig):
        raise StorageError(
            f"expected TranslationConfig, got {type(config).__name__}"
        )
    update_api_config(
        translate=TranslateCredentials(
            baidu_appid=config.baidu_appid,
            baidu_key=config.baidu_key,
            deepl_key=config.deepl_key,
            deeplx_url=config.deeplx_url,
            deeplx_token=config.deeplx_token,
            custom_base_url=config.custom_base_url,
            custom_api_key=config.custom_api_key,
            custom_model=config.custom_model,
        )
    )
    target = path if path is not None else translation_config_path()
    payload = {
        "fallback_order": list(config.fallback_order),
        "local_mt_tier": config.local_mt_tier,
        "provider": config.provider,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    atomic_write_text(target, text)
    _LOGGER.info("translation config saved (provider=%s)", config.provider)


def _parse_fallback_order(raw: object, provider: str) -> tuple[str, ...]:
    """Accept a JSON list of provider ids; anything else becomes empty."""
    if not isinstance(raw, list):
        return ()
    items = [item for item in raw if isinstance(item, str)]
    return normalize_fallback_order(provider, items)


