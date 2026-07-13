"""Persisted selection of the active translation provider + its credentials.

The GUI lets the user pick which translation service backs 翻译对照 and the
editor's inline 翻译/重译 actions: the LLM translator (default, preserving the
original behavior) or one of the built-in web providers (Google / Baidu /
DeepL). That choice plus the provider credentials live in a small JSON file
under ``app_data_dir()/translate.json`` so it survives restarts and is written
atomically.

The config is intentionally separate from the core ``config.json`` (LLM
profiles): the web-provider keys are GUI-only concerns and must not disturb the
core config schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.llm.web_translate import (
    PROVIDER_BAIDU,
    PROVIDER_DEEPL,
    PROVIDER_GOOGLE,
    PROVIDER_LLM,
)
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.resources import app_data_dir

_LOGGER = get_logger(__name__)

CONFIG_FILE_NAME = "translate.json"
# Default preserves the original behavior: translate via the configured LLM.
DEFAULT_PROVIDER = PROVIDER_LLM
# All selectable provider ids, in display order (llm first = default).
KNOWN_PROVIDERS: tuple[str, ...] = (
    PROVIDER_LLM,
    PROVIDER_GOOGLE,
    PROVIDER_BAIDU,
    PROVIDER_DEEPL,
)


@dataclass(frozen=True)
class TranslationConfig:
    """Immutable translation-provider selection + credentials."""

    provider: str = DEFAULT_PROVIDER
    baidu_appid: str = ""
    baidu_key: str = ""
    deepl_key: str = ""

    def credentials(self) -> dict[str, str]:
        """Credential mapping consumed by ``web_translate.create_provider``."""
        return {
            "baidu_appid": self.baidu_appid,
            "baidu_key": self.baidu_key,
            "deepl_key": self.deepl_key,
        }

    def with_changes(self, **changes: str) -> "TranslationConfig":
        """Return a copy with the given fields replaced (immutable update)."""
        return replace(self, **changes)


def translation_config_path() -> Path:
    """Location of the persisted translation config."""
    return app_data_dir() / CONFIG_FILE_NAME


def load_translation_config(path: Path | None = None) -> TranslationConfig:
    """Load the config, falling back to defaults on missing/corrupt files."""
    target = path if path is not None else translation_config_path()
    if not target.exists():
        return TranslationConfig()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("could not read %s (%s); using defaults", target, exc)
        return TranslationConfig()
    if not isinstance(raw, dict):
        _LOGGER.warning("translate config %s is not an object; using defaults", target)
        return TranslationConfig()
    provider = raw.get("provider", DEFAULT_PROVIDER)
    if not isinstance(provider, str) or provider not in KNOWN_PROVIDERS:
        _LOGGER.warning("unknown provider %r in %s; using default", provider, target)
        provider = DEFAULT_PROVIDER
    return TranslationConfig(
        provider=provider,
        baidu_appid=str(raw.get("baidu_appid", "")),
        baidu_key=str(raw.get("baidu_key", "")),
        deepl_key=str(raw.get("deepl_key", "")),
    )


def save_translation_config(
    config: TranslationConfig, path: Path | None = None
) -> None:
    """Persist the config atomically as UTF-8 JSON. Raises StorageError."""
    if not isinstance(config, TranslationConfig):
        raise StorageError(
            f"expected TranslationConfig, got {type(config).__name__}"
        )
    target = path if path is not None else translation_config_path()
    text = json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True)
    atomic_write_text(target, text)
    _LOGGER.info("translation config saved (provider=%s)", config.provider)
