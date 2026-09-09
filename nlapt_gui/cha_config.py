"""Persisted CHA标注 (组合分层推标) API + per-scheme model settings.

The wizard defaults to the main LLM profile (``api_mode=sync``). Own-API
endpoint fields live in ``Documents/NLapt/api.json``; ``api_mode`` and
per-scheme models stay in ``app_data_dir()/cha_annotation.json``.

Per-scheme models are :class:`ModelRef`s with three meanings:

* ``ModelRef("p", "m")`` — model ``m`` of pool profile ``p`` (any API);
* ``ModelRef("", "m")`` — a typed id used on the CHA base endpoint;
* ``ModelRef()`` — 留空: the base endpoint's 视觉模型.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from nlapt.core.config import LLMProfile, ModelRef
from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.api_config import CHAApi, load_api_config, update_api_config
from nlapt_gui.model_targets import model_ref_from_value
from nlapt_gui.resources import app_data_dir

_LOGGER = get_logger(__name__)

CONFIG_FILE_NAME = "cha_annotation.json"
PROFILE_NAME = "cha"
API_MODE_SYNC = "sync"
API_MODE_OWN = "own"
API_MODES: tuple[str, ...] = (API_MODE_SYNC, API_MODE_OWN)
DEFAULT_API_TYPE = "openai"
CARD_SLOTS = 3
EMPTY_CARD_MODELS: tuple[ModelRef, ...] = (ModelRef(), ModelRef(), ModelRef())

ProfileLookup = Callable[[ModelRef], LLMProfile | None]


@dataclass(frozen=True)
class CHASettings:
    """Immutable CHA标注 API mode + per-scheme vision models.

    ``card_models`` / ``batch_model`` accept bare strings for convenience
    (legacy files, tests) and are normalised to :class:`ModelRef`.
    """

    api_mode: str = API_MODE_SYNC
    api_type: str = DEFAULT_API_TYPE
    base_url: str = ""
    api_key: str = ""
    card_models: tuple[ModelRef, ...] = EMPTY_CARD_MODELS
    batch_model: ModelRef = ModelRef()

    def __post_init__(self) -> None:
        mode = self.api_mode if self.api_mode in API_MODES else API_MODE_SYNC
        models = _normalize_card_models(self.card_models)
        object.__setattr__(self, "api_mode", mode)
        object.__setattr__(self, "api_type", self.api_type.strip() or DEFAULT_API_TYPE)
        object.__setattr__(self, "base_url", self.base_url.strip())
        object.__setattr__(self, "api_key", self.api_key.strip())
        object.__setattr__(self, "card_models", models)
        object.__setattr__(self, "batch_model", model_ref_from_value(self.batch_model))

    def with_changes(self, **changes: object) -> "CHASettings":
        """Return a copy with the given fields replaced (immutable update)."""
        return replace(self, **changes)


def cha_settings_path() -> Path:
    """Location of the persisted CHA标注 public settings (no endpoint)."""
    return app_data_dir() / CONFIG_FILE_NAME


def load_cha_settings(path: Path | None = None) -> CHASettings:
    """Load settings, falling back to defaults on missing/corrupt files."""
    target = path if path is not None else cha_settings_path()
    api_mode = API_MODE_SYNC
    legacy_api_type = DEFAULT_API_TYPE
    legacy_base_url = ""
    card_models: tuple[ModelRef, ...] = EMPTY_CARD_MODELS
    batch_model = ModelRef()
    if target.exists():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _LOGGER.warning("could not read %s (%s); using defaults", target, exc)
            raw = None
        if raw is None:
            pass
        elif not isinstance(raw, dict):
            _LOGGER.warning("CHA settings %s is not an object; using defaults", target)
        else:
            candidate = raw.get("api_mode", API_MODE_SYNC)
            if isinstance(candidate, str) and candidate in API_MODES:
                api_mode = candidate
            legacy_api_type = str(raw.get("api_type", DEFAULT_API_TYPE) or DEFAULT_API_TYPE)
            legacy_base_url = str(raw.get("base_url", ""))
            card_models = _normalize_card_models(raw.get("card_models"))
            batch_model = model_ref_from_value(raw.get("batch_model", ""))
    endpoint = load_api_config().cha
    return CHASettings(
        api_mode=api_mode,
        api_type=endpoint.api_type or legacy_api_type,
        base_url=endpoint.base_url or legacy_base_url,
        api_key=endpoint.api_key,
        card_models=card_models,
        batch_model=batch_model,
    )


def save_cha_settings(settings: CHASettings, path: Path | None = None) -> None:
    """Persist public fields in AppData; endpoint in ``api.json``."""
    if not isinstance(settings, CHASettings):
        raise StorageError(f"expected CHASettings, got {type(settings).__name__}")
    update_api_config(
        cha=CHAApi(
            api_type=settings.api_type,
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
    )
    target = path if path is not None else cha_settings_path()
    payload = {
        "api_mode": settings.api_mode,
        "batch_model": dataclasses.asdict(settings.batch_model),
        "card_models": [dataclasses.asdict(ref) for ref in settings.card_models],
    }
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    _LOGGER.info("CHA settings saved (api_mode=%s)", settings.api_mode)


def resolve_base_profile(
    settings: CHASettings, active: LLMProfile | None
) -> LLMProfile | None:
    """Endpoint used by CHA标注: the main profile, or the own-API profile."""
    if settings.api_mode == API_MODE_OWN:
        if not settings.base_url:
            return None
        vision = active.vision_model if active is not None else ""
        text = active.text_model if active is not None else ""
        temperature = active.temperature if active is not None else 0.7
        max_tokens = active.max_tokens if active is not None else 1024
        system = active.system_prompt if active is not None else ""
        return LLMProfile(
            name=PROFILE_NAME,
            api_type=settings.api_type,
            base_url=settings.base_url,
            api_key=settings.api_key,
            text_model=text,
            vision_model=vision,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system,
        )
    if active is None or not active.base_url:
        return None
    return active


def resolve_card_profile(
    settings: CHASettings,
    active: LLMProfile | None,
    index: int,
    *,
    lookup: ProfileLookup | None = None,
) -> LLMProfile | None:
    """Profile for candidate ``index`` (empty model falls back to the base).

    ``lookup`` resolves pool refs (``profile`` set) to their own API profile
    with ``vision_model`` bound — see :meth:`AppController.profile_for_ref`.
    A pool ref without a lookup, or one that is stale, resolves to None.
    """
    return _with_model(settings, active, _card_model(settings, index), lookup)


def resolve_batch_profile(
    settings: CHASettings,
    active: LLMProfile | None,
    *,
    lookup: ProfileLookup | None = None,
) -> LLMProfile | None:
    """Profile for the whole-batch 画面段 pass (same ``lookup`` semantics)."""
    return _with_model(settings, active, settings.batch_model, lookup)


def _with_model(
    settings: CHASettings,
    active: LLMProfile | None,
    override: ModelRef,
    lookup: ProfileLookup | None,
) -> LLMProfile | None:
    if override.profile:
        return lookup(override) if lookup is not None else None
    base = resolve_base_profile(settings, active)
    if base is None:
        return None
    model = (override.model or base.vision_model).strip()
    if not model:
        return None
    return replace(base, vision_model=model)


def _card_model(settings: CHASettings, index: int) -> ModelRef:
    if index < 0 or index >= len(settings.card_models):
        return ModelRef()
    return settings.card_models[index]


def _normalize_card_models(raw: object) -> tuple[ModelRef, ...]:
    items: list[ModelRef] = []
    if isinstance(raw, (list, tuple)):
        for item in raw[:CARD_SLOTS]:
            items.append(model_ref_from_value(item))
    while len(items) < CARD_SLOTS:
        items.append(ModelRef())
    return tuple(items)


