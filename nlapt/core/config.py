"""Application configuration: models, JSON persistence, secret masking (spec 8).

API keys are stored locally only and MUST be masked (``mask_secret``) in any
log line or exported debug bundle (``masked_config_dict``). Unknown
``api_type`` values are tolerated at this layer — the LLM client registry is
the extensibility point that decides whether a type is usable.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nlapt.core.errors import StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

_LOGGER = get_logger(__name__)

KNOWN_API_TYPES: frozenset[str] = frozenset({"openai", "anthropic", "ollama"})
CONFIG_JSON_INDENT = 2

MASK_PLACEHOLDER = "***"
MASK_PREFIX_CHARS = 3
MASK_SUFFIX_CHARS = 4
# Minimum number of characters that must stay hidden when partially revealing.
# Short tokens (e.g. 8-10 chars, common for local/proxy gateways) would leak
# almost entirely with a prefix+suffix reveal, so they are fully masked.
MIN_HIDDEN_CHARS = 5
# Secrets shorter than this reveal nothing at all (fully replaced by "***").
MIN_MASKABLE_LENGTH = MASK_PREFIX_CHARS + MASK_SUFFIX_CHARS + MIN_HIDDEN_CHARS


@dataclass(frozen=True)
class LLMProfile:
    """One configured LLM API (an endpoint plus the models known on it).

    ``models`` is the catalog the user fetched or typed; ``enabled_models`` is
    the subset switched on for the shared model pool. ``text_model`` /
    ``vision_model`` are the ids a *resolved* profile is bound to (see
    :func:`resolve_text_profile`); on a stored profile they are only the
    legacy per-profile defaults used by :func:`upgrade_legacy_targets`.
    """

    name: str
    api_type: str  # "openai" | "anthropic" | "ollama" (registry-extensible)
    base_url: str
    api_key: str = ""  # stored locally only; must be masked in any log/export
    text_model: str = ""
    vision_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = ""
    models: tuple[str, ...] = ()
    enabled_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "enabled_models", tuple(self.enabled_models))


@dataclass(frozen=True)
class ModelRef:
    """A model chosen from the pool: ``model`` on the profile named ``profile``."""

    profile: str = ""
    model: str = ""

    def is_set(self) -> bool:
        return bool(self.profile) and bool(self.model)


ROLE_TEXT = "text"
ROLE_VISION = "vision"


@dataclass(frozen=True)
class RequestControl:
    """Request throttling and resilience parameters."""

    concurrency: int = 4
    min_interval: float = 0.0  # seconds between request starts
    timeout: float = 60.0
    max_retries: int = 2  # exponential backoff


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    profiles: tuple[LLMProfile, ...] = ()
    active_profile: str = ""  # legacy single-profile selection (see upgrade_legacy_targets)
    text_target: ModelRef = ModelRef()
    vision_target: ModelRef = ModelRef()
    request: RequestControl = RequestControl()
    image_max_edge: int = 1024
    revert_confirmed_on_edit: bool = True  # spec 4.1 supplementary rule
    snapshot_retention: int = 20
    prompt_orphan_cleanup: bool = False
    custom_templates: Mapping[str, str] = field(default_factory=dict)
    trigger_presets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Defensive copies so callers cannot mutate through shared references.
        object.__setattr__(self, "profiles", tuple(self.profiles))
        object.__setattr__(self, "custom_templates", dict(self.custom_templates))
        object.__setattr__(self, "trigger_presets", tuple(self.trigger_presets))


def _expect_type(value: Any, expected: type | tuple[type, ...], context: str) -> Any:
    if isinstance(value, bool) and expected in (int, float, (int, float)):
        raise ValidationError(f"config field {context} must be a number, got a boolean")
    if not isinstance(value, expected):
        names = (
            expected.__name__
            if isinstance(expected, type)
            else "/".join(t.__name__ for t in expected)
        )
        raise ValidationError(
            f"config field {context} must be {names}, got {type(value).__name__}"
        )
    return value


def _get_str(data: Mapping[str, Any], key: str, default: str, context: str) -> str:
    value = data.get(key, default)
    return str(_expect_type(value, str, f"{context}.{key}"))


def _get_int(data: Mapping[str, Any], key: str, default: int, context: str) -> int:
    value = data.get(key, default)
    return int(_expect_type(value, int, f"{context}.{key}"))


def _get_float(data: Mapping[str, Any], key: str, default: float, context: str) -> float:
    value = data.get(key, default)
    return float(_expect_type(value, (int, float), f"{context}.{key}"))


def _get_bool(data: Mapping[str, Any], key: str, default: bool, context: str) -> bool:
    value = data.get(key, default)
    return bool(_expect_type(value, bool, f"{context}.{key}"))


def _get_str_list(data: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    raw = data.get(key, [])
    _expect_type(raw, list, f"{context}.{key}")
    return tuple(
        str(_expect_type(item, str, f"{context}.{key}[{i}]")) for i, item in enumerate(raw)
    )


def model_ref_from_dict(data: Any, context: str) -> ModelRef:
    """Parse a ``{"profile": ..., "model": ...}`` object (missing -> unset)."""
    if data is None:
        return ModelRef()
    _expect_type(data, dict, context)
    return ModelRef(
        profile=_get_str(data, "profile", "", context),
        model=_get_str(data, "model", "", context),
    )


def profile_from_dict(data: Any, index: int = 0) -> LLMProfile:
    """Parse one LLM profile object (unknown ``api_type`` is tolerated)."""
    context = f"profiles[{index}]"
    _expect_type(data, dict, context)
    profile = LLMProfile(
        name=_get_str(data, "name", "", context),
        api_type=_get_str(data, "api_type", "", context),
        base_url=_get_str(data, "base_url", "", context),
        api_key=_get_str(data, "api_key", "", context),
        text_model=_get_str(data, "text_model", "", context),
        vision_model=_get_str(data, "vision_model", "", context),
        temperature=_get_float(data, "temperature", 0.7, context),
        max_tokens=_get_int(data, "max_tokens", 1024, context),
        system_prompt=_get_str(data, "system_prompt", "", context),
        models=_get_str_list(data, "models", context),
        enabled_models=_get_str_list(data, "enabled_models", context),
    )
    if profile.api_type and profile.api_type not in KNOWN_API_TYPES:
        # Tolerated here: the client registry may have extra types registered.
        _LOGGER.warning(
            "profile %r has unknown api_type %r (tolerated at config layer)",
            profile.name,
            profile.api_type,
        )
    return profile


def profiles_from_list(raw: Any) -> tuple[LLMProfile, ...]:
    """Parse a JSON list of profile objects. ``raw`` must be a list."""
    _expect_type(raw, list, "profiles")
    return tuple(profile_from_dict(item, i) for i, item in enumerate(raw))


def _request_from_dict(data: Any) -> RequestControl:
    context = "request"
    _expect_type(data, dict, context)
    return RequestControl(
        concurrency=_get_int(data, "concurrency", 4, context),
        min_interval=_get_float(data, "min_interval", 0.0, context),
        timeout=_get_float(data, "timeout", 60.0, context),
        max_retries=_get_int(data, "max_retries", 2, context),
    )


def _config_from_dict(data: Mapping[str, Any]) -> AppConfig:
    defaults = AppConfig()
    raw_profiles = data.get("profiles", [])
    profiles = profiles_from_list(raw_profiles)

    raw_templates = data.get("custom_templates", {})
    _expect_type(raw_templates, dict, "custom_templates")
    templates = {
        str(_expect_type(k, str, "custom_templates key")): str(
            _expect_type(v, str, f"custom_templates[{k!r}]")
        )
        for k, v in raw_templates.items()
    }

    raw_presets = data.get("trigger_presets", [])
    _expect_type(raw_presets, list, "trigger_presets")
    presets = tuple(
        str(_expect_type(item, str, f"trigger_presets[{i}]"))
        for i, item in enumerate(raw_presets)
    )

    request = (
        _request_from_dict(data["request"]) if "request" in data else RequestControl()
    )
    return AppConfig(
        profiles=profiles,
        active_profile=_get_str(data, "active_profile", "", "config"),
        text_target=model_ref_from_dict(data.get("text_target"), "text_target"),
        vision_target=model_ref_from_dict(data.get("vision_target"), "vision_target"),
        request=request,
        image_max_edge=_get_int(data, "image_max_edge", defaults.image_max_edge, "config"),
        revert_confirmed_on_edit=_get_bool(
            data, "revert_confirmed_on_edit", defaults.revert_confirmed_on_edit, "config"
        ),
        snapshot_retention=_get_int(
            data, "snapshot_retention", defaults.snapshot_retention, "config"
        ),
        prompt_orphan_cleanup=_get_bool(
            data, "prompt_orphan_cleanup", defaults.prompt_orphan_cleanup, "config"
        ),
        custom_templates=templates,
        trigger_presets=presets,
    )


def load_config(path: Path) -> AppConfig:
    """Load config JSON. Missing file -> defaults; invalid content -> ValidationError."""
    source = Path(path)
    if not source.exists():
        _LOGGER.info("config file %s not found; using defaults", source)
        return AppConfig()
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise StorageError(f"cannot read config file {source}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"config file {source} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(
            f"config file {source} must contain a JSON object at the top level"
        )
    return _config_from_dict(data)


def save_config(path: Path, config: AppConfig) -> None:
    """Atomically save config as UTF-8 JSON (full fidelity, including api keys)."""
    if not isinstance(config, AppConfig):
        raise ValidationError(f"config must be an AppConfig, got {type(config).__name__}")
    payload = dataclasses.asdict(config)
    text = json.dumps(payload, indent=CONFIG_JSON_INDENT, ensure_ascii=False) + "\n"
    atomic_write_text(Path(path), text)


def get_active_profile(config: AppConfig) -> LLMProfile | None:
    """Return the profile named by ``active_profile``, or None if unset/unknown."""
    if not isinstance(config, AppConfig):
        raise ValidationError(f"config must be an AppConfig, got {type(config).__name__}")
    if not config.active_profile:
        return None
    for profile in config.profiles:
        if profile.name == config.active_profile:
            return profile
    return None


# ---- model pool -----------------------------------------------------------------


def find_profile(config: AppConfig, name: str) -> LLMProfile | None:
    """The profile named ``name`` (exact match), or None."""
    if not name:
        return None
    for profile in config.profiles:
        if profile.name == name:
            return profile
    return None


def enabled_model_refs(config: AppConfig) -> tuple[ModelRef, ...]:
    """Every switched-on model across all profiles (profile order, then model order)."""
    refs: list[ModelRef] = []
    for profile in config.profiles:
        for model in profile.enabled_models:
            if model:
                refs.append(ModelRef(profile=profile.name, model=model))
    return tuple(refs)


def resolve_model_ref(config: AppConfig, ref: ModelRef) -> LLMProfile | None:
    """The profile ``ref`` points at, or None when it is unset / stale.

    Stale means the profile no longer exists or the model has been switched
    off — the caller must treat that as "unconfigured", never fall back to a
    different model silently.
    """
    if not ref.is_set():
        return None
    profile = find_profile(config, ref.profile)
    if profile is None or ref.model not in profile.enabled_models:
        return None
    return profile


def _role_field(role: str) -> str:
    if role not in (ROLE_TEXT, ROLE_VISION):
        raise ValidationError(f"unknown model role {role!r}")
    return "text_model" if role == ROLE_TEXT else "vision_model"


def bind_model_ref(config: AppConfig, ref: ModelRef, role: str) -> LLMProfile | None:
    """The pool profile ``ref`` points at, bound to ``ref.model`` for ``role``.

    ``text_model`` (ROLE_TEXT) or ``vision_model`` (ROLE_VISION) of the
    returned copy is the chosen id; None for an unset / stale ref.
    """
    field_name = _role_field(role)
    profile = resolve_model_ref(config, ref)
    if profile is None:
        return None
    return dataclasses.replace(profile, **{field_name: ref.model})


def _resolve_role(config: AppConfig, ref: ModelRef, role: str) -> LLMProfile | None:
    field_name = _role_field(role)
    if ref.is_set():
        return bind_model_ref(config, ref, role)
    # Legacy configs (single ``active_profile``, no targets) keep working.
    active = get_active_profile(config)
    if active is None or not getattr(active, field_name):
        return None
    return active


def resolve_text_profile(config: AppConfig) -> LLMProfile | None:
    """Profile bound to the current 文本模型 (``text_model`` = the chosen id)."""
    if not isinstance(config, AppConfig):
        raise ValidationError(f"config must be an AppConfig, got {type(config).__name__}")
    return _resolve_role(config, config.text_target, ROLE_TEXT)


def resolve_vision_profile(config: AppConfig) -> LLMProfile | None:
    """Profile bound to the current 视觉模型 (``vision_model`` = the chosen id)."""
    if not isinstance(config, AppConfig):
        raise ValidationError(f"config must be an AppConfig, got {type(config).__name__}")
    return _resolve_role(config, config.vision_target, ROLE_VISION)


def _upgrade_profile(profile: LLMProfile) -> LLMProfile:
    enabled = profile.enabled_models
    if not enabled and not profile.models:
        # Never touched by the pool UI: its legacy defaults become the switches.
        # (A profile with a catalog but nothing enabled was switched off on purpose.)
        enabled = tuple(
            dict.fromkeys(m for m in (profile.text_model, profile.vision_model) if m)
        )
    missing = tuple(m for m in enabled if m not in profile.models)
    if enabled == profile.enabled_models and not missing:
        return profile
    return dataclasses.replace(
        profile, models=profile.models + missing, enabled_models=enabled
    )


def upgrade_legacy_targets(config: AppConfig) -> AppConfig:
    """Derive pool state from a pre-pool config. Idempotent.

    * A profile with neither ``models`` nor ``enabled_models`` enables its
      ``text_model`` / ``vision_model`` (and lists them in ``models``).
    * Unset ``text_target`` / ``vision_target`` are pointed at the
      ``active_profile``'s models when it has them. The pool UI saves
      ``active_profile=""``, so a target the user cleared on purpose is not
      resurrected on the next load.
    """
    if not isinstance(config, AppConfig):
        raise ValidationError(f"config must be an AppConfig, got {type(config).__name__}")
    profiles = tuple(_upgrade_profile(p) for p in config.profiles)
    text_target = config.text_target
    vision_target = config.vision_target
    active = get_active_profile(config)
    if active is not None:
        if not text_target.is_set() and active.text_model:
            text_target = ModelRef(profile=active.name, model=active.text_model)
        if not vision_target.is_set() and active.vision_model:
            vision_target = ModelRef(profile=active.name, model=active.vision_model)
    if (
        profiles == config.profiles
        and text_target == config.text_target
        and vision_target == config.vision_target
    ):
        return config
    return dataclasses.replace(
        config, profiles=profiles, text_target=text_target, vision_target=vision_target
    )


def mask_secret(value: str) -> str:
    """Mask a secret for display/logging: '' -> '', short -> '***', long -> 'sk-***xyz9'."""
    if not isinstance(value, str):
        raise ValidationError(f"secret must be a string, got {type(value).__name__}")
    if not value:
        return ""
    if len(value) < MIN_MASKABLE_LENGTH:
        return MASK_PLACEHOLDER
    return f"{value[:MASK_PREFIX_CHARS]}{MASK_PLACEHOLDER}{value[-MASK_SUFFIX_CHARS:]}"


def masked_config_dict(config: AppConfig) -> dict:
    """Config as a plain dict with every api_key masked (safe for logs/bundles)."""
    if not isinstance(config, AppConfig):
        raise ValidationError(f"config must be an AppConfig, got {type(config).__name__}")
    data = dataclasses.asdict(config)
    for profile in data.get("profiles", []):
        profile["api_key"] = mask_secret(profile.get("api_key", ""))
    return data
