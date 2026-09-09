"""The shared model pool: view records and target switching (设置 ▸ LLM / 工具弹层).

Every switched-on model of every API profile forms one pool. Two roles pick
from it — 当前文本模型 (翻译 / 改写) and 当前视觉模型 (推标 / CHA) — and may
live on different APIs. This module owns the pure helpers plus the single
persistence path (``save_app_config``) so :mod:`nlapt_gui.controller` only
needs thin delegating methods.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from nlapt.core.config import (
    ROLE_TEXT,
    ROLE_VISION,
    AppConfig,
    ModelRef,
    enabled_model_refs,
    resolve_model_ref,
)
from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger
from nlapt.llm.model_list import looks_vision_capable

from nlapt_gui.api_config import save_app_config

_LOGGER = get_logger(__name__)

ROLES: tuple[str, ...] = (ROLE_TEXT, ROLE_VISION)
LABEL_ROLE_TEXT = "文本模型"
LABEL_ROLE_VISION = "视觉模型"
LABEL_UNSET = "未设置"
LABEL_STALE = "已失效"
CHOICE_SEPARATOR = " · "
VISION_HINT_TAG = " · 视觉"


@dataclass(frozen=True)
class ModelChoice:
    """One pool entry as shown in combos / menus."""

    ref: ModelRef
    label: str  # "profile · model"
    vision_hint: bool  # name heuristic only (see model_list.looks_vision_capable)


def model_ref_from_value(value: object) -> ModelRef:
    """Normalise a stored / typed entry: str -> bare ref; dict or ModelRef -> ref.

    A bare ref (``profile == ""``) means "this model id on whatever base
    endpoint the caller uses"; anything unparseable becomes the unset ref.
    """
    if isinstance(value, ModelRef):
        return ModelRef(profile=value.profile.strip(), model=value.model.strip())
    if isinstance(value, str):
        return ModelRef(profile="", model=value.strip())
    if isinstance(value, dict):
        profile = value.get("profile", "")
        model = value.get("model", "")
        return ModelRef(
            profile=profile.strip() if isinstance(profile, str) else "",
            model=model.strip() if isinstance(model, str) else "",
        )
    return ModelRef()


def choice_label(ref: ModelRef) -> str:
    """``profile · model`` for a set ref, 未设置 otherwise."""
    if not ref.is_set():
        return LABEL_UNSET
    return f"{ref.profile}{CHOICE_SEPARATOR}{ref.model}"


def pool_choices(config: AppConfig) -> tuple[ModelChoice, ...]:
    """Every enabled model, in pool order, as display records."""
    return tuple(
        ModelChoice(
            ref=ref,
            label=choice_label(ref),
            vision_hint=looks_vision_capable(ref.model),
        )
        for ref in enabled_model_refs(config)
    )


def grouped_pool_choices(
    config: AppConfig,
) -> tuple[tuple[str, tuple[ModelChoice, ...]], ...]:
    """Pool choices grouped by provider: ``((profile_name, choices), ...)``.

    Profiles with nothing switched on are omitted; order follows the profile list.
    """
    groups: list[tuple[str, tuple[ModelChoice, ...]]] = []
    for profile in config.profiles:
        choices = tuple(
            ModelChoice(
                ref=ModelRef(profile=profile.name, model=model),
                label=choice_label(ModelRef(profile=profile.name, model=model)),
                vision_hint=looks_vision_capable(model),
            )
            for model in profile.enabled_models
            if model
        )
        if choices:
            groups.append((profile.name, choices))
    return tuple(groups)


def role_label(role: str) -> str:
    """Chinese role name for toasts / menu headers."""
    _check_role(role)
    return LABEL_ROLE_TEXT if role == ROLE_TEXT else LABEL_ROLE_VISION


def current_target(config: AppConfig, role: str) -> ModelRef:
    """The ref stored for ``role`` (may be unset or stale)."""
    _check_role(role)
    return config.text_target if role == ROLE_TEXT else config.vision_target


def target_display(config: AppConfig, role: str) -> str:
    """Model id to show for ``role``: 未设置 / 已失效 / the id."""
    ref = current_target(config, role)
    if not ref.is_set():
        return LABEL_UNSET
    if resolve_model_ref(config, ref) is None:
        return LABEL_STALE
    return ref.model


def switch_target(config: AppConfig, role: str, ref: ModelRef) -> AppConfig:
    """Config with ``role`` pointed at ``ref``.

    ``ref`` must be unset (clearing the role) or a member of the pool; a
    stale ref raises ``ValidationError`` so the UI can never persist a
    selection the resolvers would reject. Clearing also drops the legacy
    ``active_profile`` so the cleared role is not re-derived on load.
    """
    _check_role(role)
    if not isinstance(ref, ModelRef):
        raise ValidationError(f"ref must be a ModelRef, got {type(ref).__name__}")
    if ref.is_set() and resolve_model_ref(config, ref) is None:
        raise ValidationError(
            f"model {ref.model!r} on profile {ref.profile!r} is not enabled in the pool"
        )
    field = "text_target" if role == ROLE_TEXT else "vision_target"
    return replace(config, **{field: ref, "active_profile": ""})


def persist_targets(config: AppConfig) -> None:
    """Write the profiles + targets of ``config`` through ``save_app_config``."""
    save_app_config(config)
    _LOGGER.info(
        "model targets saved (text=%s, vision=%s)",
        choice_label(config.text_target),
        choice_label(config.vision_target),
    )


def find_choice(choices: Sequence[ModelChoice], ref: ModelRef) -> ModelChoice | None:
    """The choice whose ref equals ``ref``, or None."""
    for choice in choices:
        if choice.ref == ref:
            return choice
    return None


def _check_role(role: str) -> None:
    if role not in ROLES:
        raise ValidationError(f"unknown model role {role!r}")
