"""Persisted custom system/user prompts for image inference (设置 ▸ 提示词).

The 重译 (vision re-inference) action sends the active system prompt as the
LLM ``system`` message and the user prompt as the request text before the
image. Prompts live in ``app_data_dir()/vision_prompts.json`` (atomic write,
corrupt file -> defaults) so they survive restarts and can be exported.

The built-in ``默认`` template is intentionally EMPTY — the user supplies the
real system prompt later; an empty system prompt is valid (the request simply
carries no system message beyond the profile's own).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.resources import app_data_dir

_LOGGER = get_logger(__name__)

PROMPTS_FILE_NAME = "vision_prompts.json"
# The built-in template: empty system prompt, cannot be deleted.
DEFAULT_PROMPT_NAME = "默认"
# Fallback user instruction used when the user prompt is left empty, so 重译
# works out of the box (the system prompt may legitimately be empty).
DEFAULT_USER_PROMPT = (
    "Describe this image in a detailed, objective natural language paragraph "
    "suitable for image-generation training. Reply with the caption text only."
)


@dataclass(frozen=True)
class VisionPrompts:
    """Immutable prompt configuration for image inference."""

    active: str = DEFAULT_PROMPT_NAME
    prompts: Mapping[str, str] = field(default_factory=dict)  # custom name -> text
    user_prompt: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompts", dict(self.prompts))

    def names(self) -> tuple[str, ...]:
        """Every selectable template name, 默认 first."""
        return (DEFAULT_PROMPT_NAME, *self.prompts)

    def system_text(self) -> str:
        """The active template's system prompt ('' for 默认 / unknown names)."""
        if self.active == DEFAULT_PROMPT_NAME:
            return ""
        return self.prompts.get(self.active, "")

    def effective_user_prompt(self) -> str:
        """The user prompt, falling back to the built-in instruction."""
        stripped = self.user_prompt.strip()
        return stripped if stripped else DEFAULT_USER_PROMPT

    def with_changes(self, **changes: object) -> "VisionPrompts":
        """Return a copy with the given fields replaced (immutable update)."""
        return replace(self, **changes)  # type: ignore[arg-type]


def vision_prompts_path() -> Path:
    """Location of the persisted prompt configuration."""
    return app_data_dir() / PROMPTS_FILE_NAME


def load_vision_prompts(path: Path | None = None) -> VisionPrompts:
    """Load the prompts, falling back to defaults on missing/corrupt files."""
    target = path if path is not None else vision_prompts_path()
    if not target.exists():
        return VisionPrompts()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("could not read %s (%s); using defaults", target, exc)
        return VisionPrompts()
    if not isinstance(raw, dict):
        _LOGGER.warning("vision prompts %s is not an object; using defaults", target)
        return VisionPrompts()
    raw_prompts = raw.get("prompts", {})
    prompts = (
        {
            str(name): str(text)
            for name, text in raw_prompts.items()
            if isinstance(name, str) and name.strip() and name != DEFAULT_PROMPT_NAME
        }
        if isinstance(raw_prompts, dict)
        else {}
    )
    active = raw.get("active", DEFAULT_PROMPT_NAME)
    if not isinstance(active, str) or (
        active != DEFAULT_PROMPT_NAME and active not in prompts
    ):
        active = DEFAULT_PROMPT_NAME
    user_prompt = raw.get("user_prompt", "")
    return VisionPrompts(
        active=active,
        prompts=prompts,
        user_prompt=user_prompt if isinstance(user_prompt, str) else "",
    )


def save_vision_prompts(prompts: VisionPrompts, path: Path | None = None) -> None:
    """Persist the prompts atomically as UTF-8 JSON. Raises StorageError."""
    if not isinstance(prompts, VisionPrompts):
        raise StorageError(f"expected VisionPrompts, got {type(prompts).__name__}")
    target = path if path is not None else vision_prompts_path()
    payload = {
        "active": prompts.active,
        "prompts": dict(prompts.prompts),
        "user_prompt": prompts.user_prompt,
    }
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
    _LOGGER.info("vision prompts saved (active=%s)", prompts.active)
