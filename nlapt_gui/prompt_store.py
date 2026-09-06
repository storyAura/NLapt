"""Persisted custom system/user prompts for image inference (设置 ▸ 提示词).

The 重译 (vision re-inference) action sends the active system prompt as the
LLM ``system`` message and the user prompt as the request text before the
image. Prompts live in ``app_data_dir()/vision_prompts.json`` (atomic write,
corrupt file -> defaults) so they survive restarts and can be exported.

The built-in ``默认`` template is intentionally EMPTY. Shipped schemes
(结构化视觉编译 / 客观视觉报告) live in :mod:`nlapt_gui.builtin_prompts` and
are always listed after 默认; they are not written to the JSON file unless
the user later saves an override under the same name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.builtin_prompts import (
    BUILTIN_PROMPT_ORDER,
    PROMPT_OBJECTIVE_REPORT,
    PROMPT_STRUCTURED_COMPILER,
    builtin_system_text,
    is_builtin_prompt,
)
from nlapt_gui.resources import app_data_dir

# Re-export so callers / tests import from this module only.
__all__ = (
    "BUILTIN_PROMPT_ORDER",
    "DEFAULT_PROMPT_NAME",
    "DEFAULT_USER_PROMPT",
    "ENGINE_LLM",
    "ENGINE_LOCAL",
    "PROMPT_OBJECTIVE_REPORT",
    "PROMPT_STRUCTURED_COMPILER",
    "PROMPTS_FILE_NAME",
    "VisionPrompts",
    "is_builtin_prompt",
    "is_locked_template",
    "load_vision_prompts",
    "save_vision_prompts",
    "vision_prompts_path",
)

_LOGGER = get_logger(__name__)

PROMPTS_FILE_NAME = "vision_prompts.json"
# The built-in template: empty system prompt, cannot be deleted.
DEFAULT_PROMPT_NAME = "默认"


def is_locked_template(name: str) -> bool:
    """True for 默认 and the shipped schemes (not deletable in the editor)."""
    return name == DEFAULT_PROMPT_NAME or is_builtin_prompt(name)


# Fallback user instruction used when the user prompt is left empty, so 重译
# works out of the box (the system prompt may legitimately be empty).
DEFAULT_USER_PROMPT = (
    "Describe this image in a detailed, objective natural language paragraph "
    "suitable for image-generation training. Reply with the caption text only."
)

# Inference engines (统一管线 by default; local may override its prompts).
ENGINE_LLM = "llm"
ENGINE_LOCAL = "local"


@dataclass(frozen=True)
class VisionPrompts:
    """Immutable prompt configuration for image inference.

    One shared prompt set drives every engine (统一管线). When
    ``local_unified`` is off, 本地推理 uses its own ``local_system`` /
    ``local_user_prompt`` pair instead.
    """

    active: str = DEFAULT_PROMPT_NAME
    prompts: Mapping[str, str] = field(default_factory=dict)  # custom name -> text
    user_prompt: str = ""
    local_unified: bool = True
    local_system: str = ""
    local_user_prompt: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompts", dict(self.prompts))

    def names(self) -> tuple[str, ...]:
        """Every selectable template name: 默认, shipped schemes, then customs."""
        extras = tuple(
            name
            for name in self.prompts
            if name != DEFAULT_PROMPT_NAME and name not in BUILTIN_PROMPT_ORDER
        )
        return (DEFAULT_PROMPT_NAME, *BUILTIN_PROMPT_ORDER, *extras)

    def system_text_of(self, name: str) -> str:
        """System prompt for ``name`` ('' for 默认 / unknown). Custom overrides win."""
        if name == DEFAULT_PROMPT_NAME:
            return ""
        if name in self.prompts:
            return self.prompts[name]
        return builtin_system_text(name)

    def system_text(self) -> str:
        """The active template's system prompt ('' for 默认 / unknown names)."""
        return self.system_text_of(self.active)

    def is_known_name(self, name: str) -> bool:
        """True when ``name`` is 默认, a shipped scheme, or a custom template."""
        return is_locked_template(name) or name in self.prompts

    def effective_user_prompt(self) -> str:
        """The user prompt, falling back to the built-in instruction."""
        stripped = self.user_prompt.strip()
        return stripped if stripped else DEFAULT_USER_PROMPT

    def system_text_for(self, engine: str) -> str:
        """System prompt for an engine (local override when not unified)."""
        if engine == ENGINE_LOCAL and not self.local_unified:
            return self.local_system
        return self.system_text()

    def user_prompt_for(self, engine: str) -> str:
        """User prompt for an engine, with the built-in fallback applied."""
        if engine == ENGINE_LOCAL and not self.local_unified:
            stripped = self.local_user_prompt.strip()
            return stripped if stripped else DEFAULT_USER_PROMPT
        return self.effective_user_prompt()

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
        active != DEFAULT_PROMPT_NAME
        and active not in prompts
        and not is_builtin_prompt(active)
    ):
        active = DEFAULT_PROMPT_NAME
    user_prompt = raw.get("user_prompt", "")
    local_system = raw.get("local_system", "")
    local_user_prompt = raw.get("local_user_prompt", "")
    return VisionPrompts(
        active=active,
        prompts=prompts,
        user_prompt=user_prompt if isinstance(user_prompt, str) else "",
        local_unified=bool(raw.get("local_unified", True)),
        local_system=local_system if isinstance(local_system, str) else "",
        local_user_prompt=(
            local_user_prompt if isinstance(local_user_prompt, str) else ""
        ),
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
        "local_unified": prompts.local_unified,
        "local_system": prompts.local_system,
        "local_user_prompt": prompts.local_user_prompt,
    }
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
    _LOGGER.info("vision prompts saved (active=%s)", prompts.active)
