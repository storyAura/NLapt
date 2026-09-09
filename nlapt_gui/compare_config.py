"""Persisted 多对比推标 settings: which pool models take part in a comparison.

Only ``ModelRef``s are stored (no secrets), in ``app_data_dir()/compare_infer.json``.
Refs that later leave the pool are kept in the file and skipped at run time
with an explicit message — never silently replaced.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, replace
from pathlib import Path

from nlapt.core.config import ModelRef
from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.model_targets import model_ref_from_value
from nlapt_gui.resources import app_data_dir

_LOGGER = get_logger(__name__)

CONFIG_FILE_NAME = "compare_infer.json"
MIN_COMPARE_MODELS = 2


@dataclass(frozen=True)
class CompareSettings:
    """Immutable list of pool models compared side by side."""

    models: tuple[ModelRef, ...] = ()

    def __post_init__(self) -> None:
        cleaned = tuple(
            dict.fromkeys(ref for ref in map(model_ref_from_value, self.models) if ref.is_set())
        )
        object.__setattr__(self, "models", cleaned)

    def with_changes(self, **changes: object) -> "CompareSettings":
        return replace(self, **changes)  # type: ignore[arg-type]


def compare_settings_path() -> Path:
    """Location of the persisted 多对比推标 settings."""
    return app_data_dir() / CONFIG_FILE_NAME


def load_compare_settings(path: Path | None = None) -> CompareSettings:
    """Load settings; a missing or corrupt file yields the defaults."""
    target = path if path is not None else compare_settings_path()
    if not target.exists():
        return CompareSettings()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("could not read %s (%s); using defaults", target, exc)
        return CompareSettings()
    if not isinstance(raw, dict):
        _LOGGER.warning("compare settings %s is not an object; using defaults", target)
        return CompareSettings()
    models = raw.get("models", [])
    if not isinstance(models, list):
        _LOGGER.warning("compare settings %s: models is not a list; ignoring", target)
        models = []
    return CompareSettings(models=tuple(model_ref_from_value(item) for item in models))


def save_compare_settings(settings: CompareSettings, path: Path | None = None) -> None:
    """Persist atomically as UTF-8 JSON. Raises StorageError on IO failure."""
    if not isinstance(settings, CompareSettings):
        raise StorageError(f"expected CompareSettings, got {type(settings).__name__}")
    target = path if path is not None else compare_settings_path()
    payload = {"models": [dataclasses.asdict(ref) for ref in settings.models]}
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    _LOGGER.info("compare settings saved (models=%s)", len(settings.models))
