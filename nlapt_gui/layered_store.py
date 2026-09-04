"""Last-used 分层推标 memory (role name / series / chosen card).

Persisted under ``app_data_dir()/layered_infer.json`` so the wizard can
prefill. Widgets must not call this module — the dialog (a sanctioned
settings-style owner) loads and saves it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.resources import app_data_dir

_LOGGER = get_logger(__name__)

MEMORY_FILE_NAME = "layered_infer.json"


@dataclass(frozen=True)
class LayeredMemory:
    """Last confirmed 分层推标 inputs (all fields optional)."""

    name: str = ""
    series: str = ""
    card_text: str = ""

    def with_changes(self, **changes: object) -> "LayeredMemory":
        return replace(self, **changes)  # type: ignore[arg-type]


def layered_memory_path() -> Path:
    """Location of the persisted wizard memory."""
    return app_data_dir() / MEMORY_FILE_NAME


def load_layered_memory(path: Path | None = None) -> LayeredMemory:
    """Load memory, falling back to empty fields on missing/corrupt files."""
    target = path if path is not None else layered_memory_path()
    if not target.exists():
        return LayeredMemory()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("could not read %s (%s); using empty memory", target, exc)
        return LayeredMemory()
    if not isinstance(raw, dict):
        return LayeredMemory()
    name = raw.get("name", "")
    series = raw.get("series", "")
    card_text = raw.get("card_text", "")
    return LayeredMemory(
        name=name if isinstance(name, str) else "",
        series=series if isinstance(series, str) else "",
        card_text=card_text if isinstance(card_text, str) else "",
    )


def save_layered_memory(memory: LayeredMemory, path: Path | None = None) -> None:
    """Persist memory atomically as UTF-8 JSON. Raises StorageError."""
    if not isinstance(memory, LayeredMemory):
        raise StorageError(f"expected LayeredMemory, got {type(memory).__name__}")
    target = path if path is not None else layered_memory_path()
    payload = {
        "name": memory.name,
        "series": memory.series,
        "card_text": memory.card_text,
    }
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
    _LOGGER.info("layered infer memory saved")
