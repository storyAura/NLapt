"""Last-used CHA标注 memory (per-slot role name / series / chosen card).

Persisted under ``app_data_dir()/layered_infer.json`` so the wizard can
prefill its card slots. Widgets must not call this module — the dialog (a
sanctioned settings-style owner) loads and saves it.

File shape (v2)::

    {"slots": [{"name": "...", "series": "...", "card_text": "..."}, ...]}

The v1 shape (top-level ``name`` / ``series`` / ``card_text``) is read as a
single slot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui.layered_prompts import MAX_CARDS
from nlapt_gui.resources import app_data_dir

_LOGGER = get_logger(__name__)

MEMORY_FILE_NAME = "layered_infer.json"
KEY_SLOTS = "slots"


@dataclass(frozen=True)
class SlotMemory:
    """Last confirmed inputs of one card slot (all fields optional)."""

    name: str = ""
    series: str = ""
    card_text: str = ""


@dataclass(frozen=True)
class LayeredMemory:
    """Last confirmed CHA标注 slots (0..MAX_CARDS entries)."""

    slots: tuple[SlotMemory, ...] = ()

    @property
    def name(self) -> str:
        """First slot's role name (legacy convenience)."""
        return self.slots[0].name if self.slots else ""

    @property
    def series(self) -> str:
        """First slot's series (legacy convenience)."""
        return self.slots[0].series if self.slots else ""

    def with_changes(self, **changes: object) -> "LayeredMemory":
        return replace(self, **changes)  # type: ignore[arg-type]


def layered_memory_path() -> Path:
    """Location of the persisted wizard memory."""
    return app_data_dir() / MEMORY_FILE_NAME


def _slot_from_raw(raw: object) -> SlotMemory | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name", "")
    series = raw.get("series", "")
    card_text = raw.get("card_text", "")
    return SlotMemory(
        name=name if isinstance(name, str) else "",
        series=series if isinstance(series, str) else "",
        card_text=card_text if isinstance(card_text, str) else "",
    )


def load_layered_memory(path: Path | None = None) -> LayeredMemory:
    """Load memory, falling back to no slots on missing/corrupt files."""
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
    slots_raw = raw.get(KEY_SLOTS)
    if isinstance(slots_raw, list):
        slots = [slot for slot in map(_slot_from_raw, slots_raw) if slot is not None]
    else:
        legacy = _slot_from_raw(raw)
        slots = [legacy] if legacy is not None and legacy != SlotMemory() else []
    return LayeredMemory(slots=tuple(slots[:MAX_CARDS]))


def save_layered_memory(memory: LayeredMemory, path: Path | None = None) -> None:
    """Persist memory atomically as UTF-8 JSON. Raises StorageError."""
    if not isinstance(memory, LayeredMemory):
        raise StorageError(f"expected LayeredMemory, got {type(memory).__name__}")
    if len(memory.slots) > MAX_CARDS:
        raise StorageError(f"at most {MAX_CARDS} slots, got {len(memory.slots)}")
    target = path if path is not None else layered_memory_path()
    payload = {
        KEY_SLOTS: [
            {"name": slot.name, "series": slot.series, "card_text": slot.card_text}
            for slot in memory.slots
        ]
    }
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2))
    _LOGGER.info("layered infer memory saved (%d slot(s))", len(memory.slots))
