"""Persisted settings of the local-inference module (设置 ▸ 本地推理).

A small standalone JSON file (the GUI stores it under
``app_data_dir()/local_llm.json``) holding the model directory, the
llama-server executable path, the runtime knobs (context length, GPU layers,
threads, request concurrency, port) and the last selected catalog entry.

Kept separate from the core ``config.json`` for the same reason as the
translation config: these are module-local concerns that must not disturb
the core config schema. Loading clamps every numeric field into its valid
range and falls back to defaults on missing / corrupt files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from nlapt.core.errors import StorageError
from nlapt.diagnostics import get_logger
from nlapt.local.florence import DEFAULT_FLORENCE_TASK, FLORENCE_TASK_TOKENS
from nlapt.storage.atomic import atomic_write_text

_LOGGER = get_logger(__name__)

SETTINGS_FILE_NAME = "local_llm.json"

DEFAULT_PORT = 18434
PORT_RANGE = (1024, 65535)
DEFAULT_CONTEXT_LENGTH = 4096
CONTEXT_RANGE = (512, 131_072)
DEFAULT_PARALLEL = 2
PARALLEL_RANGE = (1, 16)
DEFAULT_THREADS = 0  # 0 = let llama.cpp pick
THREADS_RANGE = (0, 256)
DEFAULT_GPU_LAYERS = -1  # -1 = offload everything possible
GPU_LAYERS_RANGE = (-1, 999)


@dataclass(frozen=True)
class LocalSettings:
    """Immutable settings of the local-inference module."""

    models_dir: str = ""  # primary/download dir; empty -> caller's default
    extra_dirs: tuple[str, ...] = ()  # additional search dirs (复用外部模型)
    server_path: str = ""  # llama-server executable; empty = not configured
    port: int = DEFAULT_PORT
    context_length: int = DEFAULT_CONTEXT_LENGTH
    gpu_layers: int = DEFAULT_GPU_LAYERS
    threads: int = DEFAULT_THREADS
    parallel: int = DEFAULT_PARALLEL
    family_id: str = ""  # last selected catalog family
    quant_label: str = ""  # last selected quant label
    # Instruction (指令) used by Florence-2 PromptGen families.
    florence_task: str = DEFAULT_FLORENCE_TASK
    # Official prompt preset id for caption-specialist GGUF families
    # (nlapt.local.presets). "" = the family's default preset; the
    # PRESET_CUSTOM sentinel opts out into the free-form 推理提示词.
    prompt_preset: str = ""

    def with_changes(self, **changes: object) -> LocalSettings:
        """Return a copy with the given fields replaced (immutable update)."""
        return replace(self, **changes)  # type: ignore[arg-type]


def _valid_task(value: object) -> str:
    """Coerce a stored Florence 指令 to a known token; fall back to default."""
    return value if value in FLORENCE_TASK_TOKENS else DEFAULT_FLORENCE_TASK


def _clamp(value: object, default: int, bounds: tuple[int, int]) -> int:
    """Coerce ``value`` to an int inside ``bounds``; fall back to default."""
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    low, high = bounds
    return max(low, min(high, number))


def load_local_settings(path: Path) -> LocalSettings:
    """Load settings; missing / corrupt files fall back to defaults."""
    if not path.exists():
        return LocalSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("could not read %s (%s); using defaults", path, exc)
        return LocalSettings()
    if not isinstance(raw, dict):
        _LOGGER.warning("local settings %s is not an object; using defaults", path)
        return LocalSettings()
    raw_extra = raw.get("extra_dirs", [])
    extra_dirs = (
        tuple(
            str(item).strip()
            for item in raw_extra
            if isinstance(item, str) and str(item).strip()
        )
        if isinstance(raw_extra, list)
        else ()
    )
    return LocalSettings(
        models_dir=str(raw.get("models_dir", "")),
        extra_dirs=extra_dirs,
        server_path=str(raw.get("server_path", "")),
        port=_clamp(raw.get("port", DEFAULT_PORT), DEFAULT_PORT, PORT_RANGE),
        context_length=_clamp(
            raw.get("context_length", DEFAULT_CONTEXT_LENGTH),
            DEFAULT_CONTEXT_LENGTH,
            CONTEXT_RANGE,
        ),
        gpu_layers=_clamp(
            raw.get("gpu_layers", DEFAULT_GPU_LAYERS),
            DEFAULT_GPU_LAYERS,
            GPU_LAYERS_RANGE,
        ),
        threads=_clamp(raw.get("threads", DEFAULT_THREADS), DEFAULT_THREADS, THREADS_RANGE),
        parallel=_clamp(
            raw.get("parallel", DEFAULT_PARALLEL), DEFAULT_PARALLEL, PARALLEL_RANGE
        ),
        family_id=str(raw.get("family_id", "")),
        quant_label=str(raw.get("quant_label", "")),
        florence_task=_valid_task(raw.get("florence_task", DEFAULT_FLORENCE_TASK)),
        prompt_preset=str(raw.get("prompt_preset", "")),
    )


def save_local_settings(path: Path, settings: LocalSettings) -> None:
    """Persist settings atomically as UTF-8 JSON. Raises StorageError."""
    if not isinstance(settings, LocalSettings):
        raise StorageError(f"expected LocalSettings, got {type(settings).__name__}")
    text = json.dumps(asdict(settings), ensure_ascii=False, indent=2, sort_keys=True)
    atomic_write_text(path, text)
    _LOGGER.info(
        "local settings saved (family=%s quant=%s parallel=%d)",
        settings.family_id or "-",
        settings.quant_label or "-",
        settings.parallel,
    )
