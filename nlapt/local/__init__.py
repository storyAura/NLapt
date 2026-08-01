"""Local-inference module: curated GGUF catalog, hardware advisor, runtime.

Public surface re-exported here; see the sibling modules for details:

* :mod:`nlapt.local.catalog`  — 大系列 / 小系列 / 量化档 model catalog
* :mod:`nlapt.local.hardware` — RAM / VRAM / CPU detection
* :mod:`nlapt.local.advisor`  — "can this machine run it" prediction
* :mod:`nlapt.local.settings` — persisted module settings (并发 etc.)
* :mod:`nlapt.local.download` — resumable GGUF downloader
* :mod:`nlapt.local.server`   — llama-server process manager
* :mod:`nlapt.local.runtime`  — pinned llama.cpp runtime auto-provisioning
* :mod:`nlapt.local.florence` — Florence-2 PromptGen ONNX engine (打标特化)
"""

from __future__ import annotations

from nlapt.local.advisor import (
    MemoryEstimate,
    RunAssessment,
    RunGrade,
    RunVerdict,
    assess,
    auto_gpu_layers,
    estimate_memory,
)
from nlapt.local.florence import (
    DEFAULT_FLORENCE_TASK,
    FLORENCE_TASK_LABELS,
    FLORENCE_TASK_TOKENS,
    FlorenceEngine,
    FlorenceTokenizer,
)
from nlapt.local.gguf import read_block_count
from nlapt.local.catalog import (
    CATALOG_SNAPSHOT_DATE,
    ENGINE_FLORENCE,
    ENGINE_LLAMA,
    ModelFamily,
    ModelSeries,
    QuantFile,
    all_families,
    all_series,
    download_url,
    families_for,
    find_family,
    find_quant,
    mmproj_path,
    quant_path,
    recommended_quant,
    repo_page_url,
)
from nlapt.local.download import download_file
from nlapt.local.hardware import (
    GpuInfo,
    HardwareInfo,
    detect_hardware,
    format_bytes,
)
from nlapt.local.runtime import (
    LLAMA_CPP_TAG,
    RUNTIME_SNAPSHOT_DATE,
    RuntimeAsset,
    current_asset,
    ensure_runtime,
    find_server_exe,
    runtime_dir,
    runtime_platform_key,
)
from nlapt.local.server import (
    LocalServerManager,
    ServerSpec,
    base_url,
    build_server_args,
)
from nlapt.local.settings import (
    LocalSettings,
    load_local_settings,
    save_local_settings,
)

__all__ = [
    "CATALOG_SNAPSHOT_DATE",
    "DEFAULT_FLORENCE_TASK",
    "ENGINE_FLORENCE",
    "ENGINE_LLAMA",
    "FLORENCE_TASK_LABELS",
    "FLORENCE_TASK_TOKENS",
    "FlorenceEngine",
    "FlorenceTokenizer",
    "GpuInfo",
    "HardwareInfo",
    "LLAMA_CPP_TAG",
    "LocalServerManager",
    "LocalSettings",
    "RUNTIME_SNAPSHOT_DATE",
    "RuntimeAsset",
    "MemoryEstimate",
    "ModelFamily",
    "ModelSeries",
    "QuantFile",
    "RunAssessment",
    "RunGrade",
    "RunVerdict",
    "ServerSpec",
    "all_families",
    "all_series",
    "assess",
    "auto_gpu_layers",
    "base_url",
    "build_server_args",
    "current_asset",
    "detect_hardware",
    "download_file",
    "download_url",
    "ensure_runtime",
    "estimate_memory",
    "families_for",
    "find_family",
    "find_quant",
    "find_server_exe",
    "format_bytes",
    "load_local_settings",
    "mmproj_path",
    "quant_path",
    "read_block_count",
    "recommended_quant",
    "repo_page_url",
    "runtime_dir",
    "runtime_platform_key",
    "save_local_settings",
]
