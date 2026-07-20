"""Local-inference module: curated GGUF catalog, hardware advisor, runtime.

Public surface re-exported here; see the sibling modules for details:

* :mod:`nlapt.local.catalog`  — 大系列 / 小系列 / 量化档 model catalog
* :mod:`nlapt.local.hardware` — RAM / VRAM / CPU detection
* :mod:`nlapt.local.advisor`  — "can this machine run it" prediction
* :mod:`nlapt.local.settings` — persisted module settings (并发 etc.)
* :mod:`nlapt.local.download` — resumable GGUF downloader
* :mod:`nlapt.local.server`   — llama-server process manager
"""

from __future__ import annotations

from nlapt.local.advisor import (
    MemoryEstimate,
    RunAssessment,
    RunVerdict,
    assess,
    estimate_memory,
)
from nlapt.local.catalog import (
    CATALOG_SNAPSHOT_DATE,
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
    "GpuInfo",
    "HardwareInfo",
    "LocalServerManager",
    "LocalSettings",
    "MemoryEstimate",
    "ModelFamily",
    "ModelSeries",
    "QuantFile",
    "RunAssessment",
    "RunVerdict",
    "ServerSpec",
    "all_families",
    "all_series",
    "assess",
    "base_url",
    "build_server_args",
    "detect_hardware",
    "download_file",
    "download_url",
    "estimate_memory",
    "families_for",
    "find_family",
    "find_quant",
    "format_bytes",
    "load_local_settings",
    "mmproj_path",
    "quant_path",
    "recommended_quant",
    "repo_page_url",
    "save_local_settings",
]
