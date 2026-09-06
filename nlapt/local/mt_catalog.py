"""Curated Hy-MT2 GGUF catalog for the local translation backend.

A static snapshot of Tencent's official llama.cpp-ready quants. File sizes
and SHA256 digests are exact LFS oids from the HuggingFace API on
:data:`MT_CATALOG_SNAPSHOT_DATE`. Downloads land under
``models_dir/mt/<tier>/`` so they never collide with the caption-model
catalog in :mod:`nlapt.local.catalog`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from nlapt.core.errors import ValidationError
from nlapt.llm.translate import Direction
from nlapt.local.catalog import SAFE_ID_PATTERN, download_url, repo_page_url

MT_CATALOG_SNAPSHOT_DATE = "2026-09-05"
MT_DIR_NAME = "mt"

TIER_FAST = "fast"
TIER_BALANCED = "balanced"
TIER_QUALITY = "quality"
KNOWN_TIERS: tuple[str, ...] = (TIER_FAST, TIER_BALANCED, TIER_QUALITY)
DEFAULT_TIER = TIER_BALANCED

# Official Hy-MT2 default translation prompt (no system prompt).
HYMT_PROMPT = (
    "将以下文本翻译为{target_lang}，注意只需要输出翻译后的结果，不要额外解释："
    "\n\n{source_text}"
)
HYMT_TARGET_NAMES: dict[str, str] = {
    "zh": "中文",
    "en": "英语",
    "ja": "日语",
}
HYMT_DIRECTION_LANG: dict[Direction, str] = {
    Direction.EN_TO_ZH: "zh",
    Direction.ZH_TO_EN: "en",
}


@dataclass(frozen=True)
class MTModel:
    """One downloadable Hy-MT2 quantization, keyed by a quality tier."""

    tier: str
    name: str
    repo_id: str
    filename: str
    size_bytes: int
    sha256: str
    params_label: str
    notes: str = ""


ALL_MT_MODELS: tuple[MTModel, ...] = (
    MTModel(
        tier=TIER_FAST,
        name="Hy-MT2 1.8B 2bit",
        repo_id="tencent/Hy-MT2-1.8B-2Bit-GGUF",
        filename="Hy-MT2-1.8B-2Bit.gguf",
        size_bytes=600_534_880,
        sha256="dcc33bbae9b28d923c8c76a64f6157840841d26f8774f3dfd770d5fabeeb1cd7",
        params_label="1.8B",
        notes="快速档: AngelSlim 2-bit, 约 573 MB。",
    ),
    MTModel(
        tier=TIER_BALANCED,
        name="Hy-MT2 1.8B Q4_K_M",
        repo_id="tencent/Hy-MT2-1.8B-GGUF",
        filename="Hy-MT2-1.8B-Q4_K_M.gguf",
        size_bytes=1_133_080_448,
        sha256="dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699",
        params_label="1.8B",
        notes="均衡档: Q4_K_M, 约 1.1 GB。",
    ),
    MTModel(
        tier=TIER_QUALITY,
        name="Hy-MT2 7B Q4_K_M",
        repo_id="tencent/Hy-MT2-7B-GGUF",
        filename="Hy-MT2-7B-Q4_K_M.gguf",
        size_bytes=4_624_648_896,
        sha256="9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b",
        params_label="7B",
        notes="高质量档: 7B Q4_K_M, 约 4.3 GB。",
    ),
)

_MODELS_BY_TIER: dict[str, MTModel] = {model.tier: model for model in ALL_MT_MODELS}


def find_mt_model(tier: str) -> MTModel:
    """Look up a catalog entry by tier id. Raises ValidationError."""
    model = _MODELS_BY_TIER.get(tier)
    if model is None:
        raise ValidationError(f"未知的本地翻译档位: {tier!r}")
    return model


def mt_tier_dir(models_dir: Path, tier: str) -> Path:
    """Directory holding one tier's GGUF (``models_dir/mt/<tier>/``)."""
    if not SAFE_ID_PATTERN.fullmatch(tier):
        raise ValidationError(f"非法的翻译档位目录名: {tier!r}")
    return models_dir / MT_DIR_NAME / tier


def mt_model_path(models_dir: Path, model: MTModel) -> Path:
    """Local path of a tier's GGUF (basename inside the per-tier dir)."""
    return mt_tier_dir(models_dir, model.tier) / PurePosixPath(model.filename).name


def mt_download_url(model: MTModel) -> str:
    """Direct HuggingFace download URL for the tier's GGUF."""
    return download_url(model.repo_id, model.filename)


def mt_repo_page_url(model: MTModel) -> str:
    """Human-readable HuggingFace page for the tier's repo."""
    return repo_page_url(model.repo_id)


def is_mt_downloaded(models_dir: Path, model: MTModel) -> bool:
    """Whether the GGUF exists at its pinned size."""
    path = mt_model_path(models_dir, model)
    return path.is_file() and path.stat().st_size == model.size_bytes


def hymt_prompt(text: str, target_lang: str) -> str:
    """Official Hy-MT2 user prompt for ``target_lang`` (zh/en/ja)."""
    name = HYMT_TARGET_NAMES.get(target_lang)
    if name is None:
        raise ValidationError(f"target_lang must be zh/en/ja, got {target_lang!r}")
    return HYMT_PROMPT.format(target_lang=name, source_text=text)
