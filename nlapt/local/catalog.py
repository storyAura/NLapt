"""Curated catalog of local models (设置 ▸ 本地推理).

A static, hand-picked snapshot of HuggingFace repositories: for every family
the highest-download public GGUF conversion at snapshot time, the "heretic"
(abliteration / uncensored) derivatives, and the caption-specialist vision
models this app is built around (ToriiGate / JoyCaption / Florence-2
PromptGen).

Hierarchy: :class:`ModelSeries` (大系列) -> :class:`ModelFamily` (小系列)
-> :class:`QuantFile` (量化档). File sizes and SHA256 digests are exact
values read from the HuggingFace API (LFS oids), so
:mod:`nlapt.local.advisor` can predict memory needs and
:mod:`nlapt.local.download` can verify integrity without any extra network
round-trip. Download counts are a display-only snapshot taken on
:data:`CATALOG_SNAPSHOT_DATE`.

Families run on one of three engines: :data:`ENGINE_LLAMA` (a GGUF served by
llama-server), :data:`ENGINE_FLORENCE` (the Florence-2 ONNX pipeline of
:mod:`nlapt.local.florence`, which llama.cpp cannot serve), or
:data:`ENGINE_TAGGER` (CL Tagger ONNX, CHA 角色识别 only — registered in
``_FAMILIES_BY_ID`` but not listed in :data:`ALL_FAMILIES`, so the 本地推理
tree never offers it as a captioner). Florence / tagger families list
sibling ONNX files in ``extra_files``; every listed file downloads into the
same per-family directory.

``kv_bytes_per_token`` is a deliberately coarse fp16 K+V-cache heuristic per
family (hybrid/sliding-window attention makes exact numbers configuration
dependent); the weights dominate the estimate, so coarse is fine here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from nlapt.core.errors import ValidationError
from nlapt.local.florence import (
    TASK_ANALYZE,
    TASK_BAI_JSON,
    TASK_CAPTION,
    TASK_DETAILED_CAPTION,
    TASK_GENERATE_TAGS,
    TASK_MIXED_CAPTION,
    TASK_MIXED_CAPTION_PLUS,
    TASK_MORE_DETAILED_CAPTION,
)

# Date the download counts / file listings were captured from huggingface.co.
CATALOG_SNAPSHOT_DATE = "2026-09-14"
# Base pattern for direct file downloads from a public HuggingFace repo.
HF_RESOLVE_BASE = "https://huggingface.co/{repo_id}/resolve/main/{path}"
# Base pattern for a repo's human-readable page.
HF_REPO_PAGE_BASE = "https://huggingface.co/{repo_id}"
# "owner/name" with the character set HuggingFace actually allows.
REPO_ID_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")
# Directory-safe id: no separators, no drive letters, cannot be "." / "..".
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

SERIES_GEMMA4 = "gemma4"
SERIES_GEMMA4_HERETIC = "gemma4-heretic"
SERIES_TORIIGATE = "toriigate"
SERIES_JOYCAPTION = "joycaption"
SERIES_FLORENCE = "florence2-promptgen"
SERIES_TAGGER = "cl-tagger"

# Inference engines a family can run on.
ENGINE_LLAMA = "llama"  # single GGUF (+ optional mmproj) served by llama-server
ENGINE_FLORENCE = "florence"  # ONNX pipeline run in-process (nlapt.local.florence)
ENGINE_TAGGER = "tagger"  # ONNX multi-label tagger (nlapt.local.tagger)


@dataclass(frozen=True)
class QuantFile:
    """One downloadable quantization of a model family."""

    label: str  # e.g. "Q4_K_M"
    filename: str  # repo-relative path of the .gguf file
    size_bytes: int
    # LFS oid from the HuggingFace API; verified after download. Empty only
    # for synthetic test entries — the catalog integrity test enforces it.
    sha256: str = ""
    recommended: bool = False


@dataclass(frozen=True)
class ModelFamily:
    """One model (小系列) inside a series, with its downloadable files."""

    family_id: str
    series_id: str
    name: str
    repo_id: str
    downloads: int  # snapshot count, display only
    params_label: str  # e.g. "12B"
    vision: bool
    kv_bytes_per_token: int  # coarse fp16 K+V heuristic (see module docstring)
    quants: tuple[QuantFile, ...]
    mmproj_filename: str = ""  # repo-relative; required for ENGINE_LLAMA vision
    mmproj_bytes: int = 0
    mmproj_sha256: str = ""
    license: str = ""
    notes: str = ""
    engine: str = ENGINE_LLAMA  # which runtime serves this family
    # Additional required files beside the quant (ENGINE_FLORENCE: the
    # sibling ONNX parts + tokenizer). Downloaded/verified like quants.
    extra_files: tuple[QuantFile, ...] = ()
    # ENGINE_FLORENCE only: the 指令 tokens this family was trained on
    # (official Florence-2 knows none of the PromptGen additions). The
    # first entry is the fallback when the persisted task is unsupported.
    florence_tasks: tuple[str, ...] = ()
    # Hugging Face gated repo: download needs Authorization: Bearer <token>.
    gated: bool = False
    # False = provisional / overwritten in place: skip size + SHA256 checks.
    pinned: bool = True


@dataclass(frozen=True)
class LoraEntry:
    """A curated downloadable PEFT LoRA for Florence families (内置 LoRA).

    ``files`` are downloaded into ``models_dir/loras/<lora_id>/`` and
    verified exactly like quants; ``base_url`` is the directory URL the
    filenames are appended to (kept as a full URL: the pinned LoRA lives
    on ModelScope, not HuggingFace). ``task`` is the 指令 the LoRA was
    trained for — the UI switches to it on selection.
    """

    lora_id: str
    name: str
    base_url: str
    page_url: str
    files: tuple[QuantFile, ...]
    compatible_family_ids: tuple[str, ...]
    task: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ModelSeries:
    """A major family group (大系列) shown as a top-level catalog node."""

    series_id: str
    name: str
    description: str


ALL_SERIES: tuple[ModelSeries, ...] = (
    ModelSeries(
        series_id=SERIES_GEMMA4,
        name="Gemma 4",
        description=(
            "Google 2026-04 发布的开源多模态系列(Apache 2.0),"
            "从手机级 E2B 到工作站级 31B 共五档,均支持图片输入。"
        ),
    ),
    ModelSeries(
        series_id=SERIES_GEMMA4_HERETIC,
        name="Gemma 4 · Heretic 无审查",
        description=(
            "社区用 Heretic 消融工具削除拒答倾向的 Gemma 4 衍生版,"
            "适合处理常规模型会拒绝标注的数据集。"
        ),
    ),
    ModelSeries(
        series_id=SERIES_TORIIGATE,
        name="ToriiGate",
        description="Minthy 训练的动漫 / 插画专用标注视觉模型,支持 booru 标签与自然语言输出。",
    ),
    ModelSeries(
        series_id=SERIES_JOYCAPTION,
        name="JoyCaption",
        description="fancyfeast 的自由图片标注视觉模型,写实照片类描述效果好。",
    ),
    ModelSeries(
        series_id=SERIES_FLORENCE,
        name="Florence-2",
        description=(
            "微软 Florence-2 轻量视觉模型系列(0.23B/0.77B)及社区打标微调版"
            "PromptGen,通过内置指令输出标签 / 各级标题 / 构图分析,"
            "免 llama-server,推理时自动加载;large 档可搭配内置 LoRA。"
        ),
    ),
)


def _q(
    label: str,
    filename: str,
    size_bytes: int,
    sha256: str,
    *,
    recommended: bool = False,
) -> QuantFile:
    """Compact QuantFile constructor for the catalog literals below."""
    return QuantFile(
        label=label,
        filename=filename,
        size_bytes=size_bytes,
        sha256=sha256,
        recommended=recommended,
    )


# The 指令 sets Florence families were trained on (first = fallback).
PROMPTGEN_TASKS: tuple[str, ...] = (
    TASK_GENERATE_TAGS,
    TASK_CAPTION,
    TASK_DETAILED_CAPTION,
    TASK_MORE_DETAILED_CAPTION,
    TASK_ANALYZE,
    TASK_MIXED_CAPTION,
    TASK_MIXED_CAPTION_PLUS,
)
OFFICIAL_TASKS: tuple[str, ...] = (
    TASK_MORE_DETAILED_CAPTION,
    TASK_DETAILED_CAPTION,
    TASK_CAPTION,
)
# large-arch officials additionally accept the BAI_JSON LoRA's 指令.
OFFICIAL_LARGE_TASKS: tuple[str, ...] = OFFICIAL_TASKS + (TASK_BAI_JSON,)

# Every official Florence-2 export ships the identical BART tokenizer
# (verified byte-identical across all four onnx-community repos).
_FLORENCE_TOKENIZER_SHA256 = (
    "d69dcdb2323e124ac4f800cb9863ddccea0d7bb11e16125e8df3bd60f2f8aeac"
)


def _florence_tokenizer() -> QuantFile:
    return _q("tokenizer", "tokenizer.json", 2_297_961, _FLORENCE_TOKENIZER_SHA256)


ALL_FAMILIES: tuple[ModelFamily, ...] = (
    # -- Gemma 4 (unsloth conversions: top downloads AND ship mmproj) --------------
    ModelFamily(
        family_id="gemma4-e2b",
        series_id=SERIES_GEMMA4,
        name="Gemma 4 E2B",
        repo_id="unsloth/gemma-4-E2B-it-GGUF",
        downloads=439_210,
        params_label="E2B",
        vision=True,
        kv_bytes_per_token=48_000,
        quants=(
            _q(
                "UD-Q2_K_XL",
                "gemma-4-E2B-it-UD-Q2_K_XL.gguf",
                2_403_614_816,
                "2bfeb49803da8db274b3fbac3c1d471903be64d382c237c6f509ccaa9cc141a2",
            ),
            _q(
                "Q4_K_M",
                "gemma-4-E2B-it-Q4_K_M.gguf",
                3_106_738_272,
                "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "gemma-4-E2B-it-Q8_0.gguf",
                5_048_352_864,
                "605d3c2647d7c58c1e4b5375ccb5702acf94c2611b4c8d4877812f8fdd32d053",
            ),
        ),
        mmproj_filename="mmproj-F16.gguf",
        mmproj_bytes=985_654_080,
        mmproj_sha256="140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa",
        license="Apache 2.0",
        notes="手机 / 核显级,约 2.3B 有效参数",
    ),
    ModelFamily(
        family_id="gemma4-e4b",
        series_id=SERIES_GEMMA4,
        name="Gemma 4 E4B",
        repo_id="unsloth/gemma-4-E4B-it-GGUF",
        downloads=504_402,
        params_label="E4B",
        vision=True,
        kv_bytes_per_token=64_000,
        quants=(
            _q(
                "UD-Q2_K_XL",
                "gemma-4-E4B-it-UD-Q2_K_XL.gguf",
                3_757_419_648,
                "cc92186419be169e992a1df01978828a0d4cf3a5962379ff403c8d55f6faff78",
            ),
            _q(
                "Q4_K_M",
                "gemma-4-E4B-it-Q4_K_M.gguf",
                4_977_171_584,
                "85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "gemma-4-E4B-it-Q8_0.gguf",
                8_192_953_472,
                "f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012",
            ),
        ),
        mmproj_filename="mmproj-F16.gguf",
        mmproj_bytes=990_372_672,
        mmproj_sha256="ddf46c21d7078e95338cfc22306b19b276a29a5ad089023449dd54d4b6170a51",
        license="Apache 2.0",
        notes="轻薄本级,约 4.5B 有效参数",
    ),
    ModelFamily(
        family_id="gemma4-12b",
        series_id=SERIES_GEMMA4,
        name="Gemma 4 12B",
        repo_id="unsloth/gemma-4-12b-it-GGUF",
        downloads=640_422,
        params_label="12B",
        vision=True,
        kv_bytes_per_token=96_000,
        quants=(
            _q(
                "UD-Q2_K_XL",
                "gemma-4-12b-it-UD-Q2_K_XL.gguf",
                4_661_419_840,
                "aa6e1ccf5cd1c5340c4786f0bda2478181327f14aba2359e51e455d02c90bff3",
            ),
            _q(
                "Q4_K_M",
                "gemma-4-12b-it-Q4_K_M.gguf",
                7_121_861_440,
                "0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "gemma-4-12b-it-Q8_0.gguf",
                12_669_647_680,
                "f20e7ff1be28c283eeeb18fc895733791c56a5851d5cd3fe9691b7f7d12afa72",
            ),
        ),
        mmproj_filename="mmproj-F16.gguf",
        mmproj_bytes=175_115_840,
        mmproj_sha256="91f086971e56d7a7d8d39e271873fccdb49541bd259d6e02c401a4f1cb7a219e",
        license="Apache 2.0",
        notes="统一多模态架构(图像 / 音频免编码器)",
    ),
    ModelFamily(
        family_id="gemma4-26b-a4b",
        series_id=SERIES_GEMMA4,
        name="Gemma 4 26B A4B",
        repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        downloads=1_491_605,
        params_label="26B MoE",
        vision=True,
        kv_bytes_per_token=128_000,
        quants=(
            _q(
                "UD-Q2_K_XL",
                "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf",
                10_546_934_240,
                "2a1d26dfe6ea00a467940a5728316af6edb366bbdba950d65b85d232392fb658",
            ),
            _q(
                "UD-Q4_K_M",
                "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                16_947_541_728,
                "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "gemma-4-26B-A4B-it-Q8_0.gguf",
                26_859_861_728,
                "5f7cbd0f4564e84342fc34321a09acb54b1a3da9215124e5bf444baa6dda152c",
            ),
        ),
        mmproj_filename="mmproj-F16.gguf",
        mmproj_bytes=1_193_058_784,
        mmproj_sha256="418a6d8723067cd712235facbbc5cba6c8fbbd413fc1292d2aace5a027d5a42f",
        license="Apache 2.0",
        notes="MoE:26B 总参 · 每 token 仅 3.8B 激活,内存换速度,CPU / 混合推理性价比高",
    ),
    ModelFamily(
        family_id="gemma4-31b",
        series_id=SERIES_GEMMA4,
        name="Gemma 4 31B",
        repo_id="unsloth/gemma-4-31B-it-GGUF",
        downloads=493_405,
        params_label="31B",
        vision=True,
        kv_bytes_per_token=192_000,
        quants=(
            _q(
                "UD-Q2_K_XL",
                "gemma-4-31B-it-UD-Q2_K_XL.gguf",
                11_774_991_296,
                "3c0f374d3bc5d3d8c26adf27535354404f144922cecb8c57d7967647d56f17f3",
            ),
            _q(
                "Q4_K_M",
                "gemma-4-31B-it-Q4_K_M.gguf",
                18_323_733_440,
                "38bd64c852c4b460434cc7162fa9bdcf242faf86502581a754cb72956bb17f84",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "gemma-4-31B-it-Q8_0.gguf",
                32_635_677_632,
                "d5808e5874e660a85ab45b2da00c9e3b4a003621249a333772232d1a703e4d67",
            ),
        ),
        mmproj_filename="mmproj-F16.gguf",
        mmproj_bytes=1_198_957_024,
        mmproj_sha256="6edcca228213c28d3567a35d22f849eea52d8360875093851959adf5d2f270eb",
        license="Apache 2.0",
        notes="稠密旗舰,质量最高,需要大显存 / 大内存",
    ),
    # -- Gemma 4 Heretic (uncensored derivatives, highest-download conversions) ----
    ModelFamily(
        family_id="gemma4-e4b-heretic",
        series_id=SERIES_GEMMA4_HERETIC,
        name="Gemma 4 E4B Heretic",
        repo_id="llmfan46/gemma-4-E4B-it-ultra-uncensored-heretic-GGUF",
        downloads=90_999,
        params_label="E4B",
        vision=True,
        kv_bytes_per_token=64_000,
        quants=(
            _q(
                "Q4_K_M",
                "gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf",
                5_335_289_696,
                "f9dbbe3bdf396f65aec4e801516a1b868a510d96225417a8aa06763e3c5b97dc",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "gemma-4-E4B-it-ultra-uncensored-heretic-Q8_0.gguf",
                8_031_240_032,
                "c1543610bcceeba8f51bd7cf38f373d2a002265f8873149aecae8a200b8fe4ba",
            ),
        ),
        mmproj_filename="gemma-4-E4B-it-mmproj-BF16.gguf",
        mmproj_bytes=991_552_000,
        mmproj_sha256="ffa64aebf7144bbadd5a1e143b77794d88e76a81cc0168d91de0770eb49e88e0",
        license="Apache 2.0",
        notes="Heretic 去审查版 · 轻薄本级",
    ),
    ModelFamily(
        family_id="gemma4-12b-heretic",
        series_id=SERIES_GEMMA4_HERETIC,
        name="Gemma 4 12B Heretic",
        repo_id="culturerevolt/gemma-4-12b-heretic-abliterated-GGUF",
        downloads=123_468,
        params_label="12B",
        vision=True,
        kv_bytes_per_token=96_000,
        quants=(
            _q(
                "IQ3_XS",
                "gemma-4-12b-heretic-IQ3_XS.gguf",
                5_272_393_056,
                "e3b989172aeca98f32200c8031c603cd7191d8d487acd3cfde59c26ba4d4f4b7",
            ),
            _q(
                "Q4_K_M",
                "gemma-4-12b-heretic-Q4_K_M.gguf",
                7_381_382_496,
                "6c4067ea0210d2367b2dbdd460d2dd86032a9b6e8dcbe03b83a3ea0a0a16dbee",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "gemma-4-12b-heretic-abliterated-Q8_0.gguf",
                12_669_645_440,
                "e4734aeb71209e9595eaac1b46b6370f4329cad4515ab449a788cb3b04745dec",
            ),
        ),
        mmproj_filename="gemma-4-12b-heretic-mmproj-f16.gguf",
        mmproj_bytes=175_115_840,
        mmproj_sha256="2e269f906eb15169ee9ce880ea649bd6d42d4964c21f8ede10d0d0efc738bcbb",
        license="Apache 2.0",
        notes="Heretic 去审查 + abliterated 版",
    ),
    ModelFamily(
        family_id="gemma4-26b-a4b-heretic",
        series_id=SERIES_GEMMA4_HERETIC,
        name="Gemma 4 26B A4B Heretic",
        repo_id="mradermacher/gemma-4-26B-A4B-it-ultra-uncensored-heretic-i1-GGUF",
        downloads=166_027,
        params_label="26B MoE",
        vision=False,
        kv_bytes_per_token=128_000,
        quants=(
            _q(
                "i1-IQ2_M",
                "gemma-4-26B-A4B-it-ultra-uncensored-heretic.i1-IQ2_M.gguf",
                10_377_711_648,
                "200ebc94ea16a77680a95fddc706bb49bd55bd677cc0a9cf61c933c6912d4261",
            ),
            _q(
                "i1-Q4_K_M",
                "gemma-4-26B-A4B-it-ultra-uncensored-heretic.i1-Q4_K_M.gguf",
                16_796_012_064,
                "4f73afa4aafc5a984e337ff4a878ae43696d92f011633a13bf69361e7fab83bb",
                recommended=True,
            ),
            _q(
                "i1-Q6_K",
                "gemma-4-26B-A4B-it-ultra-uncensored-heretic.i1-Q6_K.gguf",
                22_638_395_424,
                "e365decf35a07f78a078378b51d290c8e053a9e120fbb63825f3c4f8ff448b36",
            ),
        ),
        license="Apache 2.0",
        notes="Heretic 去审查版 MoE · 该仓库未提供视觉 mmproj,仅文本",
    ),
    ModelFamily(
        family_id="gemma4-31b-heretic",
        series_id=SERIES_GEMMA4_HERETIC,
        name="Gemma 4 31B Heretic",
        repo_id="llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF",
        downloads=84_391,
        params_label="31B",
        vision=True,
        kv_bytes_per_token=192_000,
        quants=(
            _q(
                "Q3_K_M",
                "gemma-4-31B-it-uncensored-heretic-Q3_K_M.gguf",
                15_287_108_736,
                "c0e2810df3603c8bcd103f9b7bbe0fa76c1422df8724c2989d6091641f5a4a10",
            ),
            _q(
                "Q4_K_M",
                "gemma-4-31B-it-uncensored-heretic-Q4_K_M.gguf",
                18_687_063_168,
                "7c65a35e7c4e53cba6c5e02cc9eeb850eb4251f4d9ad120c2caa6de23c5a6395",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "gemma-4-31B-it-uncensored-heretic-Q8_0.gguf",
                32_635_675_776,
                "fc1096ef2a43023469beecaa5a7fa3ba804d3f1684ade80f8f083e7e27a1028e",
            ),
        ),
        mmproj_filename="gemma-4-31B-it-mmproj-BF16.gguf",
        mmproj_bytes=1_200_726_208,
        mmproj_sha256="21487ff26d08f7ddd1d654d3bbfc1ae1020aab3119f5bf654742ce4697732e4e",
        license="Apache 2.0",
        notes="Heretic 去审查版稠密旗舰",
    ),
    # -- Caption specialists -------------------------------------------------------
    ModelFamily(
        family_id="toriigate-0.5",
        series_id=SERIES_TORIIGATE,
        name="ToriiGate 0.5",
        repo_id="DraconicDragon/ToriiGate-0.5-GGUF",
        downloads=8_574,
        params_label="4B",
        vision=True,
        kv_bytes_per_token=147_456,
        quants=(
            _q(
                "Q4_K_M",
                "ToriiGate-0.5-Q4_K_M.gguf",
                3_066_382_176,
                "1dd18497b1a1ee19e5ca63a58efb4614bfd24abf80ee7fd796bbbe3bc1e20cef",
                recommended=True,
            ),
            _q(
                "Q6_K",
                "ToriiGate-0.5-Q6_K.gguf",
                3_985_524_576,
                "2452c180427a127fe536e03c00cf5249f38c1378ae4d17a29e1a9f2371b5a852",
            ),
            _q(
                "Q8_0",
                "ToriiGate-0.5-Q8_0.gguf",
                5_157_830_496,
                "2b21dce659e9ba92ebf481468b7cea131e614f162fc43ddb8c834ffaf4364020",
            ),
        ),
        mmproj_filename="ToriiGate-0.5-fp16.mmproj.gguf",
        mmproj_bytes=672_423_360,
        mmproj_sha256="e0470d4bff4e932ecf3c7bf3b937d51301722c5d17c43f7ad9dfc73b65808b9c",
        license="MIT",
        notes="基于 Qwen3.5-4B · 动漫 / 插画标注特化(原仓库 Minthy/ToriiGate-0.5)",
    ),
    ModelFamily(
        family_id="joycaption-beta-one",
        series_id=SERIES_JOYCAPTION,
        name="JoyCaption Beta One",
        repo_id="concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
        downloads=7_977,
        params_label="8B",
        vision=True,
        kv_bytes_per_token=131_072,
        quants=(
            _q(
                "Q4_K",
                "Llama-Joycaption-Beta-One-Hf-Llava-Q4_K.gguf",
                4_920_735_936,
                "e8ae55dd07e61d541ab741d6ed63e7810192cea65d7ef8cda69b2a99fb06dc15",
                recommended=True,
            ),
            _q(
                "Q8_0",
                "Llama-Joycaption-Beta-One-Hf-Llava-Q8_0.gguf",
                8_540_772_544,
                "914fccbc28d0bdf87bab2937f58707bbc33556021664cd16a9666607d4058b99",
            ),
        ),
        mmproj_filename="llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
        mmproj_bytes=877_771_808,
        mmproj_sha256="94002cb5c354c7c9e538e64f37d593db9eceeca2e94573bae6cd3b2bd8bb1952",
        license="Llama 3.1",
        notes="基于 Llama-3.1-8B LLaVA · 写实图片描述",
    ),
    # -- Florence-2 PromptGen (ONNX engine, see nlapt.local.florence) --------------
    ModelFamily(
        family_id="florence2-promptgen-v2",
        series_id=SERIES_FLORENCE,
        name="Florence-2 PromptGen v2.0",
        # The only FUNCTIONAL community ONNX export of MiaoshouAI's PromptGen
        # v2.0 (llama.cpp cannot serve the Florence-2 architecture). Despite
        # the repo name, the exported weights are the base-size (0.23B)
        # architecture — verified from the graph dims; no large-size ONNX
        # export exists anywhere at snapshot time.
        repo_id="laub/Florence-2-large-PromptGen-v2.0-onnx",
        downloads=11_085,  # snapshot of the original MiaoshouAI repo (热度)
        params_label="0.23B",
        vision=True,
        kv_bytes_per_token=36_864,  # fp32 K+V of the 6-layer BART decoder
        quants=(
            _q(
                "ONNX",
                "onnx/decoder_model_merged.onnx",
                388_209_807,
                "4bd1dce482d3df8f6c592b248b0de8e62780d2ad5a3f3ae9b3d42b321b83a678",
                recommended=True,
            ),
        ),
        license="MIT",
        notes=(
            "打标特化 · 指令驱动(标签 / 标题 / 构图分析),免 llama-server;"
            "社区 ONNX 导出为 base 架构(仓库名标 large,实为 0.23B;"
            "large 版暂无可用 ONNX)"
        ),
        engine=ENGINE_FLORENCE,
        extra_files=(
            _q(
                "vision_encoder",
                "onnx/vision_encoder.onnx",
                366_564_017,
                "0894e2fd104d64e31d0fc22dd2dfba498f0455bcb5416534972777943c38ac17",
            ),
            _q(
                "embed_tokens",
                "onnx/embed_tokens.onnx",
                157_560_044,
                "b95f49725068d0addbb3c0522d842cb4d9e968106563583deccae3a462378ee1",
            ),
            _q(
                "encoder_model",
                "onnx/encoder_model.onnx",
                173_380_907,
                "3243b162d86802e572969581362b3efeba8329dbb8da1e2ff4fb053913e6c2df",
            ),
            _florence_tokenizer(),
        ),
        florence_tasks=PROMPTGEN_TASKS,
    ),
    # -- official Microsoft Florence-2 exports (onnx-community, fp32) --------------
    ModelFamily(
        family_id="florence2-base-ft",
        series_id=SERIES_FLORENCE,
        name="Florence-2 base-ft(官方)",
        repo_id="onnx-community/Florence-2-base-ft",
        downloads=3_634,
        params_label="0.23B",
        vision=True,
        kv_bytes_per_token=36_864,  # fp32 K+V of the 6-layer BART decoder
        quants=(
            _q(
                "ONNX",
                "onnx/decoder_model_merged.onnx",
                388_421_753,
                "5207affad8815294233b8679ee9ecb614906f819a1890d95a01b9ca68c392a79",
                recommended=True,
            ),
        ),
        license="MIT",
        notes="微软官方 base 任务微调版 · 标准标题三档指令",
        engine=ENGINE_FLORENCE,
        extra_files=(
            _q(
                "vision_encoder",
                "onnx/vision_encoder.onnx",
                366_549_825,
                "d67258cdfdebfa21285dad9e7bd4bd99725236d0aaef9e474a1b24a6ec471351",
            ),
            _q(
                "embed_tokens",
                "onnx/embed_tokens.onnx",
                157_560_044,
                "90cae3deb6406938c676a35b5246db02b478c9cc8cf93508361be80c05babf95",
            ),
            _q(
                "encoder_model",
                "onnx/encoder_model.onnx",
                173_380_723,
                "cb0bccc232c64290397f5e1235eb3e1fa6ccf8c5afed9216480ee4eed80737fc",
            ),
            _florence_tokenizer(),
        ),
        florence_tasks=OFFICIAL_TASKS,
    ),
    ModelFamily(
        family_id="florence2-large-ft",
        series_id=SERIES_FLORENCE,
        name="Florence-2 large-ft(官方)",
        repo_id="onnx-community/Florence-2-large-ft",
        downloads=1_088,
        params_label="0.77B",
        vision=True,
        kv_bytes_per_token=98_304,  # fp32 K+V of the 12-layer 1024-dim decoder
        quants=(
            _q(
                "ONNX",
                "onnx/decoder_model_merged.onnx",
                1_021_996_421,
                "a35016ed99f4260f584bdef0602617cf344dafe8d72631d2a7e5933f105f4ea6",
                recommended=True,
            ),
        ),
        license="MIT",
        notes="微软官方 large 任务微调版 · 标准标题三档指令",
        engine=ENGINE_FLORENCE,
        extra_files=(
            _q(
                "vision_encoder",
                "onnx/vision_encoder.onnx",
                1_453_500_335,
                "cbe2b9e3bfd5f00f9c135c3c48e4e8a2ad2b35651309d213dbf0803cc862d491",
            ),
            _q(
                "embed_tokens",
                "onnx/embed_tokens.onnx",
                210_080_043,
                "8497716365d7636307b334a665b0fe3482c9dfc3cc3e31c1d110c25ec183e31d",
            ),
            _q(
                "encoder_model",
                "onnx/encoder_model.onnx",
                609_049_107,
                "f4622b5203e353c0e660f8d2ccd50a1179a092cd2408b208bc9539f1572a321d",
            ),
            _florence_tokenizer(),
        ),
        florence_tasks=OFFICIAL_LARGE_TASKS,
    ),
    ModelFamily(
        family_id="florence2-base",
        series_id=SERIES_FLORENCE,
        name="Florence-2 base(官方)",
        repo_id="onnx-community/Florence-2-base",
        downloads=3_014,
        params_label="0.23B",
        vision=True,
        kv_bytes_per_token=36_864,
        quants=(
            _q(
                "ONNX",
                "onnx/decoder_model_merged.onnx",
                388_421_910,
                "6d6e1266d7f94f5d4ec9cc07d9c1f7b3e47049c9b0de7bbe82a91e62dfd152af",
                recommended=True,
            ),
        ),
        license="MIT",
        notes="微软官方 base 预训练版(未任务微调,建议优先 -ft)",
        engine=ENGINE_FLORENCE,
        extra_files=(
            _q(
                "vision_encoder",
                "onnx/vision_encoder.onnx",
                366_591_558,
                "2c7464fce495ea43b415b48afe7dbe84a9ebbe0cfe3c31fdd81c6dd66b39ff75",
            ),
            _q(
                "embed_tokens",
                "onnx/embed_tokens.onnx",
                157_560_107,
                "fec0fd20276af861afb6a23a11544bf1378c05e2a41001b610cc281e244c4b20",
            ),
            _q(
                "encoder_model",
                "onnx/encoder_model.onnx",
                173_380_723,
                "b155b5a0e56a4244c62060751bb1b70dfe481015d0dbfa6d19b1912d9e58da0d",
            ),
            _florence_tokenizer(),
        ),
        florence_tasks=OFFICIAL_TASKS,
    ),
    ModelFamily(
        family_id="florence2-large",
        series_id=SERIES_FLORENCE,
        name="Florence-2 large(官方)",
        repo_id="onnx-community/Florence-2-large",
        downloads=53,
        params_label="0.77B",
        vision=True,
        kv_bytes_per_token=98_304,
        quants=(
            _q(
                "ONNX",
                "onnx/decoder_model_merged.onnx",
                1_021_996_421,
                "b8cc87fea29465237fc9980ee9cf4ef5c6a3e197d78ae6fd1e0b31c20bf7fd80",
                recommended=True,
            ),
        ),
        license="MIT",
        notes="微软官方 large 预训练版 · BAI_JSON LoRA 的推荐载体(直系底座)",
        engine=ENGINE_FLORENCE,
        extra_files=(
            _q(
                "vision_encoder",
                "onnx/vision_encoder.onnx",
                1_453_500_335,
                "3579b2a9fe0e73d80a6e6152c610c80fed3739bc9e386145740ac41a0ac0dc05",
            ),
            _q(
                "embed_tokens",
                "onnx/embed_tokens.onnx",
                210_080_043,
                "af0a7ec8918e556c4f3b7931271b8cf882db2024e2b777955daa053ee6a881b3",
            ),
            _q(
                "encoder_model",
                "onnx/encoder_model.onnx",
                609_049_107,
                "3046c1c8abee3c1d5ed667db738035a4abbe2793c5298683a8ccf1d6db5f4170",
            ),
            _florence_tokenizer(),
        ),
        florence_tasks=OFFICIAL_LARGE_TASKS,
    ),
)

# CHA 角色识别 (optional). Not in ALL_FAMILIES / ALL_SERIES — the 本地推理
# tree must not offer it as a caption model. find_family still resolves it.
TAGGER_FAMILY_ID = "cl-tagger-v2"
TAGGER_QUANT_LABEL = "ONNX"
CHARACTER_CSV_FILENAME = "danbooru_character_tags.csv"
CHARACTER_CSV_URL = (
    "https://huggingface.co/datasets/StoryAura/Danbooru-Dataset-csv/"
    "resolve/main/danbooru_character_tags.csv"
)
TAGGER_FAMILY = ModelFamily(
    family_id=TAGGER_FAMILY_ID,
    series_id=SERIES_TAGGER,
    name="CL Tagger v2.01a",
    repo_id="cella110n/cl_tagger_v2",
    downloads=1,
    params_label="400M",
    vision=True,
    kv_bytes_per_token=1,
    quants=(
        _q("ONNX", "v2_01a/model.onnx.data", 2_211_645_300, "", recommended=True),
    ),
    license="CL Tagger v2 Model License v1.0（禁止再分发）",
    notes=(
        "受限模型,需 Hugging Face Token;v2_01a 为暂定版,官方可同名覆盖,"
        "故不校验大小/SHA256。角色识别可选模块,不作标注引擎。"
    ),
    engine=ENGINE_TAGGER,
    extra_files=(
        _q("model", "v2_01a/model.onnx", 791_773, ""),
        _q("vocabulary", "v2_01a/model_vocabulary.json", 14_594_140, ""),
    ),
    gated=True,
    pinned=False,
)

_FAMILIES_BY_ID: dict[str, ModelFamily] = {f.family_id: f for f in ALL_FAMILIES}
_FAMILIES_BY_ID[TAGGER_FAMILY.family_id] = TAGGER_FAMILY

# -- curated LoRAs (内置 LoRA, downloadable like models) ------------------------------
# Per-user directory name (inside models_dir) holding downloaded LoRAs.
LORA_DIR_NAME = "loras"

ALL_LORAS: tuple[LoraEntry, ...] = (
    LoraEntry(
        lora_id="bai-json-large",
        name="BAI_JSON(JSON 结构化打标)",
        base_url=(
            "https://modelscope.cn/models/silverlong/"
            "Florence-2-large-PromptGen-v2.0-BAI_JSON-LoRa/resolve/master/"
            "ep-50/lora/"
        ),
        page_url=(
            "https://modelscope.cn/models/silverlong/"
            "Florence-2-large-PromptGen-v2.0-BAI_JSON-LoRa"
        ),
        files=(
            _q(
                "adapter",
                "adapter_model.safetensors",
                106_778_680,
                "fc86f3ce5e9a8f759ef069a70e80ad47a712c0d16ca09ec2a6dc00aa56fcbb12",
            ),
            _q(
                "config",
                "adapter_config.json",
                1_170,
                "103608ef8350f9c07b9b7d288aed7f4369b3dfbccb83749fc3c8d0b76efd3ed0",
            ),
        ),
        # Trained on large-PromptGen v2.0 (1024-dim) — only the large-arch
        # families can merge it. PromptGen v2.0 itself has no ONNX export;
        # its direct parent (microsoft/Florence-2-large, per its config's
        # _name_or_path) is the best carrier: real-image runs produce clean
        # structured JSON there, while the sibling large-ft finetune
        # degenerates into token soup. Order = recommendation.
        compatible_family_ids=("florence2-large", "florence2-large-ft"),
        task=TASK_BAI_JSON,
        notes=(
            "silverlong 训练 · 指令 <BAI_JSON> 输出 JSON 结构化描述;"
            "推荐搭配 Florence-2 large(其直系底座),large-ft 上效果差;"
            "原始底座 large-PromptGen v2.0 无 ONNX,JSON 语法偶有小瑕疵"
        ),
    ),
)

_LORAS_BY_ID: dict[str, LoraEntry] = {entry.lora_id: entry for entry in ALL_LORAS}


def all_series() -> tuple[ModelSeries, ...]:
    """Every catalog series (大系列), in display order."""
    return ALL_SERIES


def families_for(series_id: str) -> tuple[ModelFamily, ...]:
    """The families (小系列) of one series, in catalog order."""
    return tuple(f for f in ALL_FAMILIES if f.series_id == series_id)


def all_families() -> tuple[ModelFamily, ...]:
    """Every family across all series, in catalog order."""
    return ALL_FAMILIES


def find_family(family_id: str) -> ModelFamily:
    """Look up a family by id. Raises ValidationError for unknown ids."""
    family = _FAMILIES_BY_ID.get(family_id)
    if family is None:
        raise ValidationError(f"未知的本地模型: {family_id!r}")
    return family


def find_quant(family: ModelFamily, label: str) -> QuantFile:
    """Look up a quant by label inside a family. Raises ValidationError."""
    for quant in family.quants:
        if quant.label == label:
            return quant
    raise ValidationError(f"模型 {family.family_id} 没有量化档 {label!r}")


def recommended_quant(family: ModelFamily) -> QuantFile:
    """The family's recommended quant (first flagged, else first listed)."""
    for quant in family.quants:
        if quant.recommended:
            return quant
    return family.quants[0]


def download_url(repo_id: str, filename: str) -> str:
    """Direct HuggingFace download URL for a repo-relative file path."""
    if not repo_id or not filename:
        raise ValidationError("repo_id 与 filename 不能为空")
    if not REPO_ID_PATTERN.fullmatch(repo_id):
        raise ValidationError(f"非法的仓库 ID: {repo_id!r}")
    encoded = "/".join(quote(part) for part in PurePosixPath(filename).parts)
    return HF_RESOLVE_BASE.format(repo_id=repo_id, path=encoded)


def repo_page_url(repo_id: str) -> str:
    """Human-readable HuggingFace page URL for a repo."""
    if not REPO_ID_PATTERN.fullmatch(repo_id):
        raise ValidationError(f"非法的仓库 ID: {repo_id!r}")
    return HF_REPO_PAGE_BASE.format(repo_id=repo_id)


def all_loras() -> tuple[LoraEntry, ...]:
    """Every curated LoRA, in catalog order."""
    return ALL_LORAS


def find_lora(lora_id: str) -> LoraEntry:
    """Look up a curated LoRA by id. Raises ValidationError for unknown ids."""
    entry = _LORAS_BY_ID.get(lora_id)
    if entry is None:
        raise ValidationError(f"未知的内置 LoRA: {lora_id!r}")
    return entry


def loras_for_family(family_id: str) -> tuple[LoraEntry, ...]:
    """The curated LoRAs merged-compatible with one family."""
    return tuple(
        entry for entry in ALL_LORAS if family_id in entry.compatible_family_ids
    )


def lora_dir(models_dir: Path, entry: LoraEntry) -> Path:
    """Local directory holding one curated LoRA's files."""
    if not SAFE_ID_PATTERN.fullmatch(entry.lora_id):
        raise ValidationError(f"非法的 LoRA 目录名: {entry.lora_id!r}")
    return models_dir / LORA_DIR_NAME / entry.lora_id


def lora_file_path(models_dir: Path, entry: LoraEntry, file: QuantFile) -> Path:
    """Local path of one LoRA file (basename inside the per-LoRA dir)."""
    return lora_dir(models_dir, entry) / PurePosixPath(file.filename).name


def lora_adapter_path(models_dir: Path, entry: LoraEntry) -> Path:
    """The .safetensors file of a curated LoRA — what the engine loads."""
    for file in entry.files:
        if file.filename.endswith(".safetensors"):
            return lora_file_path(models_dir, entry, file)
    raise ValidationError(f"内置 LoRA {entry.lora_id} 缺少 safetensors 文件")


def lora_download_url(entry: LoraEntry, file: QuantFile) -> str:
    """Direct download URL of one LoRA file (base_url + encoded filename)."""
    encoded = "/".join(quote(part) for part in PurePosixPath(file.filename).parts)
    return entry.base_url + encoded


def family_dir(models_dir: Path, family: ModelFamily) -> Path:
    """Local directory holding one family's files (namespaced by family id).

    The id is validated against :data:`SAFE_ID_PATTERN` so a malformed
    family can never escape ``models_dir`` (defense in depth — catalog ids
    are hardcoded today, but this function must stay safe if the catalog
    ever gains an external source).
    """
    if not SAFE_ID_PATTERN.fullmatch(family.family_id):
        raise ValidationError(f"非法的模型目录名: {family.family_id!r}")
    return models_dir / family.family_id


def quant_path(models_dir: Path, family: ModelFamily, quant: QuantFile) -> Path:
    """Local path of a quant file (basename of the repo path, per-family dir)."""
    return family_dir(models_dir, family) / PurePosixPath(quant.filename).name


def mmproj_path(models_dir: Path, family: ModelFamily) -> Path | None:
    """Local path of the family's mmproj file, or None for text-only families."""
    if not family.vision or not family.mmproj_filename:
        return None
    return family_dir(models_dir, family) / PurePosixPath(family.mmproj_filename).name
