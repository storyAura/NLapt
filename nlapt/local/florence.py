"""Florence-2 PromptGen ONNX inference (设置 ▸ 本地推理 · 打标特化).

Florence-2 is an encoder-decoder vision model llama.cpp cannot serve, so
this engine runs the community ONNX export in-process through onnxruntime
(optional dependency, imported lazily like httpx/Pillow elsewhere). The
pipeline mirrors the reference ``Florence2Processor``:

    pixel_values -> vision_encoder -> image features
    task prompt  -> BPE tokenizer  -> embed_tokens -> text embeds
    concat -> encoder -> greedy decoder loop (merged KV-cache model)

The model is steered by task instructions (指令), not free-form prompts.
Standard Florence-2 tasks are rewritten to their fixed English questions
(:data:`TASK_PROMPTS`, copied verbatim from the official
``processing_florence2.py``); the PromptGen-only tasks are passed through
literally — exactly what the model was trained on. Generation follows the
model config: forced BOS after the decoder start token and 3-gram
repetition blocking.

Everything slow or environment-dependent is injectable: the ONNX session
factory (tests use fakes, no onnxruntime needed) and the file set (any
directory layout). All five required files are pinned in
:mod:`nlapt.local.catalog` with exact sizes and SHA256 digests.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from nlapt.core.errors import LocalInferenceError, ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# -- required files (basenames inside the family directory) -------------------------
FILE_DECODER = "decoder_model_merged.onnx"
FILE_VISION = "vision_encoder.onnx"
FILE_EMBED = "embed_tokens.onnx"
FILE_ENCODER = "encoder_model.onnx"
FILE_TOKENIZER = "tokenizer.json"
REQUIRED_FILES: tuple[str, ...] = (
    FILE_DECODER,
    FILE_VISION,
    FILE_EMBED,
    FILE_ENCODER,
    FILE_TOKENIZER,
)

# -- task instructions (指令) --------------------------------------------------------
TASK_GENERATE_TAGS = "<GENERATE_TAGS>"
TASK_CAPTION = "<CAPTION>"
TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
TASK_MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
TASK_ANALYZE = "<ANALYZE>"
TASK_MIXED_CAPTION = "<MIXED_CAPTION>"
TASK_MIXED_CAPTION_PLUS = "<MIXED_CAPTION_PLUS>"

DEFAULT_FLORENCE_TASK = TASK_GENERATE_TAGS

# token -> Chinese label for the 指令 dropdown, in display order.
FLORENCE_TASK_LABELS: dict[str, str] = {
    TASK_GENERATE_TAGS: "生成标签(danbooru 风格)",
    TASK_CAPTION: "一行简短标题",
    TASK_DETAILED_CAPTION: "结构化详细标题",
    TASK_MORE_DETAILED_CAPTION: "超详细描述",
    TASK_ANALYZE: "构图分析",
    TASK_MIXED_CAPTION: "混合标题(描述 + 标签)",
    TASK_MIXED_CAPTION_PLUS: "混合标题 + 构图分析",
}
FLORENCE_TASK_TOKENS: tuple[str, ...] = tuple(FLORENCE_TASK_LABELS)

# Standard Florence-2 tasks are rewritten to fixed English questions before
# tokenization (verbatim from the official processing_florence2.py); tasks
# missing here — the PromptGen additions — are fed to the model literally.
TASK_PROMPTS: dict[str, str] = {
    TASK_CAPTION: "What does the image describe?",
    TASK_DETAILED_CAPTION: "Describe in detail what is shown in the image.",
    TASK_MORE_DETAILED_CAPTION: "Describe with a paragraph what is shown in the image.",
}

# -- model constants (pinned export: laub/Florence-2-large-PromptGen-v2.0-onnx) -----
BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
DECODER_START_TOKEN_ID = 2  # config.text_config.decoder_start_token_id
FORCED_BOS_TOKEN_ID = 0  # config.text_config.forced_bos_token_id
NO_REPEAT_NGRAM_SIZE = 3  # config.text_config.no_repeat_ngram_size
MAX_NEW_TOKENS = 1024
# KV-cache input/output name conventions of the merged optimum export.
PAST_INPUT_PREFIX = "past_key_values."
PRESENT_OUTPUT_PREFIX = "present."

IMAGE_SIZE = 768
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
IMAGE_RESCALE = 1.0 / 255.0

# Providers tried in order when onnxruntime supports them (CPU always works;
# DirectML/CUDA appear when the user installed that onnxruntime flavour).
PREFERRED_PROVIDERS = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)

MSG_MISSING_DEPS = (
    "本地打标模型需要 onnxruntime 与 numpy 组件 — "
    "请先运行 pip install onnxruntime numpy 后重试"
)
MSG_MISSING_PILLOW = "本地打标模型需要 Pillow 组件 — 请先运行 pip install Pillow 后重试"
MSG_FILE_MISSING = "本地打标模型文件缺失: {name} — 打开 设置 ▸ 本地推理 重新下载"
MSG_BAD_TASK = "未知的打标指令: {task!r}"
MSG_BAD_TOKENIZER = "分词器文件损坏: {path}"


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LocalInferenceError(MSG_MISSING_DEPS) from exc
    return numpy


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LocalInferenceError(MSG_MISSING_DEPS) from exc
    return onnxruntime


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LocalInferenceError(MSG_MISSING_PILLOW) from exc
    return Image


def validate_task(task: str) -> str:
    """Return ``task`` unchanged if known; raise ValidationError otherwise."""
    if task not in FLORENCE_TASK_LABELS:
        raise ValidationError(MSG_BAD_TASK.format(task=task))
    return task


def prompt_for_task(task: str) -> str:
    """The literal text fed to the tokenizer for a task instruction."""
    return TASK_PROMPTS.get(validate_task(task), task)


# -- byte-level BPE tokenizer (BART vocabulary from tokenizer.json) ------------------
@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte <-> printable-unicode table (standard)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + shift)
            shift += 1
    return dict(zip(printable, (chr(code) for code in codes)))


# ASCII version of the GPT-2 pre-tokenizer split. Sufficient here: the only
# texts ever encoded are the fixed English task prompts pinned above.
# ponytail: ASCII-only pre-tokenizer; port the full \p{L} pattern if
# free-form prompts are ever exposed.
_PRETOKEN_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


class FlorenceTokenizer:
    """Minimal byte-level BPE encoder/decoder over a ``tokenizer.json`` file."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        merges: Sequence[tuple[str, str]],
        added_tokens: Mapping[str, int],
    ) -> None:
        self._vocab = dict(vocab)
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._added = dict(added_tokens)
        self._id_to_token = {tid: tok for tok, tid in self._vocab.items()}
        self._id_to_added = {tid: tok for tok, tid in self._added.items()}
        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {char: byte for byte, char in self._byte_encoder.items()}
        self._bpe_cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_file(cls, path: Path) -> FlorenceTokenizer:
        """Parse a HuggingFace fast-tokenizer ``tokenizer.json``."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            model = data["model"]
            vocab = model["vocab"]
            raw_merges = model["merges"]
            merges = [
                tuple(entry.split(" ", 1)) if isinstance(entry, str) else tuple(entry)
                for entry in raw_merges
            ]
            added = {
                entry["content"]: int(entry["id"])
                for entry in data.get("added_tokens", ())
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise LocalInferenceError(MSG_BAD_TOKENIZER.format(path=path)) from exc
        return cls(vocab, merges, added)  # type: ignore[arg-type]

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._bpe_cache.get(token)
        if cached is not None:
            return cached
        parts = tuple(token)
        while len(parts) > 1:
            pairs = {(parts[i], parts[i + 1]) for i in range(len(parts) - 1)}
            best = min(pairs, key=lambda pair: self._ranks.get(pair, float("inf")))
            if best not in self._ranks:
                break
            first, second = best
            merged: list[str] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == first and parts[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = tuple(merged)
        self._bpe_cache[token] = parts
        return parts

    def encode(self, text: str) -> list[int]:
        """Byte-level BPE ids of ``text`` (no special tokens added)."""
        ids: list[int] = []
        for piece in _PRETOKEN_RE.findall(text):
            mapped = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
            for part in self._bpe(mapped):
                token_id = self._vocab.get(part)
                if token_id is None:  # unseen byte combo: fall back per char
                    ids.extend(
                        self._vocab[char] for char in part if char in self._vocab
                    )
                else:
                    ids.append(token_id)
        return ids

    def encode_prompt(self, text: str) -> list[int]:
        """``<s> text </s>`` — how the reference processor frames prompts."""
        return [BOS_TOKEN_ID, *self.encode(text), EOS_TOKEN_ID]

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        """Text of ``ids``; drops <s>/</s>/<pad> like the pure_text post."""
        special = {BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID} if skip_special else set()
        chunks: list[str] = []
        byte_run: list[str] = []

        def flush() -> None:
            if byte_run:
                data = bytes(self._byte_decoder[char] for char in byte_run)
                chunks.append(data.decode("utf-8", errors="replace"))
                byte_run.clear()

        for token_id in ids:
            if token_id in special:
                continue
            added = self._id_to_added.get(token_id)
            if added is not None and added not in self._vocab:
                flush()
                chunks.append(added)
                continue
            token = self._id_to_token.get(token_id)
            if token is None:
                continue
            byte_run.extend(token)
        flush()
        return "".join(chunks)


# -- ONNX session plumbing -----------------------------------------------------------
class TensorSpec(Protocol):
    """Name + shape of one session input/output (onnxruntime NodeArg)."""

    name: str
    shape: Sequence[Any]


class InferenceSession(Protocol):
    """Minimal onnxruntime.InferenceSession surface the engine relies on."""

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
    ) -> list[Any]: ...

    def get_inputs(self) -> Sequence[TensorSpec]: ...

    def get_outputs(self) -> Sequence[TensorSpec]: ...


SessionFactory = Callable[[str], InferenceSession]


def _default_session_factory(path: str) -> InferenceSession:
    ort = _require_onnxruntime()
    available = set(ort.get_available_providers())
    providers = [p for p in PREFERRED_PROVIDERS if p in available]
    return ort.InferenceSession(path, providers=providers or None)


def _load_pixels(image_path: Path) -> Any:
    """CLIP-style pixel tensor (1, 3, 768, 768) float32 for one image."""
    np = _require_numpy()
    image_module = _require_pillow()
    with image_module.open(image_path) as handle:
        rgb = handle.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), image_module.Resampling.BICUBIC
        )
        array = np.asarray(rgb, dtype=np.float32)
    array = array * IMAGE_RESCALE
    array = (array - np.asarray(IMAGE_MEAN, dtype=np.float32)) / np.asarray(
        IMAGE_STD, dtype=np.float32
    )
    return array.transpose(2, 0, 1)[np.newaxis, :]


def _banned_next_tokens(generated: Sequence[int], ngram: int) -> set[int]:
    """Ids that would repeat an already-seen ``ngram``-gram (HF semantics)."""
    if ngram <= 0 or len(generated) < ngram:
        return set()
    prefix = tuple(generated[-(ngram - 1) :]) if ngram > 1 else ()
    banned: set[int] = set()
    for start in range(len(generated) - ngram + 1):
        window = tuple(generated[start : start + ngram])
        if window[: ngram - 1] == prefix:
            banned.add(window[-1])
    return banned


class FlorenceEngine:
    """Loads the ONNX pipeline once and captions images until unloaded.

    ``files`` maps each :data:`REQUIRED_FILES` basename to its local path.
    Loading is lazy and serialized; ``caption`` may be called from several
    worker threads (onnxruntime sessions are thread-safe for ``run``).
    """

    def __init__(
        self,
        files: Mapping[str, Path],
        *,
        session_factory: SessionFactory | None = None,
    ) -> None:
        missing_keys = [name for name in REQUIRED_FILES if name not in files]
        if missing_keys:
            raise ValidationError(f"缺少模型文件映射: {', '.join(missing_keys)}")
        self._files = {name: Path(files[name]) for name in REQUIRED_FILES}
        self._session_factory = (
            session_factory if session_factory is not None else _default_session_factory
        )
        self._lock = threading.Lock()
        self._sessions: dict[str, InferenceSession] = {}
        self._tokenizer: FlorenceTokenizer | None = None

    @property
    def files(self) -> Mapping[str, Path]:
        return dict(self._files)

    def is_loaded(self) -> bool:
        return self._tokenizer is not None

    def load(self) -> None:
        """Create the four sessions + tokenizer (idempotent, thread-safe)."""
        with self._lock:
            if self._tokenizer is not None:
                return
            for name, path in self._files.items():
                if not path.is_file():
                    raise LocalInferenceError(MSG_FILE_MISSING.format(name=name))
            started = time.monotonic()
            for name in (FILE_VISION, FILE_EMBED, FILE_ENCODER, FILE_DECODER):
                self._sessions[name] = self._session_factory(str(self._files[name]))
            self._tokenizer = FlorenceTokenizer.from_file(self._files[FILE_TOKENIZER])
            _LOGGER.info(
                "florence engine loaded in %.1fs (%s)",
                time.monotonic() - started,
                self._files[FILE_DECODER].parent,
            )

    def unload(self) -> None:
        """Drop sessions and tokenizer (memory back to the OS)."""
        with self._lock:
            self._sessions.clear()
            self._tokenizer = None

    # -- inference -----------------------------------------------------------------
    def caption(self, image_path: Path, task: str = DEFAULT_FLORENCE_TASK) -> str:
        """Run one image through the pipeline; returns the decoded text."""
        prompt = prompt_for_task(task)
        self.load()
        np = _require_numpy()
        tokenizer = self._tokenizer
        assert tokenizer is not None  # load() either set it or raised
        started = time.monotonic()

        pixels = _load_pixels(image_path)
        (image_features,) = self._sessions[FILE_VISION].run(
            None, {"pixel_values": pixels}
        )
        prompt_ids = np.asarray([tokenizer.encode_prompt(prompt)], dtype=np.int64)
        (text_embeds,) = self._sessions[FILE_EMBED].run(None, {"input_ids": prompt_ids})
        inputs_embeds = np.concatenate(
            [image_features.astype(np.float32), text_embeds.astype(np.float32)], axis=1
        )
        attention_mask = np.ones(inputs_embeds.shape[:2], dtype=np.int64)
        (encoder_hidden,) = self._sessions[FILE_ENCODER].run(
            None,
            {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask},
        )
        generated = self._generate(np, encoder_hidden, attention_mask)
        text = tokenizer.decode(generated).strip()
        _LOGGER.debug(
            "florence caption in %.1fs (%d tokens, task=%s)",
            time.monotonic() - started,
            len(generated),
            task,
        )
        return text

    def _generate(
        self, np: Any, encoder_hidden: Any, encoder_attention_mask: Any
    ) -> list[int]:
        """Greedy loop over the merged decoder with forced BOS + no-repeat-3.

        KV-cache layout (layer count, head count, head dim) is read from the
        decoder session itself, so the engine serves any Florence-2 export
        size without pinned architecture constants.

        ponytail: greedy decode; the reference runs num_beams=3 — add
        KV-cache beam search if tag quality measurably falls short.
        """
        decoder = self._sessions[FILE_DECODER]
        embed = self._sessions[FILE_EMBED]
        past: dict[str, Any] = {
            spec.name: np.zeros(
                (1, int(spec.shape[1]), 0, int(spec.shape[3])), dtype=np.float32
            )
            for spec in decoder.get_inputs()
            if spec.name.startswith(PAST_INPUT_PREFIX)
        }
        output_names = [spec.name for spec in decoder.get_outputs()]

        generated = [DECODER_START_TOKEN_ID]
        use_cache = False
        for _step in range(MAX_NEW_TOKENS):
            token_ids = np.asarray([[generated[-1]]], dtype=np.int64)
            (step_embeds,) = embed.run(None, {"input_ids": token_ids})
            feeds: dict[str, Any] = {
                "encoder_attention_mask": encoder_attention_mask,
                "encoder_hidden_states": encoder_hidden,
                "inputs_embeds": step_embeds.astype(np.float32),
                "use_cache_branch": np.asarray([use_cache], dtype=bool),
            }
            feeds.update(past)
            named = dict(zip(output_names, decoder.run(None, feeds)))
            logits = named["logits"]
            for name, fresh in named.items():
                if not name.startswith(PRESENT_OUTPUT_PREFIX):
                    continue
                # Cross-attention presents come back as zero-element dummies
                # on cache steps (e.g. shape (0,1,1,1)): keep the first real
                # tensor and ignore every placeholder.
                if fresh.size > 0:
                    past[name.replace(PRESENT_OUTPUT_PREFIX, PAST_INPUT_PREFIX, 1)] = (
                        fresh
                    )
            use_cache = True
            if len(generated) == 1:
                next_id = FORCED_BOS_TOKEN_ID  # forced_bos_token_id semantics
            else:
                row = logits[0, -1]
                banned = _banned_next_tokens(generated, NO_REPEAT_NGRAM_SIZE)
                if banned:
                    row = row.copy()
                    row[list(banned)] = -np.inf
                next_id = int(np.argmax(row))
            generated.append(next_id)
            if next_id == EOS_TOKEN_ID:
                break
        return generated
