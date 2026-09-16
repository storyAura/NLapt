"""CL Tagger v2 ONNX engine (CHA 角色识别).

SigLIP2 SoViT-400m/14 @384px → multi-label Danbooru tags. The session
factory is injectable so tests never import onnxruntime. External weights
(``model.onnx.data``) must sit next to ``model.onnx``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlapt.core.errors import LocalInferenceError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.local.onnx_session import (
    InferenceSession,
    SessionFactory,
    default_session_factory,
)

_LOGGER = get_logger(__name__)

FILE_MODEL = "model.onnx"
FILE_WEIGHTS = "model.onnx.data"
FILE_VOCAB = "model_vocabulary.json"
REQUIRED_FILES: tuple[str, ...] = (FILE_MODEL, FILE_WEIGHTS, FILE_VOCAB)

IMAGE_SIZE = 384
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
IMAGE_RESCALE = 1.0 / 255.0
DEFAULT_THRESHOLD = 0.55

CATEGORY_CHARACTER = "Character"
CATEGORY_COPYRIGHT = "Copyright"
CATEGORY_GENERAL = "General"
CATEGORY_RATING = "Rating"
INPUT_PIXELS = "pixel_values"
OUTPUT_LOGITS = "logits"

MSG_MISSING_DEPS = (
    "角色识别需要 onnxruntime 与 numpy 组件 — "
    "请先运行 pip install onnxruntime numpy 后重试"
)
MSG_MISSING_PILLOW = "角色识别需要 Pillow 组件 — 请先运行 pip install Pillow 后重试"
MSG_FILE_MISSING = "角色识别模型文件缺失: {name} — 打开 设置 ▸ CHA标注 重新下载"
MSG_BAD_VOCAB = "角色识别词表损坏: {path}"


@dataclass(frozen=True)
class TagScore:
    """One predicted tag with its category and sigmoid probability."""

    tag: str
    category: str
    prob: float


@dataclass(frozen=True)
class TagResult:
    """Predicted tags grouped by category, each tuple sorted by prob desc."""

    characters: tuple[TagScore, ...] = ()
    copyrights: tuple[TagScore, ...] = ()
    general: tuple[TagScore, ...] = ()
    ratings: tuple[TagScore, ...] = ()


@dataclass(frozen=True)
class TaggerVocabulary:
    """Index → tag / category tables loaded from ``model_vocabulary.json``."""

    idx_to_tag: tuple[str, ...]
    tag_to_category: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> TaggerVocabulary:
        """Parse the official JSON; fall back to ``categories`` when needed."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LocalInferenceError(MSG_BAD_VOCAB.format(path=path)) from exc
        if not isinstance(raw, dict):
            raise LocalInferenceError(MSG_BAD_VOCAB.format(path=path))
        tags = _idx_to_tag(raw)
        categories = _categories_for(raw, tags)
        if len(categories) != len(tags):
            raise LocalInferenceError(MSG_BAD_VOCAB.format(path=path))
        return cls(tuple(tags), tuple(categories))


def _idx_to_tag(raw: dict[str, Any]) -> list[str]:
    mapping = raw.get("idx_to_tag")
    if not isinstance(mapping, dict) or not mapping:
        raise LocalInferenceError(MSG_BAD_VOCAB.format(path="idx_to_tag"))
    try:
        size = max(int(key) for key in mapping) + 1
        tags = [""] * size
        for key, value in mapping.items():
            tags[int(key)] = str(value)
    except (TypeError, ValueError) as exc:
        raise LocalInferenceError(MSG_BAD_VOCAB.format(path="idx_to_tag")) from exc
    return tags


def _categories_for(raw: dict[str, Any], tags: list[str]) -> list[str]:
    """Resolve a category name per index (tag_to_category, else categories)."""
    by_tag = _category_map(raw)
    return [by_tag.get(tag, "") for tag in tags]


def _category_map(raw: dict[str, Any]) -> dict[str, str]:
    """tag → category name. ``tag_to_category`` wins; ``categories`` fills gaps."""
    result: dict[str, str] = {}
    categories = raw.get("categories")
    if isinstance(categories, dict):
        for name, members in categories.items():
            if not isinstance(members, list):
                continue
            label = str(name)
            for tag in members:
                result[str(tag)] = label
    mapping = raw.get("tag_to_category")
    if not isinstance(mapping, dict):
        return result
    names_by_index = _category_names(raw)
    for tag, value in mapping.items():
        if isinstance(value, str) and value:
            result[str(tag)] = value
            continue
        if isinstance(value, int) and 0 <= value < len(names_by_index):
            result[str(tag)] = names_by_index[value]
    return result


def _category_names(raw: dict[str, Any]) -> tuple[str, ...]:
    """Stable index → category name when ``tag_to_category`` stores ints."""
    categories = raw.get("categories")
    if isinstance(categories, dict) and categories:
        return tuple(str(name) for name in categories)
    return (
        CATEGORY_GENERAL,
        CATEGORY_CHARACTER,
        CATEGORY_COPYRIGHT,
        "Meta",
        CATEGORY_RATING,
        "Quality",
    )


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LocalInferenceError(MSG_MISSING_DEPS) from exc
    return numpy


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LocalInferenceError(MSG_MISSING_PILLOW) from exc
    return Image


def _load_pixels(image_path: Path) -> Any:
    """SigLIP2 pixel tensor (1, 3, 384, 384) float32 for one image."""
    np = _require_numpy()
    image_module = _require_pillow()
    with image_module.open(image_path) as handle:
        rgb = handle.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), image_module.Resampling.BICUBIC
        )
        array = np.asarray(rgb, dtype=np.float32)
    array = (array * IMAGE_RESCALE - IMAGE_MEAN) / IMAGE_STD
    return array.transpose(2, 0, 1)[np.newaxis, :]


class TaggerEngine:
    """Loads the ONNX tagger once and scores images until unloaded."""

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
            session_factory if session_factory is not None else default_session_factory
        )
        self._lock = threading.Lock()
        self._session: InferenceSession | None = None
        self._vocab: TaggerVocabulary | None = None

    @property
    def files(self) -> Mapping[str, Path]:
        return dict(self._files)

    def is_loaded(self) -> bool:
        return self._session is not None and self._vocab is not None

    def load(self) -> None:
        """Create the session + vocabulary (idempotent, thread-safe)."""
        with self._lock:
            if self._session is not None and self._vocab is not None:
                return
            for name, path in self._files.items():
                if not path.is_file() or path.stat().st_size <= 0:
                    raise LocalInferenceError(MSG_FILE_MISSING.format(name=name))
            self._vocab = TaggerVocabulary.load(self._files[FILE_VOCAB])
            self._session = self._session_factory(str(self._files[FILE_MODEL]))
            _LOGGER.info("CL Tagger session loaded (%s tags)", len(self._vocab.idx_to_tag))

    def unload(self) -> None:
        """Drop the session so the ~2.2 GB weights can be released."""
        with self._lock:
            self._session = None
            self._vocab = None

    def tag(
        self, image_path: Path, threshold: float = DEFAULT_THRESHOLD
    ) -> TagResult:
        """Run one image; return tags whose sigmoid probability ≥ ``threshold``."""
        self.load()
        session = self._session
        vocab = self._vocab
        if session is None or vocab is None:  # pragma: no cover - load() guarantees
            raise LocalInferenceError(MSG_FILE_MISSING.format(name=FILE_MODEL))
        np = _require_numpy()
        pixels = _load_pixels(Path(image_path))
        with self._lock:
            outputs = session.run([OUTPUT_LOGITS], {INPUT_PIXELS: pixels})
        logits = np.asarray(outputs[0][0], dtype=np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return _split_scores(vocab, probs, threshold)


def _split_scores(
    vocab: TaggerVocabulary, probs: Any, threshold: float
) -> TagResult:
    characters: list[TagScore] = []
    copyrights: list[TagScore] = []
    general: list[TagScore] = []
    ratings: list[TagScore] = []
    limit = min(len(vocab.idx_to_tag), len(probs))
    for index in range(limit):
        prob = float(probs[index])
        if prob < threshold:
            continue
        tag = vocab.idx_to_tag[index]
        if not tag:
            continue
        category = vocab.tag_to_category[index]
        score = TagScore(tag, category, prob)
        if category == CATEGORY_CHARACTER:
            characters.append(score)
        elif category == CATEGORY_COPYRIGHT:
            copyrights.append(score)
        elif category == CATEGORY_RATING:
            ratings.append(score)
        else:
            general.append(score)
    characters.sort(key=lambda item: item.prob, reverse=True)
    copyrights.sort(key=lambda item: item.prob, reverse=True)
    general.sort(key=lambda item: item.prob, reverse=True)
    ratings.sort(key=lambda item: item.prob, reverse=True)
    return TagResult(
        characters=tuple(characters),
        copyrights=tuple(copyrights),
        general=tuple(general),
        ratings=tuple(ratings),
    )
