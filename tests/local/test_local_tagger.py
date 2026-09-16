"""Tests for nlapt.local.tagger: vocabulary, thresholds, missing files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nlapt.core.errors import LocalInferenceError, ValidationError
from nlapt.local.tagger import (
    CATEGORY_CHARACTER,
    CATEGORY_COPYRIGHT,
    CATEGORY_GENERAL,
    CATEGORY_RATING,
    FILE_MODEL,
    FILE_VOCAB,
    FILE_WEIGHTS,
    REQUIRED_FILES,
    TagScore,
    TaggerEngine,
    TaggerVocabulary,
    _split_scores,
)

TOY_VOCAB = {
    "idx_to_tag": {
        "0": "solo",
        "1": "hatsune miku",
        "2": "vocaloid",
        "3": "general",
        "4": "sensitive",
    },
    "tag_to_category": {
        "solo": "General",
        "hatsune miku": "Character",
        "vocaloid": "Copyright",
        "general": "Rating",
        "sensitive": "Rating",
    },
    "categories": {
        "General": ["solo"],
        "Character": ["hatsune miku"],
        "Copyright": ["vocaloid"],
        "Rating": ["general", "sensitive"],
    },
}


def _write_vocab(path: Path, payload: dict[str, object] | None = None) -> Path:
    path.write_text(json.dumps(payload if payload is not None else TOY_VOCAB), encoding="utf-8")
    return path


def _files(tmp_path: Path, *, vocab: dict[str, object] | None = None) -> dict[str, Path]:
    files = {
        FILE_MODEL: tmp_path / FILE_MODEL,
        FILE_WEIGHTS: tmp_path / FILE_WEIGHTS,
        FILE_VOCAB: tmp_path / FILE_VOCAB,
    }
    files[FILE_MODEL].write_bytes(b"onnx")
    files[FILE_WEIGHTS].write_bytes(b"weights")
    _write_vocab(files[FILE_VOCAB], vocab)
    return files


class FakeSession:
    def __init__(self, logits: list[float]) -> None:
        self.logits = np.array([logits], dtype=np.float32)
        self.feeds: list[object] = []

    def run(self, names: list[str], feeds: dict[str, object]) -> list[np.ndarray]:
        assert names == ["logits"]
        self.feeds.append(feeds)
        return [self.logits]


class TestVocabulary:
    def test_load_idx_and_named_categories(self, tmp_path: Path) -> None:
        path = _write_vocab(tmp_path / "vocab.json")
        vocab = TaggerVocabulary.load(path)
        assert vocab.idx_to_tag[1] == "hatsune miku"
        assert vocab.tag_to_category[1] == CATEGORY_CHARACTER
        assert vocab.tag_to_category[2] == CATEGORY_COPYRIGHT

    def test_numeric_tag_to_category_uses_categories_order(self, tmp_path: Path) -> None:
        payload = {
            "idx_to_tag": {"0": "solo", "1": "hatsune miku"},
            "tag_to_category": {"solo": 0, "hatsune miku": 1},
            "categories": {"General": ["solo"], "Character": ["hatsune miku"]},
        }
        vocab = TaggerVocabulary.load(_write_vocab(tmp_path / "v.json", payload))
        assert vocab.tag_to_category == (CATEGORY_GENERAL, CATEGORY_CHARACTER)

    def test_categories_fill_gaps_when_map_missing(self, tmp_path: Path) -> None:
        payload = {
            "idx_to_tag": {"0": "solo", "1": "hatsune miku"},
            "categories": {"General": ["solo"], "Character": ["hatsune miku"]},
        }
        vocab = TaggerVocabulary.load(_write_vocab(tmp_path / "v.json", payload))
        assert vocab.tag_to_category[1] == CATEGORY_CHARACTER

    def test_broken_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(LocalInferenceError, match="词表损坏"):
            TaggerVocabulary.load(path)


class TestSplitScores:
    def test_threshold_and_sort(self) -> None:
        vocab = TaggerVocabulary(
            idx_to_tag=("solo", "hatsune miku", "vocaloid", "general"),
            tag_to_category=(
                CATEGORY_GENERAL,
                CATEGORY_CHARACTER,
                CATEGORY_COPYRIGHT,
                CATEGORY_RATING,
            ),
        )
        result = _split_scores(vocab, [0.9, 0.8, 0.4, 0.7], 0.55)
        assert [item.tag for item in result.characters] == ["hatsune miku"]
        assert result.copyrights == ()
        assert result.general[0] == TagScore("solo", CATEGORY_GENERAL, 0.9)
        assert result.ratings[0].tag == "general"
        ranked = _split_scores(
            vocab,
            [0.1, 0.6, 0.95, 0.2],
            0.55,
        )
        assert [item.tag for item in ranked.copyrights] == ["vocaloid"]
        assert ranked.characters[0].prob == pytest.approx(0.6)


class TestEngine:
    def test_missing_mapping_rejected(self) -> None:
        with pytest.raises(ValidationError, match="缺少模型文件"):
            TaggerEngine({FILE_MODEL: Path("x")})

    def test_missing_file_raises_actionable(self, tmp_path: Path) -> None:
        files = {name: tmp_path / name for name in REQUIRED_FILES}
        engine = TaggerEngine(files, session_factory=lambda _path: FakeSession([]))
        with pytest.raises(LocalInferenceError, match="模型文件缺失"):
            engine.load()

    def test_tag_groups_sigmoid_outputs(self, tmp_path: Path) -> None:
        from PIL import Image

        files = _files(tmp_path)
        image = tmp_path / "ref.png"
        Image.new("RGB", (16, 8), (10, 20, 30)).save(image)
        # logits → sigmoid: 2≈0.88, 1.2≈0.77, 3≈0.95, -2≈0.12, 0.8≈0.69
        session = FakeSession([2.0, 1.2, 3.0, -2.0, 0.8])
        engine = TaggerEngine(files, session_factory=lambda _path: session)
        result = engine.tag(image, threshold=0.55)
        assert [item.tag for item in result.characters] == ["hatsune miku"]
        assert result.copyrights[0].tag == "vocaloid"
        assert result.general[0].tag == "solo"
        assert [item.tag for item in result.ratings] == ["sensitive"]
        pixels = session.feeds[0]["pixel_values"]
        assert pixels.shape == (1, 3, 384, 384)
        engine.unload()
        assert engine.is_loaded() is False
