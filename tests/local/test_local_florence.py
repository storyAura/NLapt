"""Tests for nlapt.local.florence: tokenizer, task prompts, ONNX pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nlapt.core.errors import LocalInferenceError, ValidationError
from nlapt.local.florence import (
    DEFAULT_FLORENCE_TASK,
    FILE_DECODER,
    FILE_EMBED,
    FILE_ENCODER,
    FILE_TOKENIZER,
    FILE_VISION,
    FLORENCE_TASK_LABELS,
    FLORENCE_TASK_TOKENS,
    REQUIRED_FILES,
    TASK_CAPTION,
    TASK_DETAILED_CAPTION,
    TASK_GENERATE_TAGS,
    TASK_MIXED_CAPTION_PLUS,
    TASK_MORE_DETAILED_CAPTION,
    FlorenceEngine,
    FlorenceTokenizer,
    _banned_next_tokens,
    _bytes_to_unicode,
    prompt_for_task,
    validate_task,
)

EMB = 8
VOCAB = 16
LAYERS = 2
HEADS = 2
HEAD_DIM = 4
IMAGE_TOKENS = 5


class Spec:
    def __init__(self, name: str, shape: object = None) -> None:
        self.name = name
        self.shape = shape


def _kv_names() -> list[str]:
    return [
        f"{layer}.{branch}.{kind}"
        for layer in range(LAYERS)
        for branch in ("decoder", "encoder")
        for kind in ("key", "value")
    ]


class FakeVision:
    def get_inputs(self):  # noqa: ANN201
        return [Spec("pixel_values")]

    def get_outputs(self):  # noqa: ANN201
        return [Spec("image_features")]

    def run(self, _names, feeds):  # noqa: ANN001, ANN201
        assert feeds["pixel_values"].shape == (1, 3, 768, 768)
        return [np.zeros((1, IMAGE_TOKENS, EMB), dtype=np.float32)]


class FakeEmbed:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def get_inputs(self):  # noqa: ANN201
        return [Spec("input_ids")]

    def get_outputs(self):  # noqa: ANN201
        return [Spec("inputs_embeds")]

    def run(self, _names, feeds):  # noqa: ANN001, ANN201
        ids = feeds["input_ids"]
        self.calls.append(list(ids[0]))
        return [np.zeros((1, ids.shape[1], EMB), dtype=np.float32)]


class FakeEncoder:
    def get_inputs(self):  # noqa: ANN201
        return [Spec("inputs_embeds"), Spec("attention_mask")]

    def get_outputs(self):  # noqa: ANN201
        return [Spec("last_hidden_state")]

    def run(self, _names, feeds):  # noqa: ANN001, ANN201
        embeds = feeds["inputs_embeds"]
        assert feeds["attention_mask"].shape == tuple(embeds.shape[:2])
        return [embeds]


class FakeDecoder:
    """Scripted merged decoder: logits argmax follows ``script`` after the
    forced-BOS step; cross-attn presents degrade to (0,1,1,1) dummies on
    cache steps exactly like the real optimum export."""

    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.steps = 0
        self.cache_flags: list[bool] = []
        self.cross_past_shapes: list[tuple[int, ...]] = []

    def get_inputs(self):  # noqa: ANN201
        specs = [
            Spec("encoder_attention_mask"),
            Spec("encoder_hidden_states"),
            Spec("inputs_embeds"),
            Spec("use_cache_branch"),
        ]
        specs.extend(
            Spec(f"past_key_values.{name}", ["batch", HEADS, "seq", HEAD_DIM])
            for name in _kv_names()
        )
        return specs

    def get_outputs(self):  # noqa: ANN201
        return [Spec("logits")] + [Spec(f"present.{name}") for name in _kv_names()]

    def run(self, _names, feeds):  # noqa: ANN001, ANN201
        self.steps += 1
        self.cache_flags.append(bool(feeds["use_cache_branch"][0]))
        self.cross_past_shapes.append(feeds["past_key_values.0.encoder.key"].shape)
        logits = np.full((1, 1, VOCAB), -1.0, dtype=np.float32)
        if self.steps >= 2:
            index = min(self.steps - 2, len(self.script) - 1)
            logits[0, 0, self.script[index]] = 10.0
        outputs: list[np.ndarray] = [logits]
        for name in _kv_names():
            if ".decoder." in f".{name}":
                outputs.append(
                    np.ones((1, HEADS, self.steps, HEAD_DIM), dtype=np.float32)
                )
            elif self.steps == 1:  # real cross-attn KV only on the first pass
                outputs.append(
                    np.ones((1, HEADS, IMAGE_TOKENS, HEAD_DIM), dtype=np.float32)
                )
            else:  # zero-element dummy, as the real export returns
                outputs.append(np.zeros((0, 1, 1, 1), dtype=np.float32))
        return outputs


def write_tokenizer(path: Path) -> None:
    """Toy tokenizer.json: specials + ' red'/' car' + one merge pair."""
    data = {
        "model": {
            "vocab": {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "Ġred": 3,
                "Ġcar": 4,
                ",": 5,
                "Ġ": 6,
                "h": 7,
                "e": 8,
                "he": 9,
                "Ġhe": 10,
            },
            "merges": [
                "h e",
                "Ġ he",
                "r e",
                "re d",
                "Ġ red",
                "c a",
                "ca r",
                "Ġ car",
            ],
        },
        "added_tokens": [
            {"id": 0, "content": "<s>"},
            {"id": 1, "content": "<pad>"},
            {"id": 2, "content": "</s>"},
            {"id": 11, "content": "<loc_1>"},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture()
def engine_files(tmp_path: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in REQUIRED_FILES:
        target = tmp_path / name
        if name == FILE_TOKENIZER:
            write_tokenizer(target)
        else:
            target.write_bytes(b"onnx")
        files[name] = target
    return files


@pytest.fixture()
def image(tmp_path: Path) -> Path:
    from PIL import Image

    target = tmp_path / "img.png"
    Image.new("RGB", (32, 16), (255, 0, 0)).save(target)
    return target


def make_engine(
    files: dict[str, Path], decoder: FakeDecoder, embed: FakeEmbed | None = None
) -> FlorenceEngine:
    sessions = {
        str(files[FILE_VISION]): FakeVision(),
        str(files[FILE_EMBED]): embed if embed is not None else FakeEmbed(),
        str(files[FILE_ENCODER]): FakeEncoder(),
        str(files[FILE_DECODER]): decoder,
    }
    return FlorenceEngine(files, session_factory=lambda path: sessions[path])


class TestTasks:
    def test_labels_cover_all_tokens(self) -> None:
        assert set(FLORENCE_TASK_TOKENS) == set(FLORENCE_TASK_LABELS)
        assert DEFAULT_FLORENCE_TASK in FLORENCE_TASK_TOKENS
        assert DEFAULT_FLORENCE_TASK == TASK_GENERATE_TAGS

    def test_standard_tasks_map_to_english_prompts(self) -> None:
        assert prompt_for_task(TASK_CAPTION) == "What does the image describe?"
        assert (
            prompt_for_task(TASK_DETAILED_CAPTION)
            == "Describe in detail what is shown in the image."
        )
        assert (
            prompt_for_task(TASK_MORE_DETAILED_CAPTION)
            == "Describe with a paragraph what is shown in the image."
        )

    def test_promptgen_tasks_pass_through_literally(self) -> None:
        assert prompt_for_task(TASK_GENERATE_TAGS) == TASK_GENERATE_TAGS
        assert prompt_for_task(TASK_MIXED_CAPTION_PLUS) == TASK_MIXED_CAPTION_PLUS

    def test_unknown_task_rejected(self) -> None:
        with pytest.raises(ValidationError, match="未知的打标指令"):
            validate_task("<NOPE>")


class TestTokenizer:
    def test_bytes_to_unicode_is_reversible(self) -> None:
        table = _bytes_to_unicode()
        assert len(table) == 256
        assert len(set(table.values())) == 256

    def test_encode_applies_merges(self, tmp_path: Path) -> None:
        write_tokenizer(tmp_path / "tok.json")
        tok = FlorenceTokenizer.from_file(tmp_path / "tok.json")
        # " he" pre-tokenizes to one piece, then merges h+e then Ġ+he.
        assert tok.encode(" he") == [10]
        assert tok.encode(" red car") == [3, 4]

    def test_encode_prompt_frames_with_bos_eos(self, tmp_path: Path) -> None:
        write_tokenizer(tmp_path / "tok.json")
        tok = FlorenceTokenizer.from_file(tmp_path / "tok.json")
        assert tok.encode_prompt(" red") == [0, 3, 2]

    def test_decode_skips_specials_and_keeps_added_tokens(
        self, tmp_path: Path
    ) -> None:
        write_tokenizer(tmp_path / "tok.json")
        tok = FlorenceTokenizer.from_file(tmp_path / "tok.json")
        assert tok.decode([0, 3, 5, 4, 2]) == " red, car"
        assert tok.decode([0, 3, 11, 2]) == " red<loc_1>"
        assert tok.decode([0, 3, 2], skip_special=False) == "<s> red</s>"

    def test_broken_tokenizer_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "tok.json"
        target.write_text("{not json", encoding="utf-8")
        with pytest.raises(LocalInferenceError, match="分词器"):
            FlorenceTokenizer.from_file(target)


class TestNgramBan:
    def test_bans_completion_of_seen_trigram(self) -> None:
        assert _banned_next_tokens([1, 2, 3, 1, 2], 3) == {3}

    def test_no_ban_for_short_or_disabled(self) -> None:
        assert _banned_next_tokens([1, 2], 3) == set()
        assert _banned_next_tokens([1, 2, 3, 1, 2], 0) == set()


class TestEngine:
    def test_missing_file_mapping_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="缺少模型文件映射"):
            FlorenceEngine({FILE_DECODER: tmp_path / "d.onnx"})

    def test_load_missing_file_raises_actionable(
        self, engine_files: dict[str, Path]
    ) -> None:
        engine_files[FILE_VISION].unlink()
        engine = make_engine(engine_files, FakeDecoder([2]))
        with pytest.raises(LocalInferenceError, match="重新下载"):
            engine.load()

    def test_unknown_task_rejected_before_load(
        self, engine_files: dict[str, Path], image: Path
    ) -> None:
        engine = make_engine(engine_files, FakeDecoder([2]))
        with pytest.raises(ValidationError):
            engine.caption(image, "<NOPE>")
        assert not engine.is_loaded()

    def test_caption_decodes_scripted_tokens(
        self, engine_files: dict[str, Path], image: Path
    ) -> None:
        decoder = FakeDecoder([3, 5, 4, 2])  # " red, car" then EOS
        engine = make_engine(engine_files, decoder)
        assert engine.caption(image, TASK_GENERATE_TAGS) == "red, car"
        assert engine.is_loaded()
        # use_cache_branch: first step False, cached afterwards.
        assert decoder.cache_flags[0] is False
        assert all(decoder.cache_flags[1:])

    def test_cross_attention_cache_survives_dummy_presents(
        self, engine_files: dict[str, Path], image: Path
    ) -> None:
        decoder = FakeDecoder([3, 4, 2])
        engine = make_engine(engine_files, decoder)
        engine.caption(image, TASK_GENERATE_TAGS)
        # Step 1 feeds the empty template; every later step must feed the
        # REAL first-step cross KV, never the (0,1,1,1) dummies.
        assert decoder.cross_past_shapes[0] == (1, HEADS, 0, HEAD_DIM)
        for shape in decoder.cross_past_shapes[1:]:
            assert shape == (1, HEADS, IMAGE_TOKENS, HEAD_DIM)

    def test_prompt_ids_framed_and_embedded(
        self, engine_files: dict[str, Path], image: Path
    ) -> None:
        embed = FakeEmbed()
        engine = make_engine(engine_files, FakeDecoder([2]), embed)
        engine.caption(image, TASK_CAPTION)
        first = embed.calls[0]
        assert first[0] == 0 and first[-1] == 2  # <s> ... </s>
        assert len(first) > 2  # the mapped English prompt tokenized

    def test_unload_drops_sessions(
        self, engine_files: dict[str, Path], image: Path
    ) -> None:
        engine = make_engine(engine_files, FakeDecoder([2]))
        engine.caption(image, TASK_GENERATE_TAGS)
        assert engine.is_loaded()
        engine.unload()
        assert not engine.is_loaded()


class TestPreprocess:
    def test_pixels_resized_and_normalized(self, image: Path) -> None:
        from nlapt.local.florence import _load_pixels

        pixels = _load_pixels(image)
        assert pixels.shape == (1, 3, 768, 768)
        assert pixels.dtype == np.float32
        # Pure red: R=(1-0.485)/0.229, G=(0-0.456)/0.224, B=(0-0.406)/0.225.
        assert pixels[0, 0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-3)
        assert pixels[0, 1, 0, 0] == pytest.approx(-0.456 / 0.224, abs=1e-3)
        assert pixels[0, 2, 0, 0] == pytest.approx(-0.406 / 0.225, abs=1e-3)
