"""Tests for nlapt.local.lora: safetensors/adapter parsing, ONNX merge."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from nlapt.core.errors import LocalInferenceError
from nlapt.local.florence import (
    FILE_DECODER,
    FILE_EMBED,
    FILE_ENCODER,
    FILE_TOKENIZER,
    FILE_VISION,
    REQUIRED_FILES,
    FlorenceEngine,
)
from nlapt.local.lora import (
    ADAPTER_CONFIG_NAME,
    LoraAdapter,
    LoraModule,
    _module_path,
    load_adapter,
    merge_into_onnx,
)

Q_PROJ = "language_model.model.decoder.layers.0.self_attn.q_proj"


# -- byte-level builders (safetensors + ONNX protobuf) -------------------------------
def write_safetensors(
    path: Path, tensors: dict[str, np.ndarray], dtype: str = "F32"
) -> None:
    codes = {"F32": "<f4", "F16": "<f2"}
    header: dict[str, object] = {}
    blob = bytearray()
    for name, array in tensors.items():
        if dtype == "BF16":  # truncate f32 to its top 16 bits
            raw = (
                (array.astype(np.float32).view(np.uint32) >> 16)
                .astype("<u2")
                .tobytes()
            )
        else:
            raw = array.astype(codes[dtype]).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(blob), len(blob) + len(raw)],
        }
        blob.extend(raw)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(blob))


def write_config(directory: Path, **overrides: object) -> None:
    config: dict[str, object] = {"r": 1, "lora_alpha": 2, "peft_type": "LORA"}
    config.update(overrides)
    (directory / ADAPTER_CONFIG_NAME).write_text(
        json.dumps(config), encoding="utf-8"
    )


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _len_field(field: int, payload: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _varint_field(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def make_tensor(name: str, array: np.ndarray, dtype_code: int = 1) -> bytes:
    numpy_dtype = "<f4" if dtype_code == 1 else "<f2"
    payload = b"".join(_varint_field(1, dim) for dim in array.shape)
    payload += _varint_field(2, dtype_code)
    payload += _len_field(8, name.encode("utf-8"))
    payload += _len_field(9, array.astype(numpy_dtype).tobytes())
    return payload


def make_node(
    name: str, op_type: str, inputs: tuple[str, ...], subgraphs: tuple[bytes, ...] = ()
) -> bytes:
    payload = b"".join(_len_field(1, item.encode("utf-8")) for item in inputs)
    payload += _len_field(3, name.encode("utf-8"))
    payload += _len_field(4, op_type.encode("utf-8"))
    for graph in subgraphs:
        payload += _len_field(5, _len_field(6, graph))  # attribute { g: graph }
    return payload


def make_graph(
    nodes: tuple[bytes, ...] = (), initializers: tuple[bytes, ...] = ()
) -> bytes:
    return b"".join(_len_field(1, node) for node in nodes) + b"".join(
        _len_field(5, tensor) for tensor in initializers
    )


def make_model(graph: bytes) -> bytes:
    return _varint_field(8, 21) + _len_field(7, graph)  # ir_version + graph


def simple_decoder_model(weight: np.ndarray) -> bytes:
    """One q_proj MatMul whose weight is the initializer ``w`` (in, out)."""
    node = make_node(
        "/model/decoder/layers.0/self_attn/q_proj/MatMul",
        "MatMul",
        ("hidden", "w"),
    )
    return make_model(make_graph((node,), (make_tensor("w", weight),)))


def adapter_for(
    module: str, down: np.ndarray, up: np.ndarray, scale: float = 2.0
) -> LoraAdapter:
    return LoraAdapter(
        path=Path("adapter_model.safetensors"),
        modules={
            module: LoraModule(
                down=down.astype(np.float32), up=up.astype(np.float32), scale=scale
            )
        },
    )


# -- safetensors + adapter_config ----------------------------------------------------
class TestLoadAdapter:
    def make_pair(self, tmp_path: Path, dtype: str = "F32") -> Path:
        target = tmp_path / "adapter_model.safetensors"
        write_safetensors(
            target,
            {
                f"base_model.model.{Q_PROJ}.lora_A.weight": np.ones((1, 2)),
                f"base_model.model.{Q_PROJ}.lora_B.weight": np.ones((3, 1)),
            },
            dtype=dtype,
        )
        write_config(tmp_path)
        return target

    def test_parses_pairs_and_strips_prefix(self, tmp_path: Path) -> None:
        adapter = load_adapter(self.make_pair(tmp_path))
        assert set(adapter.modules) == {Q_PROJ}
        module = adapter.modules[Q_PROJ]
        assert module.down.shape == (1, 2)
        assert module.up.shape == (3, 1)
        assert module.scale == pytest.approx(2.0)  # alpha 2 / r 1

    @pytest.mark.parametrize("dtype", ["F16", "BF16"])
    def test_half_precision_tensors_widen_to_f32(
        self, tmp_path: Path, dtype: str
    ) -> None:
        adapter = load_adapter(self.make_pair(tmp_path, dtype=dtype))
        module = adapter.modules[Q_PROJ]
        assert module.down.dtype == np.float32
        np.testing.assert_allclose(module.down, np.ones((1, 2)))

    def test_rslora_scale_uses_sqrt(self, tmp_path: Path) -> None:
        target = self.make_pair(tmp_path)
        write_config(tmp_path, lora_alpha=8, use_rslora=True)
        module = load_adapter(target).modules[Q_PROJ]
        assert module.scale == pytest.approx(8 / math.sqrt(1))

    def test_missing_alpha_defaults_to_rank(self, tmp_path: Path) -> None:
        target = self.make_pair(tmp_path)
        write_config(tmp_path, lora_alpha=None)
        assert load_adapter(target).modules[Q_PROJ].scale == pytest.approx(1.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LocalInferenceError, match="LoRA 文件不存在"):
            load_adapter(tmp_path / "nope.safetensors")

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        target = self.make_pair(tmp_path)
        (tmp_path / ADAPTER_CONFIG_NAME).unlink()
        with pytest.raises(LocalInferenceError, match="adapter_config.json"):
            load_adapter(target)

    def test_dora_rejected(self, tmp_path: Path) -> None:
        target = self.make_pair(tmp_path)
        write_config(tmp_path, use_dora=True)
        with pytest.raises(LocalInferenceError, match="DoRA"):
            load_adapter(target)

    def test_unpaired_tensor_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "adapter_model.safetensors"
        write_safetensors(
            target, {f"base_model.model.{Q_PROJ}.lora_A.weight": np.ones((1, 2))}
        )
        write_config(tmp_path)
        with pytest.raises(LocalInferenceError, match="不成对"):
            load_adapter(target)

    def test_rank_mismatch_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "adapter_model.safetensors"
        write_safetensors(
            target,
            {
                f"base_model.model.{Q_PROJ}.lora_A.weight": np.ones((2, 2)),
                f"base_model.model.{Q_PROJ}.lora_B.weight": np.ones((3, 1)),
            },
        )
        write_config(tmp_path)
        with pytest.raises(LocalInferenceError, match="秩不一致"):
            load_adapter(target)

    def test_corrupt_safetensors_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "adapter_model.safetensors"
        target.write_bytes(b"\xff" * 32)
        write_config(tmp_path)
        with pytest.raises(LocalInferenceError, match="无法解析"):
            load_adapter(target)


# -- node-scope to module-path mapping -----------------------------------------------
class TestModulePath:
    def test_plain_scopes(self) -> None:
        assert (
            _module_path("/model/decoder/layers.0/self_attn/q_proj/MatMul")
            == "model.decoder.layers.0.self_attn.q_proj"
        )
        assert _module_path("/lm_head/MatMul") == "lm_head"

    def test_repeated_modulelist_scope_collapses(self) -> None:
        assert (
            _module_path("/blocks.0/blocks.0.0/spatial_block/ffn/fn/net/fc1/MatMul")
            == "blocks.0.0.spatial_block.ffn.fn.net.fc1"
        )

    def test_single_segment_yields_empty(self) -> None:
        assert _module_path("MatMul_7") == ""


# -- merging into ONNX ---------------------------------------------------------------
class TestMerge:
    def test_merges_matching_weight(self, tmp_path: Path) -> None:
        weight = np.arange(6, dtype=np.float32).reshape(2, 3)  # (in, out)
        target = tmp_path / "decoder.onnx"
        target.write_bytes(simple_decoder_model(weight))
        down = np.asarray([[1.0, 2.0]])  # (r=1, in=2)
        up = np.asarray([[1.0], [0.0], [-1.0]])  # (out=3, r=1)
        result = merge_into_onnx(target, adapter_for(Q_PROJ, down, up, scale=2.0))
        assert result.matched == {Q_PROJ}
        merged = result.initializers["w"]
        expected = weight + 2.0 * (down.T @ up.T)
        np.testing.assert_allclose(merged, expected)
        assert merged.dtype == np.float32

    def test_fp16_weight_keeps_dtype(self, tmp_path: Path) -> None:
        weight = np.ones((2, 3), dtype=np.float16)
        node = make_node(
            "/model/decoder/layers.0/self_attn/q_proj/MatMul",
            "MatMul",
            ("hidden", "w"),
        )
        target = tmp_path / "decoder.onnx"
        target.write_bytes(
            make_model(make_graph((node,), (make_tensor("w", weight, 10),)))
        )
        result = merge_into_onnx(
            target,
            adapter_for(Q_PROJ, np.ones((1, 2)), np.ones((3, 1)), scale=1.0),
        )
        assert result.initializers["w"].dtype == np.float16

    def test_if_branches_share_one_weight(self, tmp_path: Path) -> None:
        weight = np.zeros((2, 3), dtype=np.float32)
        branch_node = make_node(
            "/model/decoder/layers.0/self_attn/q_proj/MatMul",
            "MatMul",
            ("hidden", "w"),
        )
        branch = make_graph((branch_node,))
        if_node = make_node("/If", "If", ("flag",), subgraphs=(branch, branch))
        target = tmp_path / "merged.onnx"
        target.write_bytes(
            make_model(make_graph((if_node,), (make_tensor("w", weight),)))
        )
        result = merge_into_onnx(
            target,
            adapter_for(Q_PROJ, np.ones((1, 2)), np.ones((3, 1)), scale=1.0),
        )
        assert set(result.initializers) == {"w"}
        assert result.matched == {Q_PROJ}

    def test_vision_scope_collapse_matches(self, tmp_path: Path) -> None:
        weight = np.zeros((2, 3), dtype=np.float32)
        node = make_node(
            "/blocks.0/blocks.0.0/spatial_block/ffn/fn/net/fc1/MatMul",
            "MatMul",
            ("hidden", "w"),
        )
        target = tmp_path / "vision.onnx"
        target.write_bytes(make_model(make_graph((node,), (make_tensor("w", weight),))))
        module = "vision_tower.blocks.0.0.spatial_block.ffn.fn.net.fc1"
        result = merge_into_onnx(
            target,
            adapter_for(module, np.ones((1, 2)), np.ones((3, 1))),
        )
        assert result.matched == {module}

    def test_bare_layer_paths_resolved_by_module_prefix(
        self, tmp_path: Path
    ) -> None:
        # onnx-community's encoder export roots node names at the submodule
        # (bare /layers.0/...): the suffix alone matches BOTH the encoder
        # and decoder q_proj modules — the caller's prefix must decide.
        weight = np.zeros((2, 3), dtype=np.float32)
        node = make_node("/layers.0/self_attn/q_proj/MatMul", "MatMul", ("h", "w"))
        target = tmp_path / "encoder.onnx"
        target.write_bytes(make_model(make_graph((node,), (make_tensor("w", weight),))))
        encoder_module = "language_model.model.encoder.layers.0.self_attn.q_proj"
        adapter = LoraAdapter(
            path=Path("adapter_model.safetensors"),
            modules={
                Q_PROJ: LoraModule(
                    down=np.ones((1, 2), dtype=np.float32),
                    up=np.ones((3, 1), dtype=np.float32),
                    scale=1.0,
                ),
                encoder_module: LoraModule(
                    down=np.ones((1, 2), dtype=np.float32),
                    up=np.ones((3, 1), dtype=np.float32),
                    scale=1.0,
                ),
            },
        )
        with pytest.raises(LocalInferenceError, match="多个模块命中"):
            merge_into_onnx(target, adapter)
        result = merge_into_onnx(
            target, adapter, module_prefix="language_model.model.encoder."
        )
        assert result.matched == {encoder_module}

    def test_unrelated_modules_match_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "decoder.onnx"
        target.write_bytes(simple_decoder_model(np.zeros((2, 3), dtype=np.float32)))
        result = merge_into_onnx(
            target,
            adapter_for("something.else.entirely", np.ones((1, 2)), np.ones((3, 1))),
        )
        assert result.initializers == {}
        assert result.matched == frozenset()

    def test_architecture_mismatch_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "decoder.onnx"
        target.write_bytes(simple_decoder_model(np.zeros((2, 3), dtype=np.float32)))
        # A "large"-trained factor pair: in=4 instead of the model's 2.
        with pytest.raises(LocalInferenceError, match="架构不匹配"):
            merge_into_onnx(
                target,
                adapter_for(Q_PROJ, np.ones((1, 4)), np.ones((3, 1))),
            )

    def test_garbage_model_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "decoder.onnx"
        target.write_bytes(b"\x07\x07not-a-protobuf")
        with pytest.raises(LocalInferenceError, match="模型文件解析失败"):
            merge_into_onnx(
                target,
                adapter_for(Q_PROJ, np.ones((1, 2)), np.ones((3, 1))),
            )


# -- engine integration --------------------------------------------------------------
def write_tokenizer(path: Path) -> None:
    data = {
        "model": {"vocab": {"<s>": 0, "<pad>": 1, "</s>": 2}, "merges": []},
        "added_tokens": [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture()
def engine_files(tmp_path: Path) -> dict[str, Path]:
    """Four minimal-but-valid ONNX files + tokenizer; decoder has q_proj."""
    files: dict[str, Path] = {}
    empty = make_model(make_graph())
    for name in REQUIRED_FILES:
        target = tmp_path / name
        if name == FILE_TOKENIZER:
            write_tokenizer(target)
        elif name == FILE_DECODER:
            target.write_bytes(
                simple_decoder_model(np.zeros((2, 3), dtype=np.float32))
            )
        else:
            target.write_bytes(empty)
        files[name] = target
    return files


def write_adapter(tmp_path: Path, module: str = Q_PROJ) -> Path:
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir(exist_ok=True)
    target = lora_dir / "adapter_model.safetensors"
    write_safetensors(
        target,
        {
            f"base_model.model.{module}.lora_A.weight": np.ones((1, 2)),
            f"base_model.model.{module}.lora_B.weight": np.ones((3, 1)),
        },
    )
    write_config(lora_dir)
    return target


class TestEngineWithLora:
    def factory_recorder(self) -> tuple[list[tuple[str, object]], object]:
        calls: list[tuple[str, object]] = []

        def factory(path: str, initializers: object = None) -> object:
            calls.append((path, initializers))
            return object()

        return calls, factory

    def test_load_passes_overrides_to_factory(
        self, tmp_path: Path, engine_files: dict[str, Path]
    ) -> None:
        calls, factory = self.factory_recorder()
        engine = FlorenceEngine(
            engine_files,
            session_factory=factory,
            lora_path=write_adapter(tmp_path),
        )
        engine.load()
        assert engine.is_loaded()
        by_path = dict(calls)
        decoder_overrides = by_path[str(engine_files[FILE_DECODER])]
        assert set(decoder_overrides) == {"w"}
        for name in (FILE_VISION, FILE_EMBED, FILE_ENCODER):
            assert by_path[str(engine_files[name])] == {}

    def test_without_lora_factory_called_unary(
        self, engine_files: dict[str, Path]
    ) -> None:
        seen: list[str] = []
        engine = FlorenceEngine(
            engine_files, session_factory=lambda path: seen.append(path) or object()
        )
        engine.load()
        assert len(seen) == 4  # unary fakes keep working without a LoRA

    def test_lora_matching_nothing_rejected_and_engine_reset(
        self, tmp_path: Path, engine_files: dict[str, Path]
    ) -> None:
        calls, factory = self.factory_recorder()
        engine = FlorenceEngine(
            engine_files,
            session_factory=factory,
            lora_path=write_adapter(tmp_path, module="unrelated.module"),
        )
        with pytest.raises(LocalInferenceError, match="不兼容"):
            engine.load()
        assert not engine.is_loaded()

    def test_missing_lora_file_rejected(
        self, tmp_path: Path, engine_files: dict[str, Path]
    ) -> None:
        calls, factory = self.factory_recorder()
        engine = FlorenceEngine(
            engine_files,
            session_factory=factory,
            lora_path=tmp_path / "gone.safetensors",
        )
        with pytest.raises(LocalInferenceError, match="LoRA 文件不存在"):
            engine.load()
        assert not engine.is_loaded()
