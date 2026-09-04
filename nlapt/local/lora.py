"""PEFT LoRA loading + ONNX weight merging (设置 ▸ 本地推理 · LoRA).

Community Florence-2 fine-tunes ship as PEFT LoRA adapters
(``adapter_model.safetensors`` + ``adapter_config.json``): low-rank
factor pairs ``A``/``B`` per targeted linear layer. onnxruntime has no
adapter format for plain sessions, so the factors are merged into the
exported weights at session-creation time:

    MatMul weight M = W^T          (optimum stores linears as (in, out))
    M' = M + scale * A^T B^T       scale = lora_alpha / r  (rsLoRA: / sqrt(r))

Matching PEFT module paths to ONNX initializers needs no ``onnx``
package: a small protobuf walk (stable wire format + frozen ONNX field
numbers) collects every MatMul node and its weight initializer across
all subgraphs — the merged decoder hides both KV-cache branches inside
an ``If`` node whose branches share one weight set. Node scope names
mirror the PyTorch module tree (``/model/decoder/layers.0/self_attn/
q_proj/MatMul``), so a suffix match against the PEFT path — after
collapsing repeated ModuleList scopes like ``blocks.0/blocks.0.0`` —
pins every factor pair to its weight, and a dimension check turns
architecture mismatches (a large-trained LoRA on the base export) into
one actionable error instead of silent garbage output.

The merged arrays are handed to onnxruntime through
``SessionOptions.add_initializer`` by the Florence engine; nothing is
ever written back to disk. numpy is imported lazily like the rest of
the optional ``local`` stack.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlapt.core.errors import LocalInferenceError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

ADAPTER_CONFIG_NAME = "adapter_config.json"

MSG_LORA_FILE_MISSING = "LoRA 文件不存在: {path} — 打开 设置 ▸ 本地推理 重新选择"
MSG_LORA_CONFIG_MISSING = (
    "LoRA 缺少 adapter_config.json(需与 safetensors 文件放在同一目录): {path}"
)
MSG_LORA_BAD = "LoRA 文件无法解析: {path}({reason})"
MSG_LORA_DORA = "暂不支持 DoRA 适配器(use_dora=true): {path}"
MSG_LORA_MISMATCH = (
    "LoRA 与当前模型架构不匹配({module}: LoRA 为 {lora_in}×{lora_out},"
    "模型为 {model_in}×{model_out})— 该 LoRA 基于其它规格的模型"
    "(如 large 架构)训练,与当前 ONNX 模型不通用"
)
MSG_LORA_NO_MATCH = "LoRA 与当前模型不兼容:没有任何层能对应到模型权重"
MSG_MODEL_PARSE = "模型文件解析失败: {path}"
MSG_MISSING_NUMPY = "加载 LoRA 需要 numpy 组件 — 请先运行 pip install numpy 后重试"

# safetensors dtype tag -> numpy dtype (BF16 needs a manual widening).
_SAFETENSORS_DTYPES = {"F32": "<f4", "F16": "<f2"}
_BF16 = "BF16"
# ONNX TensorProto.DataType codes we can patch (float32 / float16).
_ONNX_DTYPES = {1: "<f4", 10: "<f2"}

# protobuf wire types (the format is frozen; see protobuf encoding docs).
_VARINT, _I64, _LEN, _I32 = 0, 1, 2, 5
# frozen ONNX field numbers used below.
_MODEL_GRAPH = 7  # ModelProto.graph
_GRAPH_NODE = 1  # GraphProto.node
_GRAPH_INITIALIZER = 5  # GraphProto.initializer
_NODE_INPUT = 1  # NodeProto.input
_NODE_NAME = 3  # NodeProto.name
_NODE_OP_TYPE = 4  # NodeProto.op_type
_NODE_ATTRIBUTE = 5  # NodeProto.attribute
_ATTR_GRAPH = 6  # AttributeProto.g
_ATTR_GRAPHS = 11  # AttributeProto.graphs
_TENSOR_DIMS = 1  # TensorProto.dims
_TENSOR_DTYPE = 2  # TensorProto.data_type
_TENSOR_NAME = 8  # TensorProto.name
_TENSOR_RAW = 9  # TensorProto.raw_data


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LocalInferenceError(MSG_MISSING_NUMPY) from exc
    return numpy


@dataclass(frozen=True)
class LoraModule:
    """One targeted linear layer: ``delta W = scale * (up @ down)``."""

    down: Any  # lora_A, float32 (r, in_features)
    up: Any  # lora_B, float32 (out_features, r)
    scale: float


@dataclass(frozen=True)
class LoraAdapter:
    """A parsed PEFT adapter, keyed by module path (base_model. stripped)."""

    path: Path
    modules: Mapping[str, LoraModule]


@dataclass(frozen=True)
class MergeResult:
    """Merged weights of ONE onnx file + which adapter modules they cover."""

    initializers: Mapping[str, Any]  # initializer name -> merged numpy array
    matched: frozenset[str]


# -- safetensors + adapter_config parsing --------------------------------------------
def _read_safetensors(path: Path) -> dict[str, Any]:
    """name -> float32 numpy array for every float tensor in the file."""
    np = _require_numpy()
    try:
        data = path.read_bytes()
        (header_len,) = struct.unpack_from("<Q", data, 0)
        header = json.loads(bytes(data[8 : 8 + header_len]).decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("header is not an object")
        base = 8 + header_len
        tensors: dict[str, Any] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info["dtype"]
            start, end = info["data_offsets"]
            raw = data[base + start : base + end]
            if dtype == _BF16:
                array = (
                    np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
                ).view(np.float32)
            elif dtype in _SAFETENSORS_DTYPES:
                array = np.frombuffer(raw, dtype=_SAFETENSORS_DTYPES[dtype])
            else:  # non-float tensors are never LoRA factors
                continue
            tensors[name] = array.reshape(info["shape"]).astype(np.float32)
        return tensors
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        raise LocalInferenceError(
            MSG_LORA_BAD.format(path=path, reason=exc)
        ) from exc


def _module_scale(config: Mapping[str, Any], rank: int) -> float:
    """PEFT scale for one module: alpha / r (rsLoRA: alpha / sqrt(r))."""
    try:
        alpha = float(config.get("lora_alpha") or rank)
    except (TypeError, ValueError):
        alpha = float(rank)
    if config.get("use_rslora"):
        return alpha / math.sqrt(rank)
    return alpha / rank


def load_adapter(path: Path) -> LoraAdapter:
    """Parse ``path`` (a .safetensors file) + its sibling adapter_config.json.

    Raises :class:`LocalInferenceError` with an actionable Chinese message
    on every failure mode (missing files, DoRA, corrupt factors).
    """
    path = Path(path)
    if not path.is_file():
        raise LocalInferenceError(MSG_LORA_FILE_MISSING.format(path=path))
    config_path = path.parent / ADAPTER_CONFIG_NAME
    if not config_path.is_file():
        raise LocalInferenceError(MSG_LORA_CONFIG_MISSING.format(path=path.parent))
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalInferenceError(
            MSG_LORA_BAD.format(path=config_path, reason=exc)
        ) from exc
    if not isinstance(config, dict):
        raise LocalInferenceError(
            MSG_LORA_BAD.format(path=config_path, reason="配置不是 JSON 对象")
        )
    if config.get("use_dora"):
        raise LocalInferenceError(MSG_LORA_DORA.format(path=path))

    downs: dict[str, Any] = {}
    ups: dict[str, Any] = {}
    for name, array in _read_safetensors(path).items():
        for marker, bucket in ((".lora_A.", downs), (".lora_B.", ups)):
            if marker in name:
                module = name.split(marker, 1)[0]
                for prefix in ("base_model.model.", "base_model."):
                    if module.startswith(prefix):
                        module = module[len(prefix) :]
                        break
                bucket[module] = array
    if not downs or set(downs) != set(ups):
        raise LocalInferenceError(
            MSG_LORA_BAD.format(path=path, reason="lora_A/lora_B 张量不成对")
        )
    modules: dict[str, LoraModule] = {}
    for module, down in downs.items():
        up = ups[module]
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
            raise LocalInferenceError(
                MSG_LORA_BAD.format(path=path, reason=f"{module} 的秩不一致")
            )
        modules[module] = LoraModule(
            down=down, up=up, scale=_module_scale(config, int(down.shape[0]))
        )
    _LOGGER.info("lora adapter loaded: %s (%d modules)", path.name, len(modules))
    return LoraAdapter(path=path, modules=modules)


# -- minimal ONNX (protobuf) walk ----------------------------------------------------
def _read_varint(view: memoryview, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = view[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint overflow")


def _iter_fields(view: memoryview) -> Iterator[tuple[int, int, Any]]:
    """Yield (field_number, wire_type, value) over one protobuf message.

    LEN fields yield a memoryview into ``view``; varints yield ints;
    fixed 64/32-bit fields are skipped structurally (yielded as views).
    """
    pos = 0
    end = len(view)
    while pos < end:
        tag, pos = _read_varint(view, pos)
        field, wire = tag >> 3, tag & 7
        if wire == _VARINT:
            value, pos = _read_varint(view, pos)
            yield field, wire, value
        elif wire == _LEN:
            length, pos = _read_varint(view, pos)
            yield field, wire, view[pos : pos + length]
            pos += length
        elif wire == _I64:
            yield field, wire, view[pos : pos + 8]
            pos += 8
        elif wire == _I32:
            yield field, wire, view[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")


def _parse_dims(field_wire: int, value: Any, dims: list[int]) -> None:
    """TensorProto.dims: unpacked varints or one packed LEN block."""
    if field_wire == _VARINT:
        dims.append(int(value))
        return
    pos = 0
    while pos < len(value):
        dim, pos = _read_varint(value, pos)
        dims.append(dim)


def _walk_graph(
    view: memoryview,
    matmuls: list[tuple[str, tuple[str, ...]]],
    inits: dict[str, tuple[tuple[int, ...], int, Any]],
) -> None:
    """Collect MatMul nodes + initializers of a GraphProto and its subgraphs."""
    for field, wire, value in _iter_fields(view):
        if field == _GRAPH_NODE and wire == _LEN:
            name = op_type = ""
            inputs: list[str] = []
            for f2, w2, v2 in _iter_fields(value):
                if f2 == _NODE_INPUT and w2 == _LEN:
                    inputs.append(bytes(v2).decode("utf-8", errors="replace"))
                elif f2 == _NODE_NAME and w2 == _LEN:
                    name = bytes(v2).decode("utf-8", errors="replace")
                elif f2 == _NODE_OP_TYPE and w2 == _LEN:
                    op_type = bytes(v2).decode("utf-8", errors="replace")
                elif f2 == _NODE_ATTRIBUTE and w2 == _LEN:
                    for f3, w3, v3 in _iter_fields(v2):
                        if f3 in (_ATTR_GRAPH, _ATTR_GRAPHS) and w3 == _LEN:
                            _walk_graph(v3, matmuls, inits)
            if op_type == "MatMul":
                matmuls.append((name, tuple(inputs)))
        elif field == _GRAPH_INITIALIZER and wire == _LEN:
            dims: list[int] = []
            dtype = 0
            name = ""
            raw: Any = None
            for f2, w2, v2 in _iter_fields(value):
                if f2 == _TENSOR_DIMS and w2 in (_VARINT, _LEN):
                    _parse_dims(w2, v2, dims)
                elif f2 == _TENSOR_DTYPE and w2 == _VARINT:
                    dtype = int(v2)
                elif f2 == _TENSOR_NAME and w2 == _LEN:
                    name = bytes(v2).decode("utf-8", errors="replace")
                elif f2 == _TENSOR_RAW and w2 == _LEN:
                    raw = v2
            if name:
                inits[name] = (tuple(dims), dtype, raw)


def _module_path(node_name: str) -> str:
    """Dot path of a node's module scope, ModuleList repeats collapsed.

    ``/blocks.0/blocks.0.0/spatial_block/ffn/fn/net/fc1/MatMul`` becomes
    ``blocks.0.0.spatial_block.ffn.fn.net.fc1`` — the trailing op segment
    is dropped and a scope that merely prefixes its child is redundant.
    """
    segments = node_name.strip("/").split("/")[:-1]
    kept: list[str] = []
    for segment in segments:
        if kept and segment.startswith(kept[-1] + "."):
            kept.pop()
        kept.append(segment)
    return ".".join(kept)


def merge_into_onnx(
    path: Path, adapter: LoraAdapter, *, module_prefix: str = ""
) -> MergeResult:
    """Merged replacement weights of one ONNX file for ``adapter``.

    ``module_prefix`` limits the candidate adapter modules to one part of
    the model (e.g. ``language_model.model.encoder.``): some exports root
    their node names at the submodule (bare ``/layers.0/...``), where a
    suffix alone cannot tell encoder from decoder layers — the caller
    knows which part each file serves. Returns an empty result when no
    candidate module lives in this file. Raises
    :class:`LocalInferenceError` on parse failures and on dimension
    mismatches (LoRA trained on a different architecture).
    """
    np = _require_numpy()
    path = Path(path)
    candidates = {
        name: module
        for name, module in adapter.modules.items()
        if name.startswith(module_prefix)
    }
    matmuls: list[tuple[str, tuple[str, ...]]] = []
    inits: dict[str, tuple[tuple[int, ...], int, Any]] = {}
    try:
        view = memoryview(path.read_bytes())
        for field, wire, value in _iter_fields(view):
            if field == _MODEL_GRAPH and wire == _LEN:
                _walk_graph(value, matmuls, inits)
                break
    except (OSError, ValueError, IndexError, struct.error) as exc:
        raise LocalInferenceError(MSG_MODEL_PARSE.format(path=path)) from exc

    overrides: dict[str, Any] = {}
    matched: set[str] = set()
    for node_name, inputs in matmuls:
        path_key = _module_path(node_name)
        if not path_key:
            continue
        suffix = "." + path_key
        hits = [
            name for name in candidates if name == path_key or name.endswith(suffix)
        ]
        if not hits:
            continue
        if len(hits) > 1:
            raise LocalInferenceError(
                MSG_LORA_BAD.format(
                    path=adapter.path, reason=f"多个模块命中同一权重 {path_key}"
                )
            )
        weight_name = next((i for i in inputs if i in inits), None)
        if weight_name is None:  # weight is not a graph constant — cannot patch
            continue
        dims, dtype, raw = inits[weight_name]
        if dtype not in _ONNX_DTYPES or raw is None or len(dims) != 2:
            continue
        module = candidates[hits[0]]
        if module.down.shape[1] != dims[0] or module.up.shape[0] != dims[1]:
            raise LocalInferenceError(
                MSG_LORA_MISMATCH.format(
                    module=hits[0],
                    lora_in=module.down.shape[1],
                    lora_out=module.up.shape[0],
                    model_in=dims[0],
                    model_out=dims[1],
                )
            )
        matched.add(hits[0])
        if weight_name in overrides:  # If-branches share one weight set
            continue
        base = (
            np.frombuffer(raw, dtype=_ONNX_DTYPES[dtype])
            .reshape(dims)
            .astype(np.float32)
        )
        delta = (module.down.T @ module.up.T) * module.scale
        overrides[weight_name] = (base + delta).astype(_ONNX_DTYPES[dtype])
    if overrides:
        _LOGGER.info(
            "lora merged into %s: %d weights (%d modules)",
            path.name,
            len(overrides),
            len(matched),
        )
    return MergeResult(initializers=overrides, matched=frozenset(matched))
