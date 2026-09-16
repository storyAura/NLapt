"""Shared onnxruntime session factory (Florence + CL Tagger).

The session surface is injectable so tests never import onnxruntime.
Providers are tried in CUDA → DirectML → CPU order. LoRA initializers
(Florence only) are attached at session creation; the OrtValue wrappers
must outlive the session (ORT keeps raw pointers).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from nlapt.core.errors import LocalInferenceError

PREFERRED_PROVIDERS = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)

MSG_MISSING_ORT = (
    "本地模型需要 onnxruntime 与 numpy 组件 — "
    "请先运行 pip install onnxruntime numpy 后重试"
)

_ort_dlls_preloaded = False


class TensorSpec(Protocol):
    """Name + shape of one session input/output (onnxruntime NodeArg)."""

    name: str
    shape: Sequence[Any]


class InferenceSession(Protocol):
    """Minimal onnxruntime.InferenceSession surface the engines rely on."""

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any]
    ) -> list[Any]: ...

    def get_inputs(self) -> Sequence[TensorSpec]: ...

    def get_outputs(self) -> Sequence[TensorSpec]: ...


class SessionFactory(Protocol):
    """Creates a session for ``path``; called with ``initializers`` (an
    initializer-name -> numpy-array mapping to override in the graph)
    only when a LoRA is active, so plain unary callables keep working."""

    def __call__(
        self, path: str, initializers: Mapping[str, Any] | None = None
    ) -> InferenceSession: ...


def require_onnxruntime() -> Any:
    """Import onnxruntime or raise a typed, actionable error."""
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LocalInferenceError(MSG_MISSING_ORT) from exc
    return onnxruntime


def preload_ort_dlls(ort: Any) -> None:
    """Load CUDA/cuDNN DLLs from ``nvidia-*`` pip packages before sessions.

    ``onnxruntime-gpu`` lists CUDA as available even when ``cudnn64_9.dll``
    is missing from PATH; the first Conv then fails with NOT_IMPLEMENTED.
    ``preload_dlls(directory="")`` picks up
    ``pip install onnxruntime-gpu[cuda,cudnn]`` site-packages. Missing
    optional GPU DLLs must not block CPU inference.
    """
    global _ort_dlls_preloaded
    if _ort_dlls_preloaded:
        return
    _ort_dlls_preloaded = True
    preload = getattr(ort, "preload_dlls", None)
    if not callable(preload):
        return
    try:
        preload(cuda=True, cudnn=True, directory="")
    except Exception:  # pragma: no cover - environment dependent
        pass


def default_session_factory(
    path: str, initializers: Mapping[str, Any] | None = None
) -> InferenceSession:
    """Build an InferenceSession, optionally injecting LoRA initializers."""
    ort = require_onnxruntime()
    preload_ort_dlls(ort)
    available = set(ort.get_available_providers())
    providers = [p for p in PREFERRED_PROVIDERS if p in available]
    options = None
    keepalive: list[Any] = []
    if initializers:
        # Merged LoRA weights replace the baked-in graph constants at
        # session creation (nothing is written back to the model file).
        options = ort.SessionOptions()
        for name, array in initializers.items():
            value = ort.OrtValue.ortvalue_from_numpy(array)
            options.add_initializer(name, value)
            keepalive.append(value)
    session = ort.InferenceSession(
        path, sess_options=options, providers=providers or None
    )
    if keepalive:
        # ORT mandates added initializers outlive the session (it keeps raw
        # pointers into them) — tie the OrtValues to the session object so
        # both are released together.
        session.nlapt_lora_keepalive = keepalive
    return session
