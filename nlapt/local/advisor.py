"""Predict whether a catalog model can run on the detected hardware.

The estimate is a documented heuristic (NOT a llama.cpp simulation):

    required = weights + mmproj + kv_cache(context) + overhead
    overhead = OVERHEAD_BASE_BYTES + weights * WEIGHTS_OVERHEAD_FACTOR

Budgets treat the question as "can this MACHINE run it", so RAM budget uses
total (not momentarily-available) memory scaled by :data:`RAM_USABLE_SHARE`;
the GPU budget uses the reported free VRAM scaled by
:data:`VRAM_USABLE_SHARE`. Verdicts, best to worst:

* ``GPU_FULL``    — everything fits in one GPU's free VRAM (流畅)
* ``GPU_PARTIAL`` — fits in free VRAM + RAM combined (部分层进显存)
* ``CPU_ONLY``    — no usable GPU, but fits in RAM (较慢)
* ``NOT_RUNNABLE``— exceeds VRAM + RAM combined
* ``UNKNOWN``     — hardware detection failed (no RAM figure)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nlapt.core.errors import ValidationError
from nlapt.local.catalog import ModelFamily, QuantFile
from nlapt.local.hardware import HardwareInfo

OVERHEAD_BASE_BYTES = 600 * 1024 * 1024
WEIGHTS_OVERHEAD_FACTOR = 0.05
VRAM_USABLE_SHARE = 0.92
RAM_USABLE_SHARE = 0.70
DEFAULT_CONTEXT_LENGTH = 4096
MAX_CONTEXT_LENGTH = 1_048_576


class RunVerdict(str, Enum):
    """Which resource scenario the model lands in on this hardware."""

    GPU_FULL = "gpu_full"
    GPU_PARTIAL = "gpu_partial"
    CPU_ONLY = "cpu_only"
    NOT_RUNNABLE = "not_runnable"
    UNKNOWN = "unknown"


# Budget/required ratio above which a run is considered to have comfortable
# headroom (drives the PERFECT vs SMOOTH and OK vs BARELY grade split).
HEADROOM_FACTOR = 1.3


class RunGrade(str, Enum):
    """Plain five-level "can this machine run it" answer for the UI.

    Ordered best to worst: PERFECT (轻松运行) > SMOOTH (流畅运行) >
    OK (可以运行) > BARELY (勉强能跑) > NO (跑不动); UNKNOWN when the
    hardware could not be detected.
    """

    PERFECT = "perfect"
    SMOOTH = "smooth"
    OK = "ok"
    BARELY = "barely"
    NO = "no"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MemoryEstimate:
    """Predicted memory footprint of one quant at a given context length."""

    weights_bytes: int
    mmproj_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.mmproj_bytes
            + self.kv_cache_bytes
            + self.overhead_bytes
        )


@dataclass(frozen=True)
class RunAssessment:
    """Verdict + grade plus the numbers that produced them (UI breakdown)."""

    verdict: RunVerdict
    grade: RunGrade
    estimate: MemoryEstimate
    gpu_budget_bytes: int
    ram_budget_bytes: int
    shortfall_bytes: int  # > 0 only for NOT_RUNNABLE


def _validated_context(context_length: int) -> int:
    """Boundary check for a caller-supplied context length."""
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise ValidationError(
            f"context_length must be an int, got {type(context_length).__name__}"
        )
    if context_length <= 0 or context_length > MAX_CONTEXT_LENGTH:
        raise ValidationError(
            f"context_length must be in (0, {MAX_CONTEXT_LENGTH}], got {context_length}"
        )
    return context_length


def estimate_memory(
    family: ModelFamily,
    quant: QuantFile,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> MemoryEstimate:
    """Heuristic memory footprint of ``quant`` at ``context_length`` tokens."""
    context = _validated_context(context_length)
    if quant.size_bytes <= 0:
        raise ValidationError(f"quant {quant.label!r} has no size information")
    weights = quant.size_bytes
    mmproj = family.mmproj_bytes if family.vision else 0
    kv_cache = family.kv_bytes_per_token * context
    overhead = OVERHEAD_BASE_BYTES + int(weights * WEIGHTS_OVERHEAD_FACTOR)
    return MemoryEstimate(
        weights_bytes=weights,
        mmproj_bytes=mmproj,
        kv_cache_bytes=kv_cache,
        overhead_bytes=overhead,
    )


def _grade_for(
    verdict: RunVerdict, required: int, gpu_budget: int, ram_budget: int
) -> RunGrade:
    """Collapse a verdict + headroom into the five-level grade."""
    if verdict is RunVerdict.UNKNOWN:
        return RunGrade.UNKNOWN
    if verdict is RunVerdict.NOT_RUNNABLE:
        return RunGrade.NO
    if verdict is RunVerdict.GPU_FULL:
        if gpu_budget >= int(required * HEADROOM_FACTOR):
            return RunGrade.PERFECT
        return RunGrade.SMOOTH
    budget = (
        gpu_budget + ram_budget if verdict is RunVerdict.GPU_PARTIAL else ram_budget
    )
    if budget >= int(required * HEADROOM_FACTOR):
        return RunGrade.OK
    return RunGrade.BARELY


def assess(
    family: ModelFamily,
    quant: QuantFile,
    hardware: HardwareInfo,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> RunAssessment:
    """Predict how ``quant`` runs on ``hardware`` (see module docstring)."""
    estimate = estimate_memory(family, quant, context_length=context_length)

    best_gpu = hardware.best_gpu()
    gpu_budget = 0
    if best_gpu is not None:
        usable_vram = (
            best_gpu.vram_free_bytes
            if best_gpu.vram_free_bytes > 0
            else best_gpu.vram_total_bytes
        )
        gpu_budget = int(usable_vram * VRAM_USABLE_SHARE)
    ram_budget = int(hardware.ram_total_bytes * RAM_USABLE_SHARE)

    if hardware.ram_total_bytes <= 0:
        return RunAssessment(
            verdict=RunVerdict.UNKNOWN,
            grade=RunGrade.UNKNOWN,
            estimate=estimate,
            gpu_budget_bytes=gpu_budget,
            ram_budget_bytes=0,
            shortfall_bytes=0,
        )

    required = estimate.total_bytes
    if gpu_budget >= required:
        verdict = RunVerdict.GPU_FULL
    elif best_gpu is not None and gpu_budget + ram_budget >= required:
        verdict = RunVerdict.GPU_PARTIAL
    elif best_gpu is None and ram_budget >= required:
        verdict = RunVerdict.CPU_ONLY
    else:
        verdict = RunVerdict.NOT_RUNNABLE

    shortfall = 0
    if verdict is RunVerdict.NOT_RUNNABLE:
        shortfall = required - (gpu_budget + ram_budget)
    return RunAssessment(
        verdict=verdict,
        grade=_grade_for(verdict, required, gpu_budget, ram_budget),
        estimate=estimate,
        gpu_budget_bytes=gpu_budget,
        ram_budget_bytes=ram_budget,
        shortfall_bytes=shortfall,
    )
