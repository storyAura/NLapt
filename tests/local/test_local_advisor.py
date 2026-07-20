"""Tests for nlapt.local.advisor: estimates and verdict boundaries."""

from __future__ import annotations

import pytest

from nlapt.core.errors import ValidationError
from nlapt.local.advisor import (
    DEFAULT_CONTEXT_LENGTH,
    OVERHEAD_BASE_BYTES,
    RAM_USABLE_SHARE,
    VRAM_USABLE_SHARE,
    WEIGHTS_OVERHEAD_FACTOR,
    RunVerdict,
    assess,
    estimate_memory,
)
from nlapt.local.catalog import ModelFamily, QuantFile
from nlapt.local.hardware import GIB, GpuInfo, HardwareInfo

KV_PER_TOKEN = 1_000
WEIGHTS = 1_000_000_000


def make_family(*, vision: bool = False, mmproj_bytes: int = 0) -> ModelFamily:
    return ModelFamily(
        family_id="test-family",
        series_id="gemma4",
        name="Test Family",
        repo_id="tester/test-GGUF",
        downloads=1,
        params_label="1B",
        vision=vision,
        kv_bytes_per_token=KV_PER_TOKEN,
        quants=(QuantFile("Q4", "test.gguf", WEIGHTS, recommended=True),),
        mmproj_filename="mmproj.gguf" if vision else "",
        mmproj_bytes=mmproj_bytes,
    )


def hw(
    *, ram: int = 0, gpus: tuple[GpuInfo, ...] = (), available: int | None = None
) -> HardwareInfo:
    return HardwareInfo(
        cpu_cores=8,
        ram_total_bytes=ram,
        ram_available_bytes=available if available is not None else ram,
        gpus=gpus,
    )


def gpu(total: int, free: int) -> GpuInfo:
    return GpuInfo(name="gpu", vram_total_bytes=total, vram_free_bytes=free)


class TestEstimateMemory:
    def test_text_only_math(self) -> None:
        family = make_family()
        estimate = estimate_memory(family, family.quants[0], context_length=2048)
        assert estimate.weights_bytes == WEIGHTS
        assert estimate.mmproj_bytes == 0
        assert estimate.kv_cache_bytes == KV_PER_TOKEN * 2048
        assert estimate.overhead_bytes == OVERHEAD_BASE_BYTES + int(
            WEIGHTS * WEIGHTS_OVERHEAD_FACTOR
        )
        assert estimate.total_bytes == (
            estimate.weights_bytes + estimate.kv_cache_bytes + estimate.overhead_bytes
        )

    def test_vision_adds_mmproj(self) -> None:
        family = make_family(vision=True, mmproj_bytes=500)
        estimate = estimate_memory(family, family.quants[0])
        assert estimate.mmproj_bytes == 500

    def test_longer_context_costs_more(self) -> None:
        family = make_family()
        small = estimate_memory(family, family.quants[0], context_length=1024)
        large = estimate_memory(family, family.quants[0], context_length=8192)
        assert large.total_bytes > small.total_bytes

    @pytest.mark.parametrize("context", [0, -1, 2_000_000, True])
    def test_invalid_context_rejected(self, context: object) -> None:
        family = make_family()
        with pytest.raises(ValidationError):
            estimate_memory(family, family.quants[0], context_length=context)  # type: ignore[arg-type]

    def test_zero_size_quant_rejected(self) -> None:
        family = make_family()
        broken = QuantFile("bad", "bad.gguf", 0)
        with pytest.raises(ValidationError):
            estimate_memory(family, broken)


class TestAssess:
    def test_unknown_when_ram_undetected(self) -> None:
        family = make_family()
        result = assess(family, family.quants[0], hw(ram=0))
        assert result.verdict is RunVerdict.UNKNOWN
        assert result.shortfall_bytes == 0

    def test_gpu_full_when_it_fits_in_vram(self) -> None:
        family = make_family()
        machine = hw(ram=16 * GIB, gpus=(gpu(24 * GIB, 24 * GIB),))
        result = assess(family, family.quants[0], machine)
        assert result.verdict is RunVerdict.GPU_FULL
        assert result.gpu_budget_bytes == int(24 * GIB * VRAM_USABLE_SHARE)

    def test_gpu_partial_when_vram_too_small_but_ram_helps(self) -> None:
        family = make_family()
        machine = hw(ram=16 * GIB, gpus=(gpu(1 * GIB, 1 * GIB),))
        result = assess(family, family.quants[0], machine)
        assert result.verdict is RunVerdict.GPU_PARTIAL

    def test_cpu_only_without_gpu(self) -> None:
        family = make_family()
        machine = hw(ram=16 * GIB)
        result = assess(family, family.quants[0], machine)
        assert result.verdict is RunVerdict.CPU_ONLY
        assert result.ram_budget_bytes == int(16 * GIB * RAM_USABLE_SHARE)

    def test_not_runnable_with_shortfall(self) -> None:
        family = make_family()
        machine = hw(ram=1 * GIB)
        result = assess(family, family.quants[0], machine)
        assert result.verdict is RunVerdict.NOT_RUNNABLE
        assert result.shortfall_bytes > 0

    def test_not_runnable_even_with_small_gpu(self) -> None:
        family = make_family()
        machine = hw(ram=1 * GIB, gpus=(gpu(256 * 1024 * 1024, 256 * 1024 * 1024),))
        result = assess(family, family.quants[0], machine)
        assert result.verdict is RunVerdict.NOT_RUNNABLE
        assert result.shortfall_bytes > 0

    def test_zero_free_vram_falls_back_to_total(self) -> None:
        family = make_family()
        machine = hw(ram=16 * GIB, gpus=(gpu(24 * GIB, 0),))
        result = assess(family, family.quants[0], machine)
        assert result.gpu_budget_bytes == int(24 * GIB * VRAM_USABLE_SHARE)
        assert result.verdict is RunVerdict.GPU_FULL

    def test_default_context_used(self) -> None:
        family = make_family()
        machine = hw(ram=16 * GIB)
        result = assess(family, family.quants[0], machine)
        assert result.estimate.kv_cache_bytes == KV_PER_TOKEN * DEFAULT_CONTEXT_LENGTH
