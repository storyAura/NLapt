"""Tests for nlapt.local.hardware: parsing, injection, graceful failure."""

from __future__ import annotations

from nlapt.local.hardware import (
    GIB,
    MIB,
    GpuInfo,
    HardwareInfo,
    detect_hardware,
    format_bytes,
    parse_nvidia_smi,
)

SMI_ONE_GPU = "NVIDIA GeForce RTX 4070, 12282, 11005\n"
SMI_TWO_GPUS = (
    "NVIDIA GeForce RTX 4090, 24564, 20000\n"
    "NVIDIA GeForce GTX 1650, 4096, 3800\n"
)


class TestParseNvidiaSmi:
    def test_single_gpu(self) -> None:
        gpus = parse_nvidia_smi(SMI_ONE_GPU)
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA GeForce RTX 4070"
        assert gpus[0].vram_total_bytes == 12282 * MIB
        assert gpus[0].vram_free_bytes == 11005 * MIB

    def test_multiple_gpus(self) -> None:
        assert len(parse_nvidia_smi(SMI_TWO_GPUS)) == 2

    def test_malformed_lines_skipped(self) -> None:
        output = "garbage\nname only, x, y\n" + SMI_ONE_GPU + ", 1, 2\n"
        gpus = parse_nvidia_smi(output)
        assert len(gpus) == 1

    def test_empty_output(self) -> None:
        assert parse_nvidia_smi("") == ()


class TestDetectHardware:
    def test_injected_providers(self) -> None:
        info = detect_hardware(
            memory_provider=lambda: (32 * GIB, 20 * GIB),
            smi_runner=lambda: SMI_ONE_GPU,
            cpu_counter=lambda: 16,
        )
        assert info.cpu_cores == 16
        assert info.ram_total_bytes == 32 * GIB
        assert info.ram_available_bytes == 20 * GIB
        assert info.has_gpu
        assert info.gpus[0].name.endswith("4070")

    def test_missing_nvidia_smi_means_no_gpu(self) -> None:
        def missing() -> str:
            raise FileNotFoundError("nvidia-smi")

        info = detect_hardware(
            memory_provider=lambda: (8 * GIB, 4 * GIB), smi_runner=missing
        )
        assert info.gpus == ()
        assert not info.has_gpu

    def test_memory_failure_collapses_to_zero(self) -> None:
        def boom() -> tuple[int, int]:
            raise OSError("no api")

        info = detect_hardware(memory_provider=boom, smi_runner=lambda: "")
        assert info.ram_total_bytes == 0
        assert info.ram_available_bytes == 0

    def test_cpu_counter_none_becomes_zero(self) -> None:
        info = detect_hardware(
            memory_provider=lambda: (1, 1),
            smi_runner=lambda: "",
            cpu_counter=lambda: None,
        )
        assert info.cpu_cores == 0

    def test_real_detection_never_raises(self) -> None:
        info = detect_hardware()
        assert isinstance(info, HardwareInfo)
        assert info.ram_total_bytes >= 0

    def test_real_memory_probe_on_this_platform(self) -> None:
        import sys

        from nlapt.local.hardware import _posix_memory, _windows_memory

        probe = _windows_memory if sys.platform == "win32" else _posix_memory
        total, available = probe()
        assert total > 0
        assert 0 < available <= total

    def test_nvidia_smi_resolves_to_absolute_path_or_none(self) -> None:
        from pathlib import Path

        from nlapt.local.hardware import _resolve_nvidia_smi

        resolved = _resolve_nvidia_smi()
        assert resolved is None or Path(resolved).is_absolute()


class TestBestGpu:
    def test_none_without_gpus(self) -> None:
        info = HardwareInfo(cpu_cores=4, ram_total_bytes=1, ram_available_bytes=1)
        assert info.best_gpu() is None

    def test_picks_largest_vram(self) -> None:
        small = GpuInfo(name="small", vram_total_bytes=4 * GIB, vram_free_bytes=3 * GIB)
        big = GpuInfo(name="big", vram_total_bytes=24 * GIB, vram_free_bytes=1 * GIB)
        info = HardwareInfo(
            cpu_cores=4, ram_total_bytes=1, ram_available_bytes=1, gpus=(small, big)
        )
        best = info.best_gpu()
        assert best is not None and best.name == "big"


class TestFormatBytes:
    def test_zero_and_negative(self) -> None:
        assert format_bytes(0) == "0 B"
        assert format_bytes(-5) == "0 B"

    def test_bytes_kb_mb_gb(self) -> None:
        assert format_bytes(512) == "512 B"
        assert format_bytes(2048) == "2 KB"
        assert format_bytes(5 * MIB) == "5 MB"
        assert format_bytes(int(15.8 * GIB)) == "15.8 GB"
