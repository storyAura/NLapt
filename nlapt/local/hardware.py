"""Best-effort system hardware detection for the local-inference advisor.

Detects total / available RAM, logical CPU cores and NVIDIA GPU VRAM (via
``nvidia-smi``). Detection NEVER raises: every probe failure is logged and
collapses to zeros / an empty GPU tuple, and the advisor renders that as an
"unknown" verdict. All probes are injectable for tests.

Non-NVIDIA GPUs are reported as absent (no portable VRAM query exists in the
stdlib) — the advisor then falls back to a RAM-only prediction, which is the
honest answer for llama.cpp CPU/Vulkan builds anyway.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

NVIDIA_SMI_QUERY_ARGS: tuple[str, ...] = (
    "--query-gpu=name,memory.total,memory.free",
    "--format=csv,noheader,nounits",
)
NVIDIA_SMI_TIMEOUT_SECONDS = 6.0
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024
# Keep child consoles hidden on Windows (0 elsewhere).
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

MemoryProvider = Callable[[], tuple[int, int]]  # (total_bytes, available_bytes)
SmiRunner = Callable[[], str]  # stdout of nvidia-smi; raises on failure


@dataclass(frozen=True)
class GpuInfo:
    """One detected GPU and its VRAM state at detection time."""

    name: str
    vram_total_bytes: int
    vram_free_bytes: int


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of the machine resources relevant to local inference."""

    cpu_cores: int
    ram_total_bytes: int
    ram_available_bytes: int
    gpus: tuple[GpuInfo, ...] = ()

    @property
    def has_gpu(self) -> bool:
        """Whether at least one NVIDIA GPU was detected."""
        return bool(self.gpus)

    def best_gpu(self) -> GpuInfo | None:
        """The GPU with the most total VRAM (None when no GPU detected)."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda gpu: gpu.vram_total_bytes)


def format_bytes(count: int) -> str:
    """Human-readable size: '15.8 GB' / '512 MB' / '3 KB' / '0 B'."""
    if count <= 0:
        return "0 B"
    if count >= GIB:
        return f"{count / GIB:.1f} GB"
    if count >= MIB:
        return f"{count / MIB:.0f} MB"
    if count >= 1024:
        return f"{count / 1024:.0f} KB"
    return f"{count} B"


def _windows_memory() -> tuple[int, int]:
    """Total / available physical RAM via GlobalMemoryStatusEx."""
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _posix_memory() -> tuple[int, int]:
    """Total / available RAM via sysconf, refined by /proc/meminfo when present."""
    page = os.sysconf("SC_PAGE_SIZE")
    total = page * os.sysconf("SC_PHYS_PAGES")
    available = 0
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        available = 0
    if available <= 0:
        try:
            available = page * os.sysconf("SC_AV_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            available = total // 2
    return int(total), int(available)


def _default_memory() -> tuple[int, int]:
    if sys.platform == "win32":
        return _windows_memory()
    return _posix_memory()


def _resolve_nvidia_smi() -> str | None:
    """Absolute path of nvidia-smi, or None when not installed.

    On Windows only the driver's fixed install locations are trusted — never
    the PATH or the current working directory — so a same-named executable
    planted next to the app can never be launched (binary-planting hardening).
    POSIX resolves via PATH, which is the platform convention.
    """
    if sys.platform == "win32":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates = (
            Path(system_root) / "System32" / "nvidia-smi.exe",
            Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("nvidia-smi")


def _default_smi_runner() -> str:
    executable = _resolve_nvidia_smi()
    if executable is None:
        raise FileNotFoundError("nvidia-smi")
    result = subprocess.run(  # noqa: S603 - absolute path, non-shell arg vector
        [executable, *NVIDIA_SMI_QUERY_ARGS],
        capture_output=True,
        text=True,
        timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        creationflags=_CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        raise OSError(f"nvidia-smi exited with code {result.returncode}")
    return result.stdout


def parse_nvidia_smi(output: str) -> tuple[GpuInfo, ...]:
    """Parse ``name, total_mib, free_mib`` CSV lines; malformed lines skipped."""
    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0]:
            continue
        try:
            total_mib = int(float(parts[1]))
            free_mib = int(float(parts[2]))
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                name=parts[0],
                vram_total_bytes=total_mib * MIB,
                vram_free_bytes=free_mib * MIB,
            )
        )
    return tuple(gpus)


def detect_hardware(
    *,
    memory_provider: MemoryProvider | None = None,
    smi_runner: SmiRunner | None = None,
    cpu_counter: Callable[[], int | None] = os.cpu_count,
) -> HardwareInfo:
    """Probe the machine. Never raises — failed probes collapse to zeros."""
    try:
        cores = cpu_counter() or 0
    except Exception:  # noqa: BLE001 - detection must never break the caller
        _LOGGER.exception("cpu core detection failed")
        cores = 0

    ram_total = ram_available = 0
    try:
        provider = memory_provider if memory_provider is not None else _default_memory
        ram_total, ram_available = provider()
    except Exception:  # noqa: BLE001
        _LOGGER.exception("RAM detection failed")

    gpus: tuple[GpuInfo, ...] = ()
    try:
        runner = smi_runner if smi_runner is not None else _default_smi_runner
        gpus = parse_nvidia_smi(runner())
    except FileNotFoundError:
        _LOGGER.info("nvidia-smi not found; assuming no NVIDIA GPU")
    except Exception:  # noqa: BLE001
        _LOGGER.warning("GPU detection via nvidia-smi failed", exc_info=True)

    info = HardwareInfo(
        cpu_cores=int(cores),
        ram_total_bytes=int(ram_total),
        ram_available_bytes=int(ram_available),
        gpus=gpus,
    )
    _LOGGER.info(
        "hardware detected: cores=%d ram=%s free=%s gpus=%s",
        info.cpu_cores,
        format_bytes(info.ram_total_bytes),
        format_bytes(info.ram_available_bytes),
        [gpu.name for gpu in info.gpus] or "none",
    )
    return info
