"""Bundled llama.cpp runtime provisioning (设置 ▸ 本地推理 自动运行时).

The app ships "complete inference capability" without asking the user to
hunt for a llama-server binary: a pinned official llama.cpp release is
downloaded on demand (same resumable downloader + SHA256 pinning as the
model catalog) and extracted into a per-user runtime directory. A manually
selected ``server_path`` always wins; this module is the fallback that
makes 下载模型 → 推理 work with zero extra steps.

Like the model catalog this is a hand-pinned snapshot: asset names, byte
sizes and SHA256 digests are exact values from the GitHub release API
(``digest`` field), captured on :data:`RUNTIME_SNAPSHOT_DATE`. Update the
table and the date together. The Windows/Linux x64 assets are the Vulkan
builds — one binary drives NVIDIA / AMD / Intel GPUs and llama.cpp falls
back to its CPU backend when no Vulkan device exists.
"""

from __future__ import annotations

import platform
import sys
import tarfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

from nlapt.core.errors import LocalServerError
from nlapt.diagnostics import get_logger
from nlapt.local.download import OpenerFn, ProgressFn, download_file

_LOGGER = get_logger(__name__)

# Date the release digests below were captured from the GitHub API.
RUNTIME_SNAPSHOT_DATE = "2026-07-23"
# Pinned llama.cpp release tag (release assets are immutable).
LLAMA_CPP_TAG = "b10088"
RELEASE_DOWNLOAD_BASE = (
    "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{filename}"
)

SERVER_EXE_WINDOWS = "llama-server.exe"
SERVER_EXE_POSIX = "llama-server"
_EXECUTABLE_MODE = 0o755

MSG_PLATFORM_UNSUPPORTED = (
    "当前平台没有内置的 llama.cpp 运行时,请在 设置 ▸ 本地推理 手动选择 "
    "llama-server 可执行文件"
)
MSG_EXE_NOT_FOUND = (
    "运行时压缩包中未找到 llama-server 可执行文件,请重试或手动选择"
)


@dataclass(frozen=True)
class RuntimeAsset:
    """One pinned release archive for a platform."""

    platform_key: str  # "win-x64" | "win-arm64" | "linux-x64" | "macos-arm64"
    filename: str
    size_bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return RELEASE_DOWNLOAD_BASE.format(tag=LLAMA_CPP_TAG, filename=self.filename)


RUNTIME_ASSETS: dict[str, RuntimeAsset] = {
    asset.platform_key: asset
    for asset in (
        RuntimeAsset(
            platform_key="win-x64",
            filename=f"llama-{LLAMA_CPP_TAG}-bin-win-vulkan-x64.zip",
            size_bytes=33_290_390,
            sha256="ced37906bfa57dca6079b0e66163edc4f319b43ba8260bda5427fbd20a08324b",
        ),
        RuntimeAsset(
            platform_key="win-arm64",
            filename=f"llama-{LLAMA_CPP_TAG}-bin-win-cpu-arm64.zip",
            size_bytes=11_869_553,
            sha256="ca8f44b8a9d515ba28863018febdd153c483c904173d700132e045bff40d0cfc",
        ),
        RuntimeAsset(
            platform_key="linux-x64",
            filename=f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-vulkan-x64.tar.gz",
            size_bytes=32_042_886,
            sha256="ac6965fc063761571a5bef5e469150648cce3be8336b98c580e4c110d231fd95",
        ),
        RuntimeAsset(
            platform_key="macos-arm64",
            filename=f"llama-{LLAMA_CPP_TAG}-bin-macos-arm64.tar.gz",
            size_bytes=10_615_347,
            sha256="e39658aa0af5acac893b2fdf58dc6480faf6254cfbc89b3a6c5f6ce71db9442e",
        ),
    )
}

_MACHINE_X64 = frozenset({"amd64", "x86_64"})
_MACHINE_ARM64 = frozenset({"arm64", "aarch64"})


def runtime_platform_key(
    sys_platform: str = sys.platform, machine: str = ""
) -> str | None:
    """The RUNTIME_ASSETS key for this machine, or None when unsupported."""
    normalized = (machine or platform.machine()).lower()
    if normalized in _MACHINE_X64:
        arch = "x64"
    elif normalized in _MACHINE_ARM64:
        arch = "arm64"
    else:
        return None
    if sys_platform == "win32":
        key = f"win-{arch}"
    elif sys_platform == "darwin":
        key = f"macos-{arch}"
    elif sys_platform.startswith("linux"):
        key = f"linux-{arch}"
    else:
        return None
    return key if key in RUNTIME_ASSETS else None


def current_asset() -> RuntimeAsset | None:
    """The pinned archive for this machine, or None when unsupported."""
    key = runtime_platform_key()
    return RUNTIME_ASSETS.get(key) if key is not None else None


def runtime_dir(base_dir: Path) -> Path:
    """Extraction directory for the pinned release (versioned by tag)."""
    return Path(base_dir) / f"llama.cpp-{LLAMA_CPP_TAG}"


def find_server_exe(directory: Path) -> Path | None:
    """The extracted llama-server executable under ``directory`` (or None).

    Searched recursively because release archives have used both flat
    (Windows zips) and ``build/bin`` (tar.gz) layouts; the shallowest
    match wins.
    """
    target = SERVER_EXE_WINDOWS if sys.platform == "win32" else SERVER_EXE_POSIX
    root = Path(directory)
    if not root.is_dir():
        return None
    matches = sorted(root.rglob(target), key=lambda p: len(p.parts))
    return matches[0] if matches else None


def _extract_archive(archive: Path, dest: Path) -> None:
    """Extract a pinned .zip / .tar.gz archive into ``dest``.

    The archive's SHA256 was verified against the pinned release digest
    before this call; CPython's extractors additionally sanitize member
    paths (absolute paths and ``..`` components are stripped).
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(dest)
        else:
            with tarfile.open(archive, "r:*") as bundle:
                try:
                    bundle.extractall(dest, filter="data")
                except TypeError:  # pragma: no cover - pre-3.11.4 fallback
                    bundle.extractall(dest)  # noqa: S202 - digest-pinned archive
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        raise LocalServerError(f"运行时压缩包解压失败: {exc}") from exc


def ensure_runtime(
    base_dir: Path,
    *,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    opener: OpenerFn | None = None,
) -> Path:
    """Return the llama-server executable, downloading the runtime if needed.

    Fast path: an already-extracted executable is returned without any
    network access. Otherwise the pinned release archive is downloaded
    (resumable, SHA256-verified), extracted and deleted. Raises
    :class:`LocalServerError` on unsupported platforms / broken archives,
    plus the downloader's error types for network failures.
    """
    target_dir = runtime_dir(base_dir)
    existing = find_server_exe(target_dir)
    if existing is not None:
        return existing
    asset = current_asset()
    if asset is None:
        raise LocalServerError(MSG_PLATFORM_UNSUPPORTED)
    archive = Path(base_dir) / asset.filename
    _LOGGER.info("provisioning llama.cpp runtime %s -> %s", asset.filename, target_dir)
    download_file(
        asset.url,
        archive,
        expected_bytes=asset.size_bytes,
        expected_sha256=asset.sha256,
        progress=progress,
        cancel=cancel,
        opener=opener,
    )
    _extract_archive(archive, target_dir)
    archive.unlink(missing_ok=True)
    exe = find_server_exe(target_dir)
    if exe is None:
        raise LocalServerError(MSG_EXE_NOT_FOUND)
    if sys.platform != "win32":
        exe.chmod(exe.stat().st_mode | _EXECUTABLE_MODE)
    _LOGGER.info("llama.cpp runtime ready: %s", exe)
    return exe
