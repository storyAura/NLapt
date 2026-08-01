"""Tests for nlapt.local.runtime: pinned assets, extraction, provisioning."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from nlapt.core.errors import LocalServerError
from nlapt.local.runtime import (
    LLAMA_CPP_TAG,
    RUNTIME_ASSETS,
    RuntimeAsset,
    ensure_runtime,
    find_server_exe,
    runtime_dir,
    runtime_platform_key,
)

# Both platform spellings are packed so the tests pass on any OS.
_EXE_NAMES = ("build/bin/llama-server.exe", "build/bin/llama-server")
_EXE_PAYLOAD = b"exe"


def make_zip_bytes(names: tuple[str, ...] = _EXE_NAMES) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name in names:
            bundle.writestr(name, _EXE_PAYLOAD)
    return buffer.getvalue()


def make_tar_gz_bytes(names: tuple[str, ...] = _EXE_NAMES) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = len(_EXE_PAYLOAD)
            bundle.addfile(info, io.BytesIO(_EXE_PAYLOAD))
    return buffer.getvalue()


def fake_asset(filename: str, payload: bytes) -> RuntimeAsset:
    return RuntimeAsset(
        platform_key="test",
        filename=filename,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._body = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int) -> bytes:
        return self._body.read(size)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def make_opener(payload: bytes):
    def opener(request, timeout):  # noqa: ANN001
        return _FakeResponse(payload)

    return opener


def refusing_opener(request, timeout):  # noqa: ANN001
    raise AssertionError("network must not be touched")


class TestPlatformKey:
    @pytest.mark.parametrize(
        ("sys_platform", "machine", "expected"),
        [
            ("win32", "AMD64", "win-x64"),
            ("win32", "ARM64", "win-arm64"),
            ("linux", "x86_64", "linux-x64"),
            ("darwin", "arm64", "macos-arm64"),
            ("darwin", "x86_64", None),  # no pinned macos-x64 archive
            ("linux", "aarch64", None),  # no pinned linux-arm64 archive
            ("sunos5", "x86_64", None),
            ("win32", "riscv64", None),
        ],
    )
    def test_mapping(self, sys_platform: str, machine: str, expected: str | None) -> None:
        assert runtime_platform_key(sys_platform, machine) == expected


class TestAssetTable:
    def test_assets_are_pinned_and_wellformed(self) -> None:
        assert RUNTIME_ASSETS  # never empty
        for key, asset in RUNTIME_ASSETS.items():
            assert asset.platform_key == key
            assert LLAMA_CPP_TAG in asset.filename
            assert asset.url.startswith(
                f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}/"
            )
            assert asset.size_bytes > 0
            assert len(asset.sha256) == 64
            int(asset.sha256, 16)  # hex digest

    def test_runtime_dir_is_tag_versioned(self, tmp_path: Path) -> None:
        assert runtime_dir(tmp_path).name == f"llama.cpp-{LLAMA_CPP_TAG}"


class TestFindServerExe:
    def test_missing_dir_gives_none(self, tmp_path: Path) -> None:
        assert find_server_exe(tmp_path / "nope") is None

    def test_shallowest_match_wins(self, tmp_path: Path) -> None:
        import sys

        name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        deep = tmp_path / "a" / "b" / name
        deep.parent.mkdir(parents=True)
        deep.write_bytes(b"deep")
        shallow = tmp_path / name
        shallow.write_bytes(b"shallow")
        assert find_server_exe(tmp_path) == shallow


class TestEnsureRuntime:
    def _patch_asset(self, monkeypatch: pytest.MonkeyPatch, asset: RuntimeAsset) -> None:
        monkeypatch.setattr("nlapt.local.runtime.current_asset", lambda: asset)

    def test_downloads_extracts_zip_and_cleans_archive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = make_zip_bytes()
        asset = fake_asset("rt.zip", payload)
        self._patch_asset(monkeypatch, asset)
        seen: list[int] = []
        exe = ensure_runtime(
            tmp_path,
            progress=lambda done, _total: seen.append(done),
            opener=make_opener(payload),
        )
        assert exe.is_file()
        assert exe.read_bytes() == _EXE_PAYLOAD
        assert runtime_dir(tmp_path) in exe.parents
        assert not (tmp_path / "rt.zip").exists()  # archive removed
        assert seen and seen[-1] == len(payload)

    def test_extracts_tar_gz(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = make_tar_gz_bytes()
        asset = fake_asset("rt.tar.gz", payload)
        self._patch_asset(monkeypatch, asset)
        exe = ensure_runtime(tmp_path, opener=make_opener(payload))
        assert exe.is_file()

    def test_second_call_fast_paths_without_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = make_zip_bytes()
        asset = fake_asset("rt.zip", payload)
        self._patch_asset(monkeypatch, asset)
        first = ensure_runtime(tmp_path, opener=make_opener(payload))
        second = ensure_runtime(tmp_path, opener=refusing_opener)
        assert second == first

    def test_unsupported_platform_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nlapt.local.runtime.current_asset", lambda: None)
        with pytest.raises(LocalServerError, match="手动选择"):
            ensure_runtime(tmp_path, opener=refusing_opener)

    def test_archive_without_server_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = make_zip_bytes(("readme.txt",))
        asset = fake_asset("rt.zip", payload)
        self._patch_asset(monkeypatch, asset)
        with pytest.raises(LocalServerError, match="未找到 llama-server"):
            ensure_runtime(tmp_path, opener=make_opener(payload))
