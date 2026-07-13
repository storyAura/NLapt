"""Tests for nlapt.storage.atomic."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.errors import StorageError, ValidationError
from nlapt.storage.atomic import TEMP_SUFFIX, atomic_write_bytes, atomic_write_text


def _temp_files(directory: Path) -> list[Path]:
    return list(directory.glob(f"*{TEMP_SUFFIX}"))


def test_write_bytes_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "caption.txt"
    atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"
    assert _temp_files(tmp_path) == []


def test_write_bytes_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "caption.txt"
    target.write_bytes(b"old content")
    atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"


def test_write_text_is_utf8_without_bom(tmp_path: Path) -> None:
    target = tmp_path / "caption.txt"
    atomic_write_text(target, "一个女孩, smiling\n")
    raw = target.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8") == "一个女孩, smiling\n"


def test_write_failure_raises_storage_error_and_cleans_temp(tmp_path: Path) -> None:
    # A directory at the target path makes os.replace fail on every platform.
    target = tmp_path / "blocked"
    target.mkdir()
    with pytest.raises(StorageError):
        atomic_write_bytes(target, b"data")
    assert _temp_files(tmp_path) == []


def test_missing_parent_directory_raises_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        atomic_write_bytes(tmp_path / "no" / "such" / "dir.txt", b"data")


def test_original_survives_failed_overwrite(tmp_path: Path) -> None:
    # Failure before replace never corrupts the existing file.
    target = tmp_path / "keep"
    target.mkdir()
    (target / "inner.txt").write_text("safe", encoding="utf-8")
    with pytest.raises(StorageError):
        atomic_write_bytes(target, b"data")
    assert (target / "inner.txt").read_text(encoding="utf-8") == "safe"


def test_input_validation() -> None:
    with pytest.raises(ValidationError):
        atomic_write_bytes(Path("x.txt"), "not bytes")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        atomic_write_text(Path("x.txt"), b"not str")  # type: ignore[arg-type]


def test_bytearray_accepted(tmp_path: Path) -> None:
    target = tmp_path / "caption.txt"
    atomic_write_bytes(target, bytearray(b"abc"))
    assert target.read_bytes() == b"abc"
