"""Tests for nlapt.local.gguf: the minimal block_count header reader."""

from __future__ import annotations

import struct
from pathlib import Path

from nlapt.local.gguf import GGUF_MAGIC, read_block_count

T_UINT8 = 0
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(value_type: int, value: object) -> bytes:
    if value_type == T_UINT8:
        return struct.pack("<B", value)
    if value_type == T_UINT32:
        return struct.pack("<I", value)
    if value_type == T_INT32:
        return struct.pack("<i", value)
    if value_type == T_UINT64:
        return struct.pack("<Q", value)
    if value_type == T_FLOAT32:
        return struct.pack("<f", value)
    if value_type == T_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if value_type == T_STRING:
        return _string(str(value))
    raise AssertionError(f"unsupported test value type {value_type}")


def _string_array(items: list[str]) -> bytes:
    payload = struct.pack("<I", T_STRING) + struct.pack("<Q", len(items))
    for item in items:
        payload += _string(item)
    return payload


def _uint32_array(items: list[int]) -> bytes:
    payload = struct.pack("<I", T_UINT32) + struct.pack("<Q", len(items))
    for item in items:
        payload += struct.pack("<I", item)
    return payload


def build_gguf(
    kv_blobs: list[tuple[str, int, bytes]], *, version: int = 3, magic: bytes = GGUF_MAGIC
) -> bytes:
    header = magic + struct.pack("<I", version)
    header += struct.pack("<Q", 0)  # tensor count
    header += struct.pack("<Q", len(kv_blobs))
    for key, value_type, blob in kv_blobs:
        header += _string(key) + struct.pack("<I", value_type) + blob
    return header


def write(tmp_path: Path, data: bytes) -> Path:
    target = tmp_path / "model.gguf"
    target.write_bytes(data)
    return target


class TestReadBlockCount:
    def test_reads_uint32_block_count(self, tmp_path: Path) -> None:
        data = build_gguf(
            [
                ("general.architecture", T_STRING, _value(T_STRING, "gemma4")),
                ("gemma4.context_length", T_UINT32, _value(T_UINT32, 131072)),
                ("gemma4.block_count", T_UINT32, _value(T_UINT32, 48)),
            ]
        )
        assert read_block_count(write(tmp_path, data)) == 48

    def test_skips_arrays_and_mixed_values_before_the_key(
        self, tmp_path: Path
    ) -> None:
        data = build_gguf(
            [
                ("tokenizer.ggml.tokens", T_ARRAY, _string_array(["a", "bb", "ccc"])),
                ("some.scores", T_ARRAY, _uint32_array([1, 2, 3, 4])),
                ("general.finetuned", T_BOOL, _value(T_BOOL, True)),
                ("general.scale", T_FLOAT32, _value(T_FLOAT32, 0.5)),
                ("llama.block_count", T_UINT64, _value(T_UINT64, 32)),
            ]
        )
        assert read_block_count(write(tmp_path, data)) == 32

    def test_missing_key_gives_none(self, tmp_path: Path) -> None:
        data = build_gguf(
            [("general.architecture", T_STRING, _value(T_STRING, "x"))]
        )
        assert read_block_count(write(tmp_path, data)) is None

    def test_bad_magic_gives_none(self, tmp_path: Path) -> None:
        data = build_gguf(
            [("a.block_count", T_UINT32, _value(T_UINT32, 4))], magic=b"NOPE"
        )
        assert read_block_count(write(tmp_path, data)) is None

    def test_unsupported_version_gives_none(self, tmp_path: Path) -> None:
        data = build_gguf(
            [("a.block_count", T_UINT32, _value(T_UINT32, 4))], version=1
        )
        assert read_block_count(write(tmp_path, data)) is None

    def test_truncated_file_gives_none(self, tmp_path: Path) -> None:
        data = build_gguf([("a.block_count", T_UINT32, _value(T_UINT32, 4))])
        assert read_block_count(write(tmp_path, data[:-2])) is None

    def test_missing_file_gives_none(self, tmp_path: Path) -> None:
        assert read_block_count(tmp_path / "ghost.gguf") is None

    def test_non_integer_block_count_gives_none(self, tmp_path: Path) -> None:
        data = build_gguf(
            [("a.block_count", T_FLOAT32, _value(T_FLOAT32, 48.0))]
        )
        assert read_block_count(write(tmp_path, data)) is None

    def test_zero_block_count_gives_none(self, tmp_path: Path) -> None:
        data = build_gguf([("a.block_count", T_UINT32, _value(T_UINT32, 0))])
        assert read_block_count(write(tmp_path, data)) is None
