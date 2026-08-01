"""Minimal GGUF metadata reader: the model's layer count (stdlib only).

自动 GPU 层数 needs the number of transformer blocks to translate a VRAM
budget into an ``-ngl`` value. The count lives in the GGUF header as the
``<arch>.block_count`` key, so no catalog data or network is needed — the
local file is the source of truth.

Only what this app needs is implemented: scan the key/value section of a
GGUF v2/v3 file until a ``*.block_count`` integer appears. Anything
unexpected (bad magic, v1, truncated data, absurd counts) returns ``None``
instead of raising — callers treat an unknown count conservatively.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
BLOCK_COUNT_SUFFIX = ".block_count"

# GGUF metadata value types -> fixed byte size (None = variable length).
_TYPE_UINT8, _TYPE_INT8 = 0, 1
_TYPE_UINT16, _TYPE_INT16 = 2, 3
_TYPE_UINT32, _TYPE_INT32 = 4, 5
_TYPE_FLOAT32, _TYPE_BOOL = 6, 7
_TYPE_STRING, _TYPE_ARRAY = 8, 9
_TYPE_UINT64, _TYPE_INT64, _TYPE_FLOAT64 = 10, 11, 12

_FIXED_SIZES: dict[int, int] = {
    _TYPE_UINT8: 1,
    _TYPE_INT8: 1,
    _TYPE_UINT16: 2,
    _TYPE_INT16: 2,
    _TYPE_UINT32: 4,
    _TYPE_INT32: 4,
    _TYPE_FLOAT32: 4,
    _TYPE_BOOL: 1,
    _TYPE_UINT64: 8,
    _TYPE_INT64: 8,
    _TYPE_FLOAT64: 8,
}
_INT_FORMATS: dict[int, str] = {
    _TYPE_UINT8: "<B",
    _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H",
    _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I",
    _TYPE_INT32: "<i",
    _TYPE_UINT64: "<Q",
    _TYPE_INT64: "<q",
}

# Defensive caps against corrupt headers (far above any real model).
MAX_KV_COUNT = 10_000
MAX_STRING_BYTES = 64 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 100_000_000


class _Malformed(Exception):
    """Internal: header data that cannot be parsed safely."""


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise _Malformed("truncated header")
    return data


def _read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_string(handle: BinaryIO) -> bytes:
    length = _read_u64(handle)
    if length > MAX_STRING_BYTES:
        raise _Malformed(f"string length {length} exceeds cap")
    return _read_exact(handle, length)


def _skip_value(handle: BinaryIO, value_type: int) -> None:
    fixed = _FIXED_SIZES.get(value_type)
    if fixed is not None:
        handle.seek(fixed, 1)
        return
    if value_type == _TYPE_STRING:
        length = _read_u64(handle)
        if length > MAX_STRING_BYTES:
            raise _Malformed(f"string length {length} exceeds cap")
        handle.seek(length, 1)
        return
    if value_type == _TYPE_ARRAY:
        element_type = _read_u32(handle)
        count = _read_u64(handle)
        if count > MAX_ARRAY_ELEMENTS:
            raise _Malformed(f"array count {count} exceeds cap")
        element_size = _FIXED_SIZES.get(element_type)
        if element_size is not None:
            handle.seek(element_size * count, 1)
            return
        if element_type == _TYPE_STRING:
            for _ in range(count):
                length = _read_u64(handle)
                if length > MAX_STRING_BYTES:
                    raise _Malformed(f"string length {length} exceeds cap")
                handle.seek(length, 1)
            return
        raise _Malformed(f"unsupported array element type {element_type}")
    raise _Malformed(f"unsupported value type {value_type}")


def read_block_count(path: Path) -> int | None:
    """The ``<arch>.block_count`` value of a GGUF file, or None.

    Never raises: unreadable / non-GGUF / v1 / malformed files all yield
    ``None`` (logged at debug level).
    """
    try:
        with open(path, "rb") as handle:
            if _read_exact(handle, 4) != GGUF_MAGIC:
                raise _Malformed("not a GGUF file")
            version = _read_u32(handle)
            if version not in SUPPORTED_VERSIONS:
                raise _Malformed(f"unsupported GGUF version {version}")
            _read_u64(handle)  # tensor count (unused)
            kv_count = _read_u64(handle)
            if kv_count > MAX_KV_COUNT:
                raise _Malformed(f"kv count {kv_count} exceeds cap")
            for _ in range(kv_count):
                key = _read_string(handle)
                value_type = _read_u32(handle)
                if key.endswith(BLOCK_COUNT_SUFFIX.encode("ascii")):
                    fmt = _INT_FORMATS.get(value_type)
                    if fmt is None:
                        raise _Malformed(
                            f"block_count has non-integer type {value_type}"
                        )
                    value = struct.unpack(
                        fmt, _read_exact(handle, _FIXED_SIZES[value_type])
                    )[0]
                    return int(value) if value > 0 else None
                _skip_value(handle, value_type)
    except (_Malformed, OSError, struct.error) as exc:
        _LOGGER.debug("cannot read block_count from %s: %s", path, exc)
        return None
    return None
