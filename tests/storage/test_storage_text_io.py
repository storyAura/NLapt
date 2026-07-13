"""Tests for nlapt.storage.text_io (encoding detection, spec 2.2)."""

from __future__ import annotations

import codecs
from pathlib import Path

import pytest

import nlapt.storage.text_io as text_io
from nlapt.core.errors import EncodingDetectionError, StorageError, ValidationError
from nlapt.storage.text_io import CANDIDATE_ENCODINGS, read_text_detect, write_caption

CHINESE_TEXT = "一个女孩，微笑着看向镜头"


def test_candidate_encodings_contract_order() -> None:
    assert CANDIDATE_ENCODINGS == (
        "utf-8", "utf-8-sig", "gb18030", "big5", "shift_jis", "latin-1"
    )


def test_plain_utf8_no_conversion_needed(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes("1girl, smile\n".encode("utf-8"))
    result = read_text_detect(path)
    assert result.text == "1girl, smile\n"
    assert result.encoding == "utf-8"
    assert result.needs_conversion is False


def test_empty_file_is_plain_utf8(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    result = read_text_detect(path)
    assert result.text == ""
    assert result.needs_conversion is False


def test_utf8_bom_detected_and_flagged(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(codecs.BOM_UTF8 + "1girl, smile".encode("utf-8"))
    result = read_text_detect(path)
    assert result.text == "1girl, smile"  # BOM stripped from text
    assert "﻿" not in result.text
    assert result.encoding == "utf-8-sig"
    assert result.needs_conversion is True


def test_gb18030_round_trip_detection(tmp_path: Path) -> None:
    data = CHINESE_TEXT.encode("gb18030")
    # precondition: the payload really is invalid UTF-8 so detection must fall through
    with pytest.raises(UnicodeDecodeError):
        data.decode("utf-8")
    path = tmp_path / "gbk.txt"
    path.write_bytes(data)
    result = read_text_detect(path)
    assert result.text == CHINESE_TEXT
    assert result.encoding == "gb18030"
    assert result.needs_conversion is True


def test_gb18030_convert_and_resave_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "gbk.txt"
    path.write_bytes(CHINESE_TEXT.encode("gb18030"))
    detected = read_text_detect(path)
    write_caption(path, detected.text)  # convert to UTF-8 as the app would
    reread = read_text_detect(path)
    assert reread.text == CHINESE_TEXT
    assert reread.encoding == "utf-8"
    assert reread.needs_conversion is False


def test_crlf_and_cr_normalized_to_lf(tmp_path: Path) -> None:
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"line1\r\nline2\rline3\n")
    result = read_text_detect(path)
    assert result.text == "line1\nline2\nline3\n"
    assert result.encoding == "utf-8"


def test_latin1_fallback(tmp_path: Path) -> None:
    path = tmp_path / "latin.txt"
    path.write_bytes(b"caf\xe9")  # invalid for utf-8/gb18030/big5/shift_jis
    result = read_text_detect(path)
    assert result.text == "café"
    assert result.encoding == "latin-1"
    assert result.needs_conversion is True


def test_missing_file_raises_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        read_text_detect(tmp_path / "absent.txt")


def test_nothing_decodes_raises_encoding_detection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # latin-1 decodes any byte string, so shrink the candidate list to force the error.
    monkeypatch.setattr(text_io, "CANDIDATE_ENCODINGS", ("utf-8",))
    path = tmp_path / "gbk.txt"
    path.write_bytes(CHINESE_TEXT.encode("gb18030"))
    with pytest.raises(EncodingDetectionError) as excinfo:
        read_text_detect(path)
    assert excinfo.value.path == path


def test_write_caption_utf8_no_bom_lf(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    write_caption(path, "first line\r\nsecond\r third")
    raw = path.read_bytes()
    assert not raw.startswith(codecs.BOM_UTF8)
    assert raw == "first line\nsecond\n third".encode("utf-8")


def test_write_caption_rejects_non_string(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        write_caption(tmp_path / "x.txt", 123)  # type: ignore[arg-type]


def test_text_read_result_is_frozen(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    result = read_text_detect(path)
    with pytest.raises(Exception):
        result.text = "changed"  # type: ignore[misc]
