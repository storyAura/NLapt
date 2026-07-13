"""Tests for nlapt.core.errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.errors import (
    BatchCancelledError,
    EncodingDetectionError,
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMRequestError,
    LLMTimeoutError,
    NLaptError,
    OperationError,
    RegexPatternError,
    SessionError,
    SnapshotError,
    StorageError,
    ValidationError,
)


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (ValidationError, NLaptError),
        (StorageError, NLaptError),
        (EncodingDetectionError, StorageError),
        (SnapshotError, StorageError),
        (SessionError, StorageError),
        (OperationError, NLaptError),
        (RegexPatternError, OperationError),
        (LLMError, NLaptError),
        (LLMConfigError, LLMError),
        (LLMRequestError, LLMError),
        (LLMTimeoutError, LLMRequestError),
        (LLMOutputError, LLMError),
        (BatchCancelledError, NLaptError),
    ],
)
def test_hierarchy(child: type, parent: type) -> None:
    assert issubclass(child, parent)
    assert issubclass(child, Exception)


def test_message_is_required_and_stored() -> None:
    err = NLaptError("boom")
    assert err.message == "boom"
    assert str(err) == "boom"


@pytest.mark.parametrize("bad", ["", None, 42])
def test_empty_or_non_string_message_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        NLaptError(bad)  # type: ignore[arg-type]


def test_encoding_detection_error_carries_path() -> None:
    p = Path("some/file.txt")
    err = EncodingDetectionError("cannot decode", path=p)
    assert err.path == p
    assert isinstance(err, StorageError)


def test_encoding_detection_error_path_optional() -> None:
    assert EncodingDetectionError("cannot decode").path is None


def test_regex_pattern_error_carries_pattern_and_detail() -> None:
    err = RegexPatternError("bad regex", pattern="[unclosed", detail="unterminated set")
    assert err.pattern == "[unclosed"
    assert err.detail == "unterminated set"
    assert isinstance(err, OperationError)


def test_catching_base_catches_all() -> None:
    with pytest.raises(NLaptError):
        raise LLMTimeoutError("timed out")
