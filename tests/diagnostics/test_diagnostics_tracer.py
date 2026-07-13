"""Tests for nlapt.diagnostics.tracer."""

from __future__ import annotations

import logging

import pytest

from nlapt.core.errors import ValidationError
from nlapt.diagnostics.tracer import trace, truncate_repr

TEST_LOGGER_NAME = "tracer_test_logger"


@pytest.fixture()
def test_logger() -> logging.Logger:
    logger = logging.getLogger(TEST_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    return logger


# ---- truncate_repr ---------------------------------------------------------


def test_truncate_repr_short_value_unchanged() -> None:
    assert truncate_repr([1, 2, 3]) == "[1, 2, 3]"


def test_truncate_repr_long_value_cut_with_marker() -> None:
    value = "x" * 500
    result = truncate_repr(value, limit=50)
    assert result.startswith(repr(value)[:50])
    assert "...(+" in result and result.endswith("chars)")
    assert len(result) < len(repr(value))


def test_truncate_repr_never_raises_on_broken_repr() -> None:
    class Broken:
        def __repr__(self) -> str:
            raise RuntimeError("no repr for you")

    assert "Broken" in truncate_repr(Broken())


@pytest.mark.parametrize("bad_limit", [0, -5, "10", None])
def test_truncate_repr_validates_limit(bad_limit: object) -> None:
    with pytest.raises(ValidationError):
        truncate_repr("value", limit=bad_limit)  # type: ignore[arg-type]


# ---- trace decorator -------------------------------------------------------


def test_trace_logs_call_and_duration(
    test_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    @trace(test_logger)
    def add(a: int, b: int = 0) -> int:
        return a + b

    with caplog.at_level(logging.DEBUG, logger=TEST_LOGGER_NAME):
        assert add(1, b=2) == 3

    messages = [rec.message for rec in caplog.records]
    assert any(m.startswith("call ") and "add" in m and "b=2" in m for m in messages)
    assert any(m.startswith("done ") and "ms" in m for m in messages)


def test_trace_logs_and_reraises_exceptions(
    test_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    @trace(test_logger)
    def explode() -> None:
        raise ValueError("kaboom")

    with caplog.at_level(logging.DEBUG, logger=TEST_LOGGER_NAME):
        with pytest.raises(ValueError, match="kaboom"):
            explode()

    assert any(
        m.startswith("raise ") and "ValueError" in m for m in (r.message for r in caplog.records)
    )


def test_trace_truncates_huge_arguments(
    test_logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    @trace(test_logger)
    def consume(blob: str) -> int:
        return len(blob)

    with caplog.at_level(logging.DEBUG, logger=TEST_LOGGER_NAME):
        consume("y" * 5000)

    call_line = next(m for m in (r.message for r in caplog.records) if m.startswith("call "))
    assert len(call_line) < 1000
    assert "...(+" in call_line


def test_trace_preserves_function_metadata(test_logger: logging.Logger) -> None:
    @trace(test_logger)
    def documented() -> None:
        """Docstring survives."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring survives."


def test_trace_default_logger_and_bare_form() -> None:
    @trace()
    def with_parens(x: int) -> int:
        return x * 2

    @trace
    def without_parens(x: int) -> int:
        return x + 1

    assert with_parens(4) == 8
    assert without_parens(4) == 5
