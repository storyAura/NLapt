"""Tests for nlapt.llm.fallback (ordered provider chain, no network)."""

from __future__ import annotations

import pytest

from nlapt.core.errors import LLMConfigError, LLMRequestError
from nlapt.llm.fallback import MSG_ALL_FAILED, MSG_EMPTY_RESULT, run_fallback_chain


class TestRunFallbackChain:
    def test_returns_first_success(self) -> None:
        assert run_fallback_chain((("a", lambda: "你好"), ("b", lambda: "no"))) == "你好"

    def test_skips_config_error(self) -> None:
        def boom() -> str:
            raise LLMConfigError("missing key")

        assert run_fallback_chain((("a", boom), ("b", lambda: "ok"))) == "ok"

    def test_retries_after_request_error(self) -> None:
        def busy() -> str:
            raise LLMRequestError("too many")

        assert run_fallback_chain((("a", busy), ("b", lambda: "  ok  "))) == "ok"

    def test_empty_result_tries_next(self) -> None:
        assert run_fallback_chain((("a", lambda: "  "), ("b", lambda: "next"))) == "next"

    def test_last_request_error_is_raised(self) -> None:
        def first() -> str:
            raise LLMRequestError("first fail")

        def second() -> str:
            raise LLMRequestError("second fail")

        with pytest.raises(LLMRequestError, match="second fail"):
            run_fallback_chain((("a", first), ("b", second)))

    def test_all_empty_raises(self) -> None:
        with pytest.raises(LLMRequestError, match=MSG_EMPTY_RESULT):
            run_fallback_chain((("a", lambda: ""),))

    def test_no_attempts_raises(self) -> None:
        with pytest.raises(LLMRequestError, match=MSG_ALL_FAILED):
            run_fallback_chain(())
