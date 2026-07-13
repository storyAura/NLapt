"""Tests for nlapt.llm.retry: backoff delays and the min-interval limiter."""

from __future__ import annotations

import threading

import pytest

from nlapt.core.errors import LLMRequestError, LLMTimeoutError, ValidationError
from nlapt.llm.retry import MinIntervalLimiter, RetryPolicy, with_retry


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []
        self._lock = threading.Lock()

    def __call__(self, delay: float) -> None:
        with self._lock:
            self.delays.append(delay)


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


class TestRetryPolicy:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 2
        assert policy.base_delay == 1.0
        assert policy.multiplier == 2.0

    def test_delay_for_is_exponential(self) -> None:
        policy = RetryPolicy(max_retries=3, base_delay=0.5, multiplier=3.0)
        assert policy.delay_for(0) == 0.5
        assert policy.delay_for(1) == 1.5
        assert policy.delay_for(2) == 4.5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_retries": -1},
            {"base_delay": -0.1},
            {"multiplier": 0.0},
            {"multiplier": -2.0},
        ],
    )
    def test_invalid_policy_raises(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            RetryPolicy(**kwargs)


class TestWithRetry:
    def test_success_first_try_no_sleep(self) -> None:
        sleep = SleepRecorder()
        result = with_retry(lambda: 42, RetryPolicy(), sleep=sleep)
        assert result == 42
        assert sleep.delays == []

    def test_fails_then_succeeds_with_backoff_delays(self) -> None:
        sleep = SleepRecorder()
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise LLMRequestError(f"fail {attempts['n']}")
            return "ok"

        policy = RetryPolicy(max_retries=3, base_delay=1.0, multiplier=2.0)
        assert with_retry(flaky, policy, sleep=sleep) == "ok"
        assert attempts["n"] == 3
        assert sleep.delays == [1.0, 2.0]

    def test_exhausted_retries_reraise_last_error(self) -> None:
        sleep = SleepRecorder()
        errors: list[LLMRequestError] = []

        def always_fails() -> None:
            error = LLMRequestError(f"failure {len(errors) + 1}")
            errors.append(error)
            raise error

        policy = RetryPolicy(max_retries=2, base_delay=0.5, multiplier=2.0)
        with pytest.raises(LLMRequestError) as excinfo:
            with_retry(always_fails, policy, sleep=sleep)
        assert excinfo.value is errors[-1]
        assert len(errors) == 3  # initial + 2 retries
        assert sleep.delays == [0.5, 1.0]

    def test_non_retryable_error_propagates_immediately(self) -> None:
        sleep = SleepRecorder()
        calls = {"n": 0}

        def boom() -> None:
            calls["n"] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            with_retry(boom, RetryPolicy(), sleep=sleep)
        assert calls["n"] == 1
        assert sleep.delays == []

    def test_custom_retry_on_filter(self) -> None:
        sleep = SleepRecorder()

        def fails_with_request_error() -> None:
            raise LLMRequestError("nope")

        with pytest.raises(LLMRequestError):
            with_retry(
                fails_with_request_error,
                RetryPolicy(),
                sleep=sleep,
                retry_on=(LLMTimeoutError,),
            )
        assert sleep.delays == []

    def test_timeout_error_retried_by_default(self) -> None:
        # LLMTimeoutError subclasses LLMRequestError, so the default filter catches it.
        sleep = SleepRecorder()
        calls = {"n": 0}

        def times_out_once() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMTimeoutError("slow")
            return "ok"

        assert with_retry(times_out_once, RetryPolicy(), sleep=sleep) == "ok"
        assert sleep.delays == [1.0]

    def test_zero_retries_single_attempt(self) -> None:
        sleep = SleepRecorder()
        calls = {"n": 0}

        def fails() -> None:
            calls["n"] += 1
            raise LLMRequestError("no luck")

        with pytest.raises(LLMRequestError):
            with_retry(fails, RetryPolicy(max_retries=0), sleep=sleep)
        assert calls["n"] == 1
        assert sleep.delays == []

    def test_non_callable_fn_raises(self) -> None:
        with pytest.raises(ValidationError):
            with_retry("not-callable", RetryPolicy())  # type: ignore[arg-type]

    def test_non_policy_raises(self) -> None:
        with pytest.raises(ValidationError):
            with_retry(lambda: 1, "not-a-policy")  # type: ignore[arg-type]


class TestMinIntervalLimiter:
    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValidationError):
            MinIntervalLimiter(-1.0)

    def test_zero_interval_never_sleeps(self) -> None:
        sleep = SleepRecorder()
        limiter = MinIntervalLimiter(0.0, clock=FakeClock([0.0]), sleep=sleep)
        for _ in range(3):
            limiter.wait()
        assert sleep.delays == []

    def test_first_wait_is_immediate(self) -> None:
        sleep = SleepRecorder()
        limiter = MinIntervalLimiter(1.0, clock=FakeClock([10.0]), sleep=sleep)
        limiter.wait()
        assert sleep.delays == []

    def test_second_wait_sleeps_remaining_interval(self) -> None:
        sleep = SleepRecorder()
        clock = FakeClock([0.0, 0.2])
        limiter = MinIntervalLimiter(1.0, clock=clock, sleep=sleep)
        limiter.wait()  # at t=0.0, next slot at 1.0
        limiter.wait()  # at t=0.2 -> must wait 0.8
        assert sleep.delays == pytest.approx([0.8])

    def test_no_sleep_when_interval_already_elapsed(self) -> None:
        sleep = SleepRecorder()
        clock = FakeClock([0.0, 5.0])
        limiter = MinIntervalLimiter(1.0, clock=clock, sleep=sleep)
        limiter.wait()
        limiter.wait()
        assert sleep.delays == []

    def test_thread_safe_slots_are_evenly_spaced(self) -> None:
        sleep = SleepRecorder()
        # Frozen clock: every caller arrives "at the same instant".
        limiter = MinIntervalLimiter(1.0, clock=FakeClock([100.0]), sleep=sleep)
        thread_count = 4
        barrier = threading.Barrier(thread_count)

        def worker() -> None:
            barrier.wait()
            limiter.wait()

        threads = [threading.Thread(target=worker) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # First slot has no delay; the rest are spaced 1.0s apart.
        assert sorted(sleep.delays) == pytest.approx([1.0, 2.0, 3.0])
