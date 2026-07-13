"""Retry with exponential backoff and request-rate limiting (spec 8).

Both utilities take injectable ``sleep``/``clock`` callables so tests never
perform real waits.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from nlapt.core.errors import LLMRequestError, ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 2
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff parameters: delay n = base_delay * multiplier**n."""

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS
    multiplier: float = DEFAULT_BACKOFF_MULTIPLIER

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValidationError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.base_delay < 0:
            raise ValidationError(f"base_delay must be >= 0, got {self.base_delay}")
        if self.multiplier <= 0:
            raise ValidationError(f"multiplier must be > 0, got {self.multiplier}")

    def delay_for(self, retry_index: int) -> float:
        """Backoff delay before retry number ``retry_index`` (0-based)."""
        return self.base_delay * (self.multiplier**retry_index)


def with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    retry_on: tuple[type[BaseException], ...] = (LLMRequestError,),
) -> T:
    """Call ``fn`` with up to ``policy.max_retries`` retries on ``retry_on``.

    Non-matching exceptions propagate immediately. When retries are
    exhausted, the last matching error is re-raised.
    """
    if not callable(fn):
        raise ValidationError("with_retry requires a callable fn")
    if not isinstance(policy, RetryPolicy):
        raise ValidationError(
            f"policy must be a RetryPolicy, got {type(policy).__name__}"
        )
    attempts = policy.max_retries + 1
    for attempt in range(attempts):
        try:
            return fn()
        except retry_on as exc:
            remaining = attempts - attempt - 1
            if remaining == 0:
                _LOGGER.warning(
                    "retries exhausted after %d attempt(s): %s", attempts, exc
                )
                raise
            delay = policy.delay_for(attempt)
            _LOGGER.debug(
                "attempt %d/%d failed (%s); retrying in %.3fs",
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable: with_retry loop must return or raise")


class MinIntervalLimiter:
    """Enforce a minimum interval between request starts (thread-safe).

    Start slots are scheduled under a lock; the (injectable) sleep happens
    outside the lock so concurrent callers queue up evenly spaced slots.
    """

    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval < 0:
            raise ValidationError(f"min_interval must be >= 0, got {min_interval}")
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start: float | None = None

    def wait(self) -> None:
        """Block (via injected sleep) until this caller's start slot arrives."""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = self._clock()
            scheduled = now if self._next_start is None else max(now, self._next_start)
            delay = scheduled - now
            self._next_start = scheduled + self._min_interval
        if delay > 0:
            _LOGGER.debug("rate limiter sleeping %.3fs", delay)
            self._sleep(delay)
