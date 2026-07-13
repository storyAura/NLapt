"""Lightweight call tracing for QA/debugging (DEBUG-level, zero deps)."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from nlapt.core.errors import ValidationError
from nlapt.diagnostics.logging_setup import get_logger

REPR_LIMIT = 200
_MS_PER_SECOND = 1000.0

_F = TypeVar("_F", bound=Callable[..., Any])


def truncate_repr(value: Any, limit: int = REPR_LIMIT) -> str:
    """``repr(value)`` cut to ``limit`` chars with an overflow marker.

    Never raises: un-repr-able objects yield a type-based placeholder.
    """
    if not isinstance(limit, int) or limit <= 0:
        raise ValidationError(f"limit must be a positive integer, got {limit!r}")
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - diagnostics must never raise from repr
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(+{len(text) - limit} chars)"


def _format_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    parts = [truncate_repr(a) for a in args]
    parts += [f"{k}={truncate_repr(v)}" for k, v in kwargs.items()]
    return truncate_repr(", ".join(parts))


def trace(logger: logging.Logger | None = None) -> Callable[[_F], _F]:
    """Decorator: DEBUG-log call name, truncated args, duration ms, exceptions.

    Exceptions are logged and re-raised untouched. Usable as ``@trace()`` or
    ``@trace(my_logger)``; the bare ``@trace`` form is also tolerated.
    """
    # Tolerate the decorator being applied without parentheses.
    if callable(logger) and not isinstance(logger, logging.Logger):
        return trace()(logger)  # type: ignore[return-value]

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger if logger is not None else get_logger(fn.__module__)
            log.debug("call %s(%s)", fn.__qualname__, _format_call(args, kwargs))
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                elapsed_ms = (time.perf_counter() - start) * _MS_PER_SECOND
                log.debug(
                    "raise %s from %s after %.2f ms: %s",
                    type(exc).__name__,
                    fn.__qualname__,
                    elapsed_ms,
                    truncate_repr(str(exc)),
                )
                raise
            elapsed_ms = (time.perf_counter() - start) * _MS_PER_SECOND
            log.debug("done %s in %.2f ms", fn.__qualname__, elapsed_ms)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
