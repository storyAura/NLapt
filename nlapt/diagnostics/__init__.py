"""Diagnostics: logging setup, call tracing, debug manager, crash handler.

``get_logger`` / ``configure_logging`` / ``trace`` are exported eagerly.
``DebugManager`` / ``get_debug_manager`` / ``install_crash_handler`` are
resolved lazily (PEP 562) because :mod:`nlapt.diagnostics.debug_manager`
depends on :mod:`nlapt.core.config`, which itself needs ``get_logger`` —
laziness keeps that import chain acyclic.
"""

from __future__ import annotations

from typing import Any

from nlapt.diagnostics.logging_setup import LOGGER_ROOT, configure_logging, get_logger
from nlapt.diagnostics.tracer import trace, truncate_repr

__all__ = [
    "LOGGER_ROOT",
    "configure_logging",
    "get_logger",
    "trace",
    "truncate_repr",
    "DebugManager",
    "get_debug_manager",
    "install_crash_handler",
]

_LAZY_EXPORTS = {
    "DebugManager": "nlapt.diagnostics.debug_manager",
    "get_debug_manager": "nlapt.diagnostics.debug_manager",
    "install_crash_handler": "nlapt.diagnostics.crash",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
