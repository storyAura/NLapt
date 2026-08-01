"""NLapt - Natural Language Annotation Processing Tools (core library).

The package root stays import-light: ``NLaptApp`` is resolved lazily via
PEP 562 so importing :mod:`nlapt` never pulls in the full facade wiring.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.3.0"

__all__ = ["NLaptApp", "__version__"]

_LAZY_EXPORTS = {"NLaptApp": "nlapt.app"}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for heavyweight exports."""
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
