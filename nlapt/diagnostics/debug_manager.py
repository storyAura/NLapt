"""Process-wide debug/QA hub: runtime log levels, metrics, error history,
event tapping, and QA bundle export.

The exported bundle NEVER contains raw API keys — configuration is always
serialized through :func:`nlapt.core.config.masked_config_dict`.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import threading
import time
import zipfile
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from nlapt.core.errors import StorageError, ValidationError
from nlapt.diagnostics.logging_setup import get_logger
from nlapt.diagnostics.tracer import truncate_repr

if TYPE_CHECKING:
    from nlapt.core.config import AppConfig
    from nlapt.core.events import Event, EventBus

_LOGGER = get_logger(__name__)

ERROR_HISTORY_LIMIT = 50
EVENT_COUNTER_PREFIX = "events."
LOG_FILE_GLOB = "*.log*"
BUNDLE_METRICS_NAME = "metrics.json"
BUNDLE_ERRORS_NAME = "errors.json"
BUNDLE_ENV_NAME = "environment.json"
BUNDLE_CONFIG_NAME = "config.json"
BUNDLE_LOGS_DIR = "logs"
BUNDLE_EXTRA_DIR = "extra"
_ERROR_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class DebugManager:
    """Thread-safe debug/QA hub. Get the process singleton via get_debug_manager()."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._metrics: dict[str, int] = {}
        self._errors: deque[str] = deque(maxlen=ERROR_HISTORY_LIMIT)

    def set_level(self, subsystem: str, level: str) -> None:
        """Change a subsystem logger level at runtime, e.g. ('nlapt.llm', 'DEBUG')."""
        if not isinstance(subsystem, str) or not subsystem:
            raise ValidationError(f"subsystem must be a non-empty string, got {subsystem!r}")
        if not isinstance(level, str) or level.upper() not in logging.getLevelNamesMapping():
            raise ValidationError(f"unknown log level: {level!r}")
        logging.getLogger(subsystem).setLevel(level.upper())
        _LOGGER.info("log level of %s set to %s", subsystem, level.upper())

    def incr(self, counter: str, by: int = 1) -> None:
        """Increment a named metric counter (thread-safe)."""
        if not isinstance(counter, str) or not counter:
            raise ValidationError(f"counter must be a non-empty string, got {counter!r}")
        if not isinstance(by, int) or isinstance(by, bool):
            raise ValidationError(f"by must be an integer, got {by!r}")
        with self._lock:
            self._metrics[counter] = self._metrics.get(counter, 0) + by

    def metrics(self) -> Mapping[str, int]:
        """Immutable snapshot of all counters."""
        with self._lock:
            return MappingProxyType(dict(self._metrics))

    def record_error(self, source: str, error: BaseException) -> None:
        """Remember an error (keeps the most recent ERROR_HISTORY_LIMIT records)."""
        if not isinstance(source, str) or not source:
            raise ValidationError(f"source must be a non-empty string, got {source!r}")
        if not isinstance(error, BaseException):
            raise ValidationError(f"error must be an exception, got {type(error).__name__}")
        stamp = time.strftime(_ERROR_TIME_FORMAT)
        record = f"{stamp} {source}: {type(error).__name__}: {error}"
        with self._lock:
            self._errors.append(record)

    def recent_errors(self) -> tuple[str, ...]:
        """Recorded error strings, oldest first."""
        with self._lock:
            return tuple(self._errors)

    def tap_events(self, bus: EventBus) -> Callable[[], None]:
        """DEBUG-log (and count) every event on ``bus``. Returns an untap callable."""

        def _on_event(event: Event) -> None:
            _LOGGER.debug(
                "event %s payload=%s", event.name, truncate_repr(dict(event.payload))
            )
            self.incr(f"{EVENT_COUNTER_PREFIX}{event.name}")

        return bus.subscribe(None, _on_event)

    def export_bundle(
        self,
        target_zip: Path,
        *,
        log_dir: Path | None = None,
        config: AppConfig | None = None,
        extra_files: Sequence[Path] = (),
    ) -> Path:
        """Export a QA zip: logs + masked config + metrics + errors + env info.

        API keys are always masked via ``masked_config_dict``; the raw config
        is never written. Raises StorageError on IO failure.
        """
        target = Path(target_zip)
        missing = [str(p) for p in extra_files if not Path(p).is_file()]
        if missing:
            raise ValidationError(f"extra_files not found: {', '.join(missing)}")
        try:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(BUNDLE_METRICS_NAME, _to_json(dict(self.metrics())))
                bundle.writestr(BUNDLE_ERRORS_NAME, _to_json(list(self.recent_errors())))
                bundle.writestr(BUNDLE_ENV_NAME, _to_json(_environment_info()))
                if config is not None:
                    from nlapt.core.config import masked_config_dict

                    bundle.writestr(BUNDLE_CONFIG_NAME, _to_json(masked_config_dict(config)))
                if log_dir is not None:
                    for log_file in sorted(Path(log_dir).glob(LOG_FILE_GLOB)):
                        bundle.write(log_file, arcname=f"{BUNDLE_LOGS_DIR}/{log_file.name}")
                for extra in extra_files:
                    extra_path = Path(extra)
                    bundle.write(extra_path, arcname=f"{BUNDLE_EXTRA_DIR}/{extra_path.name}")
        except OSError as exc:
            _cleanup_partial(target)
            raise StorageError(f"debug bundle export to {target} failed: {exc}") from exc
        _LOGGER.info("debug bundle exported to %s", target)
        return target


def _to_json(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _environment_info() -> dict[str, str]:
    import nlapt

    return {
        "nlapt_version": nlapt.__version__,
        "python": sys.version,
        "platform": platform.platform(),
    }


def _cleanup_partial(target: Path) -> None:
    """Best-effort removal of a partially written bundle; never raises."""
    try:
        target.unlink(missing_ok=True)
    except OSError:
        _LOGGER.warning("could not remove partial bundle %s", target)


_MANAGER_LOCK = threading.Lock()
_MANAGER: DebugManager | None = None


def get_debug_manager() -> DebugManager:
    """Return the process-wide DebugManager singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = DebugManager()
        return _MANAGER
