"""Crash handler: dump uncaught exceptions to timestamped crash log files.

Installs ``sys.excepthook`` and ``threading.excepthook`` wrappers that write a
full traceback to ``<log_dir>/crash-YYYYMMDD-HHMMSS.log`` and then delegate to
the previously installed hooks. The handler itself must never raise.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path
from types import TracebackType

from nlapt.core.errors import StorageError, ValidationError
from nlapt.diagnostics.logging_setup import get_logger

_LOGGER = get_logger(__name__)

CRASH_FILE_PREFIX = "crash-"
CRASH_FILE_SUFFIX = ".log"
CRASH_TIME_FORMAT = "%Y%m%d-%H%M%S"
CRASH_FILE_ENCODING = "utf-8"
# Exceptions that mean "user asked to stop", not a crash.
_PASSTHROUGH_EXCEPTIONS: tuple[type[BaseException], ...] = (KeyboardInterrupt, SystemExit)


def _crash_file_path(log_dir: Path) -> Path:
    stamp = time.strftime(CRASH_TIME_FORMAT)
    candidate = log_dir / f"{CRASH_FILE_PREFIX}{stamp}{CRASH_FILE_SUFFIX}"
    counter = 2
    while candidate.exists():
        candidate = log_dir / f"{CRASH_FILE_PREFIX}{stamp}_{counter}{CRASH_FILE_SUFFIX}"
        counter += 1
    return candidate


def _write_crash_file(
    log_dir: Path,
    exc_type: type[BaseException],
    exc_value: BaseException | None,
    exc_tb: TracebackType | None,
) -> None:
    """Best-effort crash dump; never raises (we are already crashing)."""
    try:
        target = _crash_file_path(log_dir)
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        target.write_text(text, encoding=CRASH_FILE_ENCODING)
        _LOGGER.critical("uncaught %s written to %s", exc_type.__name__, target)
    except Exception:  # noqa: BLE001 - crash handling must never raise
        _LOGGER.exception("failed to write crash log to %s", log_dir)


def install_crash_handler(log_dir: Path) -> None:
    """Install sys/threading excepthooks that dump crashes into ``log_dir``."""
    if not isinstance(log_dir, Path):
        raise ValidationError(f"log_dir must be a Path, got {type(log_dir).__name__}")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError(f"cannot create crash log directory {log_dir}: {exc}") from exc

    previous_sys_hook = sys.excepthook
    previous_threading_hook = threading.excepthook

    def _sys_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        if not issubclass(exc_type, _PASSTHROUGH_EXCEPTIONS):
            _write_crash_file(log_dir, exc_type, exc_value, exc_tb)
        previous_sys_hook(exc_type, exc_value, exc_tb)

    def _threading_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not None and not issubclass(
            args.exc_type, _PASSTHROUGH_EXCEPTIONS
        ):
            _write_crash_file(log_dir, args.exc_type, args.exc_value, args.exc_traceback)
        previous_threading_hook(args)

    sys.excepthook = _sys_hook
    threading.excepthook = _threading_hook
    _LOGGER.info("crash handler installed; dumps go to %s", log_dir)
