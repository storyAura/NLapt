"""Central logging configuration for the ``nlapt`` logger hierarchy."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from nlapt.core.errors import StorageError, ValidationError

LOGGER_ROOT = "nlapt"
LOG_FILE_NAME = "nlapt.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s"

# Attribute stamped on handlers created here so reconfiguration replaces
# exactly our handlers and never duplicates or removes foreign ones.
_MANAGED_ATTR = "_nlapt_managed"


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per log line (for machine-readable QA logs)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "func": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _resolve_level(level: str) -> int:
    """Translate a level name to its numeric value; ValidationError if unknown."""
    if not isinstance(level, str) or not level:
        raise ValidationError(f"log level must be a non-empty string, got {level!r}")
    mapping = logging.getLevelNamesMapping()
    upper = level.upper()
    if upper not in mapping:
        raise ValidationError(f"unknown log level: {level!r}")
    return mapping[upper]


def _remove_managed_handlers(logger: logging.Logger) -> None:
    for handler in [h for h in logger.handlers if getattr(h, _MANAGED_ATTR, False)]:
        logger.removeHandler(handler)
        handler.close()


def configure_logging(
    log_dir: Path | None = None, level: str = "INFO", json_lines: bool = False
) -> logging.Logger:
    """Configure the root ``nlapt`` logger. Idempotent (no duplicate handlers).

    Adds a console handler always, plus a rotating file handler
    (``nlapt.log``, 5 MB x 3) when ``log_dir`` is given.
    """
    numeric_level = _resolve_level(level)
    logger = logging.getLogger(LOGGER_ROOT)
    _remove_managed_handlers(logger)

    formatter: logging.Formatter = (
        JsonLinesFormatter() if json_lines else logging.Formatter(LOG_FORMAT)
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    setattr(console, _MANAGED_ATTR, True)
    logger.addHandler(console)

    if log_dir is not None:
        directory = Path(log_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"cannot create log directory {directory}: {exc}") from exc
        file_handler = RotatingFileHandler(
            directory / LOG_FILE_NAME,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)
        setattr(file_handler, _MANAGED_ATTR, True)
        logger.addHandler(file_handler)

    logger.setLevel(numeric_level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``nlapt`` hierarchy.

    Names already under ``nlapt`` are used as-is; anything else becomes a
    child of the root (``get_logger("foo") -> "nlapt.foo"``).
    """
    if not isinstance(name, str) or not name:
        raise ValidationError(f"logger name must be a non-empty string, got {name!r}")
    if name == LOGGER_ROOT or name.startswith(LOGGER_ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_ROOT}.{name}")
