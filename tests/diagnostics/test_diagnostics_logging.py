"""Tests for nlapt.diagnostics.logging_setup and nlapt.diagnostics.crash."""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path

import pytest

from nlapt.core.errors import ValidationError
from nlapt.diagnostics.crash import install_crash_handler
from nlapt.diagnostics.logging_setup import (
    LOG_FILE_NAME,
    LOGGER_ROOT,
    configure_logging,
    get_logger,
)

_MANAGED_ATTR = "_nlapt_managed"


def _managed_handlers() -> list[logging.Handler]:
    logger = logging.getLogger(LOGGER_ROOT)
    return [h for h in logger.handlers if getattr(h, _MANAGED_ATTR, False)]


@pytest.fixture(autouse=True)
def _reset_nlapt_logger():
    yield
    logger = logging.getLogger(LOGGER_ROOT)
    for handler in _managed_handlers():
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


@pytest.fixture()
def _restore_excepthooks():
    prev_sys = sys.excepthook
    prev_threading = threading.excepthook
    yield
    sys.excepthook = prev_sys
    threading.excepthook = prev_threading


# ---- configure_logging -----------------------------------------------------


def test_configure_creates_log_file_on_first_record(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, level="INFO")
    get_logger("nlapt.test.filewriter").info("hello file")
    log_file = tmp_path / LOG_FILE_NAME
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hello file" in content
    assert "INFO" in content


def test_configure_is_idempotent_no_duplicate_handlers(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path)
    configure_logging(log_dir=tmp_path)
    configure_logging(log_dir=tmp_path)
    assert len(_managed_handlers()) == 2  # console + file, never more
    get_logger("nlapt.test.once").warning("only-once-marker")
    content = (tmp_path / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert content.count("only-once-marker") == 1


def test_configure_without_log_dir_console_only() -> None:
    configure_logging()
    handlers = _managed_handlers()
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)


def test_level_filtering(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, level="WARNING")
    logger = get_logger("nlapt.test.level")
    logger.info("info-suppressed")
    logger.error("error-shown")
    content = (tmp_path / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "info-suppressed" not in content
    assert "error-shown" in content


def test_json_lines_format(tmp_path: Path) -> None:
    configure_logging(log_dir=tmp_path, level="DEBUG", json_lines=True)
    get_logger("nlapt.test.json").debug("structured message")
    lines = (tmp_path / LOG_FILE_NAME).read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]
    match = [r for r in records if r["message"] == "structured message"]
    assert match and match[0]["level"] == "DEBUG"
    assert match[0]["logger"] == "nlapt.test.json"
    assert "line" in match[0] and "func" in match[0]


def test_invalid_level_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        configure_logging(level="LOUD")
    with pytest.raises(ValidationError):
        configure_logging(level="")


# ---- get_logger ------------------------------------------------------------


def test_get_logger_names() -> None:
    assert get_logger("nlapt").name == "nlapt"
    assert get_logger("nlapt.llm").name == "nlapt.llm"
    assert get_logger("myplugin").name == "nlapt.myplugin"
    # a name merely *starting* with "nlapt" is not inside the hierarchy
    assert get_logger("nlaptx").name == "nlapt.nlaptx"


def test_get_logger_rejects_bad_names() -> None:
    with pytest.raises(ValidationError):
        get_logger("")
    with pytest.raises(ValidationError):
        get_logger(None)  # type: ignore[arg-type]


# ---- crash handler ---------------------------------------------------------


def test_crash_handler_writes_crash_file(
    tmp_path: Path, _restore_excepthooks: None
) -> None:
    seen: list[type[BaseException]] = []
    sys.excepthook = lambda t, v, tb: seen.append(t)  # silent previous hook
    install_crash_handler(tmp_path)
    try:
        raise RuntimeError("boom for crash test")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    crash_files = list(tmp_path.glob("crash-*.log"))
    assert len(crash_files) == 1
    content = crash_files[0].read_text(encoding="utf-8")
    assert "RuntimeError" in content
    assert "boom for crash test" in content
    assert seen == [RuntimeError]  # previous hook still chained


def test_crash_handler_collision_gets_suffixed_file(
    tmp_path: Path, _restore_excepthooks: None
) -> None:
    sys.excepthook = lambda t, v, tb: None
    install_crash_handler(tmp_path)
    for _ in range(2):  # same second -> name collision handled with _2 suffix
        try:
            raise ValueError("twice")
        except ValueError:
            sys.excepthook(*sys.exc_info())
    assert len(list(tmp_path.glob("crash-*.log"))) == 2


def test_crash_handler_ignores_keyboard_interrupt(
    tmp_path: Path, _restore_excepthooks: None
) -> None:
    sys.excepthook = lambda t, v, tb: None
    install_crash_handler(tmp_path)
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        sys.excepthook(*sys.exc_info())
    assert list(tmp_path.glob("crash-*.log")) == []


def test_crash_handler_covers_thread_exceptions(
    tmp_path: Path, _restore_excepthooks: None
) -> None:
    threading.excepthook = lambda args: None  # silence previous hook
    install_crash_handler(tmp_path)

    def boom() -> None:
        raise RuntimeError("thread crash")

    thread = threading.Thread(target=boom)
    thread.start()
    thread.join()
    crash_files = list(tmp_path.glob("crash-*.log"))
    assert len(crash_files) == 1
    assert "thread crash" in crash_files[0].read_text(encoding="utf-8")


def test_install_crash_handler_validates_log_dir() -> None:
    with pytest.raises(ValidationError):
        install_crash_handler("not-a-path")  # type: ignore[arg-type]
