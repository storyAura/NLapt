"""Tests for nlapt.diagnostics.debug_manager."""

from __future__ import annotations

import json
import logging
import threading
import zipfile
from pathlib import Path

import pytest

from nlapt.core.config import AppConfig, LLMProfile
from nlapt.core.errors import StorageError, ValidationError
from nlapt.core.events import EventBus
from nlapt.diagnostics.debug_manager import (
    ERROR_HISTORY_LIMIT,
    DebugManager,
    get_debug_manager,
)

SECRET = "sk-topsecret0987654321"


@pytest.fixture()
def manager() -> DebugManager:
    return DebugManager()  # fresh instance; the singleton is tested separately


@pytest.fixture(autouse=True)
def _propagate_nlapt_logs():
    logger = logging.getLogger("nlapt")
    previous = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = previous


def _config_with_secret() -> AppConfig:
    return AppConfig(
        profiles=(LLMProfile(name="p", api_type="openai", base_url="u", api_key=SECRET),),
        active_profile="p",
    )


# ---- metrics ----------------------------------------------------------------


def test_incr_and_metrics(manager: DebugManager) -> None:
    manager.incr("llm.requests")
    manager.incr("llm.requests", by=2)
    manager.incr("saves")
    assert manager.metrics() == {"llm.requests": 3, "saves": 1}


def test_metrics_snapshot_is_immutable(manager: DebugManager) -> None:
    manager.incr("c")
    snapshot = manager.metrics()
    with pytest.raises(TypeError):
        snapshot["c"] = 99  # type: ignore[index]
    manager.incr("c")
    assert snapshot["c"] == 1  # old snapshot unaffected


def test_incr_thread_safety(manager: DebugManager) -> None:
    def bump() -> None:
        for _ in range(500):
            manager.incr("hits")

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert manager.metrics()["hits"] == 2000


def test_incr_validates_input(manager: DebugManager) -> None:
    with pytest.raises(ValidationError):
        manager.incr("")
    with pytest.raises(ValidationError):
        manager.incr("x", by="2")  # type: ignore[arg-type]


# ---- error history ----------------------------------------------------------


def test_record_error_keeps_last_n(manager: DebugManager) -> None:
    for i in range(ERROR_HISTORY_LIMIT + 10):
        manager.record_error("src", ValueError(f"err-{i}"))
    errors = manager.recent_errors()
    assert len(errors) == ERROR_HISTORY_LIMIT
    assert "err-10" in errors[0]  # oldest surviving record
    assert f"err-{ERROR_HISTORY_LIMIT + 9}" in errors[-1]


def test_record_error_format(manager: DebugManager) -> None:
    manager.record_error("llm.translate", TimeoutError("too slow"))
    (record,) = manager.recent_errors()
    assert "llm.translate" in record
    assert "TimeoutError" in record
    assert "too slow" in record


def test_record_error_validates_input(manager: DebugManager) -> None:
    with pytest.raises(ValidationError):
        manager.record_error("", ValueError("x"))
    with pytest.raises(ValidationError):
        manager.record_error("src", "not an exception")  # type: ignore[arg-type]


# ---- runtime log levels -------------------------------------------------------


def test_set_level_changes_logger(manager: DebugManager) -> None:
    manager.set_level("nlapt.test.subsystem", "DEBUG")
    assert logging.getLogger("nlapt.test.subsystem").level == logging.DEBUG
    manager.set_level("nlapt.test.subsystem", "warning")  # case-insensitive
    assert logging.getLogger("nlapt.test.subsystem").level == logging.WARNING


def test_set_level_validates(manager: DebugManager) -> None:
    with pytest.raises(ValidationError):
        manager.set_level("", "DEBUG")
    with pytest.raises(ValidationError):
        manager.set_level("nlapt.x", "SHOUTING")


# ---- event tap ----------------------------------------------------------------


def test_tap_events_logs_and_counts(
    manager: DebugManager, caplog: pytest.LogCaptureFixture
) -> None:
    bus = EventBus()
    untap = manager.tap_events(bus)
    with caplog.at_level(logging.DEBUG, logger="nlapt.diagnostics.debug_manager"):
        bus.publish("file_saved", key="a.txt")
    assert manager.metrics()["events.file_saved"] == 1
    assert any("file_saved" in rec.message for rec in caplog.records)

    untap()
    bus.publish("file_saved", key="b.txt")
    assert manager.metrics()["events.file_saved"] == 1  # no longer counted


# ---- bundle export -------------------------------------------------------------


def test_export_bundle_contents_and_masking(manager: DebugManager, tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "nlapt.log").write_text("log line one\n", encoding="utf-8")
    (log_dir / "nlapt.log.1").write_text("rotated\n", encoding="utf-8")
    extra = tmp_path / "notes.txt"
    extra.write_text("qa notes", encoding="utf-8")

    manager.incr("llm.requests", by=7)
    manager.record_error("batch", RuntimeError("worker died"))

    target = tmp_path / "bundle.zip"
    result = manager.export_bundle(
        target, log_dir=log_dir, config=_config_with_secret(), extra_files=[extra]
    )
    assert result == target

    with zipfile.ZipFile(target) as bundle:
        names = set(bundle.namelist())
        assert {
            "metrics.json", "errors.json", "environment.json", "config.json",
            "logs/nlapt.log", "logs/nlapt.log.1", "extra/notes.txt",
        } <= names

        everything = "".join(bundle.read(n).decode("utf-8") for n in names)
        assert SECRET not in everything  # NEVER raw api keys

        config_data = json.loads(bundle.read("config.json"))
        assert config_data["profiles"][0]["api_key"] == "sk-***4321"
        metrics = json.loads(bundle.read("metrics.json"))
        assert metrics["llm.requests"] == 7
        errors = json.loads(bundle.read("errors.json"))
        assert any("worker died" in e for e in errors)
        env = json.loads(bundle.read("environment.json"))
        assert env["nlapt_version"] == "0.1.0"


def test_export_bundle_minimal_without_optionals(manager: DebugManager, tmp_path: Path) -> None:
    target = tmp_path / "minimal.zip"
    manager.export_bundle(target)
    with zipfile.ZipFile(target) as bundle:
        names = set(bundle.namelist())
    assert names == {"metrics.json", "errors.json", "environment.json"}


def test_export_bundle_missing_extra_file_raises(manager: DebugManager, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        manager.export_bundle(tmp_path / "b.zip", extra_files=[tmp_path / "ghost.txt"])
    assert not (tmp_path / "b.zip").exists()


def test_export_bundle_unwritable_target_raises_storage_error(
    manager: DebugManager, tmp_path: Path
) -> None:
    with pytest.raises(StorageError):
        manager.export_bundle(tmp_path / "no" / "dir" / "bundle.zip")


# ---- singleton -------------------------------------------------------------------


def test_get_debug_manager_is_singleton() -> None:
    assert get_debug_manager() is get_debug_manager()
    assert isinstance(get_debug_manager(), DebugManager)


# ---- lazy package exports (PEP 562 in nlapt.diagnostics) --------------------


def test_diagnostics_lazy_exports_resolve() -> None:
    import nlapt.diagnostics as diagnostics

    # Force the lazy path even if another test already cached the attributes.
    for name in ("DebugManager", "get_debug_manager", "install_crash_handler"):
        diagnostics.__dict__.pop(name, None)
        assert name in dir(diagnostics)
        assert callable(getattr(diagnostics, name))
    assert diagnostics.get_debug_manager is get_debug_manager
    with pytest.raises(AttributeError):
        diagnostics.no_such_symbol  # noqa: B018
