"""Tests for nlapt_gui.local_bridge: settings, files, downloads, server."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from nlapt.core.errors import DownloadCancelledError, DownloadError, LocalServerError
from nlapt.local.catalog import ModelFamily, QuantFile
from nlapt.local.hardware import HardwareInfo
from nlapt.local.settings import load_local_settings

from nlapt_gui.local_bridge import (
    DOWNLOAD_CANCELLED,
    DOWNLOAD_ERROR,
    DOWNLOAD_OK,
    SERVER_ERROR,
    SERVER_RUNNING,
    SERVER_STARTING,
    SERVER_STOPPED,
    LocalBridge,
)
from nlapt_gui.resources import app_data_dir

FAMILY_ID = "test-fam"
QUANT_LABEL = "Q4"
MODEL_BYTES = 10
MMPROJ_BYTES = 4


def make_family(*, vision: bool = True) -> ModelFamily:
    return ModelFamily(
        family_id=FAMILY_ID,
        series_id="gemma4",
        name="Test Fam",
        repo_id="tester/fam-GGUF",
        downloads=5,
        params_label="1B",
        vision=vision,
        kv_bytes_per_token=1_000,
        quants=(QuantFile(QUANT_LABEL, "fam-Q4.gguf", MODEL_BYTES, recommended=True),),
        mmproj_filename="fam-mmproj.gguf" if vision else "",
        mmproj_bytes=MMPROJ_BYTES if vision else 0,
    )


class FakeManager:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.running = False
        self.started_specs: list[object] = []
        self.stop_calls = 0

    def is_running(self) -> bool:
        return self.running

    @property
    def current_base_url(self) -> str:
        return "http://127.0.0.1:18434/v1" if self.running else ""

    def start(self, spec) -> str:  # noqa: ANN001
        self.started_specs.append(spec)
        if self.fail:
            raise LocalServerError("boom")
        self.running = True
        return f"http://127.0.0.1:{spec.port}/v1"

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


@pytest.fixture()
def family(monkeypatch: pytest.MonkeyPatch) -> ModelFamily:
    fam = make_family()
    monkeypatch.setattr("nlapt_gui.local_bridge.find_family", lambda fid: fam)
    monkeypatch.setattr(
        "nlapt_gui.local_bridge.find_quant", lambda f, label: f.quants[0]
    )
    return fam


@pytest.fixture()
def bridge(qtbot) -> LocalBridge:
    return LocalBridge(manager=FakeManager())


def write_downloaded(bridge: LocalBridge, fam: ModelFamily) -> None:
    quant = fam.quants[0]
    model = bridge.model_file(fam, quant)
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"x" * quant.size_bytes)
    mmproj = bridge.mmproj_file(fam)
    if mmproj is not None:
        mmproj.write_bytes(b"y" * fam.mmproj_bytes)


class TestSettings:
    def test_update_settings_persists(self, bridge: LocalBridge) -> None:
        bridge.update_settings(parallel=8, server_path="srv.exe")
        assert bridge.settings.parallel == 8
        stored = load_local_settings(app_data_dir() / "local_llm.json")
        assert stored.parallel == 8
        assert stored.server_path == "srv.exe"

    def test_models_dir_default_under_app_data(self, bridge: LocalBridge) -> None:
        assert bridge.models_dir() == app_data_dir() / "models"

    def test_models_dir_override(self, bridge: LocalBridge, tmp_path: Path) -> None:
        bridge.update_settings(models_dir=str(tmp_path / "elsewhere"))
        assert bridge.models_dir() == tmp_path / "elsewhere"


class TestDownloadState:
    def test_is_downloaded_requires_model_and_mmproj(
        self, bridge: LocalBridge, family: ModelFamily
    ) -> None:
        quant = family.quants[0]
        assert not bridge.is_downloaded(family, quant)
        model = bridge.model_file(family, quant)
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"x" * quant.size_bytes)
        assert not bridge.is_downloaded(family, quant)  # mmproj missing
        mmproj = bridge.mmproj_file(family)
        assert mmproj is not None
        mmproj.write_bytes(b"y" * family.mmproj_bytes)
        assert bridge.is_downloaded(family, quant)

    def test_wrong_size_means_not_downloaded(
        self, bridge: LocalBridge, family: ModelFamily
    ) -> None:
        write_downloaded(bridge, family)
        bridge.model_file(family, family.quants[0]).write_bytes(b"short")
        assert not bridge.is_downloaded(family, family.quants[0])

    def test_text_only_family_ignores_mmproj(
        self, bridge: LocalBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fam = make_family(vision=False)
        quant = fam.quants[0]
        model = bridge.model_file(fam, quant)
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"x" * quant.size_bytes)
        assert bridge.is_downloaded(fam, quant)


class TestStartDownload:
    def test_success_emits_progress_and_finished(
        self, qtbot, bridge: LocalBridge, family: ModelFamily, monkeypatch
    ) -> None:
        def fake_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * int(expected_bytes or 0))
            if progress is not None:
                progress(int(expected_bytes or 0), int(expected_bytes or 0))
            return dest

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", fake_download)
        seen: list[tuple[object, object]] = []
        bridge.download_progress.connect(
            lambda _f, _q, done, total: seen.append((done, total))
        )
        with qtbot.waitSignal(bridge.download_finished, timeout=2000) as blocker:
            assert bridge.start_download(FAMILY_ID, QUANT_LABEL)
        assert blocker.args[:3] == [FAMILY_ID, QUANT_LABEL, DOWNLOAD_OK]
        assert not bridge.is_downloading()
        assert bridge.is_downloaded(family, family.quants[0])
        qtbot.waitUntil(lambda: bool(seen), timeout=2000)
        total_bytes = MODEL_BYTES + MMPROJ_BYTES
        assert seen[-1] == (total_bytes, total_bytes)

    def test_failure_reports_message(
        self, qtbot, bridge: LocalBridge, family: ModelFamily, monkeypatch
    ) -> None:
        def broken_download(url, dest, **kw):  # noqa: ANN001, ANN003
            raise DownloadError("磁盘已满")

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", broken_download)
        with qtbot.waitSignal(bridge.download_finished, timeout=2000) as blocker:
            assert bridge.start_download(FAMILY_ID, QUANT_LABEL)
        assert blocker.args[2] == DOWNLOAD_ERROR
        assert "磁盘已满" in blocker.args[3]
        assert not bridge.is_downloading()

    def test_busy_rejects_second_download_and_cancel_works(
        self, qtbot, bridge: LocalBridge, family: ModelFamily, monkeypatch
    ) -> None:
        started = threading.Event()

        def blocking_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            started.set()
            assert cancel.wait(timeout=2)
            raise DownloadCancelledError("下载已取消")

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", blocking_download)
        assert bridge.start_download(FAMILY_ID, QUANT_LABEL)
        assert not bridge.start_download(FAMILY_ID, QUANT_LABEL)
        # A FRESH bridge (e.g. the dialog was closed and reopened) must also
        # refuse to race the still-running download of the same files.
        other = LocalBridge(manager=FakeManager())
        assert not other.start_download(FAMILY_ID, QUANT_LABEL)
        assert started.wait(timeout=2)
        with qtbot.waitSignal(bridge.download_finished, timeout=2000) as blocker:
            bridge.cancel_download()
        assert blocker.args[2] == DOWNLOAD_CANCELLED
        assert not bridge.is_downloading()

    def test_finished_download_releases_the_path_lock(
        self, qtbot, bridge: LocalBridge, family: ModelFamily, monkeypatch
    ) -> None:
        def fake_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * int(expected_bytes or 0))
            return dest

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", fake_download)
        with qtbot.waitSignal(bridge.download_finished, timeout=2000):
            assert bridge.start_download(FAMILY_ID, QUANT_LABEL)
        with qtbot.waitSignal(bridge.download_finished, timeout=2000):
            assert bridge.start_download(FAMILY_ID, QUANT_LABEL)


class TestHardware:
    def test_detect_emits_hardware_ready(self, qtbot, bridge: LocalBridge, monkeypatch) -> None:
        info = HardwareInfo(cpu_cores=8, ram_total_bytes=16, ram_available_bytes=8)
        monkeypatch.setattr("nlapt_gui.local_bridge.detect_hardware", lambda: info)
        with qtbot.waitSignal(bridge.hardware_ready, timeout=2000) as blocker:
            bridge.detect()
        assert blocker.args[0] == info
        assert bridge.hardware == info


class TestServer:
    def test_start_server_emits_starting_then_running(
        self, qtbot, family: ModelFamily
    ) -> None:
        manager = FakeManager()
        bridge = LocalBridge(manager=manager)
        states: list[tuple[str, str]] = []
        bridge.server_changed.connect(lambda state, detail: states.append((state, detail)))
        bridge.start_server(FAMILY_ID, QUANT_LABEL)
        qtbot.waitUntil(lambda: len(states) >= 2, timeout=2000)
        assert states[0] == (SERVER_STARTING, "")
        assert states[1][0] == SERVER_RUNNING
        assert states[1][1].endswith("/v1")
        assert bridge.server_running()
        assert len(manager.started_specs) == 1

    def test_start_server_failure_emits_error(self, qtbot, family: ModelFamily) -> None:
        bridge = LocalBridge(manager=FakeManager(fail=True))
        states: list[tuple[str, str]] = []
        bridge.server_changed.connect(lambda state, detail: states.append((state, detail)))
        bridge.start_server(FAMILY_ID, QUANT_LABEL)
        qtbot.waitUntil(lambda: len(states) >= 2, timeout=2000)
        assert states[1][0] == SERVER_ERROR
        assert "boom" in states[1][1]

    def test_stop_server_emits_stopped(self, qtbot, family: ModelFamily) -> None:
        manager = FakeManager()
        manager.running = True
        bridge = LocalBridge(manager=manager)
        with qtbot.waitSignal(bridge.server_changed, timeout=2000) as blocker:
            bridge.stop_server()
        assert blocker.args == [SERVER_STOPPED, ""]
        assert manager.stop_calls == 1

    def test_build_server_spec_uses_settings(
        self, bridge: LocalBridge, family: ModelFamily
    ) -> None:
        bridge.update_settings(
            server_path="srv.exe", port=2048, context_length=1024, parallel=5,
            gpu_layers=7, threads=3,
        )
        spec = bridge.build_server_spec(family, family.quants[0])
        assert spec.server_path == "srv.exe"
        assert spec.port == 2048
        assert spec.context_length == 1024
        assert spec.parallel == 5
        assert spec.gpu_layers == 7
        assert spec.threads == 3
        assert spec.model_path.endswith("fam-Q4.gguf")
        assert spec.mmproj_path.endswith("fam-mmproj.gguf")
