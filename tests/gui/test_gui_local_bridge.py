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
def bridge(qtbot, tmp_path: Path) -> LocalBridge:
    """Bridge with the primary models dir pinned inside tmp_path.

    A manual server_path is set so download flows never try to provision
    the real llama.cpp runtime (that path has its own dedicated tests).
    """
    instance = LocalBridge(manager=FakeManager())
    instance.update_settings(
        models_dir=str(tmp_path / "models"), server_path="srv.exe"
    )
    return instance


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

    def test_models_dir_defaults_inside_the_app(self, qtbot) -> None:
        from nlapt_gui.local_bridge import default_models_dir

        fresh = LocalBridge(manager=FakeManager())
        assert fresh.models_dir() == default_models_dir()
        assert fresh.models_dir().name == "models"

    def test_models_dir_override(self, bridge: LocalBridge, tmp_path: Path) -> None:
        bridge.update_settings(models_dir=str(tmp_path / "elsewhere"))
        assert bridge.models_dir() == tmp_path / "elsewhere"

    def test_models_dirs_include_extras_deduplicated(
        self, bridge: LocalBridge, tmp_path: Path
    ) -> None:
        primary = bridge.models_dir()
        bridge.update_settings(
            extra_dirs=(str(tmp_path / "shared"), str(primary), "  ")
        )
        assert bridge.models_dirs() == (primary, tmp_path / "shared")


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


class TestReuseDirs:
    def write_into(self, base: Path, fam: ModelFamily) -> tuple[Path, Path]:
        from nlapt.local.catalog import mmproj_path, quant_path

        model = quant_path(base, fam, fam.quants[0])
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"x" * fam.quants[0].size_bytes)
        mmproj = mmproj_path(base, fam)
        assert mmproj is not None
        mmproj.write_bytes(b"y" * fam.mmproj_bytes)
        return model, mmproj

    def test_model_found_in_extra_dir(
        self, bridge: LocalBridge, family: ModelFamily, tmp_path: Path
    ) -> None:
        shared = tmp_path / "shared"
        bridge.update_settings(extra_dirs=(str(shared),))
        model, mmproj = self.write_into(shared, family)
        assert bridge.is_downloaded(family, family.quants[0])
        assert bridge.find_model_file(family, family.quants[0]) == model
        spec = bridge.build_server_spec(family, family.quants[0])
        assert spec.model_path == str(model)
        assert spec.mmproj_path == str(mmproj)

    def test_download_skipped_when_files_reused(
        self, qtbot, bridge: LocalBridge, family: ModelFamily, tmp_path: Path, monkeypatch
    ) -> None:
        shared = tmp_path / "shared"
        bridge.update_settings(extra_dirs=(str(shared),))
        self.write_into(shared, family)

        def must_not_download(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("download_file must not be called for reused files")

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", must_not_download)
        with qtbot.waitSignal(bridge.download_finished, timeout=2000) as blocker:
            assert bridge.start_download(FAMILY_ID, QUANT_LABEL)
        assert blocker.args[2] == DOWNLOAD_OK
        assert not bridge.is_downloading()


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

    def test_fresh_bridge_reattaches_to_running_download(
        self, qtbot, bridge: LocalBridge, family: ModelFamily, monkeypatch
    ) -> None:
        """A dialog reopened mid-download sees live state and can cancel."""
        started = threading.Event()

        def blocking_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            if progress is not None:
                progress(3, None)  # some bytes land before the dialog reopens
            started.set()
            assert cancel.wait(timeout=2)
            raise DownloadCancelledError("下载已取消")

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", blocking_download)
        assert bridge.start_download(FAMILY_ID, QUANT_LABEL)
        assert started.wait(timeout=2)

        fresh = LocalBridge(manager=FakeManager())
        assert fresh.is_downloading()
        active = fresh.active_download()
        assert active is not None
        assert active[0] == FAMILY_ID
        assert active[1] == QUANT_LABEL
        assert active[2] >= 3  # snapshot carries the already-reported bytes
        assert active[3] == MODEL_BYTES + MMPROJ_BYTES
        # The fresh bridge cancels the task its predecessor started and gets
        # the finish signal through the process-wide hub.
        with qtbot.waitSignal(fresh.download_finished, timeout=2000) as blocker:
            fresh.cancel_download()
        assert blocker.args[2] == DOWNLOAD_CANCELLED
        assert not fresh.is_downloading()

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
        # Explicit gpu_layers: the 自动 (-1) path probes real hardware.
        bridge.update_settings(server_path="srv.exe", gpu_layers=7)
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
        bridge.update_settings(server_path="srv.exe", gpu_layers=7)
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


def _fake_runtime_exe() -> Path:
    """Create a fake extracted llama-server under the isolated data dir."""
    import sys

    from nlapt.local.runtime import runtime_dir

    from nlapt_gui.local_bridge import runtime_base_dir

    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    exe = runtime_dir(runtime_base_dir()) / "sub" / name
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"exe")
    return exe


class TestRuntimeResolution:
    def test_manual_server_path_wins(self, qtbot) -> None:
        from nlapt.local.settings import LocalSettings

        from nlapt_gui.local_bridge import resolve_server_path

        _fake_runtime_exe()
        assert resolve_server_path(LocalSettings(server_path="manual.exe")) == "manual.exe"

    def test_extracted_runtime_found_and_no_pending(self, qtbot) -> None:
        from nlapt.local.settings import LocalSettings

        from nlapt_gui.local_bridge import pending_runtime_asset, resolve_server_path

        exe = _fake_runtime_exe()
        settings = LocalSettings()
        assert resolve_server_path(settings) == str(exe)
        assert pending_runtime_asset(settings) is None

    def test_pending_runtime_when_nothing_available(self, qtbot) -> None:
        from nlapt.local.runtime import current_asset
        from nlapt.local.settings import LocalSettings

        from nlapt_gui.local_bridge import pending_runtime_asset

        assert pending_runtime_asset(LocalSettings()) == current_asset()

    def test_start_server_auto_provisions_runtime(
        self, qtbot, family: ModelFamily, monkeypatch
    ) -> None:
        manager = FakeManager()
        bridge = LocalBridge(manager=manager)
        bridge.update_settings(gpu_layers=5)  # keep the 自动 probe out of tests
        calls: list[Path] = []

        def fake_ensure(base_dir, **_kw):  # noqa: ANN001, ANN003
            calls.append(base_dir)
            return Path("auto/llama-server.exe")

        monkeypatch.setattr("nlapt_gui.local_bridge.ensure_runtime", fake_ensure)
        states: list[str] = []
        bridge.server_changed.connect(lambda state, _d: states.append(state))
        bridge.start_server(FAMILY_ID, QUANT_LABEL)
        qtbot.waitUntil(lambda: SERVER_RUNNING in states, timeout=2000)
        assert calls  # runtime was provisioned first
        assert manager.started_specs[0].server_path == str(
            Path("auto/llama-server.exe")
        )

    def test_start_download_provisions_runtime_first(
        self, qtbot, family: ModelFamily, tmp_path: Path, monkeypatch
    ) -> None:
        from nlapt.local.runtime import current_asset

        if current_asset() is None:
            pytest.skip("no pinned runtime for this platform")
        bridge = LocalBridge(manager=FakeManager())
        bridge.update_settings(models_dir=str(tmp_path / "models"))
        order: list[str] = []

        def fake_ensure(base_dir, *, progress=None, cancel=None, **_kw):  # noqa: ANN001, ANN003
            order.append("runtime")
            if progress is not None:
                progress(current_asset().size_bytes, None)
            return _fake_runtime_exe()

        def fake_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            order.append("model")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * int(expected_bytes or 0))
            if progress is not None:
                progress(int(expected_bytes or 0), int(expected_bytes or 0))
            return dest

        monkeypatch.setattr("nlapt_gui.local_bridge.ensure_runtime", fake_ensure)
        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", fake_download)
        seen: list[tuple[object, object]] = []
        bridge.download_progress.connect(
            lambda _f, _q, done, total: seen.append((done, total))
        )
        with qtbot.waitSignal(bridge.download_finished, timeout=2000) as blocker:
            assert bridge.start_download(FAMILY_ID, QUANT_LABEL)
        assert blocker.args[2] == DOWNLOAD_OK
        assert order[0] == "runtime"  # runtime first, then the model files
        assert "model" in order
        qtbot.waitUntil(lambda: bool(seen), timeout=2000)
        expected_total = current_asset().size_bytes + MODEL_BYTES + MMPROJ_BYTES
        assert seen[-1] == (expected_total, expected_total)


class TestResolveLocalTarget:
    def make_settings(self, tmp_path: Path, **overrides: object):
        from nlapt.local.settings import LocalSettings

        fields: dict[str, object] = {
            "models_dir": str(tmp_path / "models"),
            "server_path": "srv.exe",
            "family_id": FAMILY_ID,
            "quant_label": QUANT_LABEL,
        }
        fields.update(overrides)
        return LocalSettings(**fields)  # type: ignore[arg-type]

    def test_no_selection_raises(self, qtbot, tmp_path: Path) -> None:
        from nlapt.core.errors import LocalInferenceError

        from nlapt_gui.local_bridge import resolve_local_target

        with pytest.raises(LocalInferenceError, match="未选择本地模型"):
            resolve_local_target(self.make_settings(tmp_path, family_id=""))

    def test_unknown_family_raises_selection_error(
        self, qtbot, tmp_path: Path
    ) -> None:
        from nlapt.core.errors import LocalInferenceError

        from nlapt_gui.local_bridge import resolve_local_target

        with pytest.raises(LocalInferenceError, match="未选择本地模型"):
            resolve_local_target(
                self.make_settings(tmp_path, family_id="nope", quant_label="Q4")
            )

    def test_text_only_family_rejected_for_vision(
        self, qtbot, tmp_path: Path, monkeypatch
    ) -> None:
        from nlapt.core.errors import LocalInferenceError

        from nlapt_gui.local_bridge import resolve_local_target

        fam = make_family(vision=False)
        monkeypatch.setattr("nlapt_gui.local_bridge.find_family", lambda fid: fam)
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.find_quant", lambda f, label: f.quants[0]
        )
        with pytest.raises(LocalInferenceError, match="不支持图片输入"):
            resolve_local_target(self.make_settings(tmp_path))

    def test_not_downloaded_raises(
        self, qtbot, tmp_path: Path, family: ModelFamily
    ) -> None:
        from nlapt.core.errors import LocalInferenceError

        from nlapt_gui.local_bridge import resolve_local_target

        with pytest.raises(LocalInferenceError, match="尚未下载"):
            resolve_local_target(self.make_settings(tmp_path))

    def test_ready_selection_builds_spec(
        self, qtbot, tmp_path: Path, family: ModelFamily, bridge: LocalBridge
    ) -> None:
        from nlapt_gui.local_bridge import resolve_local_target

        write_downloaded(bridge, family)
        target = resolve_local_target(
            self.make_settings(tmp_path, parallel=3, port=2001)
        )
        assert target.family.family_id == FAMILY_ID
        assert target.spec.server_path == "srv.exe"
        assert target.spec.parallel == 3
        assert target.spec.port == 2001
        assert target.spec.model_path.endswith("fam-Q4.gguf")


class TestPrepareLaunchSpec:
    """v1.7 fix: 自动 GPU 层数 resolves to a hardware-fitting -ngl value."""

    GIB = 1024**3

    def make_spec(self, tmp_path: Path, **overrides: object):
        from nlapt.local.server import ServerSpec

        fields: dict[str, object] = {
            "server_path": "srv.exe",
            "model_path": str(tmp_path / "m.gguf"),
            "port": 18434,
            "gpu_layers": -1,
        }
        fields.update(overrides)
        return ServerSpec(**fields)  # type: ignore[arg-type]

    def patch_probes(self, monkeypatch, hardware: HardwareInfo, blocks: int | None):
        probes: list[int] = []

        def fake_detect() -> HardwareInfo:
            probes.append(1)
            return hardware

        monkeypatch.setattr("nlapt_gui.local_bridge.detect_hardware", fake_detect)
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.read_block_count", lambda path: blocks
        )
        return probes

    def test_auto_resolves_to_full_offload_on_roomy_gpu(
        self, qtbot, tmp_path: Path, family: ModelFamily, monkeypatch
    ) -> None:
        from nlapt.local.hardware import GpuInfo
        from nlapt.local.server import GPU_LAYERS_ALL

        from nlapt_gui.local_bridge import prepare_launch_spec

        roomy = GpuInfo(
            name="big", vram_total_bytes=100 * self.GIB, vram_free_bytes=100 * self.GIB
        )
        self.patch_probes(
            monkeypatch,
            HardwareInfo(
                cpu_cores=8,
                ram_total_bytes=32 * self.GIB,
                ram_available_bytes=16 * self.GIB,
                gpus=(roomy,),
            ),
            blocks=48,
        )
        spec = self.make_spec(tmp_path)
        resolved = prepare_launch_spec(spec, family, family.quants[0])
        assert resolved.gpu_layers == GPU_LAYERS_ALL

    def test_auto_without_gpu_resolves_to_cpu_only_and_caches(
        self, qtbot, tmp_path: Path, family: ModelFamily, monkeypatch
    ) -> None:
        from nlapt_gui.local_bridge import prepare_launch_spec

        probes = self.patch_probes(
            monkeypatch,
            HardwareInfo(
                cpu_cores=8,
                ram_total_bytes=32 * self.GIB,
                ram_available_bytes=16 * self.GIB,
            ),
            blocks=48,
        )
        spec = self.make_spec(tmp_path)
        first = prepare_launch_spec(spec, family, family.quants[0])
        second = prepare_launch_spec(spec, family, family.quants[0])
        assert first.gpu_layers == 0
        # Identical resolved specs (ensure() must keep reusing the server)
        # and a single hardware probe thanks to the cache.
        assert first == second
        assert len(probes) == 1

    def test_manual_layers_pass_through_untouched(
        self, qtbot, tmp_path: Path, family: ModelFamily, monkeypatch
    ) -> None:
        from nlapt_gui.local_bridge import prepare_launch_spec

        probes = self.patch_probes(
            monkeypatch,
            HardwareInfo(cpu_cores=8, ram_total_bytes=1, ram_available_bytes=1),
            blocks=1,
        )
        spec = self.make_spec(tmp_path, gpu_layers=7)
        resolved = prepare_launch_spec(spec, family, family.quants[0])
        assert resolved.gpu_layers == 7
        assert not probes  # manual value never probes hardware

    def test_auto_budget_uses_total_context(
        self, qtbot, tmp_path: Path, family: ModelFamily, monkeypatch
    ) -> None:
        """KV budget must follow ctx × parallel (the server's real -c)."""
        from nlapt_gui.local_bridge import prepare_launch_spec

        self.patch_probes(
            monkeypatch,
            HardwareInfo(
                cpu_cores=8,
                ram_total_bytes=32 * self.GIB,
                ram_available_bytes=16 * self.GIB,
            ),
            blocks=48,
        )
        captured: list[int] = []

        def fake_auto(family, quant, hardware, *, context_length, block_count):  # noqa: ANN001
            captured.append(context_length)
            return 5

        monkeypatch.setattr("nlapt_gui.local_bridge.auto_gpu_layers", fake_auto)
        spec = self.make_spec(tmp_path, context_length=4096, parallel=4)
        resolved = prepare_launch_spec(spec, family, family.quants[0])
        assert resolved.gpu_layers == 5
        assert captured == [4096 * 4]


class TestPresetCaptioner:
    """Official presets replace free-form prompts for caption specialists."""

    def make_joy_family(self) -> ModelFamily:
        return ModelFamily(
            family_id="joycaption-beta-one",  # real id -> real official presets
            series_id="joycaption",
            name="Joy Test",
            repo_id="tester/joy-GGUF",
            downloads=5,
            params_label="8B",
            vision=True,
            kv_bytes_per_token=1_000,
            quants=(QuantFile("Q4", "joy-Q4.gguf", MODEL_BYTES, recommended=True),),
            mmproj_filename="joy-mmproj.gguf",
            mmproj_bytes=MMPROJ_BYTES,
        )

    def setup_captioner(self, bridge: LocalBridge, monkeypatch, captured: list) -> None:
        from nlapt.llm.base import LLMResponse

        fam = self.make_joy_family()
        monkeypatch.setattr("nlapt_gui.local_bridge.find_family", lambda fid: fam)
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.find_quant", lambda f, label: f.quants[0]
        )
        bridge.update_settings(
            family_id="joycaption-beta-one", quant_label="Q4", gpu_layers=5
        )
        write_downloaded(bridge, fam)

        class EnsureManager:
            def ensure(self, spec):  # noqa: ANN001
                return "http://127.0.0.1:1/v1"

        monkeypatch.setattr(
            "nlapt_gui.local_bridge.get_server_manager", lambda: EnsureManager()
        )
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.prepare_launch_spec", lambda spec, f, q: spec
        )
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.prepare_image", lambda path, *, max_edge: b"img"
        )

        class FakeClient:
            def complete(self, request):  # noqa: ANN001
                captured.append(request)
                return LLMResponse(text="1girl, solo", model=request.model)

        monkeypatch.setattr(
            "nlapt_gui.local_bridge.create_client", lambda profile: FakeClient()
        )

    def test_default_preset_replaces_free_form_prompts(
        self, qtbot, bridge: LocalBridge, monkeypatch
    ) -> None:
        from nlapt.local.presets import JOYCAPTION_PRESETS

        from nlapt_gui.local_bridge import make_local_vision_captioner

        captured: list = []
        self.setup_captioner(bridge, monkeypatch, captured)
        captioner = make_local_vision_captioner(image_max_edge=512)
        assert captioner(Path("x.png"), "free-sys", "free-user") == "1girl, solo"
        default = JOYCAPTION_PRESETS[0]
        assert captured[0].system == default.system
        assert captured[0].messages[0].text == default.user_prompt

    def test_custom_choice_keeps_free_form_prompts(
        self, qtbot, bridge: LocalBridge, monkeypatch
    ) -> None:
        from nlapt.local.presets import PRESET_CUSTOM

        from nlapt_gui.local_bridge import make_local_vision_captioner

        captured: list = []
        self.setup_captioner(bridge, monkeypatch, captured)
        bridge.update_settings(prompt_preset=PRESET_CUSTOM)
        captioner = make_local_vision_captioner(image_max_edge=512)
        captioner(Path("x.png"), "free-sys", "free-user")
        assert captured[0].system == "free-sys"
        assert captured[0].messages[0].text == "free-user"


# -- Florence engine families --------------------------------------------------------
def make_florence_family() -> ModelFamily:
    from nlapt.local.catalog import ENGINE_FLORENCE

    return ModelFamily(
        family_id=FAMILY_ID,
        series_id="florence2-promptgen",
        name="Flor Test",
        repo_id="tester/flor-onnx",
        downloads=5,
        params_label="0.2B",
        vision=True,
        kv_bytes_per_token=1_000,
        quants=(
            QuantFile(QUANT_LABEL, "onnx/decoder_model_merged.onnx", MODEL_BYTES,
                      recommended=True),
        ),
        engine=ENGINE_FLORENCE,
        extra_files=(
            QuantFile("vision", "onnx/vision_encoder.onnx", 6),
            QuantFile("tokenizer", "tokenizer.json", 3),
        ),
    )


@pytest.fixture()
def florence_family(monkeypatch: pytest.MonkeyPatch) -> ModelFamily:
    fam = make_florence_family()
    monkeypatch.setattr("nlapt_gui.local_bridge.find_family", lambda fid: fam)
    monkeypatch.setattr(
        "nlapt_gui.local_bridge.find_quant", lambda f, label: f.quants[0]
    )
    return fam


def write_florence_files(bridge: LocalBridge, fam: ModelFamily) -> None:
    for item in (fam.quants[0], *fam.extra_files):
        dest = bridge.model_file(fam, item)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * item.size_bytes)


class TestFlorenceDownload:
    def test_extra_files_required_for_downloaded(
        self, bridge: LocalBridge, florence_family: ModelFamily
    ) -> None:
        fam = florence_family
        quant = fam.quants[0]
        model = bridge.model_file(fam, quant)
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"x" * quant.size_bytes)
        assert not bridge.is_downloaded(fam, quant)  # extras missing
        write_florence_files(bridge, fam)
        assert bridge.is_downloaded(fam, quant)

    def test_download_queues_extras_and_skips_runtime(
        self, qtbot, bridge: LocalBridge, florence_family: ModelFamily, monkeypatch
    ) -> None:
        # Even with NO llama-server configured, florence must not provision
        # the llama.cpp runtime.
        bridge.update_settings(server_path="")
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.ensure_runtime",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("runtime must not be provisioned for florence")
            ),
        )
        names: list[str] = []

        def fake_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            names.append(dest.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * int(expected_bytes or 0))
            return dest

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", fake_download)
        with qtbot.waitSignal(bridge.download_finished, timeout=2000) as blocker:
            assert bridge.start_download(FAMILY_ID, QUANT_LABEL)
        assert blocker.args[2] == DOWNLOAD_OK
        assert set(names) == {
            "decoder_model_merged.onnx",
            "vision_encoder.onnx",
            "tokenizer.json",
        }
        assert bridge.is_downloaded(florence_family, florence_family.quants[0])


class TestFlorenceCaptioner:
    def save_settings(self, bridge: LocalBridge, **overrides: object) -> None:
        bridge.update_settings(
            family_id=FAMILY_ID, quant_label=QUANT_LABEL, **overrides
        )

    def test_resolve_target_skips_llama_runtime_check(
        self, qtbot, bridge: LocalBridge, florence_family: ModelFamily, monkeypatch
    ) -> None:
        from nlapt_gui.local_bridge import resolve_local_target

        monkeypatch.setattr(
            "nlapt_gui.local_bridge.runtime_supported", lambda: False
        )
        self.save_settings(bridge, server_path="")
        write_florence_files(bridge, florence_family)
        target = resolve_local_target(bridge.settings)
        assert target.family.engine == "florence"

    def test_captioner_uses_persisted_task(
        self, qtbot, bridge: LocalBridge, florence_family: ModelFamily, monkeypatch
    ) -> None:
        from nlapt.local.florence import TASK_ANALYZE

        from nlapt_gui.local_bridge import make_local_vision_captioner

        self.save_settings(bridge, server_path="", florence_task=TASK_ANALYZE)
        write_florence_files(bridge, florence_family)

        class StubEngine:
            def __init__(self) -> None:
                self.calls: list[tuple[Path, str]] = []

            def caption(self, image_path: Path, task: str) -> str:
                self.calls.append((image_path, task))
                return "structured analysis"

        stub = StubEngine()
        seen_files: list[dict[str, Path]] = []
        seen_loras: list[Path | None] = []

        def fake_get_engine(files, lora_path=None):  # noqa: ANN001, ANN202
            seen_files.append(files)
            seen_loras.append(lora_path)
            return stub

        monkeypatch.setattr(
            "nlapt_gui.local_bridge.get_florence_engine", fake_get_engine
        )
        captioner = make_local_vision_captioner(image_max_edge=1024)
        result = captioner(Path("img.png"), "system ignored", "prompt ignored")
        assert result == "structured analysis"
        assert stub.calls == [(Path("img.png"), TASK_ANALYZE)]
        assert set(seen_files[0]) == {
            "decoder_model_merged.onnx",
            "vision_encoder.onnx",
            "tokenizer.json",
        }
        assert seen_loras == [None]  # no LoRA configured

    def test_captioner_passes_selected_lora(
        self, qtbot, bridge: LocalBridge, florence_family: ModelFamily, monkeypatch
    ) -> None:
        from nlapt_gui.local_bridge import make_local_vision_captioner

        lora_file = bridge.models_dir() / "style.safetensors"
        lora_file.parent.mkdir(parents=True, exist_ok=True)
        lora_file.write_bytes(b"st")
        self.save_settings(bridge, server_path="", florence_lora=str(lora_file))
        write_florence_files(bridge, florence_family)
        seen_loras: list[Path | None] = []

        class StubEngine:
            def caption(self, image_path: Path, task: str) -> str:
                return "ok"

        def fake_get_engine(files, lora_path=None):  # noqa: ANN001, ANN202
            seen_loras.append(lora_path)
            return StubEngine()

        monkeypatch.setattr(
            "nlapt_gui.local_bridge.get_florence_engine", fake_get_engine
        )
        captioner = make_local_vision_captioner(image_max_edge=1024)
        assert captioner(Path("img.png"), "", "") == "ok"
        assert seen_loras == [Path(str(lora_file))]

    def test_captioner_rejects_missing_lora_file(
        self, qtbot, bridge: LocalBridge, florence_family: ModelFamily
    ) -> None:
        from nlapt.core.errors import LocalInferenceError

        from nlapt_gui.local_bridge import make_local_vision_captioner

        self.save_settings(
            bridge,
            server_path="",
            florence_lora=str(bridge.models_dir() / "gone.safetensors"),
        )
        write_florence_files(bridge, florence_family)
        with pytest.raises(LocalInferenceError, match="LoRA 文件不存在"):
            make_local_vision_captioner(image_max_edge=1024)

    def test_get_florence_engine_reuses_until_files_change(self, qtbot) -> None:
        from nlapt.local.florence import REQUIRED_FILES

        import nlapt_gui.local_bridge as lb

        files_a = {name: Path(f"a-{name}") for name in REQUIRED_FILES}
        files_b = {name: Path(f"b-{name}") for name in REQUIRED_FILES}
        first = lb.get_florence_engine(dict(files_a))
        assert lb.get_florence_engine(dict(files_a)) is first
        assert lb.get_florence_engine(files_b) is not first

    def test_captioner_clamps_task_to_family_set(
        self, qtbot, bridge: LocalBridge, monkeypatch
    ) -> None:
        from dataclasses import replace

        from nlapt.local.florence import TASK_CAPTION, TASK_GENERATE_TAGS

        from nlapt_gui.local_bridge import make_local_vision_captioner

        # An official-style family that never learned the PromptGen 指令.
        fam = replace(make_florence_family(), florence_tasks=(TASK_CAPTION,))
        monkeypatch.setattr("nlapt_gui.local_bridge.find_family", lambda fid: fam)
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.find_quant", lambda f, label: f.quants[0]
        )
        self.save_settings(bridge, server_path="", florence_task=TASK_GENERATE_TAGS)
        write_florence_files(bridge, fam)

        class StubEngine:
            def __init__(self) -> None:
                self.tasks: list[str] = []

            def caption(self, image_path: Path, task: str) -> str:
                self.tasks.append(task)
                return "ok"

        stub = StubEngine()
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.get_florence_engine",
            lambda files, lora_path=None: stub,
        )
        captioner = make_local_vision_captioner(image_max_edge=1024)
        captioner(Path("img.png"), "", "")
        assert stub.tasks == [TASK_CAPTION]  # clamped, not <GENERATE_TAGS>

    def test_get_florence_engine_keyed_by_lora(self, qtbot, tmp_path: Path) -> None:
        from nlapt.local.florence import REQUIRED_FILES

        import nlapt_gui.local_bridge as lb

        files = {name: Path(f"k-{name}") for name in REQUIRED_FILES}
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"one")
        plain = lb.get_florence_engine(dict(files))
        with_lora = lb.get_florence_engine(dict(files), lora)
        assert with_lora is not plain
        assert lb.get_florence_engine(dict(files), lora) is with_lora
        # A retrained (rewritten) LoRA file must rebuild the engine.
        lora.write_bytes(b"retrained-longer")
        assert lb.get_florence_engine(dict(files), lora) is not with_lora


class TestLoraDownload:
    LORA_ID = "bai-json-large"

    def test_download_writes_files_and_reports(
        self, qtbot, bridge: LocalBridge, monkeypatch
    ) -> None:
        from nlapt.local.catalog import find_lora

        from nlapt_gui.local_bridge import LORA_FAMILY_PREFIX

        urls: list[str] = []

        def fake_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            urls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * int(expected_bytes or 0))
            return dest

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", fake_download)
        entry = find_lora(self.LORA_ID)
        assert not bridge.is_lora_downloaded(entry)
        with qtbot.waitSignal(bridge.download_finished, timeout=2000) as blocker:
            assert bridge.start_lora_download(self.LORA_ID)
        assert blocker.args[0] == LORA_FAMILY_PREFIX + self.LORA_ID
        assert blocker.args[1] == ""
        assert blocker.args[2] == DOWNLOAD_OK
        assert bridge.is_lora_downloaded(entry)
        assert bridge.lora_adapter_file(entry).is_file()
        assert all(url.startswith("https://modelscope.cn/") for url in urls)
        assert len(urls) == len(entry.files)

    def test_already_downloaded_short_circuits(
        self, qtbot, bridge: LocalBridge, monkeypatch
    ) -> None:
        from nlapt.local.catalog import find_lora, lora_file_path

        entry = find_lora(self.LORA_ID)
        for file in entry.files:
            dest = lora_file_path(bridge.models_dir(), entry, file)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * file.size_bytes)
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.download_file",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")),
        )
        with qtbot.waitSignal(bridge.download_finished, timeout=2000) as blocker:
            assert bridge.start_lora_download(self.LORA_ID)
        assert blocker.args[2] == DOWNLOAD_OK

    def test_refused_while_another_download_runs(
        self, qtbot, bridge: LocalBridge, monkeypatch
    ) -> None:
        import nlapt_gui.local_bridge as lb

        monkeypatch.setattr(
            lb,
            "_ACTIVE_TASK",
            lb._ActiveDownload(
                family_id="other", quant_label="Q4", cancel=threading.Event()
            ),
        )
        assert not bridge.start_lora_download(self.LORA_ID)


# -- idle auto-stop (推理完先不卸载,空闲 30 秒后再卸载) ------------------------------
class RecordingTimer:
    """Fake threading.Timer: records the delay, fires only on demand."""

    def __init__(self, delay: float, fn) -> None:  # noqa: ANN001
        self.delay = delay
        self.fn = fn
        self.daemon = False
        self.cancelled = False

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.fn()


def make_idle_stopper(manager: FakeManager):
    from nlapt_gui.local_bridge import IdleServerStopper

    timers: list[RecordingTimer] = []

    def factory(delay: float, fn):  # noqa: ANN001
        timer = RecordingTimer(delay, fn)
        timers.append(timer)
        return timer

    return IdleServerStopper(manager, timer_factory=factory), timers


class TestIdleServerStopper:
    def test_finish_arms_default_delay_and_fire_stops(self) -> None:
        from nlapt_gui.local_bridge import IDLE_STOP_DELAY_SECONDS

        manager = FakeManager()
        stopper, timers = make_idle_stopper(manager)
        stopper.note_request()  # server not running -> this run loads it
        manager.running = True
        stopper.note_finished()
        assert stopper.pending()
        assert timers[-1].delay == IDLE_STOP_DELAY_SECONDS
        assert timers[-1].daemon  # must never delay interpreter exit
        assert manager.stop_calls == 0  # 先不卸载
        timers[-1].fire()
        assert manager.stop_calls == 1
        assert not stopper.pending()

    def test_new_request_cancels_pending_stop(self) -> None:
        manager = FakeManager()
        stopper, timers = make_idle_stopper(manager)
        stopper.note_request()
        manager.running = True
        stopper.note_finished()
        armed = timers[-1]
        stopper.note_request()  # 反复推理: cancels the pending unload
        assert armed.cancelled
        assert not stopper.pending()
        armed.fire()  # late fire of a cancelled timer is inert
        assert manager.stop_calls == 0
        stopper.note_finished()  # run ends again -> re-armed
        assert stopper.pending()

    def test_overlapping_runs_arm_only_after_the_last(self) -> None:
        manager = FakeManager()
        stopper, timers = make_idle_stopper(manager)
        stopper.note_request()
        manager.running = True
        stopper.note_request()  # second run overlaps the first
        stopper.note_finished()
        assert not stopper.pending()  # one run still in flight
        stopper.note_finished()
        assert stopper.pending()
        assert len(timers) == 1

    def test_user_started_server_is_never_auto_stopped(self) -> None:
        manager = FakeManager()
        manager.running = True  # user pre-started via 启动本地服务
        stopper, timers = make_idle_stopper(manager)
        stopper.note_request()
        stopper.note_finished()
        assert not timers
        assert manager.stop_calls == 0

    def test_user_control_clears_pending_stop(self) -> None:
        manager = FakeManager()
        stopper, timers = make_idle_stopper(manager)
        stopper.note_request()
        manager.running = True
        stopper.note_finished()
        stopper.note_user_control()  # user clicked 启动/停止 themselves
        assert timers[-1].cancelled
        assert not stopper.pending()
        stopper.note_request()  # server still running (user's) -> no re-arm
        stopper.note_finished()
        assert not stopper.pending()

    def test_fire_during_new_run_does_not_stop(self) -> None:
        manager = FakeManager()
        stopper, timers = make_idle_stopper(manager)
        stopper.note_request()
        manager.running = True
        stopper.note_finished()
        armed = timers[-1]
        stopper.note_request()
        armed.cancelled = False  # simulate the timer racing the cancel
        armed.fire()
        assert manager.stop_calls == 0  # in-flight run guards the server
