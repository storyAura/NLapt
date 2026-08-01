"""Tests for nlapt.local.server: args, lifecycle, health-wait, stop."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from nlapt.core.errors import LocalServerError, ValidationError
from nlapt.local.server import (
    GPU_LAYERS_ALL,
    LocalServerManager,
    ServerSpec,
    base_url,
    build_server_args,
    health_url,
)


class FakeProcess:
    def __init__(self, *, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = 0

    def wait(self, timeout: float | None = None) -> int | None:
        return self.exit_code

    def kill(self) -> None:
        self.killed = True
        self.exit_code = -9


def make_clock(step: float = 0.5) -> tuple[Callable[[], float], Callable[[float], None]]:
    state = {"now": 0.0}

    def clock() -> float:
        return state["now"]

    def sleep(seconds: float) -> None:
        state["now"] += max(seconds, step)

    return clock, sleep


def make_spec(tmp_path: Path, **overrides: object) -> ServerSpec:
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"exe")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    fields: dict[str, object] = {
        "server_path": str(server),
        "model_path": str(model),
        "port": 18434,
    }
    fields.update(overrides)
    return ServerSpec(**fields)  # type: ignore[arg-type]


class TestBuildServerArgs:
    def test_full_vector(self) -> None:
        spec = ServerSpec(
            server_path="srv",
            model_path="m.gguf",
            port=18434,
            mmproj_path="proj.gguf",
            context_length=8192,
            gpu_layers=20,
            threads=8,
            parallel=4,
        )
        args = build_server_args(spec)
        assert args == (
            "srv",
            "-m",
            "m.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "18434",
            "-c",
            "32768",  # per-slot 8192 × 4 slots (llama-server splits -c)
            "--parallel",
            "4",
            "-ngl",
            "20",
            "--reasoning",
            "off",
            "-t",
            "8",
            "--mmproj",
            "proj.gguf",
        )

    def test_context_is_per_slot_times_parallel(self) -> None:
        spec = ServerSpec(
            server_path="srv", model_path="m", port=18434,
            context_length=4096, parallel=4,
        )
        args = build_server_args(spec)
        assert args[args.index("-c") + 1] == "16384"

    def test_reasoning_disabled_for_captioning(self) -> None:
        # Thinking models burned the whole budget in their reasoning
        # channel and returned empty captions (field bug).
        spec = ServerSpec(server_path="srv", model_path="m", port=18434)
        args = build_server_args(spec)
        assert args[args.index("--reasoning") + 1] == "off"

    def test_auto_gpu_layers_maps_to_all(self) -> None:
        spec = ServerSpec(server_path="srv", model_path="m", port=18434, gpu_layers=-1)
        args = build_server_args(spec)
        assert args[args.index("-ngl") + 1] == str(GPU_LAYERS_ALL)

    def test_zero_threads_and_empty_mmproj_omitted(self) -> None:
        spec = ServerSpec(server_path="srv", model_path="m", port=18434)
        args = build_server_args(spec)
        assert "-t" not in args
        assert "--mmproj" not in args

    @pytest.mark.parametrize(
        "overrides",
        [
            {"server_path": ""},
            {"model_path": ""},
            {"port": 80},
            {"context_length": 0},
            {"parallel": 0},
        ],
    )
    def test_invalid_specs_rejected(self, overrides: dict[str, object]) -> None:
        fields: dict[str, object] = {
            "server_path": "srv",
            "model_path": "m",
            "port": 18434,
        }
        fields.update(overrides)
        with pytest.raises(ValidationError):
            build_server_args(ServerSpec(**fields))  # type: ignore[arg-type]


class TestUrls:
    def test_health_and_base_url(self) -> None:
        assert health_url(18434) == "http://127.0.0.1:18434/health"
        assert base_url(18434) == "http://127.0.0.1:18434/v1"


class TestManagerLifecycle:
    def test_start_waits_for_health(self, tmp_path: Path) -> None:
        process = FakeProcess()
        health_results = [False, False, True]
        clock, sleep = make_clock()
        manager = LocalServerManager(
            popen=lambda args: process,
            health_check=lambda url: health_results.pop(0),
            sleep=sleep,
            clock=clock,
        )
        url = manager.start(make_spec(tmp_path))
        assert url == "http://127.0.0.1:18434/v1"
        assert manager.is_running()
        assert manager.current_base_url == url

    def test_early_exit_raises_with_code(self, tmp_path: Path) -> None:
        process = FakeProcess(exit_code=3)
        clock, sleep = make_clock()
        manager = LocalServerManager(
            popen=lambda args: process,
            health_check=lambda url: False,
            sleep=sleep,
            clock=clock,
        )
        with pytest.raises(LocalServerError, match="3"):
            manager.start(make_spec(tmp_path))
        assert not manager.is_running()

    def test_health_timeout_stops_process(self, tmp_path: Path) -> None:
        process = FakeProcess()
        clock, sleep = make_clock()
        manager = LocalServerManager(
            popen=lambda args: process,
            health_check=lambda url: False,
            sleep=sleep,
            clock=clock,
        )
        with pytest.raises(LocalServerError, match="未就绪"):
            manager.start(make_spec(tmp_path), ready_timeout=2.0)
        assert process.terminated
        assert not manager.is_running()

    def test_missing_server_file_rejected(self, tmp_path: Path) -> None:
        spec = make_spec(tmp_path, server_path=str(tmp_path / "missing.exe"))
        manager = LocalServerManager(popen=lambda args: FakeProcess())
        with pytest.raises(ValidationError, match="llama-server"):
            manager.start(spec)

    def test_missing_mmproj_rejected(self, tmp_path: Path) -> None:
        spec = make_spec(tmp_path, mmproj_path=str(tmp_path / "missing.mmproj.gguf"))
        manager = LocalServerManager(popen=lambda args: FakeProcess())
        with pytest.raises(ValidationError, match="视觉组件"):
            manager.start(spec)

    def test_restart_stops_previous_process(self, tmp_path: Path) -> None:
        first, second = FakeProcess(), FakeProcess()
        processes = [first, second]
        clock, sleep = make_clock()
        manager = LocalServerManager(
            popen=lambda args: processes.pop(0),
            health_check=lambda url: True,
            sleep=sleep,
            clock=clock,
        )
        manager.start(make_spec(tmp_path))
        manager.start(make_spec(tmp_path))
        assert first.terminated
        assert manager.is_running()

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        process = FakeProcess()
        clock, sleep = make_clock()
        manager = LocalServerManager(
            popen=lambda args: process,
            health_check=lambda url: True,
            sleep=sleep,
            clock=clock,
        )
        manager.start(make_spec(tmp_path))
        manager.stop()
        manager.stop()
        assert process.terminated
        assert not manager.is_running()
        assert manager.current_base_url == ""

    def test_popen_failure_wrapped(self, tmp_path: Path) -> None:
        def boom(args: tuple[str, ...]) -> FakeProcess:
            raise OSError("no exec permission")

        manager = LocalServerManager(popen=boom)
        with pytest.raises(LocalServerError, match="无法启动"):
            manager.start(make_spec(tmp_path))


class TestEnsure:
    def make_manager(self) -> tuple[LocalServerManager, list[FakeProcess]]:
        spawned: list[FakeProcess] = []

        def popen(args):  # noqa: ANN001
            process = FakeProcess()
            spawned.append(process)
            return process

        clock, sleep = make_clock()
        manager = LocalServerManager(
            popen=popen, health_check=lambda url: True, sleep=sleep, clock=clock
        )
        return manager, spawned

    def test_ensure_starts_when_stopped(self, tmp_path: Path) -> None:
        manager, spawned = self.make_manager()
        spec = make_spec(tmp_path)
        assert manager.ensure(spec) == base_url(spec.port)
        assert len(spawned) == 1
        assert manager.current_spec == spec

    def test_ensure_same_spec_reuses_running_server(self, tmp_path: Path) -> None:
        manager, spawned = self.make_manager()
        spec = make_spec(tmp_path)
        manager.ensure(spec)
        # Equal spec (fresh instance): NO reload of the model.
        assert manager.ensure(make_spec(tmp_path)) == base_url(spec.port)
        assert len(spawned) == 1

    def test_ensure_different_spec_restarts(self, tmp_path: Path) -> None:
        manager, spawned = self.make_manager()
        manager.ensure(make_spec(tmp_path))
        manager.ensure(make_spec(tmp_path, port=18435))
        assert len(spawned) == 2
        assert spawned[0].terminated  # previous server stopped first
        assert manager.current_spec is not None
        assert manager.current_spec.port == 18435

    def test_stop_clears_current_spec(self, tmp_path: Path) -> None:
        manager, _spawned = self.make_manager()
        manager.ensure(make_spec(tmp_path))
        manager.stop()
        assert manager.current_spec is None
