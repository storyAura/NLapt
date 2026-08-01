"""Manage a local ``llama-server`` process (llama.cpp OpenAI-compatible API).

The app talks to local models through the same OpenAI-compatible client it
already uses for remote endpoints; this module only builds the argument
vector, spawns the process, waits for ``/health`` to answer and stops it
again. Process creation, health probing and sleeping are injectable so
tests never spawn real processes or wait on wall-clock time.

Vision models pass their mmproj projector via ``--mmproj``; a negative
``gpu_layers`` maps to :data:`GPU_LAYERS_ALL` ("offload everything").

Two launch decisions came from field debugging (v1.7):

* ``-c`` is ``context_length × parallel`` — llama-server SPLITS the total
  context across its ``--parallel`` slots, so passing the raw setting gave
  each request only a fraction of the configured 上下文长度.
* ``--reasoning off`` — thinking-capable models (Gemma 4) spend the whole
  token budget inside their reasoning channel and return an EMPTY
  ``content`` for captioning requests; this app wants captions, not
  chain-of-thought.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from nlapt.core.errors import LocalServerError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.local.settings import PORT_RANGE

_LOGGER = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
HEALTH_ENDPOINT = "/health"
OPENAI_SUFFIX = "/v1"
READY_TIMEOUT_SECONDS = 180.0
POLL_INTERVAL_SECONDS = 0.5
STOP_TIMEOUT_SECONDS = 10.0
HEALTH_TIMEOUT_SECONDS = 2.0
GPU_LAYERS_ALL = 999
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@runtime_checkable
class ManagedProcess(Protocol):
    """Minimal Popen surface the manager relies on."""

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int | None: ...

    def kill(self) -> None: ...


PopenFn = Callable[[tuple[str, ...]], ManagedProcess]
HealthFn = Callable[[str], bool]  # health URL -> is the server answering?


@dataclass(frozen=True)
class ServerSpec:
    """Everything needed to launch one llama-server instance."""

    server_path: str
    model_path: str
    port: int
    mmproj_path: str = ""
    context_length: int = 4096
    gpu_layers: int = -1
    threads: int = 0
    parallel: int = 2


def total_context(spec: ServerSpec) -> int:
    """Total ``-c`` value: per-slot 上下文长度 × parallel slots."""
    return spec.context_length * spec.parallel


def build_server_args(spec: ServerSpec) -> tuple[str, ...]:
    """The llama-server argument vector for ``spec`` (validated, no shell).

    ``spec.context_length`` is PER SLOT: llama-server divides ``-c`` by
    ``--parallel``, so the emitted ``-c`` is :func:`total_context`.
    Reasoning/thinking is disabled (see module docstring).
    """
    if not spec.server_path:
        raise ValidationError("请先在设置中选择 llama-server 可执行文件")
    if not spec.model_path:
        raise ValidationError("请先选择要启动的模型文件")
    low, high = PORT_RANGE
    if not (low <= spec.port <= high):
        raise ValidationError(f"端口必须在 {low}-{high} 之间,得到 {spec.port}")
    if spec.context_length <= 0:
        raise ValidationError(f"上下文长度必须为正数,得到 {spec.context_length}")
    if spec.parallel <= 0:
        raise ValidationError(f"并发数必须为正数,得到 {spec.parallel}")
    gpu_layers = GPU_LAYERS_ALL if spec.gpu_layers < 0 else spec.gpu_layers
    args: list[str] = [
        spec.server_path,
        "-m",
        spec.model_path,
        "--host",
        DEFAULT_HOST,
        "--port",
        str(spec.port),
        "-c",
        str(total_context(spec)),
        "--parallel",
        str(spec.parallel),
        "-ngl",
        str(gpu_layers),
        "--reasoning",
        "off",
    ]
    if spec.threads > 0:
        args.extend(["-t", str(spec.threads)])
    if spec.mmproj_path:
        args.extend(["--mmproj", spec.mmproj_path])
    return tuple(args)


def health_url(port: int) -> str:
    """The llama-server health endpoint for ``port``."""
    return f"http://{DEFAULT_HOST}:{port}{HEALTH_ENDPOINT}"


def base_url(port: int) -> str:
    """OpenAI-compatible base URL of the local server on ``port``."""
    return f"http://{DEFAULT_HOST}:{port}{OPENAI_SUFFIX}"


def _default_popen(args: tuple[str, ...]) -> ManagedProcess:
    """Spawn the server detached from any console window (no shell)."""
    return subprocess.Popen(  # noqa: S603 - validated non-shell argument vector
        list(args),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_CREATE_NO_WINDOW,
    )


def _default_health(url: str) -> bool:
    """Whether the localhost health endpoint answers 200 within the timeout."""
    request = urllib.request.Request(url, headers={"User-Agent": "NLapt"})
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed localhost http URL
            request, timeout=HEALTH_TIMEOUT_SECONDS
        ) as response:
            return int(getattr(response, "status", 200) or 200) == 200
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False


class LocalServerManager:
    """Owns at most one llama-server process at a time.

    ``start``/``stop`` are serialized by a re-entrant lock so concurrent
    callers (worker-pool threads) cannot interleave process management.
    """

    def __init__(
        self,
        *,
        popen: PopenFn | None = None,
        health_check: HealthFn | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._popen = popen if popen is not None else _default_popen
        self._health = health_check if health_check is not None else _default_health
        self._sleep = sleep
        self._clock = clock
        self._process: ManagedProcess | None = None
        self._port: int | None = None
        self._spec: ServerSpec | None = None
        # RLock: start() calls stop() when a previous server is running.
        self._lock = threading.RLock()

    # -- state -------------------------------------------------------------------
    def is_running(self) -> bool:
        """Whether the managed process is alive right now."""
        return self._process is not None and self._process.poll() is None

    @property
    def current_spec(self) -> ServerSpec | None:
        """The spec of the running server (None when stopped)."""
        return self._spec if self.is_running() else None

    @property
    def current_base_url(self) -> str:
        """Base URL of the running server ('' when stopped)."""
        if not self.is_running() or self._port is None:
            return ""
        return base_url(self._port)

    # -- lifecycle ---------------------------------------------------------------
    def start(
        self, spec: ServerSpec, *, ready_timeout: float = READY_TIMEOUT_SECONDS
    ) -> str:
        """Launch llama-server and wait until ``/health`` answers.

        Returns the OpenAI-compatible base URL. Any previously managed
        process is stopped first. Raises :class:`ValidationError` for bad
        specs / missing files and :class:`LocalServerError` when the process
        exits early or never becomes healthy (it is stopped again then).
        """
        args = build_server_args(spec)
        if not Path(spec.server_path).is_file():
            raise ValidationError(f"llama-server 不存在: {spec.server_path}")
        if not Path(spec.model_path).is_file():
            raise ValidationError(f"模型文件不存在: {spec.model_path}")
        if spec.mmproj_path and not Path(spec.mmproj_path).is_file():
            raise ValidationError(f"视觉组件不存在: {spec.mmproj_path}")
        with self._lock:
            if self.is_running():
                self.stop()

            _LOGGER.info("starting llama-server: %s", " ".join(args))
            try:
                self._process = self._popen(args)
            except OSError as exc:
                raise LocalServerError(f"无法启动 llama-server: {exc}") from exc
            self._port = spec.port
            self._spec = spec

            probe = health_url(spec.port)
            deadline = self._clock() + ready_timeout
            while self._clock() < deadline:
                exit_code = self._process.poll()
                if exit_code is not None:
                    self._process = None
                    raise LocalServerError(
                        f"llama-server 启动失败,提前退出(退出码 {exit_code})。"
                        "常见原因:显存不足或模型文件损坏;可在 设置 ▸ 本地推理 "
                        "调低 GPU 层数 / 上下文长度,或换更小的量化档后重试。"
                    )
                if self._health(probe):
                    _LOGGER.info("llama-server ready on port %d", spec.port)
                    return base_url(spec.port)
                self._sleep(POLL_INTERVAL_SECONDS)
            self.stop()
            raise LocalServerError(
                f"llama-server 在 {ready_timeout:.0f} 秒内未就绪,已停止。"
                "大模型首次加载较慢,可换更小的量化档重试。"
            )

    def ensure(
        self, spec: ServerSpec, *, ready_timeout: float = READY_TIMEOUT_SECONDS
    ) -> str:
        """Return a server serving exactly ``spec``, starting one only if needed.

        The keep-alive primitive behind batch inference (加载一次,推理全部):
        when the running server was started with an identical spec its base
        URL is returned immediately — no model unload/reload; any other
        state starts (and, via :meth:`start`, first stops) a server.
        """
        with self._lock:
            if self.is_running() and self._spec == spec:
                _LOGGER.debug("llama-server already serving this spec; reusing")
                return base_url(spec.port)
            return self.start(spec, ready_timeout=ready_timeout)

    def stop(self) -> None:
        """Terminate the managed process (terminate → wait → kill). Idempotent."""
        with self._lock:
            process = self._process
            self._process = None
            self._port = None
            self._spec = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                _LOGGER.warning("llama-server ignored terminate; killing")
                process.kill()
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except (OSError, subprocess.SubprocessError):
            # stop() must never raise — a process stuck even after kill()
            # (e.g. hung in un-interruptible I/O) is logged and abandoned.
            _LOGGER.exception("error while stopping llama-server")
        _LOGGER.info("llama-server stopped")
