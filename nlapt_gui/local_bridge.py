"""Async bridge between the 本地推理 tab and :mod:`nlapt.local`.

Owns the persisted :class:`LocalSettings`, runs hardware detection, GGUF
downloads and llama-server lifecycle on the shared ``QThreadPool`` and
reports back through Qt signals only — the tab never blocks the GUI thread
and never touches ``nlapt.local`` directly for slow work.

The llama-server process manager is a PROCESS-WIDE singleton
(:func:`get_server_manager`): the settings dialog is recreated on every
open, but a started server must survive dialog closes and be terminated
when the application exits (``atexit``).

Downloads are process-wide too: one task at a time, tracked in module state
and broadcast through the :func:`get_download_hub` singleton, so a dialog
reopened mid-download re-attaches to the live progress (and can cancel)
instead of only being told a task exists.
"""

from __future__ import annotations

import atexit
import sys
import threading
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Callable

import shiboken6
from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.core.config import LLMProfile
from nlapt.core.errors import LocalInferenceError, NLaptError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import LLMMessage, LLMRequest, create_client
from nlapt.llm.cleaning import clean_llm_output, ensure_not_refusal
from nlapt.llm.retry import RetryPolicy, with_retry
from nlapt.llm.vision import prepare_image
from nlapt.local.advisor import auto_gpu_layers
from nlapt.local.catalog import (
    ENGINE_FLORENCE,
    LoraEntry,
    ModelFamily,
    QuantFile,
    download_url,
    find_family,
    find_lora,
    find_quant,
    lora_adapter_path,
    lora_download_url,
    lora_file_path,
    mmproj_path,
    quant_path,
)
from nlapt.local.florence import FlorenceEngine
from nlapt.local.gguf import read_block_count
from nlapt.local.lora import MSG_LORA_FILE_MISSING
from nlapt.local.hardware import HardwareInfo, detect_hardware
from nlapt.local.presets import resolve_preset
from nlapt.local.runtime import (
    MSG_PLATFORM_UNSUPPORTED,
    RuntimeAsset,
    current_asset,
    ensure_runtime,
    find_server_exe,
    runtime_dir,
)
from nlapt.local.server import LocalServerManager, ServerSpec
from nlapt.local.settings import (
    SETTINGS_FILE_NAME,
    LocalSettings,
    load_local_settings,
    save_local_settings,
)

from nlapt_gui import download_hub as _download_hub
from nlapt_gui import idle_server as _idle_server
from nlapt_gui.idle_server import get_idle_stopper
from nlapt_gui.resources import app_data_dir, resource_path
from nlapt_gui.workers import run_async

# Public download-slot API — implementation lives in download_hub so this
# module stays under the 800-line cap. Tests/UI keep importing these names.
DOWNLOAD_OK = _download_hub.DOWNLOAD_OK
DOWNLOAD_CANCELLED = _download_hub.DOWNLOAD_CANCELLED
DOWNLOAD_ERROR = _download_hub.DOWNLOAD_ERROR
_ActiveDownload = _download_hub._ActiveDownload
active_download = _download_hub.active_download
cancel_active_download = _download_hub.cancel_active_download
get_download_hub = _download_hub.get_download_hub
launch_download_jobs = _download_hub.launch_download_jobs
IDLE_STOP_DELAY_SECONDS = _idle_server.IDLE_STOP_DELAY_SECONDS
IdleServerStopper = _idle_server.IdleServerStopper

_LOGGER = get_logger(__name__)

MODELS_DIR_NAME = "models"
# Per-user directory holding the auto-provisioned llama.cpp runtime.
RUNTIME_DIR_NAME = "runtime"
# Local inference is slow on CPU offload; give requests generous headroom.
LOCAL_VISION_TIMEOUT_SECONDS = 300.0
LOCAL_VISION_PROFILE_NAME = "local-vision"
LOCAL_VISION_API_TYPE = "openai"

# Actionable guidance when local inference is not ready yet.
MSG_NO_MODEL_SELECTED = "未选择本地模型 — 打开 设置 ▸ 本地推理,选择并下载一个模型"
MSG_NOT_DOWNLOADED = "本地模型尚未下载完成 — 打开 设置 ▸ 本地推理 下载"
MSG_NOT_VISION = "所选本地模型不支持图片输入 — 请选择带「视觉」标记的模型"

SERVER_STOPPED = "stopped"
SERVER_STARTING = "starting"
SERVER_RUNNING = "running"
SERVER_ERROR = "error"

# family_id sentinel prefix in download-hub signals for curated LoRAs.
LORA_FAMILY_PREFIX = "lora:"

_SHARED_MANAGER: LocalServerManager | None = None


def get_server_manager() -> LocalServerManager:
    """Process-wide llama-server manager, stopped automatically at exit."""
    global _SHARED_MANAGER
    if _SHARED_MANAGER is None:
        _SHARED_MANAGER = LocalServerManager()
        atexit.register(_SHARED_MANAGER.stop)
    return _SHARED_MANAGER


def _alive(obj: QObject) -> bool:
    """Whether the underlying C++ object still exists (async-reply guard)."""
    return shiboken6.isValid(obj)


def default_models_dir() -> Path:
    """Default download dir INSIDE the app (用户要求: 默认放在项目内).

    Next to the executable in a frozen build, the repo root in a source
    checkout. Users can point the primary dir elsewhere and add extra
    reuse directories on top.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / MODELS_DIR_NAME
    return resource_path(MODELS_DIR_NAME)


# -- settings-driven path resolution (module level: reused OUTSIDE the dialog) ------
def local_settings_path() -> Path:
    """Location of the persisted local-inference settings."""
    return app_data_dir() / SETTINGS_FILE_NAME


def models_dir_for(settings: LocalSettings) -> Path:
    """Primary (download) directory: user override or the in-app default."""
    configured = settings.models_dir.strip()
    return Path(configured) if configured else default_models_dir()


def search_dirs(settings: LocalSettings) -> tuple[Path, ...]:
    """Primary dir + the extra reuse dirs (order kept, deduplicated)."""
    dirs: list[Path] = [models_dir_for(settings)]
    for raw in settings.extra_dirs:
        text = raw.strip()
        if not text:
            continue
        candidate = Path(text)
        if candidate not in dirs:
            dirs.append(candidate)
    return tuple(dirs)


def find_model_file_in(
    settings: LocalSettings, family: ModelFamily, quant: QuantFile
) -> Path:
    """An existing right-size copy in ANY dir, else the primary path.

    Works for the main quant AND for ``family.extra_files`` entries — both
    are :class:`QuantFile` values stored under their basename in the
    per-family directory.
    """
    for base in search_dirs(settings):
        candidate = quant_path(base, family, quant)
        if candidate.is_file() and candidate.stat().st_size == quant.size_bytes:
            return candidate
    return quant_path(models_dir_for(settings), family, quant)


def find_mmproj_file_in(settings: LocalSettings, family: ModelFamily) -> Path | None:
    """An existing right-size mmproj in ANY dir, else the primary path."""
    if not family.vision or not family.mmproj_filename:
        return None
    for base in search_dirs(settings):
        candidate = mmproj_path(base, family)
        if (
            candidate is not None
            and candidate.is_file()
            and candidate.stat().st_size == family.mmproj_bytes
        ):
            return candidate
    return mmproj_path(models_dir_for(settings), family)


def is_downloaded_in(
    settings: LocalSettings, family: ModelFamily, quant: QuantFile
) -> bool:
    """Whether the quant + extra files (+ mmproj, for vision) all exist."""

    def complete(path: Path, size: int) -> bool:
        return path.is_file() and path.stat().st_size == size

    if not complete(find_model_file_in(settings, family, quant), quant.size_bytes):
        return False
    for extra in family.extra_files:
        if not complete(find_model_file_in(settings, family, extra), extra.size_bytes):
            return False
    if not family.vision or not family.mmproj_filename:
        return True
    mmproj = find_mmproj_file_in(settings, family)
    return mmproj is not None and complete(mmproj, family.mmproj_bytes)


def is_lora_downloaded_in(settings: LocalSettings, entry: LoraEntry) -> bool:
    """Whether every file of a curated LoRA exists at its expected size.

    LoRAs live in the PRIMARY dir only (they are small; the reuse-dirs
    machinery exists for multi-GB model files).
    """
    base = models_dir_for(settings)
    for file in entry.files:
        path = lora_file_path(base, entry, file)
        if not path.is_file() or path.stat().st_size != file.size_bytes:
            return False
    return True


# -- llama.cpp runtime resolution ---------------------------------------------------
def runtime_base_dir() -> Path:
    """Per-user directory where the auto-provisioned runtime is kept."""
    return app_data_dir() / RUNTIME_DIR_NAME


def runtime_supported() -> bool:
    """Whether a pinned runtime archive exists for this platform."""
    return current_asset() is not None


def pending_runtime_asset(settings: LocalSettings) -> RuntimeAsset | None:
    """The runtime archive that still needs downloading (None when ready).

    Ready means: a manual ``server_path`` is configured, the runtime is
    already extracted, or the platform has no pinned archive (manual-only).
    """
    if resolve_server_path(settings):
        return None
    return current_asset()


def resolve_server_path(settings: LocalSettings) -> str:
    """The llama-server executable: manual setting first, else the runtime.

    Empty string when neither exists yet — the runtime may still be
    auto-provisioned later by :func:`nlapt.local.runtime.ensure_runtime`.
    """
    manual = settings.server_path.strip()
    if manual:
        return manual
    exe = find_server_exe(runtime_dir(runtime_base_dir()))
    return str(exe) if exe is not None else ""


def build_spec_for(
    settings: LocalSettings, family: ModelFamily, quant: QuantFile
) -> ServerSpec:
    """ServerSpec for ``settings`` + a catalog selection (found files used)."""
    mmproj = find_mmproj_file_in(settings, family)
    return ServerSpec(
        server_path=resolve_server_path(settings),
        model_path=str(find_model_file_in(settings, family, quant)),
        port=settings.port,
        mmproj_path=str(mmproj) if mmproj is not None else "",
        context_length=settings.context_length,
        gpu_layers=settings.gpu_layers,
        threads=settings.threads,
        parallel=settings.parallel,
    )


# -- launch-time spec finalization ---------------------------------------------------
# Resolved 自动 GPU layer counts per (model_path, context_length). The cache
# keeps every caller launching an IDENTICAL spec (free VRAM fluctuates between
# probes; a changing value would make manager.ensure() restart the server
# mid-batch) and avoids re-running nvidia-smi per request.
# ponytail: cache lives for the process; restart the app after a big VRAM
# change (e.g. closing a game) to re-probe.
_AUTO_LAYERS_CACHE: dict[tuple[str, int], int] = {}
_AUTO_LAYERS_LOCK = threading.Lock()


def _auto_gpu_layers_for(
    spec: ServerSpec, family: ModelFamily, quant: QuantFile
) -> int:
    # KV cache scales with the TOTAL context the server allocates
    # (per-slot 上下文长度 × parallel slots, see server.total_context).
    context = spec.context_length * spec.parallel
    key = (spec.model_path, context)
    with _AUTO_LAYERS_LOCK:
        cached = _AUTO_LAYERS_CACHE.get(key)
        if cached is not None:
            return cached
        hardware = detect_hardware()
        block_count = read_block_count(Path(spec.model_path))
        layers = auto_gpu_layers(
            family,
            quant,
            hardware,
            context_length=context,
            block_count=block_count,
        )
        _AUTO_LAYERS_CACHE[key] = layers
        _LOGGER.info(
            "auto GPU layers for %s: -ngl %d (blocks=%s, total ctx=%d)",
            Path(spec.model_path).name,
            layers,
            block_count,
            context,
        )
        return layers


def prepare_launch_spec(
    spec: ServerSpec, family: ModelFamily, quant: QuantFile
) -> ServerSpec:
    """Blocking pre-launch finalization (call on a worker thread).

    Provisions the pinned runtime when no llama-server is set, and resolves
    自动 GPU layers (-1) into a count that actually fits the detected
    hardware — blindly passing "offload everything" OOM-crashed llama-server
    on GPUs where the advisor itself predicted partial offload.
    """
    if not spec.server_path:
        spec = _dc_replace(
            spec, server_path=str(ensure_runtime(runtime_base_dir()))
        )
    if spec.gpu_layers < 0:
        spec = _dc_replace(
            spec, gpu_layers=_auto_gpu_layers_for(spec, family, quant)
        )
    return spec


# -- local inference target + captioner --------------------------------------------
@dataclass(frozen=True)
class LocalTarget:
    """A validated, launchable local-model selection."""

    family: ModelFamily
    quant: QuantFile
    spec: ServerSpec  # server_path may be "" until the runtime is ensured


def resolve_local_target(
    settings: LocalSettings | None = None, *, require_vision: bool = True
) -> LocalTarget:
    """The local model 推理 would use right now, or a typed actionable error.

    Raises :class:`LocalInferenceError` when no model is selected /
    downloaded / vision-capable, and :class:`LocalServerError` (from the
    runtime module) when the platform needs a manual llama-server pick.
    """
    resolved = (
        settings if settings is not None else load_local_settings(local_settings_path())
    )
    if not resolved.family_id or not resolved.quant_label:
        raise LocalInferenceError(MSG_NO_MODEL_SELECTED)
    try:
        family = find_family(resolved.family_id)
        quant = find_quant(family, resolved.quant_label)
    except NLaptError as exc:
        raise LocalInferenceError(MSG_NO_MODEL_SELECTED) from exc
    if require_vision and not family.vision:
        raise LocalInferenceError(MSG_NOT_VISION)
    if not is_downloaded_in(resolved, family, quant):
        raise LocalInferenceError(MSG_NOT_DOWNLOADED)
    spec = build_spec_for(resolved, family, quant)
    # Florence runs in-process — no llama-server / pinned runtime involved.
    if (
        family.engine != ENGINE_FLORENCE
        and not spec.server_path
        and not runtime_supported()
    ):
        raise LocalInferenceError(MSG_PLATFORM_UNSUPPORTED)
    return LocalTarget(family=family, quant=quant, spec=spec)


# -- Florence engine (in-process, PROCESS-wide like the server manager) -------------
_FLORENCE_ENGINE: FlorenceEngine | None = None
_FLORENCE_KEY: tuple[tuple[str, str], ...] = ()
_FLORENCE_LOCK = threading.Lock()


def florence_files_for(
    settings: LocalSettings, family: ModelFamily, quant: QuantFile
) -> dict[str, Path]:
    """basename -> found local path for a Florence family's file set."""
    files = {Path(quant.filename).name: find_model_file_in(settings, family, quant)}
    for extra in family.extra_files:
        files[Path(extra.filename).name] = find_model_file_in(settings, family, extra)
    return files


def get_florence_engine(
    files: dict[str, Path], lora_path: Path | None = None
) -> FlorenceEngine:
    """Process-wide Florence engine for ``files`` (replaced when they change).

    Keeping one engine alive between requests is the Florence version of
    加载一次,推理全部 — sessions stay in memory until the file set OR the
    LoRA selection changes (a retrained LoRA file counts as a change:
    the key includes its mtime + size) or the process exits.
    """
    global _FLORENCE_ENGINE, _FLORENCE_KEY
    lora_token = ""
    if lora_path is not None:
        try:
            stat = lora_path.stat()
            lora_token = f"{lora_path}|{stat.st_mtime_ns}|{stat.st_size}"
        except OSError:
            lora_token = str(lora_path)
    key = tuple(
        sorted((name, str(path)) for name, path in files.items())
    ) + (("__lora__", lora_token),)
    with _FLORENCE_LOCK:
        if _FLORENCE_ENGINE is None or _FLORENCE_KEY != key:
            _FLORENCE_ENGINE = FlorenceEngine(files, lora_path=lora_path)
            _FLORENCE_KEY = key
        return _FLORENCE_ENGINE


def make_local_vision_captioner(
    *, image_max_edge: int
) -> Callable[[Path, str, str], str]:
    """A ``(image_path, system, user_prompt) -> caption`` callable on the
    CURRENT local settings, validating them now (typed errors, see
    :func:`resolve_local_target`).

    The callable blocks (runtime provisioning + server ensure + HTTP
    round-trip) — run it on the worker pool. ``ensure`` reuses a running
    server with the same spec, so consecutive calls never reload the model.

    Florence families skip the server entirely: the ONNX engine runs
    in-process, steered by the persisted 指令 (``settings.florence_task``);
    the system/user prompts do not apply to it.

    Caption specialists with official presets (:mod:`nlapt.local.presets`)
    replace the passed prompts with the persisted preset's system/user pair
    unless the user opted into 自定义 (``PRESET_CUSTOM``).
    """
    target = resolve_local_target()
    family = target.family
    settings = load_local_settings(local_settings_path())

    if family.engine == ENGINE_FLORENCE:
        lora_text = settings.florence_lora.strip()
        lora_path = Path(lora_text) if lora_text else None
        if lora_path is not None and not lora_path.is_file():
            raise LocalInferenceError(MSG_LORA_FILE_MISSING.format(path=lora_path))
        engine = get_florence_engine(
            florence_files_for(settings, family, target.quant), lora_path
        )
        task = settings.florence_task
        # Official Florence-2 knows none of the PromptGen 指令 — clamp a
        # persisted task the selected family was never trained on.
        if family.florence_tasks and task not in family.florence_tasks:
            task = family.florence_tasks[0]

        def florence_caption(image_path: Path, system: str, user_prompt: str) -> str:
            return engine.caption(image_path, task)

        return florence_caption

    preset = resolve_preset(family.family_id, settings.prompt_preset)

    def caption(image_path: Path, system: str, user_prompt: str) -> str:
        if preset is not None:
            system = preset.system
            user_prompt = preset.user_prompt
        spec = prepare_launch_spec(target.spec, target.family, target.quant)
        url = get_server_manager().ensure(spec)
        profile = LLMProfile(
            name=LOCAL_VISION_PROFILE_NAME,
            api_type=LOCAL_VISION_API_TYPE,
            base_url=url,
            api_key="",
            text_model=family.family_id,
            vision_model=family.family_id,
        )
        client = create_client(profile)
        image = prepare_image(image_path, max_edge=image_max_edge)
        request = LLMRequest(
            messages=(LLMMessage(role="user", text=user_prompt, images=(image,)),),
            model=family.family_id,
            system=system,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            timeout=LOCAL_VISION_TIMEOUT_SECONDS,
        )
        # Transient failures (busy slots, empty completions under load)
        # back off and retry like the cloud path does.
        response = with_retry(lambda: client.complete(request), RetryPolicy())
        return ensure_not_refusal(clean_llm_output(response.text))

    return caption


class LocalBridge(QObject):
    """UI-facing async facade over catalog / hardware / download / server."""

    hardware_ready = Signal(object)  # HardwareInfo
    # family_id, quant_label, done_bytes, total_bytes (None when unknown)
    download_progress = Signal(str, str, object, object)
    # family_id, quant_label, status (DOWNLOAD_OK/CANCELLED/ERROR), message
    download_finished = Signal(str, str, str, str)
    server_changed = Signal(str, str)  # state, detail (base_url or error message)

    def __init__(
        self,
        *,
        settings_path: Path | None = None,
        pool: QThreadPool | None = None,
        manager: LocalServerManager | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings_path = (
            settings_path
            if settings_path is not None
            else app_data_dir() / SETTINGS_FILE_NAME
        )
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._manager = manager if manager is not None else get_server_manager()
        self._settings = load_local_settings(self._settings_path)
        self._hardware: HardwareInfo | None = None
        # Downloads are process-wide (they outlive the dialog that started
        # them): forward the hub's signals so THIS bridge's listeners see the
        # progress of a download any bridge started. Qt drops the connection
        # automatically when this bridge is destroyed.
        hub = get_download_hub()
        hub.progress.connect(self.download_progress)
        hub.finished.connect(self.download_finished)

    # -- settings ----------------------------------------------------------------------
    @property
    def settings(self) -> LocalSettings:
        return self._settings

    def update_settings(self, **changes: object) -> LocalSettings:
        """Replace fields, persist atomically, and return the new settings."""
        self._settings = self._settings.with_changes(**changes)
        save_local_settings(self._settings_path, self._settings)
        return self._settings

    def models_dir(self) -> Path:
        """Primary (download) directory: user override or the in-app default."""
        return models_dir_for(self._settings)

    def models_dirs(self) -> tuple[Path, ...]:
        """Primary dir + the extra reuse dirs (order kept, deduplicated)."""
        return search_dirs(self._settings)

    # -- files -------------------------------------------------------------------------
    # File checks are deliberately synchronous: a few ``stat`` calls on
    # selection change is the accepted small exception to the never-block
    # rule (worst case is a sleeping network drive among the dirs).
    def model_file(self, family: ModelFamily, quant: QuantFile) -> Path:
        """Download destination for a quant (always in the primary dir)."""
        return quant_path(self.models_dir(), family, quant)

    def mmproj_file(self, family: ModelFamily) -> Path | None:
        """Download destination for the mmproj (always in the primary dir)."""
        return mmproj_path(self.models_dir(), family)

    def find_model_file(self, family: ModelFamily, quant: QuantFile) -> Path:
        """An existing right-size copy in ANY dir, else the primary path."""
        return find_model_file_in(self._settings, family, quant)

    def find_mmproj_file(self, family: ModelFamily) -> Path | None:
        """An existing right-size mmproj in ANY dir, else the primary path."""
        return find_mmproj_file_in(self._settings, family)

    def is_downloaded(self, family: ModelFamily, quant: QuantFile) -> bool:
        """Whether a complete quant (and mmproj, for vision) exists in any dir."""
        return is_downloaded_in(self._settings, family, quant)

    def lora_adapter_file(self, entry: LoraEntry) -> Path:
        """Local safetensors path of a curated LoRA (primary dir)."""
        return lora_adapter_path(self.models_dir(), entry)

    def is_lora_downloaded(self, entry: LoraEntry) -> bool:
        """Whether every file of a curated LoRA exists at its pinned size."""
        return is_lora_downloaded_in(self._settings, entry)

    # -- hardware ----------------------------------------------------------------------
    @property
    def hardware(self) -> HardwareInfo | None:
        return self._hardware

    def detect(self) -> None:
        """Async hardware probe; result arrives via ``hardware_ready``."""

        def done(info: object) -> None:
            if not _alive(self):
                return
            if isinstance(info, HardwareInfo):
                self._hardware = info
                self.hardware_ready.emit(info)

        # detect_hardware never raises; on_error is a formality.
        run_async(self._pool, detect_hardware, on_done=done, on_error=lambda _m: None)

    # -- downloads ---------------------------------------------------------------------
    def is_downloading(self) -> bool:
        """Whether ANY download is running process-wide (not just ours)."""
        return active_download() is not None

    def active_download(self) -> tuple[str, str, int, int | None] | None:
        """Snapshot of the running download for UI re-attach (module state)."""
        return active_download()

    def start_download(self, family_id: str, quant_label: str) -> bool:
        """Download the quant + mmproj (resumable). False when already busy.

        就绪即可用: when no llama-server is available yet (no manual path, no
        extracted runtime) the pinned llama.cpp runtime is fetched in the
        same run, so a finished download means the model can be inferred
        immediately. One download runs at a time PROCESS-wide; progress and
        completion are broadcast through the download hub so every open
        dialog (including ones opened later) sees them.
        """
        if self.is_downloading():
            return False
        family = find_family(family_id)
        quant = find_quant(family, quant_label)

        def complete(path: Path | None, size: int) -> bool:
            return path is not None and path.is_file() and path.stat().st_size == size

        # Only fetch what no directory (incl. reuse dirs) already provides.
        jobs: list[tuple[str, Path, int, str]] = []
        mmproj_dest = self.mmproj_file(family)
        if mmproj_dest is not None and not complete(
            self.find_mmproj_file(family), family.mmproj_bytes
        ):
            jobs.append(
                (
                    download_url(family.repo_id, family.mmproj_filename),
                    mmproj_dest,
                    family.mmproj_bytes,
                    family.mmproj_sha256,
                )
            )
        if not complete(self.find_model_file(family, quant), quant.size_bytes):
            jobs.append(
                (
                    download_url(family.repo_id, quant.filename),
                    self.model_file(family, quant),
                    quant.size_bytes,
                    quant.sha256,
                )
            )
        for extra in family.extra_files:
            if not complete(
                find_model_file_in(self._settings, family, extra), extra.size_bytes
            ):
                jobs.append(
                    (
                        download_url(family.repo_id, extra.filename),
                        quant_path(self.models_dir(), family, extra),
                        extra.size_bytes,
                        extra.sha256,
                    )
                )
        # Florence needs no llama-server, so never provision the runtime for it.
        runtime_asset = (
            None
            if family.engine == ENGINE_FLORENCE
            else pending_runtime_asset(self._settings)
        )
        if not jobs and runtime_asset is None:
            # Everything already available (possibly from a reuse dir).
            self.download_finished.emit(family_id, quant_label, DOWNLOAD_OK, "")
            return True
        return self._launch_jobs(family_id, quant_label, jobs, runtime_asset)

    def start_lora_download(self, lora_id: str) -> bool:
        """Download a curated LoRA's files (resumable). False when busy.

        Broadcast identity on the shared hub is ``lora:<lora_id>`` with an
        empty quant label — the same one-at-a-time pipeline as models.
        """
        if self.is_downloading():
            return False
        entry = find_lora(lora_id)
        base = self.models_dir()
        jobs: list[tuple[str, Path, int, str]] = []
        for file in entry.files:
            dest = lora_file_path(base, entry, file)
            if dest.is_file() and dest.stat().st_size == file.size_bytes:
                continue
            jobs.append(
                (lora_download_url(entry, file), dest, file.size_bytes, file.sha256)
            )
        family_key = LORA_FAMILY_PREFIX + entry.lora_id
        if not jobs:
            self.download_finished.emit(family_key, "", DOWNLOAD_OK, "")
            return True
        return self._launch_jobs(family_key, "", jobs, None)

    def _launch_jobs(
        self,
        family_id: str,
        quant_label: str,
        jobs: list[tuple[str, Path, int, str]],
        runtime_asset: RuntimeAsset | None,
    ) -> bool:
        """Run download jobs on the pool under the process-wide single slot."""
        return launch_download_jobs(family_id, quant_label, jobs, runtime_asset)

    def cancel_download(self) -> None:
        """Signal the running download to stop (partial files are kept).

        Works from ANY bridge — a dialog reopened mid-download can cancel
        the task its predecessor started.
        """
        cancel_active_download()

    # -- server ------------------------------------------------------------------------
    def server_running(self) -> bool:
        return self._manager.is_running()

    def server_base_url(self) -> str:
        return self._manager.current_base_url

    def build_server_spec(self, family: ModelFamily, quant: QuantFile) -> ServerSpec:
        """ServerSpec for the current settings + a catalog selection.

        Uses the FOUND files (a model reused from an extra directory is
        served from where it actually lives) and the RESOLVED llama-server
        (manual setting first, else the auto-provisioned runtime).
        """
        return build_spec_for(self._settings, family, quant)

    def start_server(self, family_id: str, quant_label: str) -> None:
        """Async llama-server start; progress via ``server_changed``.

        When no llama-server executable is available yet, the pinned
        llama.cpp runtime is downloaded and extracted first (on the pool);
        自动 GPU layers resolve to a hardware-fitting count — both via
        :func:`prepare_launch_spec`, so 启动本地服务 and the inference
        captioner launch identical specs (``ensure`` keeps reusing one).
        """
        family = find_family(family_id)
        quant = find_quant(family, quant_label)
        spec = self.build_server_spec(family, quant)
        # Explicit user start: this server must never be idle-stopped.
        get_idle_stopper().note_user_control()
        self.server_changed.emit(SERVER_STARTING, "")

        def work() -> str:
            return self._manager.start(prepare_launch_spec(spec, family, quant))

        def done(url: object) -> None:
            if _alive(self):
                self.server_changed.emit(SERVER_RUNNING, str(url))

        def failed(message: str) -> None:
            _LOGGER.warning("llama-server start failed: %s", message)
            if _alive(self):
                self.server_changed.emit(SERVER_ERROR, message)

        run_async(self._pool, work, on_done=done, on_error=failed)

    def stop_server(self) -> None:
        """Async llama-server stop; emits ``server_changed('stopped', '')``."""
        get_idle_stopper().note_user_control()

        def done(_result: Any) -> None:
            if _alive(self):
                self.server_changed.emit(SERVER_STOPPED, "")

        def failed(message: str) -> None:
            # Never leave the UI wedged in "stopping": surface the failure.
            _LOGGER.warning("stopping llama-server failed: %s", message)
            if _alive(self):
                self.server_changed.emit(SERVER_ERROR, message)

        run_async(self._pool, self._manager.stop, on_done=done, on_error=failed)
