"""Async bridge between the 本地推理 tab and :mod:`nlapt.local`.

Owns the persisted :class:`LocalSettings`, runs hardware detection, GGUF
downloads and llama-server lifecycle on the shared ``QThreadPool`` and
reports back through Qt signals only — the tab never blocks the GUI thread
and never touches ``nlapt.local`` directly for slow work.

The llama-server process manager is a PROCESS-WIDE singleton
(:func:`get_server_manager`): the settings dialog is recreated on every
open, but a started server must survive dialog closes and be terminated
when the application exits (``atexit``).
"""

from __future__ import annotations

import atexit
import threading
from pathlib import Path
from typing import Any

import shiboken6
from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.core.errors import DownloadCancelledError
from nlapt.diagnostics import get_logger
from nlapt.local.catalog import (
    ModelFamily,
    QuantFile,
    download_url,
    find_family,
    find_quant,
    mmproj_path,
    quant_path,
)
from nlapt.local.download import download_file
from nlapt.local.hardware import HardwareInfo, detect_hardware
from nlapt.local.server import LocalServerManager, ServerSpec
from nlapt.local.settings import (
    SETTINGS_FILE_NAME,
    LocalSettings,
    load_local_settings,
    save_local_settings,
)

from nlapt_gui.resources import app_data_dir
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

MODELS_DIR_NAME = "models"
# Emit a progress signal at most every this many new bytes (UI flood guard).
PROGRESS_EMIT_STEP_BYTES = 8 * 1024 * 1024

SERVER_STOPPED = "stopped"
SERVER_STARTING = "starting"
SERVER_RUNNING = "running"
SERVER_ERROR = "error"

# download_finished statuses — a user cancel is NOT a failure.
DOWNLOAD_OK = "ok"
DOWNLOAD_CANCELLED = "cancelled"
DOWNLOAD_ERROR = "error"

_SHARED_MANAGER: LocalServerManager | None = None

# Destinations of in-flight downloads, PROCESS-wide: a download outlives the
# dialog (and bridge) that started it, and two writers on one ``.part`` file
# would corrupt it. Guarded by _ACTIVE_LOCK.
_ACTIVE_DOWNLOAD_PATHS: set[str] = set()
_ACTIVE_LOCK = threading.Lock()


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
        self._cancel_event: threading.Event | None = None
        self._downloading = False

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
        """The configured model directory (default: app data dir / models)."""
        configured = self._settings.models_dir.strip()
        if configured:
            return Path(configured)
        return app_data_dir() / MODELS_DIR_NAME

    # -- files -------------------------------------------------------------------------
    def model_file(self, family: ModelFamily, quant: QuantFile) -> Path:
        return quant_path(self.models_dir(), family, quant)

    def mmproj_file(self, family: ModelFamily) -> Path | None:
        return mmproj_path(self.models_dir(), family)

    def is_downloaded(self, family: ModelFamily, quant: QuantFile) -> bool:
        """Whether the quant (and its mmproj, for vision models) is complete.

        Deliberately synchronous: a couple of ``stat`` calls on selection
        change is the accepted small exception to the never-block rule
        (worst case is a sleeping network drive chosen as 模型目录).
        """
        model = self.model_file(family, quant)
        if not model.is_file() or model.stat().st_size != quant.size_bytes:
            return False
        mmproj = self.mmproj_file(family)
        if mmproj is None:
            return True
        return mmproj.is_file() and mmproj.stat().st_size == family.mmproj_bytes

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
        return self._downloading

    def start_download(self, family_id: str, quant_label: str) -> bool:
        """Download the quant + mmproj (resumable). False when already busy."""
        if self._downloading:
            return False
        family = find_family(family_id)
        quant = find_quant(family, quant_label)
        jobs: list[tuple[str, Path, int, str]] = []
        mmproj = self.mmproj_file(family)
        if mmproj is not None:
            jobs.append(
                (
                    download_url(family.repo_id, family.mmproj_filename),
                    mmproj,
                    family.mmproj_bytes,
                    family.mmproj_sha256,
                )
            )
        jobs.append(
            (
                download_url(family.repo_id, quant.filename),
                self.model_file(family, quant),
                quant.size_bytes,
                quant.sha256,
            )
        )
        total_bytes = sum(expected for _url, _dest, expected, _sha in jobs)
        dest_keys = tuple(str(dest) for _url, dest, _expected, _sha in jobs)
        with _ACTIVE_LOCK:
            # A download started from a previous (now closed) dialog may still
            # be writing these files — refuse to race it.
            if any(key in _ACTIVE_DOWNLOAD_PATHS for key in dest_keys):
                return False
            _ACTIVE_DOWNLOAD_PATHS.update(dest_keys)
        cancel = threading.Event()
        self._cancel_event = cancel
        self._downloading = True
        bridge = self

        def work() -> str:
            finished_prefix = 0
            last_emitted = -PROGRESS_EMIT_STEP_BYTES
            try:
                for url, dest, expected, sha256 in jobs:

                    def report(
                        done: int, _total: int | None, *, base: int = finished_prefix
                    ) -> None:
                        nonlocal last_emitted
                        overall = base + done
                        if (
                            overall - last_emitted < PROGRESS_EMIT_STEP_BYTES
                            and overall != total_bytes
                        ):
                            return
                        last_emitted = overall
                        # Worker-thread emit: _alive() alone is racy against a
                        # GUI-thread teardown, so a stale-object RuntimeError
                        # is swallowed as well (progress is best-effort).
                        try:
                            if _alive(bridge):
                                bridge.download_progress.emit(
                                    family_id, quant_label, overall, total_bytes
                                )
                        except RuntimeError:
                            _LOGGER.debug("progress emit after bridge teardown")

                    download_file(
                        url,
                        dest,
                        expected_bytes=expected,
                        expected_sha256=sha256 or None,
                        progress=report,
                        cancel=cancel,
                    )
                    finished_prefix += expected
            except DownloadCancelledError:
                return DOWNLOAD_CANCELLED
            return DOWNLOAD_OK

        def finish(status: str, message: str) -> None:
            with _ACTIVE_LOCK:
                _ACTIVE_DOWNLOAD_PATHS.difference_update(dest_keys)
            self._downloading = False
            self._cancel_event = None
            if status == DOWNLOAD_ERROR:
                _LOGGER.warning(
                    "download failed for %s/%s: %s", family_id, quant_label, message
                )
            elif status == DOWNLOAD_CANCELLED:
                _LOGGER.info("download cancelled for %s/%s", family_id, quant_label)
            if _alive(self):
                self.download_finished.emit(family_id, quant_label, status, message)

        run_async(
            self._pool,
            work,
            on_done=lambda status: finish(str(status), ""),
            on_error=lambda message: finish(DOWNLOAD_ERROR, message),
        )
        return True

    def cancel_download(self) -> None:
        """Signal the running download to stop (partial files are kept)."""
        event = self._cancel_event
        if event is not None:
            event.set()

    # -- server ------------------------------------------------------------------------
    def server_running(self) -> bool:
        return self._manager.is_running()

    def server_base_url(self) -> str:
        return self._manager.current_base_url

    def build_server_spec(self, family: ModelFamily, quant: QuantFile) -> ServerSpec:
        """ServerSpec for the current settings + a catalog selection."""
        settings = self._settings
        mmproj = self.mmproj_file(family)
        return ServerSpec(
            server_path=settings.server_path,
            model_path=str(self.model_file(family, quant)),
            port=settings.port,
            mmproj_path=str(mmproj) if mmproj is not None else "",
            context_length=settings.context_length,
            gpu_layers=settings.gpu_layers,
            threads=settings.threads,
            parallel=settings.parallel,
        )

    def start_server(self, family_id: str, quant_label: str) -> None:
        """Async llama-server start; progress via ``server_changed``."""
        family = find_family(family_id)
        quant = find_quant(family, quant_label)
        spec = self.build_server_spec(family, quant)
        self.server_changed.emit(SERVER_STARTING, "")

        def done(url: object) -> None:
            if _alive(self):
                self.server_changed.emit(SERVER_RUNNING, str(url))

        def failed(message: str) -> None:
            _LOGGER.warning("llama-server start failed: %s", message)
            if _alive(self):
                self.server_changed.emit(SERVER_ERROR, message)

        run_async(
            self._pool,
            lambda: self._manager.start(spec),
            on_done=done,
            on_error=failed,
        )

    def stop_server(self) -> None:
        """Async llama-server stop; emits ``server_changed('stopped', '')``."""

        def done(_result: Any) -> None:
            if _alive(self):
                self.server_changed.emit(SERVER_STOPPED, "")

        def failed(message: str) -> None:
            # Never leave the UI wedged in "stopping": surface the failure.
            _LOGGER.warning("stopping llama-server failed: %s", message)
            if _alive(self):
                self.server_changed.emit(SERVER_ERROR, message)

        run_async(self._pool, self._manager.stop, on_done=done, on_error=failed)
