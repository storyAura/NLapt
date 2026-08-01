"""Async vision inference bridge (标注工作区 ▸ LLM 推理 / 本地推理).

Two engines produce captions from images:

- ``ENGINE_LLM``   — the configured cloud/API profile
  (``controller.make_vision_captioner_or_none``).
- ``ENGINE_LOCAL`` — the 设置 ▸ 本地推理 model served by llama-server
  (:func:`nlapt_gui.local_bridge.make_local_vision_captioner`); the runtime
  and server are provisioned lazily on the worker pool, and consecutive
  requests reuse the running server (the model loads once).

Single requests (``request``) power the caption workspace buttons; batch
requests (``request_batch``) power the file panel's 文件夹推标 and run
through :meth:`AppController.run_caption_batch` (snapshot + oplog +
checkpoint). For a local batch that had to start the server, the model is
unloaded after the WHOLE batch finishes — never between items.

Prompts come from :mod:`nlapt_gui.prompt_store` (统一管线 by default; the
local engine may carry its own system/user prompts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger
from nlapt.local.settings import load_local_settings

from nlapt_gui.controller import AppController
from nlapt_gui.local_bridge import (
    get_server_manager,
    local_settings_path,
    make_local_vision_captioner,
    resolve_local_target,
)
from nlapt_gui.prompt_store import (
    ENGINE_LLM,
    ENGINE_LOCAL,
    VisionPrompts,
    load_vision_prompts,
)
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

# Result note when no vision model is configured (guided state).
NOTE_VISION_UNCONFIGURED = "(未配置视觉模型)"

# Chinese engine words used in descriptions / history labels / toasts.
ENGINE_WORDS: dict[str, str] = {ENGINE_LLM: "LLM", ENGINE_LOCAL: "本地模型"}
BATCH_DESCRIPTION_FMT = "推标({word}) · {n} 张"
BATCH_HISTORY_FMT = "推标({word})"

Captioner = Callable[[Path, str, str], str]


class VisionBridge(QObject):
    """Non-blocking image captioner over the LLM profile or the local model."""

    caption_ready = Signal(str, str, bool)  # key, result_or_error, ok

    def __init__(
        self,
        controller: AppController,
        *,
        pool: QThreadPool | None = None,
        prompts: VisionPrompts | None = None,
        local_captioner_factory: Callable[[], Captioner] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        # When prompts are injected (tests) they are fixed; otherwise the
        # active prompt selection is read fresh from disk per request so a
        # 设置 save is picked up without explicit wiring.
        self._fixed_prompts = prompts
        # Test seam: builds the local captioner (may raise NLaptError).
        self._local_captioner_factory = (
            local_captioner_factory
            if local_captioner_factory is not None
            else self._default_local_factory
        )

    def _default_local_factory(self) -> Captioner:
        return make_local_vision_captioner(
            image_max_edge=self._controller.app.config.image_max_edge
        )

    def _prompts(self) -> VisionPrompts:
        if self._fixed_prompts is not None:
            return self._fixed_prompts
        return load_vision_prompts()

    # -- readiness ---------------------------------------------------------------------
    def configured(self, engine: str = ENGINE_LLM) -> bool:
        """Whether ``engine`` could serve a caption request right now."""
        if engine == ENGINE_LOCAL:
            try:
                resolve_local_target()
            except NLaptError:
                return False
            return True
        return self._controller.make_vision_captioner_or_none() is not None

    def _make_captioner(self, engine: str) -> tuple[Captioner | None, str]:
        """(captioner, error message) — exactly one side is meaningful."""
        if engine == ENGINE_LOCAL:
            try:
                return self._local_captioner_factory(), ""
            except NLaptError as exc:
                return None, str(exc)
        captioner = self._controller.make_vision_captioner_or_none()
        if captioner is None:
            return None, NOTE_VISION_UNCONFIGURED
        return captioner, ""

    # -- single request ----------------------------------------------------------------
    def request(self, key: str, engine: str = ENGINE_LLM) -> None:
        """Caption the image behind ``key``; emits ``caption_ready``."""
        captioner, error = self._make_captioner(engine)
        if captioner is None:
            _LOGGER.info("vision caption unavailable for %r (%s): %s", key, engine, error)
            self.caption_ready.emit(key, error, False)
            return
        try:
            image_path: Path = self._controller.image_path(key)
        except NLaptError as exc:
            _LOGGER.warning("vision caption for unknown key %r: %s", key, exc)
            self.caption_ready.emit(key, str(exc), False)
            return
        prompts = self._prompts()
        system = prompts.system_text_for(engine)
        user_prompt = prompts.user_prompt_for(engine)

        def work() -> str:
            return captioner(image_path, system, user_prompt)

        def done(result: object) -> None:
            text = result if isinstance(result, str) else ""
            if text.strip():
                self.caption_ready.emit(key, text, True)
            else:  # defensive: the captioner must return non-empty text
                _LOGGER.error("empty vision caption payload for %r", key)
                self.caption_ready.emit(key, "", False)

        def failed(message: str) -> None:
            _LOGGER.warning("vision caption failed for %r: %s", key, message)
            self.caption_ready.emit(key, message, False)

        run_async(self._pool, work, on_done=done, on_error=failed)

    # -- batch request (文件夹推标) ------------------------------------------------------
    def request_batch(self, keys: tuple[str, ...], engine: str = ENGINE_LLM) -> bool:
        """Run 推标 over ``keys`` via the controller's caption batch.

        Returns False (with a warn toast) when the engine is not ready or a
        batch is already running. Local engine: the server is started once,
        reused for every item, and stopped after the LAST item only if this
        batch started it (a user-prestarted server is left running).
        """
        captioner, error = self._make_captioner(engine)
        if captioner is None:
            self._controller.toast_requested.emit(error, "warn")
            return False
        prompts = self._prompts()
        system = prompts.system_text_for(engine)
        user_prompt = prompts.user_prompt_for(engine)
        word = ENGINE_WORDS.get(engine, engine)
        config = self._controller.app.config

        if engine == ENGINE_LOCAL:
            concurrency = load_local_settings(local_settings_path()).parallel
        else:
            concurrency = config.request.concurrency

        manager = get_server_manager()
        server_was_running = manager.is_running()

        def caption_one(_key: str, image_path: Path) -> str:
            return captioner(image_path, system, user_prompt)

        def finished(_report: object) -> None:
            # 全部推理完再卸载: unload only after the whole batch, and only
            # when this batch was the one that loaded the model.
            if engine == ENGINE_LOCAL and not server_was_running:
                run_async(
                    self._pool,
                    manager.stop,
                    on_done=lambda _r: None,
                    on_error=lambda message: _LOGGER.warning(
                        "post-batch llama-server stop failed: %s", message
                    ),
                )

        return self._controller.run_caption_batch(
            keys,
            caption_one,
            description=BATCH_DESCRIPTION_FMT.format(word=word, n=len(keys)),
            history_label=BATCH_HISTORY_FMT.format(word=word),
            engine=engine,
            concurrency=max(1, concurrency),
            on_finished=finished,
        )
