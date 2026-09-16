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
checkpoint). ``request_custom`` / ``request_layered_batch`` power the
CHA标注 wizard (caller-supplied prompts; final caption = matched card(s) +
blank + scene, ``CARD: NONE`` skips the image; ``layered_finished`` carries
the closing :class:`LayeredSummary`). For local inference that had to start
the server, the model is
NOT unloaded when the run finishes: an idle timer
(:class:`nlapt_gui.local_bridge.IdleServerStopper`) stops the server only
after 30s without a new local request, so repeated runs keep the model
loaded. A user-prestarted server is never auto-stopped.

Prompts come from :mod:`nlapt_gui.prompt_store` (统一管线 by default; the
local engine may carry its own system/user prompts).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.batch.progress import BatchReport
from nlapt.core.config import LLMProfile
from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger
from nlapt.local.catalog import ENGINE_FLORENCE, find_family
from nlapt.local.settings import load_local_settings

from nlapt_gui.controller import AppController
from nlapt_gui.layered_prompts import (
    LAYERED_BATCH_DESCRIPTION_FMT,
    LAYERED_BATCH_HISTORY,
    CardRoster,
    LayeredSummary,
    assemble_roster_caption,
    parse_layered_reply,
)
from nlapt_gui.local_bridge import (
    get_idle_stopper,
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
    custom_ready = Signal(str, str, bool)  # request_id, result_or_error, ok
    layered_finished = Signal(object)  # LayeredSummary of a CHA标注 batch

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
    def configured(
        self, engine: str = ENGINE_LLM, *, profile: LLMProfile | None = None
    ) -> bool:
        """Whether ``engine`` could serve a caption request right now."""
        if engine == ENGINE_LOCAL:
            try:
                resolve_local_target()
            except NLaptError:
                return False
            return True
        return self._controller.make_vision_captioner_or_none(profile=profile) is not None

    def _make_captioner(
        self, engine: str, profile: LLMProfile | None = None
    ) -> tuple[Captioner | None, str]:
        """(captioner, error message) — exactly one side is meaningful.

        ``profile`` overrides the active LLM archive. The local engine
        ignores it (the running local model is the model).
        """
        if engine == ENGINE_LOCAL:
            try:
                return self._local_captioner_factory(), ""
            except NLaptError as exc:
                return None, str(exc)
        captioner = self._controller.make_vision_captioner_or_none(profile=profile)
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
        # 空闲卸载: a new local request cancels any pending idle stop before
        # the worker touches the server; finishing (re-)arms the window.
        stopper = get_idle_stopper() if engine == ENGINE_LOCAL else None
        if stopper is not None:
            stopper.note_request()

        def work() -> str:
            return captioner(image_path, system, user_prompt)

        def done(result: object) -> None:
            if stopper is not None:
                stopper.note_finished()
            text = result if isinstance(result, str) else ""
            if text.strip():
                self.caption_ready.emit(key, text, True)
            else:  # defensive: the captioner must return non-empty text
                _LOGGER.error("empty vision caption payload for %r", key)
                self.caption_ready.emit(key, "", False)

        def failed(message: str) -> None:
            if stopper is not None:
                stopper.note_finished()
            _LOGGER.warning("vision caption failed for %r: %s", key, message)
            self.caption_ready.emit(key, message, False)

        run_async(self._pool, work, on_done=done, on_error=failed)

    def request_custom(
        self,
        request_id: str,
        key: str,
        engine: str,
        *,
        system: str,
        user_prompt: str,
        profile: LLMProfile | None = None,
    ) -> bool:
        """Caption ``key`` with caller-supplied prompts; emit ``custom_ready``.

        Bypasses ``load_vision_prompts``. Used by the CHA标注 wizard to
        generate character-card candidates. Returns False (and emits a
        failed ``custom_ready``) when the engine is not ready or the key
        has no image. Local idle-stopper pairing matches :meth:`request`.
        ``profile`` overrides the active LLM archive; ignored for local.
        """
        captioner, error = self._make_captioner(engine, profile)
        if captioner is None:
            self.custom_ready.emit(request_id, error, False)
            return False
        try:
            image_path = self._controller.image_path(key)
        except NLaptError as exc:
            self.custom_ready.emit(request_id, str(exc), False)
            return False
        stopper = get_idle_stopper() if engine == ENGINE_LOCAL else None
        if stopper is not None:
            stopper.note_request()

        def work() -> str:
            return captioner(image_path, system, user_prompt)

        def done(result: object) -> None:
            if stopper is not None:
                stopper.note_finished()
            text = result if isinstance(result, str) else ""
            if text.strip():
                self.custom_ready.emit(request_id, text, True)
            else:
                _LOGGER.error("empty custom vision payload for %r", request_id)
                self.custom_ready.emit(request_id, "", False)

        def failed(message: str) -> None:
            if stopper is not None:
                stopper.note_finished()
            _LOGGER.warning("custom vision failed for %r: %s", request_id, message)
            self.custom_ready.emit(request_id, message, False)

        run_async(self._pool, work, on_done=done, on_error=failed)
        return True

    def request_layered_batch(
        self,
        keys: tuple[str, ...],
        engine: str,
        *,
        roster: CardRoster,
        scene_system: str,
        scene_user: str,
        profile: LLMProfile | None = None,
    ) -> bool:
        """Batch-caption ``keys`` as matched 人物卡(s) + blank + 画面段.

        Reuses :meth:`AppController.run_caption_batch` so snapshot / progress
        / cancel / history stay on the existing 推标 path. Returns False
        (with a warn toast) when the engine is not ready or a batch is
        already running. ``profile`` overrides the active LLM archive;
        ignored for local.

        Each reply goes through :func:`parse_layered_reply` /
        :func:`assemble_roster_caption`: ``CARD: NONE`` returns ``None`` so
        the core leaves the file untouched; per-card ``OUTFIT n:`` rewrites
        replace only that card's clothing block. When the batch ends,
        ``layered_finished`` emits a :class:`LayeredSummary` (per-card match
        counts, skipped and failed keys) for the closing card dialog.
        """
        captioner, error = self._make_captioner(engine, profile)
        if captioner is None:
            self._controller.toast_requested.emit(error, "warn")
            return False
        word = ENGINE_WORDS.get(engine, engine)
        config = self._controller.app.config
        if engine == ENGINE_LOCAL:
            concurrency = load_local_settings(local_settings_path()).parallel
        else:
            concurrency = config.request.concurrency

        stopper = get_idle_stopper() if engine == ENGINE_LOCAL else None
        if stopper is not None:
            stopper.note_request()

        tally_lock = threading.Lock()
        counts = [0] * len(roster)
        none_keys: list[str] = []
        failed_keys: list[str] = []

        def caption_one(key: str, image_path: Path) -> str | None:
            try:
                reply = captioner(image_path, scene_system, scene_user)
                verdict = parse_layered_reply(reply, len(roster))
                caption = assemble_roster_caption(roster, verdict)
            except Exception:
                with tally_lock:
                    failed_keys.append(key)
                raise
            with tally_lock:
                if caption is None:
                    none_keys.append(key)
                else:
                    for index in verdict.matched:
                        counts[index] += 1
            if caption is None:
                _LOGGER.info("layered reply for %s matched no card; skipped", key)
            return caption

        def finished(report: object) -> None:
            if stopper is not None:
                stopper.note_finished()
            failed = set(failed_keys)
            cancelled = False
            if isinstance(report, BatchReport):
                failed.update(report.failed_keys)
                cancelled = report.status.value == "cancelled"
            with tally_lock:
                summary = LayeredSummary(
                    roster=roster,
                    matched_counts=tuple(counts),
                    none_keys=tuple(none_keys),
                    failed_keys=tuple(sorted(failed)),
                    cancelled=cancelled,
                )
            self.layered_finished.emit(summary)

        started = self._controller.run_caption_batch(
            keys,
            caption_one,
            description=LAYERED_BATCH_DESCRIPTION_FMT.format(n=len(keys)),
            history_label=LAYERED_BATCH_HISTORY,
            engine=engine,
            concurrency=max(1, concurrency),
            on_finished=finished,
        )
        if not started and stopper is not None:
            stopper.note_finished()
        _LOGGER.info(
            "layered batch %s: %d file(s) via %s",
            "started" if started else "refused",
            len(keys),
            word,
        )
        return started

    # -- batch request (文件夹推标) ------------------------------------------------------
    def request_batch(self, keys: tuple[str, ...], engine: str = ENGINE_LLM) -> bool:
        """Run 推标 over ``keys`` via the controller's caption batch.

        Returns False (with a warn toast) when the engine is not ready or a
        batch is already running. Local engine: the server is started once
        and reused for every item; after the batch it stays loaded and is
        stopped by the idle timer only after 30s without a new local run —
        and only if inference (not the user) started it.
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

        # 推理完先不卸载: the whole batch counts as one run for the idle
        # stopper — a pending stop is cancelled now, and the 30s window only
        # starts once the LAST item finished.
        stopper = get_idle_stopper() if engine == ENGINE_LOCAL else None
        if stopper is not None:
            stopper.note_request()

        def caption_one(_key: str, image_path: Path) -> str:
            return captioner(image_path, system, user_prompt)

        def finished(_report: object) -> None:
            if stopper is not None:
                stopper.note_finished()

        started = self._controller.run_caption_batch(
            keys,
            caption_one,
            description=BATCH_DESCRIPTION_FMT.format(word=word, n=len(keys)),
            history_label=BATCH_HISTORY_FMT.format(word=word),
            engine=engine,
            concurrency=max(1, concurrency),
            on_finished=finished,
        )
        if not started and stopper is not None:
            stopper.note_finished()  # nothing queued: balance note_request
        return started


def local_engine_is_florence() -> bool:
    """True when the configured local family is Florence-2 (instruction mode).

    Does not require the weights to be downloaded — the wizard uses this
    to disable the local engine option, because Florence ignores free-form
    prompts and cannot run the 分层推标 skills.
    """
    settings = load_local_settings(local_settings_path())
    if not settings.family_id:
        return False
    try:
        family = find_family(settings.family_id)
    except NLaptError:
        return False
    return family.engine == ENGINE_FLORENCE
