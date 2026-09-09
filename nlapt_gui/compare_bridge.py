"""多对比推标 fan-out: caption every image with several pool models at once.

``CompareRunner`` builds one vision captioner per model (each keeps its own
spec-8 pacing limiter), then runs the ``keys × models`` job matrix on the
worker pool with a bounded number in flight. Results stream back as
``item_ready(key, ref, text_or_error, ok)``; nothing is written — the review
dialog decides what goes into the txt files afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import QObject, QThreadPool, Signal
from shiboken6 import isValid

from nlapt.core.config import ModelRef
from nlapt.diagnostics import get_logger

from nlapt_gui.compare_config import MIN_COMPARE_MODELS
from nlapt_gui.controller import AppController
from nlapt_gui.model_targets import choice_label
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

Captioner = Callable[[Path, str, str], str]

ERROR_TOO_FEW = "请先在设置 ▸ 多对比推标 中勾选至少 {n} 个模型"
ERROR_STALE_MODEL = "模型「{label}」已不在模型池中,请到设置 ▸ 多对比推标 重新勾选"
ERROR_NOT_CONFIGURED = "模型「{label}」所在 API 未配置 Base URL"
ERROR_NO_KEYS = "尚未选择任何图片"
ERROR_RUNNING = "上一次对比仍在进行"


@dataclass(frozen=True)
class CompareJob:
    key: str
    ref: ModelRef


class CompareRunner(QObject):
    """Runs the compare matrix; see module docstring."""

    item_ready = Signal(str, object, str, bool)  # key, ModelRef, text_or_error, ok
    progress = Signal(int, int)  # done, total
    finished = Signal()

    def __init__(
        self,
        controller: AppController,
        *,
        pool: QThreadPool | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._captioners: dict[ModelRef, Captioner] = {}
        self._pending: list[CompareJob] = []
        self._in_flight = 0
        self._done = 0
        self._total = 0
        self._concurrency = 1
        self._system = ""
        self._user_prompt = ""
        self._cancelled = False
        self._running = False

    # -- public -----------------------------------------------------------------
    def running(self) -> bool:
        return self._running

    def start(
        self,
        keys: Sequence[str],
        refs: Sequence[ModelRef],
        *,
        system: str,
        user_prompt: str,
        concurrency: int = 1,
    ) -> str | None:
        """Begin the matrix; returns a Chinese error instead when it cannot start.

        Every ref must resolve to an enabled pool model whose API has a Base
        URL — a stale ref aborts the whole run rather than being skipped, so
        the user never compares fewer models than they configured.
        """
        if self._running:
            return ERROR_RUNNING
        keys = tuple(dict.fromkeys(keys))
        refs = tuple(dict.fromkeys(refs))
        if not keys:
            return ERROR_NO_KEYS
        if len(refs) < MIN_COMPARE_MODELS:
            return ERROR_TOO_FEW.format(n=MIN_COMPARE_MODELS)
        captioners: dict[ModelRef, Captioner] = {}
        for ref in refs:
            profile = self._controller.profile_for_ref(ref)
            if profile is None:
                return ERROR_STALE_MODEL.format(label=choice_label(ref))
            captioner = self._controller.make_vision_captioner_or_none(profile=profile)
            if captioner is None:
                return ERROR_NOT_CONFIGURED.format(label=choice_label(ref))
            captioners[ref] = captioner
        self._captioners = captioners
        self._pending = [CompareJob(key, ref) for key in keys for ref in refs]
        self._total = len(self._pending)
        self._done = 0
        self._in_flight = 0
        self._concurrency = max(1, concurrency)
        self._system = system
        self._user_prompt = user_prompt
        self._cancelled = False
        self._running = True
        _LOGGER.info(
            "compare run started (images=%s, models=%s, concurrency=%s)",
            len(keys),
            len(refs),
            self._concurrency,
        )
        self.progress.emit(0, self._total)
        self._pump()
        return None

    def cancel(self) -> None:
        """Stop dispatching; jobs already in flight finish and are ignored."""
        if not self._running:
            return
        self._cancelled = True
        self._pending.clear()
        if self._in_flight == 0:
            self._finish()

    # -- internals ---------------------------------------------------------------
    def _pump(self) -> None:
        while self._pending and self._in_flight < self._concurrency:
            job = self._pending.pop(0)
            self._in_flight += 1
            self._dispatch(job)
        if not self._pending and self._in_flight == 0 and self._running:
            self._finish()

    def _dispatch(self, job: CompareJob) -> None:
        captioner = self._captioners[job.ref]
        image_path = self._controller.image_path(job.key)
        system = self._system
        user_prompt = self._user_prompt

        def work() -> str:
            return captioner(image_path, system, user_prompt)

        def done(result: object) -> None:
            self._settle(job, str(result), True)

        def failed(message: str) -> None:
            self._settle(job, message, False)

        run_async(self._pool, work, on_done=done, on_error=failed)

    def _settle(self, job: CompareJob, text: str, ok: bool) -> None:
        if not isValid(self):
            return
        self._in_flight -= 1
        if self._cancelled:
            if self._in_flight == 0:
                self._finish()
            return
        self._done += 1
        self.item_ready.emit(job.key, job.ref, text, ok)
        self.progress.emit(self._done, self._total)
        self._pump()

    def _finish(self) -> None:
        if not self._running:
            return
        self._running = False
        self._captioners = {}
        _LOGGER.info(
            "compare run %s (done=%s/%s)",
            "cancelled" if self._cancelled else "finished",
            self._done,
            self._total,
        )
        self.finished.emit()
