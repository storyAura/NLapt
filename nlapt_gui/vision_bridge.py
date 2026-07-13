"""Async vision re-inference bridge (标注工作区 ▸ 重译).

The caption workspace asks :class:`VisionBridge` to re-caption the current
image with the configured vision model; the blocking LLM round-trip runs on
the QThreadPool through :func:`nlapt_gui.workers.run_async` and the result
comes back through the ``caption_ready`` signal on the GUI thread.

The system / user prompts come from :mod:`nlapt_gui.prompt_store` (设置 ▸
提示词): the active custom system prompt (may be empty) and the user prompt
(falls back to a built-in instruction so the feature works out of the box).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController
from nlapt_gui.prompt_store import VisionPrompts, load_vision_prompts
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

# Result note when no vision model is configured (guided state).
NOTE_VISION_UNCONFIGURED = "(未配置视觉模型)"


class VisionBridge(QObject):
    """Non-blocking image captioner over ``controller.make_vision_captioner_or_none``."""

    caption_ready = Signal(str, str, bool)  # key, result_or_error, ok

    def __init__(
        self,
        controller: AppController,
        *,
        pool: QThreadPool | None = None,
        prompts: VisionPrompts | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        # When prompts are injected (tests) they are fixed; otherwise the
        # active prompt selection is read fresh from disk per request so a
        # 设置 save is picked up without explicit wiring.
        self._fixed_prompts = prompts

    def _prompts(self) -> VisionPrompts:
        if self._fixed_prompts is not None:
            return self._fixed_prompts
        return load_vision_prompts()

    def configured(self) -> bool:
        """True when an active profile with a vision model is configured."""
        return self._controller.make_vision_captioner_or_none() is not None

    def request(self, key: str) -> None:
        """Re-caption the image behind ``key``; emits ``caption_ready``."""
        captioner = self._controller.make_vision_captioner_or_none()
        if captioner is None:
            _LOGGER.info("vision caption requested while unconfigured (%r)", key)
            self.caption_ready.emit(key, NOTE_VISION_UNCONFIGURED, False)
            return
        try:
            image_path: Path = self._controller.image_path(key)
        except NLaptError as exc:
            _LOGGER.warning("vision caption for unknown key %r: %s", key, exc)
            self.caption_ready.emit(key, str(exc), False)
            return
        prompts = self._prompts()
        system = prompts.system_text()
        user_prompt = prompts.effective_user_prompt()

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
