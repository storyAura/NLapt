"""Async per-segment translation bridge between widgets and a translator.

Widgets (translate section, editor floating toolbar) call
:meth:`TranslateBridge.request` / :meth:`translate_all_cjk` and react to the
``segment_ready`` / ``all_done`` signals; the actual translation call runs on
the QThreadPool through :func:`nlapt_gui.workers.run_async`, so the GUI thread
never blocks.

The active provider comes from :class:`nlapt_gui.translate_config`: ``"llm"``
(default) translates via the controller's configured LLM profile, while
``"google"`` / ``"baidu"`` / ``"deepl"`` use the built-in web providers in
:mod:`nlapt.llm.web_translate`. ``configured()`` reflects whether the selected
provider is usable right now (Google always is; Baidu/DeepL need keys; the LLM
needs an active profile).

Caching: results live in a core :class:`TranslationCache` keyed by a hash of
the source text, so identical tags across files/segments share one LLM call.
``fresh=True`` (重译) bypasses the cache and overwrites the shared entry with
the new alternative.
"""

from __future__ import annotations

import hashlib
import time
from itertools import count
from typing import Any, Callable, Sequence

from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.core.errors import LLMError
from nlapt.diagnostics import get_logger
from nlapt.llm.translate import (
    CJK_CHAR_RANGES,
    CachedTranslation,
    Direction,
    TranslationCache,
    detect_direction,
)
from nlapt.llm.web_translate import PROVIDER_LLM, create_provider

from nlapt_gui.controller import AppController
from nlapt_gui.translate_config import TranslationConfig, load_translation_config
from nlapt_gui.workers import run_async

# A resolved single-segment translation callable ``(text, direction) -> str``.
TranslateFn = Callable[[str, Direction], str]

_LOGGER = get_logger(__name__)

# Cache key namespace (keys are text-hash based, independent of file/segment).
_CACHE_KEY_PREFIX = "seg::"
# Shown by rows when no LLM profile is configured (design guided state).
NOTE_UNCONFIGURED = "(未配置翻译 API)"


def has_cjk(text: str) -> bool:
    """True when the text contains at least one Han character."""
    return any(
        low <= ord(char) <= high
        for char in text
        for low, high in CJK_CHAR_RANGES
    )


def _text_cache_key(text: str) -> str:
    """Stable cache key from the segment text only (identical tags share)."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{_CACHE_KEY_PREFIX}{digest}"


class TranslateBridge(QObject):
    """Non-blocking segment translator over ``controller.make_translator_or_none``."""

    segment_ready = Signal(str, str, str, bool)  # key, source_text, result, ok
    all_done = Signal(str, int, int)  # key, succeeded, failed
    # Caption-level target-language translation (中/英/日 workspace):
    # key, source_text, target_lang, result, ok.
    target_ready = Signal(str, str, str, str, bool)

    def __init__(
        self,
        controller: AppController,
        *,
        pool: QThreadPool | None = None,
        config: TranslationConfig | None = None,
        transport: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._cache = TranslationCache()
        self._sequence = count()
        # When a config is injected (tests) it is fixed; otherwise the active
        # provider selection is read fresh from disk so a 设置 save is picked
        # up without any explicit wiring. ``transport`` lets tests drive web
        # providers through httpx.MockTransport.
        self._fixed_config = config
        self._transport = transport

    def _config(self) -> TranslationConfig:
        if self._fixed_config is not None:
            return self._fixed_config
        return load_translation_config()

    def configured(self) -> bool:
        """True when the selected provider is usable right now.

        The LLM provider needs an active profile; Baidu/DeepL need their keys;
        Google is always usable.
        """
        return self._resolve_translate_fn() is not None

    def _resolve_translate_fn(self) -> TranslateFn | None:
        """Build the callable for the selected provider, or None if unusable."""
        config = self._config()
        if config.provider == PROVIDER_LLM:
            translator = self._controller.make_translator_or_none()
            if translator is None:
                return None

            def llm_translate(text: str, direction: Direction) -> str:
                # A unique per-call key keeps the translator's own cache out of
                # the way: the bridge cache (text-hash keyed) is the single
                # authority, so fresh requests genuinely reach the LLM.
                request_key = f"{_text_cache_key(text)}::{next(self._sequence)}"
                return translator.translate(request_key, text, direction).translated

            return llm_translate
        try:
            provider = create_provider(
                config.provider,
                config.credentials(),
                transport=self._transport,
            )
        except LLMError as exc:
            _LOGGER.info("translation provider %r unusable: %s", config.provider, exc)
            return None
        return provider.translate

    def request(self, key: str, text: str, *, fresh: bool = False) -> None:
        """Translate one segment asynchronously; emits ``segment_ready``.

        ``fresh=True`` skips the cache lookup (重译) and replaces the shared
        cache entry with the newly produced translation.
        """
        self._start(key, text, fresh=fresh, on_finish=None)

    def _resolve_target_fn(self, target_lang: str) -> Callable[[str], str] | None:
        """A ``text -> translated`` callable for a target language, or None."""
        config = self._config()
        if config.provider == PROVIDER_LLM:
            translator = self._controller.make_translator_or_none()
            if translator is None:
                return None
            return lambda text: translator.translate_to(
                f"{_text_cache_key(text)}::{target_lang}", text, target_lang
            ).translated
        try:
            provider = create_provider(
                config.provider,
                config.credentials(),
                transport=self._transport,
            )
        except LLMError as exc:
            _LOGGER.info("translation provider %r unusable: %s", config.provider, exc)
            return None
        translate_to = getattr(provider, "translate_to", None)
        if translate_to is None:  # defensive: every built-in provider has it
            _LOGGER.error("provider %r lacks translate_to", config.provider)
            return None
        return lambda text: translate_to(text, target_lang)

    def request_to(self, key: str, text: str, target_lang: str) -> None:
        """Translate a whole caption into ``target_lang``; emits ``target_ready``.

        Results are cached per (text, target language) so repeated requests
        for the same caption cost one provider call.
        """
        stripped = text.strip()
        if not stripped:
            _LOGGER.warning("empty caption text ignored for key %r", key)
            self.target_ready.emit(key, text, target_lang, "", False)
            return
        translate_fn = self._resolve_target_fn(target_lang)
        if translate_fn is None:
            _LOGGER.info("caption translation requested while unconfigured (%r)", key)
            self.target_ready.emit(key, text, target_lang, NOTE_UNCONFIGURED, False)
            return
        cache_key = f"{_text_cache_key(stripped)}::{target_lang}"
        if not self._cache.is_stale(cache_key, stripped):
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.target_ready.emit(key, text, target_lang, cached.translated, True)
                return

        def work() -> str:
            return translate_fn(stripped)

        def done(result: object) -> None:
            if isinstance(result, str) and result.strip():
                entry = CachedTranslation(
                    source_text=stripped,
                    translated=result,
                    direction=detect_direction(stripped),
                    created_at=time.time(),
                )
                self._cache.put(cache_key, entry)
                self.target_ready.emit(key, text, target_lang, result, True)
            else:  # defensive: providers must return non-empty text
                _LOGGER.error("empty/invalid translation payload %r", type(result))
                self.target_ready.emit(key, text, target_lang, "", False)

        def failed(message: str) -> None:
            _LOGGER.warning("caption translation failed for %r: %s", key, message)
            self.target_ready.emit(key, text, target_lang, message, False)

        run_async(self._pool, work, on_done=done, on_error=failed)

    def translate_all_cjk(self, key: str, texts: Sequence[str]) -> None:
        """Translate every CJK-containing text; emits ``all_done`` when finished.

        ``segment_ready`` still fires per segment so rows can update live.
        """
        targets = [text for text in texts if text.strip() and has_cjk(text)]
        if not targets:
            self.all_done.emit(key, 0, 0)
            return
        state = {"remaining": len(targets), "ok": 0, "failed": 0}

        def finish_one(ok: bool) -> None:
            state["ok" if ok else "failed"] += 1
            state["remaining"] -= 1
            if state["remaining"] == 0:
                self.all_done.emit(key, state["ok"], state["failed"])

        for text in targets:
            self._start(key, text, fresh=False, on_finish=finish_one)

    # -- internals -----------------------------------------------------------------
    def _start(
        self,
        key: str,
        text: str,
        *,
        fresh: bool,
        on_finish: Callable[[bool], None] | None,
    ) -> None:
        stripped = text.strip()
        if not stripped:
            _LOGGER.warning("empty segment text ignored for key %r", key)
            self._finish(key, text, "", False, on_finish)
            return
        translate_fn = self._resolve_translate_fn()
        if translate_fn is None:
            _LOGGER.info("translation requested while unconfigured (key %r)", key)
            self._finish(key, text, NOTE_UNCONFIGURED, False, on_finish)
            return
        cache_key = _text_cache_key(stripped)
        if not fresh and not self._cache.is_stale(cache_key, stripped):
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._finish(key, text, cached.translated, True, on_finish)
                return
        direction = detect_direction(stripped)

        def work() -> str:
            return translate_fn(stripped, direction)

        def done(result: object) -> None:
            if isinstance(result, str) and result.strip():
                entry = CachedTranslation(
                    source_text=stripped,
                    translated=result,
                    direction=direction,
                    created_at=time.time(),
                )
                self._cache.put(cache_key, entry)
                self._finish(key, text, result, True, on_finish)
            else:  # defensive: providers must return non-empty text
                _LOGGER.error("empty/invalid translation payload %r", type(result))
                self._finish(key, text, "", False, on_finish)

        def failed(message: str) -> None:
            _LOGGER.warning("translation failed for %r: %s", stripped, message)
            self._finish(key, text, message, False, on_finish)

        run_async(self._pool, work, on_done=done, on_error=failed)

    def _finish(
        self,
        key: str,
        text: str,
        result: str,
        ok: bool,
        on_finish: Callable[[bool], None] | None,
    ) -> None:
        self.segment_ready.emit(key, text, result, ok)
        if on_finish is not None:
            on_finish(ok)
