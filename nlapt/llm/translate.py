"""Translation with direction auto-detection and a staleness-aware cache (spec 7.3).

Translations are auxiliary reading aids: they are cached per file key with the
source text they were produced from, so the UI can flag "source changed" and
offer re-translation. Translations never write into caption txt files here.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum

from nlapt.core.config import LLMProfile, RequestControl
from nlapt.core.errors import LLMConfigError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import DEFAULT_TIMEOUT_SECONDS, LLMClient, LLMMessage, LLMRequest
from nlapt.llm.cleaning import OUTPUT_CONSTRAINT, clean_llm_output
from nlapt.llm.retry import MinIntervalLimiter

# Shared spec-8 request-control plumbing lives next to the rewrite service;
# both services apply identical timeout/retry/pacing semantics.
from nlapt.llm.rewrite import (
    _paced_complete,
    _resolve_limiter,
    _validated_request_control,
    _validated_retry_sleep,
)
from nlapt.llm.templates import DEFAULT_TEMPLATES, render_template

_LOGGER = get_logger(__name__)


class Direction(str, Enum):
    EN_TO_ZH = "en->zh"
    ZH_TO_EN = "zh->en"


# Target languages for the caption-level 中/英/日 translation (spec module 1).
LANG_ZH = "zh"
LANG_EN = "en"
LANG_JA = "ja"
TARGET_LANGS: tuple[str, ...] = (LANG_ZH, LANG_EN, LANG_JA)
LANG_LABELS: Mapping[str, str] = {
    LANG_ZH: "中文",
    LANG_EN: "English",
    LANG_JA: "日本語",
}
TEMPLATE_KEY_BY_TARGET: Mapping[str, str] = {
    LANG_ZH: "translate_to_zh",
    LANG_EN: "translate_to_en",
    LANG_JA: "translate_to_ja",
}


# CJK ratio at or above this threshold means the text is treated as Chinese.
CJK_RATIO_THRESHOLD = 0.3
# Han character ranges (CJK Unified Ideographs, Extension A, Compatibility).
CJK_CHAR_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
)

TEMPLATE_KEY_BY_DIRECTION: Mapping[Direction, str] = {
    Direction.EN_TO_ZH: "translate_en_zh",
    Direction.ZH_TO_EN: "translate_zh_en",
}


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in CJK_CHAR_RANGES)


def detect_direction(text: str) -> Direction:
    """CJK char ratio >= 0.3 -> ZH_TO_EN, else EN_TO_ZH. Empty -> EN_TO_ZH."""
    if not isinstance(text, str):
        raise ValidationError(f"text must be a string, got {type(text).__name__}")
    considered = [char for char in text if not char.isspace()]
    if not considered:
        return Direction.EN_TO_ZH
    cjk_count = sum(1 for char in considered if _is_cjk(char))
    ratio = cjk_count / len(considered)
    return Direction.ZH_TO_EN if ratio >= CJK_RATIO_THRESHOLD else Direction.EN_TO_ZH


@dataclass(frozen=True)
class CachedTranslation:
    """A translation result bound to the exact source text it came from."""

    source_text: str
    translated: str
    direction: Direction
    created_at: float


class TranslationCache:
    """Per-key translation cache. Entries are replaced, never mutated."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, CachedTranslation] = {}

    def get(self, key: str) -> CachedTranslation | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, entry: CachedTranslation) -> None:
        if not isinstance(key, str) or not key:
            raise ValidationError("cache key must be a non-empty string")
        if not isinstance(entry, CachedTranslation):
            raise ValidationError(
                f"entry must be a CachedTranslation, got {type(entry).__name__}"
            )
        with self._lock:
            self._entries[key] = entry

    def is_stale(self, key: str, current_text: str) -> bool:
        """True when there is no entry or the source text has changed since."""
        entry = self.get(key)
        return entry is None or entry.source_text != current_text


class Translator:
    """Translate captions via the profile's text model, with caching."""

    def __init__(
        self,
        client: LLMClient,
        profile: LLMProfile,
        *,
        cache: TranslationCache | None = None,
        templates: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.time,
        request: RequestControl | None = None,
        retry_sleep: Callable[[float], None] = time.sleep,
        limiter: MinIntervalLimiter | None = None,
    ) -> None:
        """``request`` (spec 8) wires timeout/retries/min-interval into every
        LLM call; ``None`` keeps the legacy direct-call behavior. One shared
        ``limiter`` paces requests across all callers of this instance (built
        from ``request.min_interval`` unless injected). ``retry_sleep`` is the
        backoff sleep, injectable so tests never really wait.
        """
        if not isinstance(client, LLMClient):
            raise ValidationError(
                f"client must be an LLMClient, got {type(client).__name__}"
            )
        if not isinstance(profile, LLMProfile):
            raise ValidationError(
                f"profile must be an LLMProfile, got {type(profile).__name__}"
            )
        self._client = client
        self._profile = profile
        self._cache = cache
        self._templates = dict(templates) if templates else {}
        self._clock = clock
        self._request_control = _validated_request_control(request)
        self._retry_sleep = _validated_retry_sleep(retry_sleep)
        self._limiter = _resolve_limiter(request, limiter)

    def _template_for(self, direction: Direction) -> str:
        template_key = TEMPLATE_KEY_BY_DIRECTION[direction]
        return self._templates.get(template_key, DEFAULT_TEMPLATES[template_key])

    def _complete_prompt(self, prompt: str) -> str:
        """Run one text completion under the shared request controls."""
        timeout = (
            self._request_control.timeout
            if self._request_control is not None
            else DEFAULT_TIMEOUT_SECONDS
        )
        request = LLMRequest(
            messages=(LLMMessage(role="user", text=prompt),),
            model=self._profile.text_model,
            system=self._profile.system_prompt,
            temperature=self._profile.temperature,
            max_tokens=self._profile.max_tokens,
            timeout=timeout,
        )
        response = _paced_complete(
            self._client,
            request,
            control=self._request_control,
            limiter=self._limiter,
            retry_sleep=self._retry_sleep,
        )
        return clean_llm_output(response.text)

    def translate_to(self, key: str, text: str, target: str) -> CachedTranslation:
        """Translate ``text`` into a target language (``zh`` / ``en`` / ``ja``).

        Unlike :meth:`translate` there is no direction detection and the
        instance cache is not consulted — callers (the GUI bridge) key their
        own cache on ``(text, target)``.
        """
        if not isinstance(key, str) or not key:
            raise ValidationError("translation key must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("there is no text to translate")
        if target not in TARGET_LANGS:
            raise ValidationError(
                f"target must be one of {TARGET_LANGS}, got {target!r}"
            )
        if not self._profile.text_model:
            raise LLMConfigError(
                f"profile {self._profile.name!r} has no text_model configured"
            )
        template_key = TEMPLATE_KEY_BY_TARGET[target]
        template = self._templates.get(template_key, DEFAULT_TEMPLATES[template_key])
        prompt = render_template(template, caption=text) + "\n\n" + OUTPUT_CONSTRAINT
        translated = self._complete_prompt(prompt)
        entry = CachedTranslation(
            source_text=text,
            translated=translated,
            direction=detect_direction(text),
            created_at=self._clock(),
        )
        _LOGGER.debug("translated %r to %s (%d chars)", key, target, len(translated))
        return entry

    def translate(
        self, key: str, text: str, direction: Direction | None = None
    ) -> CachedTranslation:
        """Translate ``text``; fresh cache hits are returned without an LLM call."""
        if not isinstance(key, str) or not key:
            raise ValidationError("translation key must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("there is no text to translate")
        if not self._profile.text_model:
            raise LLMConfigError(
                f"profile {self._profile.name!r} has no text_model configured"
            )
        resolved = direction if direction is not None else detect_direction(text)
        if self._cache is not None and not self._cache.is_stale(key, text):
            cached = self._cache.get(key)
            if cached is not None and cached.direction == resolved:
                _LOGGER.debug("translation cache hit for %r", key)
                return cached
        prompt = (
            render_template(self._template_for(resolved), caption=text)
            + "\n\n"
            + OUTPUT_CONSTRAINT
        )
        translated = self._complete_prompt(prompt)
        entry = CachedTranslation(
            source_text=text,
            translated=translated,
            direction=resolved,
            created_at=self._clock(),
        )
        if self._cache is not None:
            self._cache.put(key, entry)
        _LOGGER.debug("translated %r (%s, %d chars)", key, resolved.value, len(translated))
        return entry
