"""Built-in web translation providers (Google / Baidu / DeepL).

These are lightweight, self-contained alternatives to the LLM translator so a
user can translate captions without configuring a large-language-model
profile. Every provider implements the small :class:`TranslationProvider`
protocol ``translate(text, direction) -> str`` and runs a single blocking HTTP
round-trip; callers push that call onto a worker thread (the GUI uses
``nlapt_gui.workers.run_async``) so the UI thread never blocks.

Providers:

* :class:`GoogleFreeProvider` -- unofficial, keyless ``translate_a/single``
  endpoint. Free and needs no registration, but it is an undocumented,
  best-effort interface that may rate-limit or change without notice.
* :class:`BaiduProvider` -- the official 百度翻译 open API. Requires an APP ID
  and secret key obtained by registering at https://fanyi-api.baidu.com .
* :class:`DeepLProvider` -- the DeepL API Free tier. Requires an API key from
  https://www.deepl.com/pro-api .
* :class:`DeepLXProvider` -- DeepLX free ``POST /translate`` (self-hosted
  or a community instance from https://deeplx.owo.network/endpoints/free.html ).
* :class:`CustomOpenAIProvider` -- user-supplied OpenAI-compatible
  ``/chat/completions`` endpoint (Base URL + model + optional key).

``httpx`` is imported lazily (via :func:`nlapt.llm.base.require_httpx`) so the
core package keeps no hard HTTP dependency, and every provider accepts a
``transport`` so tests can drive it with ``httpx.MockTransport`` -- no real
network is ever required.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from nlapt.core.errors import LLMConfigError, LLMRequestError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import http_post_json, join_url, require_httpx
from nlapt.llm.retry import MinIntervalLimiter
from nlapt.llm.translate import Direction, TARGET_LANGS

_LOGGER = get_logger(__name__)

# Web translation endpoints answer in a couple of seconds when healthy; a
# short per-attempt timeout + the caller's retry beats one 60 s hang.
WEB_TRANSLATE_TIMEOUT_SECONDS = 15.0
# Unofficial Google endpoint rate-limits aggressively; retry 429s locally
# before bubbling up (sleep is injectable so tests never wait).
GOOGLE_429_RETRIES = 2
GOOGLE_429_BACKOFF_SECONDS = 0.5
# Community DeepLX instances rate-limit concurrent caption segments; a
# process-wide gap plus local 429/418 backoff keeps the free endpoints usable.
DEEPLX_MIN_INTERVAL_SECONDS = 1.0
DEEPLX_429_RETRIES = 3
DEEPLX_429_BACKOFF_SECONDS = 1.0
# api.deeplx.org uses HTTP 418 "I'm a teapot" as a soft ban / rate-limit.
DEEPLX_RETRY_STATUS_CODES = frozenset({418, 429, 503})
MSG_DEEPLX_BUSY = "DeepLX 请求过于频繁，请稍后再试或换用自建实例"

# -- provider identifiers ----------------------------------------------------
PROVIDER_GOOGLE = "google"
PROVIDER_BAIDU = "baidu"
PROVIDER_DEEPL = "deepl"
# The LLM translator is not a web provider (it is served by the core
# Translator through the GUI controller) but shares the same selection UI, so
# its id + registration note live here for a single source of truth.
PROVIDER_LLM = "llm"
PROVIDER_CUSTOM = "custom"
PROVIDER_DEEPLX = "deeplx"
PROVIDER_LOCAL_MT = "local_mt"
DEEPLX_TRANSLATE_PATH = "/translate"
DEEPLX_AUTO_SOURCE = "auto"
CUSTOM_CHAT_ENDPOINT = "/chat/completions"
CUSTOM_SYSTEM = (
    "You translate image-caption tags. Output only the translation. "
    "Keep comma-separated tag structure. No quotes or commentary."
)
CUSTOM_USER_FMT = "Translate into {language}:\n\n{text}"
_CUSTOM_DIRECTION_LANG: Mapping[Direction, str] = {
    Direction.EN_TO_ZH: "Simplified Chinese",
    Direction.ZH_TO_EN: "English",
}
_CUSTOM_TARGET_LANG: Mapping[str, str] = {
    "zh": "Simplified Chinese",
    "en": "English",
    "ja": "Japanese",
}

# -- endpoints ---------------------------------------------------------------
GOOGLE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
BAIDU_ENDPOINT = "https://fanyi-api.baidu.com/api/trans/vip/translate"
DEEPL_ENDPOINT = "https://api-free.deepl.com/v2/translate"

# -- per-provider language codes keyed by translation direction --------------
_GOOGLE_TARGET: Mapping[Direction, str] = {
    Direction.EN_TO_ZH: "zh-CN",
    Direction.ZH_TO_EN: "en",
}
_BAIDU_LANGS: Mapping[Direction, tuple[str, str]] = {
    Direction.EN_TO_ZH: ("en", "zh"),
    Direction.ZH_TO_EN: ("zh", "en"),
}
_DEEPL_TARGET: Mapping[Direction, str] = {
    Direction.EN_TO_ZH: "ZH",
    Direction.ZH_TO_EN: "EN-US",
}

# -- per-provider language codes keyed by target language (中/英/日) ----------
# Source language is auto-detected by the provider ("auto" where applicable).
_GOOGLE_TARGET_LANG: Mapping[str, str] = {"zh": "zh-CN", "en": "en", "ja": "ja"}
_BAIDU_TARGET_LANG: Mapping[str, str] = {"zh": "zh", "en": "en", "ja": "jp"}
_BAIDU_AUTO_SOURCE = "auto"
_DEEPL_TARGET_LANG: Mapping[str, str] = {"zh": "ZH", "en": "EN-US", "ja": "JA"}
_DEEPLX_TARGET: Mapping[Direction, str] = {
    Direction.EN_TO_ZH: "ZH",
    Direction.ZH_TO_EN: "EN",
}
_DEEPLX_TARGET_LANG: Mapping[str, str] = {"zh": "ZH", "en": "EN", "ja": "JA"}


def _require_target_lang(target_lang: str) -> str:
    if target_lang not in TARGET_LANGS:
        raise LLMConfigError(
            f"target_lang must be one of {TARGET_LANGS}, got {target_lang!r}"
        )
    return target_lang


@runtime_checkable
class TranslationProvider(Protocol):
    """A single-segment text translator."""

    def translate(self, text: str, direction: Direction) -> str:
        """Return the translation of ``text`` for ``direction``.

        Raises :class:`nlapt.core.errors.LLMConfigError` when the provider is
        not usable (missing credentials) and
        :class:`nlapt.core.errors.LLMRequestError` on network/HTTP/parse
        failures.
        """
        ...


def _require_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise LLMRequestError("there is no text to translate")
    return stripped


class GoogleFreeProvider:
    """Unofficial keyless Google translate endpoint (best-effort, may throttle)."""

    def __init__(
        self,
        *,
        transport: Any = None,
        timeout: float = WEB_TRANSLATE_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._transport = transport
        self._timeout = timeout
        self._sleep = time.sleep if sleep is None else sleep

    def translate(self, text: str, direction: Direction) -> str:
        return self._request(text, _GOOGLE_TARGET[direction])

    def translate_to(self, text: str, target_lang: str) -> str:
        """Translate into a target language (``zh``/``en``/``ja``), auto source."""
        return self._request(text, _GOOGLE_TARGET_LANG[_require_target_lang(target_lang)])

    def _request(self, text: str, target_code: str) -> str:
        stripped = _require_text(text)
        httpx = require_httpx()
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": target_code,
            "dt": "t",
            "q": stripped,
        }
        attempts = 1 + GOOGLE_429_RETRIES
        response = None
        for attempt in range(attempts):
            try:
                with httpx.Client(
                    timeout=self._timeout, transport=self._transport
                ) as http:
                    response = http.get(GOOGLE_ENDPOINT, params=params)
            except httpx.TimeoutException as exc:
                raise LLMRequestError(f"Google 翻译请求超时: {exc}") from exc
            except httpx.HTTPError as exc:
                raise LLMRequestError(f"Google 翻译请求失败: {exc}") from exc
            if response.status_code == 429 and attempt + 1 < attempts:
                self._sleep(GOOGLE_429_BACKOFF_SECONDS * (attempt + 1))
                continue
            break
        if response is None:  # pragma: no cover - loop always assigns or raises
            raise LLMRequestError("Google 翻译请求失败")
        if response.status_code >= 400:
            raise LLMRequestError(
                f"Google 翻译返回 HTTP {response.status_code}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMRequestError(f"Google 翻译响应不是合法 JSON: {exc}") from exc
        return _parse_google(data)


def _parse_google(data: Any) -> str:
    """Concatenate the translated segments of a translate_a/single payload."""
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise LLMRequestError("Google 翻译响应结构异常")
    parts: list[str] = []
    for segment in data[0]:
        if isinstance(segment, list) and segment and isinstance(segment[0], str):
            parts.append(segment[0])
    result = "".join(parts).strip()
    if not result:
        raise LLMRequestError("Google 翻译未返回译文")
    return result


class BaiduProvider:
    """百度翻译开放平台 (needs APP ID + secret key; register at fanyi-api.baidu.com)."""

    def __init__(
        self,
        appid: str,
        key: str,
        *,
        transport: Any = None,
        timeout: float = WEB_TRANSLATE_TIMEOUT_SECONDS,
    ) -> None:
        appid = (appid or "").strip()
        key = (key or "").strip()
        if not appid or not key:
            raise LLMConfigError(
                "百度翻译未配置:请在设置中填写 APPID 与密钥"
                "(在 https://fanyi-api.baidu.com 注册开通)"
            )
        self._appid = appid
        self._key = key
        self._transport = transport
        self._timeout = timeout

    def _sign(self, query: str, salt: str) -> str:
        raw = f"{self._appid}{query}{salt}{self._key}".encode("utf-8")
        # md5 is mandated by Baidu's signing scheme; not a security primitive.
        return hashlib.md5(raw, usedforsecurity=False).hexdigest()

    def translate(self, text: str, direction: Direction) -> str:
        src, dst = _BAIDU_LANGS[direction]
        return self._request(text, src, dst)

    def translate_to(self, text: str, target_lang: str) -> str:
        """Translate into a target language (``zh``/``en``/``ja``), auto source."""
        dst = _BAIDU_TARGET_LANG[_require_target_lang(target_lang)]
        return self._request(text, _BAIDU_AUTO_SOURCE, dst)

    def _request(self, text: str, src: str, dst: str) -> str:
        stripped = _require_text(text)
        httpx = require_httpx()
        salt = secrets.token_hex(8)
        payload = {
            "q": stripped,
            "from": src,
            "to": dst,
            "appid": self._appid,
            "salt": salt,
            "sign": self._sign(stripped, salt),
        }
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as http:
                response = http.post(BAIDU_ENDPOINT, data=payload)
        except httpx.TimeoutException as exc:
            raise LLMRequestError(f"百度翻译请求超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMRequestError(f"百度翻译请求失败: {exc}") from exc
        if response.status_code >= 400:
            raise LLMRequestError(f"百度翻译返回 HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMRequestError(f"百度翻译响应不是合法 JSON: {exc}") from exc
        return _parse_baidu(data)


def _parse_baidu(data: Any) -> str:
    if not isinstance(data, Mapping):
        raise LLMRequestError("百度翻译响应结构异常")
    if data.get("error_code"):
        code = data.get("error_code")
        message = data.get("error_msg", "")
        raise LLMRequestError(f"百度翻译错误 {code}: {message}")
    results = data.get("trans_result")
    if not isinstance(results, list) or not results:
        raise LLMRequestError("百度翻译未返回译文")
    parts = [
        str(item.get("dst", ""))
        for item in results
        if isinstance(item, Mapping)
    ]
    result = "\n".join(part for part in parts if part).strip()
    if not result:
        raise LLMRequestError("百度翻译未返回译文")
    return result


class DeepLProvider:
    """DeepL API Free (needs an API key; register at www.deepl.com/pro-api)."""

    def __init__(
        self,
        key: str,
        *,
        endpoint: str = DEEPL_ENDPOINT,
        transport: Any = None,
        timeout: float = WEB_TRANSLATE_TIMEOUT_SECONDS,
    ) -> None:
        key = (key or "").strip()
        if not key:
            raise LLMConfigError(
                "DeepL 未配置:请在设置中填写 API Key"
                "(在 https://www.deepl.com/pro-api 注册获取免费额度)"
            )
        self._key = key
        self._endpoint = endpoint
        self._transport = transport
        self._timeout = timeout

    def translate(self, text: str, direction: Direction) -> str:
        return self._request(text, _DEEPL_TARGET[direction])

    def translate_to(self, text: str, target_lang: str) -> str:
        """Translate into a target language (``zh``/``en``/``ja``), auto source."""
        return self._request(text, _DEEPL_TARGET_LANG[_require_target_lang(target_lang)])

    def _request(self, text: str, target_code: str) -> str:
        stripped = _require_text(text)
        httpx = require_httpx()
        payload = {
            "text": stripped,
            "target_lang": target_code,
        }
        headers = {"Authorization": f"DeepL-Auth-Key {self._key}"}
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as http:
                response = http.post(self._endpoint, data=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMRequestError(f"DeepL 请求超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMRequestError(f"DeepL 请求失败: {exc}") from exc
        if response.status_code >= 400:
            raise LLMRequestError(f"DeepL 返回 HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMRequestError(f"DeepL 响应不是合法 JSON: {exc}") from exc
        return _parse_deepl(data)


def _parse_deepl(data: Any) -> str:
    if not isinstance(data, Mapping):
        raise LLMRequestError("DeepL 响应结构异常")
    translations = data.get("translations")
    if not isinstance(translations, list) or not translations:
        raise LLMRequestError("DeepL 未返回译文")
    parts = [
        str(item.get("text", ""))
        for item in translations
        if isinstance(item, Mapping)
    ]
    result = "\n".join(part for part in parts if part).strip()
    if not result:
        raise LLMRequestError("DeepL 未返回译文")
    return result


class DeepLXProvider:
    """DeepLX free ``POST /translate`` (self-hosted or community instance)."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        *,
        transport: Any = None,
        timeout: float = WEB_TRANSLATE_TIMEOUT_SECONDS,
        limiter: MinIntervalLimiter | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        base_url = (base_url or "").strip()
        if not base_url:
            raise LLMConfigError(
                "DeepLX 未配置:请填写实例地址"
                "(自建或 https://deeplx.owo.network/endpoints/free.html )"
            )
        self._base_url = base_url
        self._token = (token or "").strip()
        self._transport = transport
        self._timeout = timeout
        self._limiter = limiter if limiter is not None else _DEEPLX_LIMITER
        self._sleep = time.sleep if sleep is None else sleep

    def translate(self, text: str, direction: Direction) -> str:
        return self._request(text, _DEEPLX_TARGET[direction])

    def translate_to(self, text: str, target_lang: str) -> str:
        """Translate into a target language (``zh``/``en``/``ja``), auto source."""
        return self._request(text, _DEEPLX_TARGET_LANG[_require_target_lang(target_lang)])

    def _request(self, text: str, target_code: str) -> str:
        stripped = _require_text(text)
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        payload = {
            "text": stripped,
            "source_lang": DEEPLX_AUTO_SOURCE,
            "target_lang": target_code,
        }
        url = _deeplx_translate_url(self._base_url)
        display = display_deeplx_url(url)
        attempts = 1 + DEEPLX_429_RETRIES
        last_error: LLMRequestError | None = None
        for attempt in range(attempts):
            self._limiter.wait()
            try:
                data = http_post_json(
                    url=url,
                    payload=payload,
                    headers=headers,
                    timeout=self._timeout,
                    provider="deeplx",
                    transport=self._transport,
                    display_url=display,
                )
                return _parse_deeplx(data)
            except LLMRequestError as exc:
                last_error = exc
                if _deeplx_retryable(exc) and attempt + 1 < attempts:
                    self._sleep(DEEPLX_429_BACKOFF_SECONDS * (2**attempt))
                    continue
                if _deeplx_retryable(exc):
                    raise _deeplx_busy_error(exc) from exc
                raise
        if last_error is not None and _deeplx_retryable(last_error):
            raise _deeplx_busy_error(last_error) from last_error
        raise last_error or LLMRequestError("DeepLX 请求失败")


def _deeplx_translate_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    path = stripped.split("?", 1)[0]
    if path.endswith(DEEPLX_TRANSLATE_PATH):
        return stripped
    return join_url(stripped, DEEPLX_TRANSLATE_PATH)


def display_deeplx_url(url: str) -> str:
    """Public host + ``/translate``, with secret path/query segments removed."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    if path.endswith(DEEPLX_TRANSLATE_PATH):
        prefix = path[: -len(DEEPLX_TRANSLATE_PATH)].rstrip("/")
        path = "/…/translate" if prefix else DEEPLX_TRANSLATE_PATH
    elif path not in ("", "/"):
        path = "/…"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


_DEEPLX_LIMITER = MinIntervalLimiter(DEEPLX_MIN_INTERVAL_SECONDS)


def reset_deeplx_limiter(
    *,
    min_interval: float = DEEPLX_MIN_INTERVAL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> MinIntervalLimiter:
    """Replace the process-wide DeepLX limiter (tests; never a real wait)."""
    global _DEEPLX_LIMITER
    _DEEPLX_LIMITER = MinIntervalLimiter(min_interval, clock=clock, sleep=sleep)
    return _DEEPLX_LIMITER


def _deeplx_retryable(exc: LLMRequestError) -> bool:
    """True for community rate-limit / soft-ban status codes."""
    return getattr(exc, "status_code", None) in DEEPLX_RETRY_STATUS_CODES


def _deeplx_busy_error(exc: LLMRequestError) -> LLMRequestError:
    """User-facing busy message; keeps the HTTP status for callers/tests."""
    wrapped = LLMRequestError(MSG_DEEPLX_BUSY)
    code = getattr(exc, "status_code", None)
    if code is not None:
        wrapped.status_code = code  # type: ignore[attr-defined]
    return wrapped


def _parse_deeplx(data: Any) -> str:
    if not isinstance(data, Mapping):
        raise LLMRequestError("DeepLX 响应结构异常")
    code = data.get("code")
    if code is not None and code != 200:
        message = str(data.get("message") or data.get("msg") or "").strip()
        suffix = f": {message}" if message else ""
        error = LLMRequestError(f"DeepLX 错误 {code}{suffix}")
        try:
            error.status_code = int(code)
        except (TypeError, ValueError):
            pass
        raise error
    result = data.get("data")
    if isinstance(result, str) and result.strip():
        return result.strip()
    raise LLMRequestError("DeepLX 未返回译文")


class CustomOpenAIProvider:
    """User-supplied OpenAI-compatible chat endpoint for translation."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        transport: Any = None,
        timeout: float = WEB_TRANSLATE_TIMEOUT_SECONDS,
    ) -> None:
        base_url = (base_url or "").strip()
        model = (model or "").strip()
        if not base_url or not model:
            raise LLMConfigError(
                "自定义翻译未配置:请填写 Base URL 与模型"
            )
        self._base_url = base_url
        self._api_key = (api_key or "").strip()
        self._model = model
        self._transport = transport
        self._timeout = timeout

    def translate(self, text: str, direction: Direction) -> str:
        return self._complete(text, _CUSTOM_DIRECTION_LANG[direction])

    def translate_to(self, text: str, target_lang: str) -> str:
        """Translate into a target language (``zh``/``en``/``ja``), auto source."""
        language = _CUSTOM_TARGET_LANG[_require_target_lang(target_lang)]
        return self._complete(text, language)

    def _complete(self, text: str, language: str) -> str:
        stripped = _require_text(text)
        url = _custom_completions_url(self._base_url)
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": CUSTOM_SYSTEM},
                {
                    "role": "user",
                    "content": CUSTOM_USER_FMT.format(
                        language=language, text=stripped
                    ),
                },
            ],
        }
        data = http_post_json(
            url=url,
            payload=payload,
            headers=headers,
            timeout=self._timeout,
            provider="custom-translate",
            transport=self._transport,
        )
        return _parse_openai_translation(data)


def _custom_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith(CUSTOM_CHAT_ENDPOINT):
        return stripped
    return join_url(stripped, CUSTOM_CHAT_ENDPOINT)


def _parse_openai_translation(data: Any) -> str:
    if not isinstance(data, Mapping):
        raise LLMRequestError("自定义翻译响应结构异常")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMRequestError("自定义翻译未返回译文")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise LLMRequestError("自定义翻译响应结构异常")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise LLMRequestError("自定义翻译响应结构异常")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMRequestError("自定义翻译未返回译文")
    return content.strip()


# -- registry ----------------------------------------------------------------
# A factory takes the credential mapping (keys documented in
# nlapt_gui.translate_config) plus an optional httpx transport (tests) and
# returns a ready provider. A missing-credential provider raises LLMConfigError.
ProviderFactory = Callable[..., TranslationProvider]


def _google_factory(_creds: Mapping[str, str], *, transport: Any = None) -> GoogleFreeProvider:
    return GoogleFreeProvider(transport=transport)


def _baidu_factory(creds: Mapping[str, str], *, transport: Any = None) -> BaiduProvider:
    return BaiduProvider(
        creds.get("baidu_appid", ""),
        creds.get("baidu_key", ""),
        transport=transport,
    )


def _deepl_factory(creds: Mapping[str, str], *, transport: Any = None) -> DeepLProvider:
    return DeepLProvider(creds.get("deepl_key", ""), transport=transport)


def _deeplx_factory(creds: Mapping[str, str], *, transport: Any = None) -> DeepLXProvider:
    return DeepLXProvider(
        creds.get("deeplx_url", ""),
        creds.get("deeplx_token", ""),
        transport=transport,
    )


def _custom_factory(creds: Mapping[str, str], *, transport: Any = None) -> CustomOpenAIProvider:
    return CustomOpenAIProvider(
        creds.get("custom_base_url", ""),
        creds.get("custom_api_key", ""),
        creds.get("custom_model", ""),
        transport=transport,
    )


PROVIDER_REGISTRY: dict[str, ProviderFactory] = {
    PROVIDER_GOOGLE: _google_factory,
    PROVIDER_BAIDU: _baidu_factory,
    PROVIDER_DEEPL: _deepl_factory,
    PROVIDER_DEEPLX: _deeplx_factory,
    PROVIDER_CUSTOM: _custom_factory,
}


def create_provider(
    provider_id: str,
    creds: Mapping[str, str],
    *,
    transport: Any = None,
) -> TranslationProvider:
    """Build a web provider from its id + credentials.

    Raises :class:`nlapt.core.errors.LLMConfigError` for an unknown id or when
    required credentials are missing.
    """
    factory = PROVIDER_REGISTRY.get(provider_id)
    if factory is None:
        raise LLMConfigError(f"未知的翻译服务: {provider_id!r}")
    return factory(creds, transport=transport)


# -- registration guidance (shown in the settings dialog, in Chinese) --------
REGISTRATION_INFO: dict[str, dict[str, Any]] = {
    PROVIDER_LLM: {
        "needs_key": False,
        "url": "",
        "note_zh": "使用已配置的大模型档案(在下方 LLM 设置中填写)。",
    },
    PROVIDER_GOOGLE: {
        "needs_key": False,
        "url": "https://translate.google.com",
        "note_zh": "免费,无需注册。为非官方接口,可能限流或偶尔不可用。",
    },
    PROVIDER_BAIDU: {
        "needs_key": True,
        "url": "https://fanyi-api.baidu.com",
        "note_zh": (
            "需注册:在 https://fanyi-api.baidu.com 注册开通通用翻译 API,"
            "填入 APPID 与密钥。"
        ),
    },
    PROVIDER_DEEPL: {
        "needs_key": True,
        "url": "https://www.deepl.com/pro-api",
        "note_zh": (
            "需注册:在 https://www.deepl.com/pro-api 注册获取免费 API Key"
            "(每月有免费额度)。"
        ),
    },
    PROVIDER_DEEPLX: {
        "needs_key": True,
        "url": "https://deeplx.owo.network/endpoints/free.html",
        "note_zh": (
            "DeepLX 免费接口,无需官方 DeepL Key。"
            "填写自建地址或社区实例(列表: "
            "https://deeplx.owo.network/endpoints/free.html )。"
            "自建启用了 -token 时再填访问令牌。"
            "地址与令牌保存在「文档/NLapt/api.json」。"
            "社区实例常限流(HTTP 429/418)，自建更稳。"
        ),
    },
    PROVIDER_LOCAL_MT: {
        "needs_key": True,
        "url": "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF",
        "note_zh": (
            "本地 Hy-MT2 翻译模型,走本机 llama-server,无需联网。"
            "选择档位后点「管理模型」下载。快速=1.8B 2bit,"
            "均衡=1.8B Q4,高质量=7B Q4。"
        ),
    },
    PROVIDER_CUSTOM: {
        "needs_key": True,
        "url": "",
        "note_zh": (
            "OpenAI 兼容接口。填写 Base URL、模型与可选 API Key。"
            "密钥保存在「文档/NLapt/api.json」，不写入应用配置目录。"
        ),
    },
}
