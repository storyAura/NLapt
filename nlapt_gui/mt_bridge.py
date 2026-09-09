"""Local Hy-MT2 translation: llama-server singleton + TranslationProvider.

Caption inference already owns one llama-server (vision GGUF). The MT
backend uses a second manager on ``settings.port + 1`` plus its own idle
stopper so 打标 and 翻译 never unload each other.

Widgets talk to this module and :mod:`nlapt.local.mt_catalog` only — they
do not spawn processes or hit HuggingFace themselves.
"""

from __future__ import annotations

import atexit
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThreadPool

from nlapt.core.errors import LLMConfigError, LLMRequestError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import http_post_json, join_url
from nlapt.llm.cleaning import clean_llm_output
from nlapt.llm.retry import RetryPolicy, with_retry
from nlapt.llm.translate import Direction
from nlapt.local.mt_catalog import (
    DEFAULT_TIER,
    HYMT_DIRECTION_LANG,
    MT_PARAGRAPH_SEPARATOR,
    MTModel,
    find_mt_model,
    hymt_prompt,
    is_mt_downloaded,
    mt_download_url,
    mt_model_path,
    split_mt_paragraphs,
)
from nlapt.local.runtime import ensure_runtime
from nlapt.local.server import GPU_LAYERS_ALL, LocalServerManager, ServerSpec
from nlapt.local.settings import (
    LocalSettings,
    PORT_RANGE,
    load_local_settings,
)

from nlapt_gui.local_bridge import (
    DOWNLOAD_OK,
    IdleServerStopper,
    cancel_active_download,
    get_download_hub,
    launch_download_jobs,
    local_settings_path,
    models_dir_for,
    pending_runtime_asset,
    resolve_server_path,
    runtime_base_dir,
)

_LOGGER = get_logger(__name__)

MT_DOWNLOAD_PREFIX = "mt:"
LOCAL_MT_TIMEOUT_SECONDS = 180.0
# Reserve room for the source alongside the official generation budget.
MT_CONTEXT_LENGTH = 8192
HYMT_TEMPERATURE = 0.7
HYMT_TOP_P = 0.6
HYMT_TOP_K = 20
HYMT_REPEAT_PENALTY = 1.05
HYMT_MAX_TOKENS = 4096
HYMT_RETRY_POLICY = RetryPolicy(max_retries=2, base_delay=1.0, multiplier=2.0)
COMPLETIONS_ENDPOINT = "/chat/completions"
MSG_MT_NOT_DOWNLOADED = "本地翻译模型尚未下载 — 打开 设置 ▸ 翻译服务 ▸ 管理模型"
MSG_MT_INVALID_RESPONSE = "本地翻译响应缺少有效的 choices[0].message.content: {response!r}"
MSG_MT_INCOMPLETE = (
    "本地翻译未正常完成 (finish_reason={reason!r})，"
    "请缩短原文后重试；未采用不完整译文。响应: {response!r}"
)

_MT_MANAGER: LocalServerManager | None = None
_MT_STOPPER: IdleServerStopper | None = None


def get_mt_server_manager() -> LocalServerManager:
    """Process-wide MT llama-server manager, stopped at interpreter exit."""
    global _MT_MANAGER
    if _MT_MANAGER is None:
        _MT_MANAGER = LocalServerManager()
        atexit.register(_MT_MANAGER.stop)
    return _MT_MANAGER


def get_mt_idle_stopper() -> IdleServerStopper:
    """Idle stopper bound to the MT server (never the caption server)."""
    global _MT_STOPPER
    if _MT_STOPPER is None:
        _MT_STOPPER = IdleServerStopper(get_mt_server_manager())
    return _MT_STOPPER


def reset_mt_singletons_for_tests(
    *,
    manager: LocalServerManager | None = None,
    stopper: IdleServerStopper | None = None,
) -> None:
    """Replace process-wide MT server objects (tests only)."""
    global _MT_MANAGER, _MT_STOPPER
    _MT_MANAGER = manager
    _MT_STOPPER = stopper


def mt_models_root(models_dir: Path | None = None) -> Path:
    """Primary models dir used for Hy-MT2 files (caption catalog sibling)."""
    if models_dir is not None:
        return Path(models_dir)
    settings = load_local_settings(local_settings_path())
    return models_dir_for(settings)


def is_tier_downloaded(tier: str, *, models_dir: Path | None = None) -> bool:
    """Whether the selected quality tier's GGUF is present at its pinned size."""
    try:
        model = find_mt_model(tier)
    except ValidationError:
        return False
    return is_mt_downloaded(mt_models_root(models_dir), model)


def mt_server_port(settings: LocalSettings) -> int:
    """Caption server port + 1, clamped into the legal range."""
    _low, high = PORT_RANGE
    candidate = settings.port + 1
    if candidate <= high:
        return candidate
    return max(_low, settings.port - 1)


def start_mt_download(
    tier: str,
    pool: QThreadPool | None = None,
    *,
    models_dir: Path | None = None,
) -> bool:
    """Download the tier's GGUF (and llama.cpp runtime if needed)."""
    model = find_mt_model(tier)
    root = mt_models_root(models_dir)
    dest = mt_model_path(root, model)
    settings = load_local_settings(local_settings_path())
    jobs: list[tuple[str, Path, int, str]] = []
    if not is_mt_downloaded(root, model):
        jobs.append((mt_download_url(model), dest, model.size_bytes, model.sha256))
    runtime_asset = pending_runtime_asset(settings)
    family_key = MT_DOWNLOAD_PREFIX + model.tier
    if not jobs and runtime_asset is None:
        get_download_hub().finished.emit(family_key, "", DOWNLOAD_OK, "")
        return True
    return launch_download_jobs(family_key, "", jobs, runtime_asset, pool)


def cancel_mt_download() -> None:
    """Cancel the in-flight download (shared slot with caption models)."""
    cancel_active_download()


def delete_mt_model(tier: str, *, models_dir: Path | None = None) -> None:
    """Stop the MT server and delete the tier's GGUF if present."""
    get_mt_server_manager().stop()
    path = mt_model_path(mt_models_root(models_dir), find_mt_model(tier))
    if path.is_file():
        path.unlink()


def ensure_mt_server(
    tier: str,
    *,
    models_dir: Path | None = None,
    settings: LocalSettings | None = None,
) -> str:
    """Return the OpenAI-compatible base URL of a server serving ``tier``."""
    model = find_mt_model(tier)
    resolved = (
        settings if settings is not None else load_local_settings(local_settings_path())
    )
    root = mt_models_root(models_dir)
    path = mt_model_path(root, model)
    if not is_mt_downloaded(root, model):
        raise LLMConfigError(MSG_MT_NOT_DOWNLOADED)
    server_path = resolve_server_path(resolved)
    if not server_path:
        server_path = str(ensure_runtime(runtime_base_dir()))
    gpu_layers = resolved.gpu_layers
    if gpu_layers < 0:
        gpu_layers = GPU_LAYERS_ALL
    spec = ServerSpec(
        server_path=server_path,
        model_path=str(path),
        port=mt_server_port(resolved),
        context_length=MT_CONTEXT_LENGTH,
        gpu_layers=gpu_layers,
        threads=resolved.threads,
        parallel=1,
    )
    return get_mt_server_manager().ensure(spec)


class LocalMTProvider:
    """Hy-MT2 translator: official prompt over the local OpenAI-compatible API."""

    def __init__(
        self,
        tier: str = DEFAULT_TIER,
        *,
        retry_sleep: Callable[[float], None],
        models_dir: Path | None = None,
        transport: Any = None,
        timeout: float = LOCAL_MT_TIMEOUT_SECONDS,
        ensure: Callable[[], str] | None = None,
        idle_stopper: Any = None,
    ) -> None:
        self._model: MTModel = find_mt_model(tier)
        self._models_dir = models_dir
        self._transport = transport
        self._timeout = timeout
        self._ensure = ensure
        self._idle_stopper = idle_stopper
        self._retry_sleep = retry_sleep

    def translate(self, text: str, direction: Direction) -> str:
        return self.translate_to(text, HYMT_DIRECTION_LANG[direction])

    def translate_to(self, text: str, target_lang: str) -> str:
        """Translate ``text`` paragraph by paragraph and rejoin with blank lines.

        Hy-MT2 only translates the block after the last blank line of the
        prompt (earlier blocks read as untranslated context), so a caption
        with its own blank lines is split and sent one paragraph per request.
        """
        stripped = text.strip()
        if not stripped:
            raise LLMRequestError("there is no text to translate")
        paragraphs = split_mt_paragraphs(stripped)
        stopper = (
            self._idle_stopper
            if self._idle_stopper is not None
            else get_mt_idle_stopper()
        )
        stopper.note_request()
        try:
            base = self._ensure() if self._ensure is not None else ensure_mt_server(
                self._model.tier, models_dir=self._models_dir
            )
            translated = [
                self._translate_paragraph(base, paragraph, target_lang)
                for paragraph in paragraphs
            ]
            return MT_PARAGRAPH_SEPARATOR.join(translated)
        finally:
            stopper.note_finished()

    def _translate_paragraph(self, base: str, paragraph: str, target_lang: str) -> str:
        """One completed, cleaned translation of a single paragraph."""
        prompt = hymt_prompt(paragraph, target_lang)
        data = with_retry(
            lambda: self._request(base, prompt),
            HYMT_RETRY_POLICY,
            sleep=self._retry_sleep,
            retry_on=(LLMRequestError,),
        )
        return clean_llm_output(_parse_mt_text(data))

    def _request(self, base: str, prompt: str) -> dict[str, object]:
        """Send one chat request; log transport failures before retrying."""
        try:
            return http_post_json(
                url=join_url(base, COMPLETIONS_ENDPOINT),
                payload={
                    "model": self._model.filename,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "max_tokens": HYMT_MAX_TOKENS,
                    "temperature": HYMT_TEMPERATURE,
                    "top_p": HYMT_TOP_P,
                    "top_k": HYMT_TOP_K,
                    "repeat_penalty": HYMT_REPEAT_PENALTY,
                    "min_p": 0.0,
                },
                headers={},
                timeout=self._timeout,
                provider="local-mt",
                transport=self._transport,
            )
        except LLMRequestError as exc:
            _LOGGER.warning(
                "Local MT chat request failed",
                extra={"model": self._model.filename, "endpoint": base, "error": str(exc)},
            )
            raise


def _parse_mt_text(data: Mapping[str, object]) -> str:
    """Accept only a completed chat reply, never partial or reasoning text."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMRequestError(MSG_MT_INVALID_RESPONSE.format(response=data))
    first = choices[0]
    if not isinstance(first, Mapping):
        raise LLMRequestError(MSG_MT_INVALID_RESPONSE.format(response=data))
    reason = first.get("finish_reason")
    if reason != "stop":
        raise LLMRequestError(MSG_MT_INCOMPLETE.format(reason=reason, response=data))
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    raise LLMRequestError(MSG_MT_INVALID_RESPONSE.format(response=data))
