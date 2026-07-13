"""List the models an LLM endpoint offers (设置 ▸ 获取模型).

One blocking HTTP GET per call; callers run it on a worker thread (the GUI
uses ``nlapt_gui.workers.run_async``). Endpoints per api_type:

* ``openai``   — ``GET {base_url}/models`` (Bearer auth), ``data[].id``
* ``anthropic``— ``GET {base_url}/v1/models`` (x-api-key), ``data[].id``
* ``ollama``   — ``GET {base_url}/api/tags``, ``models[].name``

``httpx`` is imported lazily via :func:`nlapt.llm.base.require_httpx` so the
core package keeps no hard HTTP dependency; ``transport`` lets tests drive the
call through ``httpx.MockTransport``.
"""

from __future__ import annotations

from typing import Any, Mapping

from nlapt.core.config import LLMProfile
from nlapt.core.errors import LLMConfigError, LLMRequestError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import join_url, require_httpx

_LOGGER = get_logger(__name__)

DEFAULT_LIST_TIMEOUT = 20.0

API_OPENAI = "openai"
API_ANTHROPIC = "anthropic"
API_OLLAMA = "ollama"

_OPENAI_ENDPOINT = "/models"
_ANTHROPIC_ENDPOINT = "/v1/models"
_OLLAMA_ENDPOINT = "/api/tags"
_ANTHROPIC_VERSION = "2023-06-01"

# Substrings that mark a model id as vision-capable (best-effort heuristic —
# listing APIs do not expose modality, so this only guides the UI hint).
VISION_NAME_HINTS: tuple[str, ...] = (
    "vision",
    "-vl",
    "vl-",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o3",
    "o4",
    "claude",
    "gemini",
    "llava",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "glm-4v",
    "glm-4.5v",
    "internvl",
    "minicpm-v",
    "pixtral",
    "moondream",
    "grok-vision",
    "doubao-vision",
)


def looks_vision_capable(model_id: str) -> bool:
    """Best-effort guess whether a model id names a multimodal model."""
    lowered = model_id.lower()
    return any(hint in lowered for hint in VISION_NAME_HINTS)


def _http_get_json(
    *,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    provider: str,
    transport: Any = None,
) -> Any:
    httpx = require_httpx()
    try:
        with httpx.Client(timeout=timeout, transport=transport) as http:
            response = http.get(url, headers=dict(headers))
    except httpx.TimeoutException as exc:
        raise LLMRequestError(f"{provider} 模型列表请求超时: {exc}") from exc
    except httpx.HTTPError as exc:
        raise LLMRequestError(f"{provider} 模型列表请求失败: {exc}") from exc
    if response.status_code >= 400:
        raise LLMRequestError(
            f"{provider} 模型列表返回 HTTP {response.status_code}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise LLMRequestError(f"{provider} 模型列表响应不是合法 JSON: {exc}") from exc


def _parse_id_list(data: Any, list_key: str, id_key: str, provider: str) -> tuple[str, ...]:
    if not isinstance(data, Mapping):
        raise LLMRequestError(f"{provider} 模型列表响应结构异常")
    entries = data.get(list_key)
    if not isinstance(entries, list):
        raise LLMRequestError(f"{provider} 模型列表响应缺少 {list_key!r}")
    names: list[str] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            value = entry.get(id_key)
            if isinstance(value, str) and value:
                names.append(value)
    return tuple(dict.fromkeys(sorted(names)))


def list_models(
    profile: LLMProfile,
    *,
    timeout: float = DEFAULT_LIST_TIMEOUT,
    transport: Any = None,
) -> tuple[str, ...]:
    """The model ids the profile's endpoint offers, sorted and de-duplicated.

    Raises :class:`LLMConfigError` for a missing base_url / unknown api_type
    and :class:`LLMRequestError` on network/HTTP/parse failures.
    """
    if not isinstance(profile, LLMProfile):
        raise ValidationError(
            f"profile must be an LLMProfile, got {type(profile).__name__}"
        )
    if not profile.base_url:
        raise LLMConfigError("请先填写 Base URL")
    if profile.api_type == API_OPENAI:
        headers = (
            {"Authorization": f"Bearer {profile.api_key}"} if profile.api_key else {}
        )
        data = _http_get_json(
            url=join_url(profile.base_url, _OPENAI_ENDPOINT),
            headers=headers,
            timeout=timeout,
            provider="OpenAI",
            transport=transport,
        )
        models = _parse_id_list(data, "data", "id", "OpenAI")
    elif profile.api_type == API_ANTHROPIC:
        headers = {"anthropic-version": _ANTHROPIC_VERSION}
        if profile.api_key:
            headers["x-api-key"] = profile.api_key
        data = _http_get_json(
            url=join_url(profile.base_url, _ANTHROPIC_ENDPOINT),
            headers=headers,
            timeout=timeout,
            provider="Anthropic",
            transport=transport,
        )
        models = _parse_id_list(data, "data", "id", "Anthropic")
    elif profile.api_type == API_OLLAMA:
        data = _http_get_json(
            url=join_url(profile.base_url, _OLLAMA_ENDPOINT),
            headers={},
            timeout=timeout,
            provider="Ollama",
            transport=transport,
        )
        models = _parse_id_list(data, "models", "name", "Ollama")
    else:
        raise LLMConfigError(f"接口类型 {profile.api_type!r} 不支持获取模型列表")
    _LOGGER.info("listed %d models from %s endpoint", len(models), profile.api_type)
    return models
