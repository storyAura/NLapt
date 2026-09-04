"""Ollama local chat client (``/api/chat``, spec 8).

``httpx`` is imported lazily inside :func:`nlapt.llm.base.http_post_json`;
this module can be imported without the optional ``llm`` extra installed.
"""

from __future__ import annotations

from typing import Any

from nlapt.core.config import LLMProfile
from nlapt.core.errors import LLMRequestError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    encode_image_base64,
    http_post_json,
    join_url,
    register_client,
    validate_request,
)

_LOGGER = get_logger(__name__)

PROVIDER_NAME = "ollama"
CHAT_ENDPOINT = "/api/chat"
AUTHORIZATION_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "
SYSTEM_ROLE = "system"


def _message_payload(request: LLMRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": SYSTEM_ROLE, "content": request.system})
    for message in request.messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.text}
        if message.images:
            entry["images"] = [encode_image_base64(image) for image in message.images]
        messages.append(entry)
    return messages


def _extract_text(data: dict[str, Any]) -> str:
    message = data.get("message")
    if not isinstance(message, dict):
        raise LLMRequestError(f"{PROVIDER_NAME} response missing 'message' object")
    content = message.get("content")
    if not isinstance(content, str):
        raise LLMRequestError(f"{PROVIDER_NAME} response missing message content")
    if not content.strip():
        # Empty completions are transient (retryable), never a usable caption.
        raise LLMRequestError(f"{PROVIDER_NAME} returned empty message content")
    return content


class OllamaClient(LLMClient):
    """Thin httpx-based client for a local Ollama server."""

    def __init__(self, profile: LLMProfile, *, transport: Any = None) -> None:
        self._profile = profile
        self._transport = transport

    def complete(self, request: LLMRequest) -> LLMResponse:
        validate_request(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _message_payload(request),
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        headers: dict[str, str] = {}
        if self._profile.api_key:
            headers[AUTHORIZATION_HEADER] = BEARER_PREFIX + self._profile.api_key
        data = http_post_json(
            url=join_url(self._profile.base_url, CHAT_ENDPOINT),
            payload=payload,
            headers=headers,
            timeout=request.timeout,
            provider=PROVIDER_NAME,
            transport=self._transport,
        )
        text = _extract_text(data)
        _LOGGER.debug("ollama completion ok (model=%s, chars=%d)", request.model, len(text))
        return LLMResponse(text=text, model=request.model)


register_client(PROVIDER_NAME, OllamaClient)
