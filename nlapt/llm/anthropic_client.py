"""Anthropic Messages API client (``/v1/messages``, spec 8).

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
    detect_image_media_type,
    encode_image_base64,
    http_post_json,
    join_url,
    register_client,
    validate_request,
)

_LOGGER = get_logger(__name__)

PROVIDER_NAME = "anthropic"
MESSAGES_ENDPOINT = "/v1/messages"
API_KEY_HEADER = "x-api-key"
VERSION_HEADER = "anthropic-version"
ANTHROPIC_VERSION = "2023-06-01"


def _image_block(image: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": detect_image_media_type(image),
            "data": encode_image_base64(image),
        },
    }


def _message_payload(request: LLMRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.images:
            content: Any = [_image_block(image) for image in message.images]
            content.append({"type": "text", "text": message.text})
        else:
            content = message.text
        messages.append({"role": message.role, "content": content})
    return messages


def _extract_text(data: dict[str, Any]) -> str:
    blocks = data.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise LLMRequestError(f"{PROVIDER_NAME} response missing content blocks")
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        raise LLMRequestError(f"{PROVIDER_NAME} response contained no text blocks")
    return "".join(parts)


class AnthropicClient(LLMClient):
    """Thin httpx-based client for the Anthropic Messages API."""

    def __init__(self, profile: LLMProfile, *, transport: Any = None) -> None:
        self._profile = profile
        self._transport = transport

    def complete(self, request: LLMRequest) -> LLMResponse:
        validate_request(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _message_payload(request),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.system:
            payload["system"] = request.system
        headers: dict[str, str] = {VERSION_HEADER: ANTHROPIC_VERSION}
        if self._profile.api_key:
            headers[API_KEY_HEADER] = self._profile.api_key
        data = http_post_json(
            url=join_url(self._profile.base_url, MESSAGES_ENDPOINT),
            payload=payload,
            headers=headers,
            timeout=request.timeout,
            provider=PROVIDER_NAME,
            transport=self._transport,
        )
        text = _extract_text(data)
        _LOGGER.debug(
            "anthropic completion ok (model=%s, chars=%d)", request.model, len(text)
        )
        return LLMResponse(text=text, model=request.model)


register_client(PROVIDER_NAME, AnthropicClient)
