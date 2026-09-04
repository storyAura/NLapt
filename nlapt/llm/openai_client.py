"""OpenAI-compatible chat client (``/chat/completions``, spec 8).

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

PROVIDER_NAME = "openai"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"
AUTHORIZATION_HEADER = "Authorization"
BEARER_PREFIX = "Bearer "
SYSTEM_ROLE = "system"
# Providers flag safety interception here; the content (if any) is a
# sanitized replacement, not the caption that was asked for.
FINISH_REASON_CONTENT_FILTER = "content_filter"
MSG_CONTENT_FILTERED = (
    "请求被服务商内容安全拦截(finish_reason=content_filter),"
    "可重试或调整提示词/图片"
)
MSG_EMPTY_COMPLETION = (
    "服务商返回了空回复(finish_reason={reason})— "
    "常见于高并发或思考型模型输出预算耗尽,可稍后重试或降低并发"
)


def _image_part(image: bytes) -> dict[str, Any]:
    media_type = detect_image_media_type(image)
    data_url = f"data:{media_type};base64,{encode_image_base64(image)}"
    return {"type": "image_url", "image_url": {"url": data_url}}


def _message_payload(request: LLMRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": SYSTEM_ROLE, "content": request.system})
    for message in request.messages:
        if message.images:
            content: Any = [{"type": "text", "text": message.text}]
            content.extend(_image_part(image) for image in message.images)
        else:
            content = message.text
        messages.append({"role": message.role, "content": content})
    return messages


def _extract_text(data: dict[str, Any]) -> str:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMRequestError(
            f"{PROVIDER_NAME} response missing choices[0].message.content"
        ) from exc
    # Check the interception flag BEFORE the content: a filtered choice often
    # carries a null/replaced message that would otherwise raise a generic
    # parse error and hide the real cause.
    if (
        isinstance(choice, dict)
        and choice.get("finish_reason") == FINISH_REASON_CONTENT_FILTER
    ):
        raise LLMRequestError(MSG_CONTENT_FILTERED)
    try:
        content = choice["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise LLMRequestError(
            f"{PROVIDER_NAME} response missing choices[0].message.content"
        ) from exc
    if not isinstance(content, str):
        raise LLMRequestError(
            f"{PROVIDER_NAME} response content has unexpected type "
            f"{type(content).__name__}"
        )
    if not content.strip():
        # An HTTP-200 body with empty content (proxies under load, thinking
        # models exhausting max_tokens on reasoning). LLMRequestError so the
        # spec-8 retry path treats it as transient instead of writing "".
        reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        raise LLMRequestError(MSG_EMPTY_COMPLETION.format(reason=reason))
    return content


class OpenAIClient(LLMClient):
    """Thin httpx-based client for OpenAI-compatible chat APIs."""

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
        headers: dict[str, str] = {}
        if self._profile.api_key:
            headers[AUTHORIZATION_HEADER] = BEARER_PREFIX + self._profile.api_key
        data = http_post_json(
            url=join_url(self._profile.base_url, CHAT_COMPLETIONS_ENDPOINT),
            payload=payload,
            headers=headers,
            timeout=request.timeout,
            provider=PROVIDER_NAME,
            transport=self._transport,
        )
        text = _extract_text(data)
        _LOGGER.debug("openai completion ok (model=%s, chars=%d)", request.model, len(text))
        return LLMResponse(text=text, model=request.model)


register_client(PROVIDER_NAME, OpenAIClient)
