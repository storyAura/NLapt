"""LLM client abstractions: request/response models, client ABC, registry (spec 8).

Concrete HTTP clients live in sibling modules and import ``httpx`` lazily so
``import nlapt.llm`` succeeds even when the optional ``llm`` extra is not
installed. Shared HTTP plumbing (:func:`http_post_json`) is kept here so the
per-provider modules stay thin.
"""

from __future__ import annotations

import base64
import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from nlapt.core.config import LLMProfile
from nlapt.core.errors import (
    LLMConfigError,
    LLMRequestError,
    LLMTimeoutError,
    ValidationError,
)
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

ALLOWED_ROLES: frozenset[str] = frozenset({"user", "assistant"})

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 60.0

CONNECTION_TEST_PROMPT = "Reply with the single word: ok"
CONNECTION_TEST_MAX_TOKENS = 8

HTTPX_MISSING_MESSAGE = (
    "The 'httpx' package is required for LLM HTTP clients. "
    "Install it with: pip install nlapt[llm]"
)
ERROR_BODY_PREVIEW_CHARS = 300

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
WEBP_RIFF_MAGIC = b"RIFF"
WEBP_FORMAT_MAGIC = b"WEBP"
MEDIA_TYPE_PNG = "image/png"
MEDIA_TYPE_JPEG = "image/jpeg"
MEDIA_TYPE_WEBP = "image/webp"

# api_type of the built-in clients -> module that registers them on import.
_BUILTIN_CLIENT_MODULES: dict[str, str] = {
    "openai": "nlapt.llm.openai_client",
    "anthropic": "nlapt.llm.anthropic_client",
    "ollama": "nlapt.llm.ollama_client",
}


@dataclass(frozen=True)
class LLMMessage:
    """One chat message. Images are raw JPEG/PNG bytes (vision calls only)."""

    role: str  # "user" | "assistant"
    text: str
    images: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ALLOWED_ROLES:
            raise ValidationError(
                f"message role must be one of {sorted(ALLOWED_ROLES)}, got {self.role!r}"
            )
        if not isinstance(self.text, str):
            raise ValidationError(
                f"message text must be a string, got {type(self.text).__name__}"
            )
        images = tuple(self.images)
        for i, image in enumerate(images):
            if not isinstance(image, bytes):
                raise ValidationError(
                    f"message images[{i}] must be bytes, got {type(image).__name__}"
                )
        object.__setattr__(self, "images", images)


@dataclass(frozen=True)
class LLMRequest:
    """A complete request for :meth:`LLMClient.complete`."""

    messages: tuple[LLMMessage, ...]
    model: str
    system: str = ""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not messages:
            raise ValidationError("LLMRequest requires at least one message")
        for i, message in enumerate(messages):
            if not isinstance(message, LLMMessage):
                raise ValidationError(
                    f"messages[{i}] must be an LLMMessage, got {type(message).__name__}"
                )
        object.__setattr__(self, "messages", messages)
        if not isinstance(self.model, str) or not self.model:
            raise ValidationError("LLMRequest.model must be a non-empty string")
        if self.timeout <= 0:
            raise ValidationError(f"LLMRequest.timeout must be > 0, got {self.timeout}")
        if self.max_tokens <= 0:
            raise ValidationError(
                f"LLMRequest.max_tokens must be > 0, got {self.max_tokens}"
            )


@dataclass(frozen=True)
class LLMResponse:
    """The assistant's reply text plus the model that produced it."""

    text: str
    model: str


class LLMClient(ABC):
    """Abstract synchronous LLM client."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion. Raises LLMRequestError/LLMTimeoutError on failure."""

    def test_connection(self, model: str) -> bool:
        """Send a minimal request to verify connectivity/auth. Raises on failure."""
        if not isinstance(model, str) or not model:
            raise LLMConfigError("a model name is required for the connection test")
        request = LLMRequest(
            messages=(LLMMessage(role="user", text=CONNECTION_TEST_PROMPT),),
            model=model,
            max_tokens=CONNECTION_TEST_MAX_TOKENS,
        )
        self.complete(request)
        return True


ClientFactory = Callable[[LLMProfile], LLMClient]

CLIENT_REGISTRY: dict[str, ClientFactory] = {}


def register_client(api_type: str, factory: ClientFactory) -> None:
    """Register (or replace) a client factory for an api_type."""
    if not isinstance(api_type, str) or not api_type:
        raise ValidationError("api_type must be a non-empty string")
    if not callable(factory):
        raise ValidationError(f"factory for api_type {api_type!r} must be callable")
    if api_type in CLIENT_REGISTRY:
        _LOGGER.warning("replacing existing client factory for api_type %r", api_type)
    CLIENT_REGISTRY[api_type] = factory


def _ensure_builtin_registered(api_type: str) -> None:
    """Import the built-in client module for api_type so it self-registers."""
    module_name = _BUILTIN_CLIENT_MODULES.get(api_type)
    if module_name is None or api_type in CLIENT_REGISTRY:
        return
    importlib.import_module(module_name)


def create_client(profile: LLMProfile) -> LLMClient:
    """Build the client for a profile. LLMConfigError for unknown type/missing URL."""
    if not isinstance(profile, LLMProfile):
        raise ValidationError(
            f"profile must be an LLMProfile, got {type(profile).__name__}"
        )
    if not profile.api_type:
        raise LLMConfigError(f"profile {profile.name!r} has no api_type configured")
    if not profile.base_url:
        raise LLMConfigError(f"profile {profile.name!r} has no base_url configured")
    _ensure_builtin_registered(profile.api_type)
    factory = CLIENT_REGISTRY.get(profile.api_type)
    if factory is None:
        known = ", ".join(sorted(CLIENT_REGISTRY)) or "(none)"
        raise LLMConfigError(
            f"unknown api_type {profile.api_type!r} for profile {profile.name!r}; "
            f"registered types: {known}"
        )
    _LOGGER.debug(
        "creating %r client for profile %r", profile.api_type, profile.name
    )
    return factory(profile)


# ---------------------------------------------------------------------------
# Shared plumbing for the httpx-based clients (kept private to nlapt.llm).
# ---------------------------------------------------------------------------


def require_httpx() -> ModuleType:
    """Import httpx lazily; raise LLMConfigError with install hint if missing."""
    try:
        return importlib.import_module("httpx")
    except ImportError as exc:
        raise LLMConfigError(HTTPX_MISSING_MESSAGE) from exc


def encode_image_base64(data: bytes) -> str:
    """Base64-encode image bytes as ASCII text."""
    return base64.b64encode(data).decode("ascii")


def detect_image_media_type(data: bytes) -> str:
    """Best-effort media type from magic bytes; defaults to JPEG."""
    if data.startswith(PNG_MAGIC):
        return MEDIA_TYPE_PNG
    if data.startswith(WEBP_RIFF_MAGIC) and data[8:12] == WEBP_FORMAT_MAGIC:
        return MEDIA_TYPE_WEBP
    return MEDIA_TYPE_JPEG


def join_url(base_url: str, endpoint: str) -> str:
    """Join a base URL and an endpoint path without doubling slashes."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise LLMConfigError("base_url must be a non-empty string")
    return base_url.rstrip("/") + endpoint


def http_post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    provider: str,
    transport: Any = None,
) -> dict[str, Any]:
    """POST JSON and return the decoded JSON object.

    Raises LLMTimeoutError on timeout, LLMRequestError on transport errors,
    HTTP >= 400, or non-object JSON bodies. Error messages never include
    request headers (and therefore never include API keys).
    """
    httpx = require_httpx()
    _LOGGER.debug("POST %s (provider=%s, model=%s)", url, provider, payload.get("model"))
    try:
        with httpx.Client(timeout=timeout, transport=transport) as http:
            response = http.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(
            f"{provider} request to {url} timed out after {timeout}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMRequestError(f"{provider} request to {url} failed: {exc}") from exc
    if response.status_code >= 400:
        body_preview = response.text[:ERROR_BODY_PREVIEW_CHARS]
        error = LLMRequestError(
            f"{provider} request failed with HTTP {response.status_code}: {body_preview}"
        )
        error.status_code = response.status_code  # type: ignore[attr-defined]
        raise error
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMRequestError(
            f"{provider} returned a non-JSON response from {url}"
        ) from exc
    if not isinstance(data, dict):
        raise LLMRequestError(
            f"{provider} returned unexpected JSON of type {type(data).__name__}"
        )
    return data


def validate_request(request: LLMRequest) -> None:
    """Boundary check shared by all clients."""
    if not isinstance(request, LLMRequest):
        raise ValidationError(
            f"request must be an LLMRequest, got {type(request).__name__}"
        )
