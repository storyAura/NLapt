"""LLM integration layer: clients, retry, cleaning, templates, services.

Importing this package registers the built-in OpenAI/Anthropic/Ollama client
factories. It does NOT require ``httpx`` — the HTTP dependency is imported
lazily when a concrete client actually sends a request.
"""

from __future__ import annotations

from nlapt.llm.base import (
    CLIENT_REGISTRY,
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    create_client,
    register_client,
)

# Import for their registration side effect (they self-register factories).
from nlapt.llm import anthropic_client, ollama_client, openai_client  # noqa: F401
from nlapt.llm.cleaning import OUTPUT_CONSTRAINT, clean_llm_output
from nlapt.llm.mock import MockLLMClient
from nlapt.llm.retry import MinIntervalLimiter, RetryPolicy, with_retry
from nlapt.llm.rewrite import RewriteResult, RewriteService, RewriteSpec, RewriteType
from nlapt.llm.templates import (
    ALLOWED_VARIABLES,
    DEFAULT_TEMPLATES,
    TemplateStore,
    render_template,
)
from nlapt.llm.translate import (
    CachedTranslation,
    Direction,
    TranslationCache,
    Translator,
    detect_direction,
)
from nlapt.llm.vision import prepare_image

__all__ = [
    "ALLOWED_VARIABLES",
    "CLIENT_REGISTRY",
    "CachedTranslation",
    "DEFAULT_TEMPLATES",
    "Direction",
    "LLMClient",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "MinIntervalLimiter",
    "MockLLMClient",
    "OUTPUT_CONSTRAINT",
    "RetryPolicy",
    "RewriteResult",
    "RewriteService",
    "RewriteSpec",
    "RewriteType",
    "TemplateStore",
    "TranslationCache",
    "Translator",
    "clean_llm_output",
    "create_client",
    "detect_direction",
    "prepare_image",
    "register_client",
    "render_template",
    "with_retry",
]
