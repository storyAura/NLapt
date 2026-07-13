"""AI polish/rewrite service (spec 7.4).

Builds prompts from typed templates (with the fixed output constraint
appended), routes vision requests to the vision model and text requests to
the text model, cleans outputs, and supports deterministic dry-runs. Results
are returned to the caller — they never overwrite caption bodies directly.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from nlapt.core.config import LLMProfile, RequestControl
from nlapt.core.errors import LLMConfigError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import (
    DEFAULT_TIMEOUT_SECONDS,
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from nlapt.llm.cleaning import OUTPUT_CONSTRAINT, clean_llm_output
from nlapt.llm.retry import MinIntervalLimiter, RetryPolicy, with_retry
from nlapt.llm.templates import (
    DEFAULT_TEMPLATES,
    TARGET_TOKENS_VARIABLE,
    render_template,
)

_LOGGER = get_logger(__name__)

DEFAULT_TARGET_TOKENS = 60
DEFAULT_DRY_RUN_SAMPLE_SIZE = 5
# Appended to a custom instruction that does not reference {caption} itself.
CAPTION_SUFFIX_TEMPLATE = "\n\nCaption: {caption}"
CAPTION_PLACEHOLDER = "{caption}"


class RewriteType(str, Enum):
    POLISH = "polish"
    REWRITE = "rewrite"
    EXPAND = "expand"
    CONDENSE = "condense"
    CUSTOM = "custom"


@dataclass(frozen=True)
class RewriteSpec:
    """One configured rewrite action."""

    type: RewriteType
    custom_instruction: str = ""  # required when type=CUSTOM
    target_tokens: int = DEFAULT_TARGET_TOKENS  # used by CONDENSE
    use_vision: bool = False
    template_override: str = ""  # optional explicit template text


@dataclass(frozen=True)
class RewriteResult:
    """Original caption paired with the cleaned LLM result."""

    key: str
    original: str
    result: str


ImageProvider = Callable[[str], bytes]
RewriteItem = tuple[str, str, str]  # (key, caption, filename)


def _validated_request_control(
    request: RequestControl | None,
) -> RequestControl | None:
    if request is not None and not isinstance(request, RequestControl):
        raise ValidationError(
            f"request must be a RequestControl or None, got {type(request).__name__}"
        )
    return request


def _validated_retry_sleep(
    retry_sleep: Callable[[float], None],
) -> Callable[[float], None]:
    if not callable(retry_sleep):
        raise ValidationError(
            f"retry_sleep must be callable, got {type(retry_sleep).__name__}"
        )
    return retry_sleep


def _resolve_limiter(
    request: RequestControl | None, limiter: MinIntervalLimiter | None
) -> MinIntervalLimiter | None:
    """One shared limiter per service instance (spec 8 min_interval)."""
    if limiter is not None:
        if not isinstance(limiter, MinIntervalLimiter):
            raise ValidationError(
                f"limiter must be a MinIntervalLimiter or None, "
                f"got {type(limiter).__name__}"
            )
        return limiter
    if request is None:
        return None
    return MinIntervalLimiter(request.min_interval)


def _paced_complete(
    client: LLMClient,
    request: LLMRequest,
    *,
    control: RequestControl | None,
    limiter: MinIntervalLimiter | None,
    retry_sleep: Callable[[float], None],
) -> LLMResponse:
    """Run one completion under the spec 8 request controls.

    The limiter is consulted once per logical request; retries (exponential
    backoff via ``with_retry``) apply only when a ``RequestControl`` is set.
    """
    if limiter is not None:
        limiter.wait()
    if control is None:
        return client.complete(request)
    policy = RetryPolicy(max_retries=control.max_retries)
    return with_retry(lambda: client.complete(request), policy, sleep=retry_sleep)


class RewriteService:
    """Run rewrite operations through the configured LLM clients."""

    def __init__(
        self,
        *,
        text_client: LLMClient,
        vision_client: LLMClient | None,
        profile: LLMProfile,
        trigger: str = "",
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
        if not isinstance(text_client, LLMClient):
            raise ValidationError(
                f"text_client must be an LLMClient, got {type(text_client).__name__}"
            )
        if vision_client is not None and not isinstance(vision_client, LLMClient):
            raise ValidationError(
                f"vision_client must be an LLMClient or None, "
                f"got {type(vision_client).__name__}"
            )
        if not isinstance(profile, LLMProfile):
            raise ValidationError(
                f"profile must be an LLMProfile, got {type(profile).__name__}"
            )
        self._text_client = text_client
        self._vision_client = vision_client
        self._profile = profile
        self._trigger = trigger
        self._request_control = _validated_request_control(request)
        self._retry_sleep = _validated_retry_sleep(retry_sleep)
        self._limiter = _resolve_limiter(request, limiter)

    def _validate_spec(self, spec: RewriteSpec) -> None:
        if not isinstance(spec, RewriteSpec):
            raise ValidationError(
                f"spec must be a RewriteSpec, got {type(spec).__name__}"
            )
        if spec.type is RewriteType.CUSTOM and not spec.custom_instruction.strip():
            raise ValidationError(
                "a custom instruction is required for the CUSTOM rewrite type"
            )
        if spec.type is RewriteType.CONDENSE and spec.target_tokens <= 0:
            raise ValidationError(
                f"target_tokens must be > 0, got {spec.target_tokens}"
            )

    def _template_for(self, spec: RewriteSpec) -> str:
        if spec.template_override:
            template = spec.template_override
        elif spec.type is RewriteType.CUSTOM:
            template = spec.custom_instruction
        else:
            template = DEFAULT_TEMPLATES[spec.type.value]
        if CAPTION_PLACEHOLDER not in template:
            template += CAPTION_SUFFIX_TEMPLATE
        return template

    def build_prompt(self, spec: RewriteSpec, *, caption: str, filename: str) -> str:
        """Render the template for spec and append the fixed output constraint."""
        self._validate_spec(spec)
        extra = (
            {TARGET_TOKENS_VARIABLE: str(spec.target_tokens)}
            if spec.type is RewriteType.CONDENSE
            else None
        )
        rendered = render_template(
            self._template_for(spec),
            caption=caption,
            filename=filename,
            trigger=self._trigger,
            extra=extra,
        )
        return rendered.rstrip() + "\n\n" + OUTPUT_CONSTRAINT

    def _select_client_and_model(self, spec: RewriteSpec) -> tuple[LLMClient, str]:
        if spec.use_vision:
            if self._vision_client is None:
                raise LLMConfigError(
                    "vision is enabled but no vision-capable client is configured"
                )
            if not self._profile.vision_model:
                raise LLMConfigError(
                    f"profile {self._profile.name!r} has no vision_model configured"
                )
            return self._vision_client, self._profile.vision_model
        if not self._profile.text_model:
            raise LLMConfigError(
                f"profile {self._profile.name!r} has no text_model configured"
            )
        return self._text_client, self._profile.text_model

    def run_one(
        self,
        spec: RewriteSpec,
        *,
        key: str,
        caption: str,
        filename: str,
        image: bytes | None = None,
    ) -> RewriteResult:
        """Execute one rewrite. Vision requests attach the image and use the
        vision model; text requests use the text model. Output is cleaned."""
        prompt = self.build_prompt(spec, caption=caption, filename=filename)
        client, model = self._select_client_and_model(spec)
        images: tuple[bytes, ...] = ()
        if spec.use_vision:
            if image is None:
                raise ValidationError(
                    f"vision rewrite for {key!r} requires image bytes"
                )
            images = (image,)
        timeout = (
            self._request_control.timeout
            if self._request_control is not None
            else DEFAULT_TIMEOUT_SECONDS
        )
        request = LLMRequest(
            messages=(LLMMessage(role="user", text=prompt, images=images),),
            model=model,
            system=self._profile.system_prompt,
            temperature=self._profile.temperature,
            max_tokens=self._profile.max_tokens,
            timeout=timeout,
        )
        response = _paced_complete(
            client,
            request,
            control=self._request_control,
            limiter=self._limiter,
            retry_sleep=self._retry_sleep,
        )
        result = clean_llm_output(response.text)
        _LOGGER.debug(
            "rewrite %s for %r done (%d -> %d chars)",
            spec.type.value,
            key,
            len(caption),
            len(result),
        )
        return RewriteResult(key=key, original=caption, result=result)

    def dry_run(
        self,
        spec: RewriteSpec,
        items: Sequence[RewriteItem],
        *,
        sample_size: int = DEFAULT_DRY_RUN_SAMPLE_SIZE,
        rng: random.Random | None = None,
        image_provider: ImageProvider | None = None,
    ) -> tuple[RewriteResult, ...]:
        """Run spec on a random sample of items (all of them when the sample
        size is not smaller than the item count)."""
        self._validate_spec(spec)
        if sample_size < 1:
            raise ValidationError(f"sample_size must be >= 1, got {sample_size}")
        if spec.use_vision and image_provider is None:
            raise ValidationError(
                "an image_provider is required for vision dry runs"
            )
        pool = list(items)
        if sample_size >= len(pool):
            sample = pool
        else:
            generator = rng if rng is not None else random.Random()
            sample = generator.sample(pool, sample_size)
        results: list[RewriteResult] = []
        for key, caption, filename in sample:
            image = image_provider(key) if spec.use_vision and image_provider else None
            results.append(
                self.run_one(
                    spec, key=key, caption=caption, filename=filename, image=image
                )
            )
        _LOGGER.debug("dry run produced %d result(s)", len(results))
        return tuple(results)
