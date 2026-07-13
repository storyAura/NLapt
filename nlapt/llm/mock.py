"""Deterministic in-memory LLM client for tests and offline development.

``MockLLMClient`` is the sanctioned test double for every module that talks
to an :class:`nlapt.llm.base.LLMClient`. It records each request it receives
(exposed as an immutable snapshot via :attr:`requests`) so tests can assert
on models, prompts, and attached images without any network access.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence

from nlapt.core.errors import LLMRequestError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import LLMClient, LLMRequest, LLMResponse, validate_request

_LOGGER = get_logger(__name__)

ResponseProvider = Callable[[LLMRequest], str]

DEFAULT_FAILURE_MESSAGE = "mock LLM failure"


class MockLLMClient(LLMClient):
    """Scripted LLM client.

    Args:
        responses: either a sequence of reply strings (returned in order, the
            last one repeating once exhausted) or a callable mapping the
            incoming request to a reply string.
        fail_times: the first N ``complete()`` calls raise ``failure`` before
            any response is produced (for retry tests).
        failure: exception instance raised for the failing calls.
    """

    def __init__(
        self,
        responses: Sequence[str] | ResponseProvider,
        fail_times: int = 0,
        *,
        failure: BaseException | None = None,
    ) -> None:
        if callable(responses):
            self._provider: ResponseProvider | None = responses
            self._scripted: tuple[str, ...] = ()
        else:
            # A bare string is one scripted response, not a sequence of chars.
            scripted = (responses,) if isinstance(responses, str) else tuple(responses)
            if not scripted:
                raise ValidationError(
                    "MockLLMClient requires at least one scripted response"
                )
            for i, item in enumerate(scripted):
                if not isinstance(item, str):
                    raise ValidationError(
                        f"responses[{i}] must be a string, got {type(item).__name__}"
                    )
            self._provider = None
            self._scripted = scripted
        if fail_times < 0:
            raise ValidationError(f"fail_times must be >= 0, got {fail_times}")
        self._fail_times = fail_times
        self._failure = failure or LLMRequestError(DEFAULT_FAILURE_MESSAGE)
        self._lock = threading.Lock()
        self._requests: list[LLMRequest] = []
        self._calls = 0

    @property
    def requests(self) -> tuple[LLMRequest, ...]:
        """Immutable snapshot of every request passed to complete()."""
        with self._lock:
            return tuple(self._requests)

    @property
    def calls(self) -> int:
        """Total number of complete() invocations (including failed ones)."""
        with self._lock:
            return self._calls

    def complete(self, request: LLMRequest) -> LLMResponse:
        validate_request(request)
        with self._lock:
            self._requests.append(request)
            self._calls += 1
            call_index = self._calls
        if call_index <= self._fail_times:
            _LOGGER.debug("mock client failing call %d/%d", call_index, self._fail_times)
            raise self._failure
        if self._provider is not None:
            text = self._provider(request)
        else:
            reply_index = min(call_index - self._fail_times, len(self._scripted)) - 1
            text = self._scripted[reply_index]
        if not isinstance(text, str):
            raise ValidationError(
                f"mock response must be a string, got {type(text).__name__}"
            )
        return LLMResponse(text=text, model=request.model)
