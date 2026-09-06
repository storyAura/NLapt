"""Try translation providers in order until one returns text.

Used by the GUI translate bridge so a flaky first choice (community DeepLX)
can fall through to Google / Baidu / local Hy-MT2 without the widget caring
which backend succeeded.
"""

from __future__ import annotations

from typing import Callable, Sequence

from nlapt.core.errors import LLMConfigError, LLMError, LLMRequestError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

MSG_ALL_FAILED = "所有翻译备选均失败"
MSG_EMPTY_RESULT = "翻译结果为空"


def run_fallback_chain(
    attempts: Sequence[tuple[str, Callable[[], str]]],
) -> str:
    """Return the first non-empty result; skip config errors, retry on request errors.

    ``attempts`` is ``(provider_id, thunk)``. A thunk that raises
    :class:`LLMConfigError` is skipped (not configured). Other
    :class:`LLMError` values try the next provider. The last error is
    re-raised; if every thunk returned empty text, raise
    :class:`LLMRequestError` with :data:`MSG_ALL_FAILED`.
    """
    if not attempts:
        raise LLMRequestError(MSG_ALL_FAILED)
    last_error: LLMError | None = None
    for name, thunk in attempts:
        try:
            result = thunk()
        except LLMConfigError as exc:
            _LOGGER.info("skip unusable translate provider %s: %s", name, exc)
            last_error = exc
            continue
        except LLMError as exc:
            _LOGGER.info(
                "translate provider %s failed; trying next: %s", name, exc
            )
            last_error = exc
            continue
        if isinstance(result, str) and result.strip():
            if last_error is not None:
                _LOGGER.info("translate fallback succeeded via %s", name)
            return result.strip()
        last_error = LLMRequestError(MSG_EMPTY_RESULT)
        _LOGGER.info("translate provider %s returned empty text; trying next", name)
    if last_error is not None:
        raise last_error
    raise LLMRequestError(MSG_ALL_FAILED)
