"""Unified cleaning of raw LLM output (spec 7.4).

Strips markdown code fences, surrounding quotes (including full-width and
corner quotes), and leading label prefixes such as ``Caption:`` / ``标注：``.
Cleaning runs to a fixpoint so combined wrappers (fence around a quoted,
labelled caption) are fully unwrapped.
"""

from __future__ import annotations

from nlapt.core.errors import LLMOutputError, ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

OUTPUT_CONSTRAINT = (
    "Output only the final caption text. No explanations, quotes, or code blocks."
)

CODE_FENCE = "```"
# (opening, closing) pairs stripped when they wrap the whole text.
QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),  # “ ”
    ("‘", "’"),  # ‘ ’
    ("「", "」"),  # 「 」
)
# Case-insensitive labels stripped from the start (with ASCII or full-width colon).
LABEL_WORDS: tuple[str, ...] = ("caption", "output", "result", "标注", "输出")
LABEL_SEPARATORS: tuple[str, ...] = (":", "：")  # : ：
MAX_CLEAN_PASSES = 10

# -- refusal / safety-interception detection (spec 7.4 hardening) -------------------
# When a provider's safety layer intercepts a caption request, the reply is a
# refusal ("I'm sorry, I can't ...") or a policy blurb instead of a caption.
# Callers that WRITE captions run the cleaned text through
# :func:`ensure_not_refusal` so an intercepted reply becomes a visible failure
# instead of silently landing in the dataset. Matching is against the lowered
# text: prefixes anchor at the start, markers match anywhere.
REFUSAL_PREFIXES: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i apologise",
    "i cannot",
    "i can't",
    "i can not",
    "i'm unable",
    "i am unable",
    "i won't",
    "i will not",
    "as an ai",
    "sorry, but",
    "unfortunately, i",
    "抱歉",
    "很抱歉",
    "对不起",
    "我不能",
    "我无法",
    "作为ai",
    "作为一个ai",
    "作为人工智能",
)
REFUSAL_MARKERS: tuple[str, ...] = (
    "cannot assist",
    "can't assist",
    "unable to assist",
    "cannot help with",
    "can't help with",
    "against my guidelines",
    "content policy",
    "i must decline",
    "无法协助",
    "无法帮助",
    "不能提供",
    "无法提供",
)
REFUSAL_PREVIEW_CHARS = 80
MSG_REFUSAL = "模型拒绝了本次请求(疑似触发内容安全拦截),该图已按失败处理:{preview}"


def _strip_code_fence(text: str) -> str:
    if not text.startswith(CODE_FENCE) or not text.endswith(CODE_FENCE):
        return text
    lines = text.splitlines()
    if len(lines) == 1:
        # Single-line fence: ```content```
        if len(text) >= 2 * len(CODE_FENCE):
            return text[len(CODE_FENCE) : -len(CODE_FENCE)]
        return ""
    # Multi-line: first line is ``` plus an optional language tag, last is ```.
    if lines[-1].strip() != CODE_FENCE:
        return text
    return "\n".join(lines[1:-1])


def _strip_matching_quotes(text: str) -> str:
    """Strip an outer quote pair only when it wraps the WHOLE string.

    The closing quote character must not occur anywhere in the inner text —
    otherwise the first and last characters belong to two different quoted
    phrases (e.g. ``"blue eyes" and "red hair"``) and stripping them would
    corrupt the caption. Nested distinct pairs keep being unwrapped.
    """
    while len(text) >= 2:
        pair = (text[0], text[-1])
        if pair not in QUOTE_PAIRS:
            break
        inner = text[1:-1]
        closing = pair[1]
        if closing in inner:
            break
        text = inner.strip()
    return text


def _strip_label_prefix(text: str) -> str:
    lowered = text.lower()
    for label in LABEL_WORDS:
        for separator in LABEL_SEPARATORS:
            prefix = f"{label}{separator}"
            if lowered.startswith(prefix):
                return text[len(prefix) :].lstrip()
    return text


def clean_llm_output(raw: str) -> str:
    """Normalize raw LLM output to bare caption text.

    Raises LLMOutputError when nothing usable remains after cleaning.
    """
    if not isinstance(raw, str):
        raise ValidationError(f"raw output must be a string, got {type(raw).__name__}")
    text = raw.strip()
    for _ in range(MAX_CLEAN_PASSES):
        previous = text
        text = _strip_code_fence(text).strip()
        text = _strip_matching_quotes(text).strip()
        text = _strip_label_prefix(text).strip()
        if text == previous:
            break
    if not text:
        _LOGGER.debug("LLM output empty after cleaning (raw length %d)", len(raw))
        raise LLMOutputError("LLM output was empty after cleaning")
    return text


def ensure_not_refusal(text: str) -> str:
    """Return ``text`` unchanged unless it reads like a safety refusal.

    Applied to vision captions only (translations may legitimately start
    with words like 抱歉 when the source text does). Raises
    :class:`LLMOutputError` so batch items fail visibly instead of writing
    an intercepted reply into the dataset.
    """
    if not isinstance(text, str):
        raise ValidationError(f"text must be a string, got {type(text).__name__}")
    lowered = text.strip().lower()
    refused = lowered.startswith(REFUSAL_PREFIXES) or any(
        marker in lowered for marker in REFUSAL_MARKERS
    )
    if refused:
        preview = text.strip()[:REFUSAL_PREVIEW_CHARS]
        _LOGGER.warning("LLM reply looks like a safety refusal: %r", preview)
        raise LLMOutputError(MSG_REFUSAL.format(preview=preview))
    return text
