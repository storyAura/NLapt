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
