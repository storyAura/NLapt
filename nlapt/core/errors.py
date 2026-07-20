"""Typed exception hierarchy for NLapt.

Every module converts external failures (bad input, IO errors, API failures)
into one of these exception types so callers can handle them uniformly.
"""

from __future__ import annotations

from pathlib import Path


class NLaptError(Exception):
    """Base class for all NLapt errors. A human-readable message is required."""

    def __init__(self, message: str) -> None:
        if not isinstance(message, str) or not message:
            raise TypeError(
                f"{type(self).__name__} requires a non-empty message string, "
                f"got {message!r}"
            )
        super().__init__(message)
        self.message = message


class ValidationError(NLaptError):
    """Bad user or configuration input detected at a boundary."""


class StorageError(NLaptError):
    """Filesystem or persistence failure."""


class EncodingDetectionError(StorageError):
    """No candidate encoding could decode a text file.

    Attributes:
        path: the file whose encoding could not be detected (may be None).
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


class SnapshotError(StorageError):
    """Snapshot creation, listing, or restore failure."""


class SessionError(StorageError):
    """Session persistence failure (corrupt or unreadable session file)."""


class OperationError(NLaptError):
    """A text operation or history action could not be performed."""


class RegexPatternError(OperationError):
    """User-supplied regex pattern failed to compile.

    Attributes:
        pattern: the offending pattern text.
        detail: the underlying regex engine error message.
    """

    def __init__(self, message: str, *, pattern: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.pattern = pattern
        self.detail = detail


class LLMError(NLaptError):
    """Base class for LLM-related failures."""


class LLMConfigError(LLMError):
    """LLM profile/config problem (unknown api_type, missing dependency, ...)."""


class LLMRequestError(LLMError):
    """Network, API, or HTTP-level failure during an LLM request."""


class LLMTimeoutError(LLMRequestError):
    """LLM request exceeded its timeout."""


class LLMOutputError(LLMError):
    """LLM response was empty or unusable after cleaning."""


class BatchCancelledError(NLaptError):
    """A batch run was cancelled by the user."""


class LocalInferenceError(NLaptError):
    """Base class for local-inference (模型下载 / 本地服务) failures."""


class DownloadError(LocalInferenceError):
    """A model file download failed or produced a wrong-sized file."""


class DownloadCancelledError(LocalInferenceError):
    """A model download was cancelled by the user (partial file is kept)."""


class LocalServerError(LocalInferenceError):
    """The local inference server could not start, respond, or stop."""
