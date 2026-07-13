"""Caption text IO with encoding detection (spec 2.2).

All caption reads go through :func:`read_text_detect`, which tries a fixed
list of candidate encodings, normalizes newlines to ``\\n``, and flags files
that are not plain UTF-8 so the caller can offer a convert-and-resave prompt.
All caption writes go through :func:`write_caption` (atomic UTF-8, no BOM).
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path

from nlapt.core.errors import EncodingDetectionError, StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

_LOGGER = get_logger(__name__)

PLAIN_UTF8 = "utf-8"
UTF8_WITH_BOM = "utf-8-sig"
CANDIDATE_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "gb18030",
    "big5",
    "shift_jis",
    "latin-1",
)


@dataclass(frozen=True)
class TextReadResult:
    """Decoded caption text plus detection metadata."""

    text: str  # newlines normalized to "\n"
    encoding: str  # encoding actually used to decode
    needs_conversion: bool  # True when the file is not plain UTF-8


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text_detect(path: Path) -> TextReadResult:
    """Read a text file, detecting its encoding from CANDIDATE_ENCODINGS.

    A UTF-8 BOM is honored first (decoded as ``utf-8-sig`` and flagged
    ``needs_conversion``). When the BOM is present but the body is not valid
    UTF-8, the BOM bytes are stripped and the remaining candidates are tried
    against the body alone (result reports the body encoding, still flagged
    ``needs_conversion``). Raises EncodingDetectionError when no candidate
    decodes, StorageError when the file cannot be read at all.
    """
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise StorageError(f"cannot read text file {source}: {exc}") from exc

    body = data
    has_bom = data.startswith(codecs.BOM_UTF8)
    if has_bom:
        try:
            text = data.decode(UTF8_WITH_BOM)
        except UnicodeDecodeError:
            # BOM prefixed onto a non-UTF-8 body (e.g. a Windows tool prepended
            # a BOM to a GBK file). Strip the BOM bytes so the candidate
            # encodings are tried against the body alone — otherwise the BOM
            # breaks multibyte alignment and detection falls through to latin-1.
            _LOGGER.warning("file %s has a UTF-8 BOM but invalid UTF-8 body", source)
            body = data[len(codecs.BOM_UTF8):]
        else:
            return TextReadResult(
                text=_normalize_newlines(text),
                encoding=UTF8_WITH_BOM,
                needs_conversion=True,
            )

    for encoding in CANDIDATE_ENCODINGS:
        if encoding == UTF8_WITH_BOM:
            continue  # BOM case handled above; without a BOM it equals utf-8
        try:
            text = body.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        needs_conversion = has_bom or encoding != PLAIN_UTF8
        if needs_conversion:
            _LOGGER.info("file %s decoded as %s (needs conversion to UTF-8)", source, encoding)
        return TextReadResult(
            text=_normalize_newlines(text),
            encoding=encoding,
            needs_conversion=needs_conversion,
        )

    raise EncodingDetectionError(
        f"could not detect text encoding for {source}", path=source
    )


def write_caption(path: Path, text: str) -> None:
    """Atomically write caption text as UTF-8 (no BOM) with ``\\n`` newlines."""
    if not isinstance(text, str):
        raise ValidationError(f"caption text must be str, got {type(text).__name__}")
    atomic_write_text(Path(path), _normalize_newlines(text))
