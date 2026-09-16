"""Runtime SVG glyphs for QSS ``url()`` (checkbox / combo / partial).

Qt stylesheets do not load ``data:`` URIs, so the themed chevron and
check marks are written as real SVG files under the per-user data
directory and referenced by absolute path. Generation is keyed by the
token hex so a custom accent or theme swap just writes a new file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlapt.diagnostics import get_logger

from nlapt_gui.resources import app_data_dir
from nlapt_gui.theme.tokens import ThemeTokens

_LOGGER = get_logger(__name__)

GLYPH_DIR_NAME = "theme"


@dataclass(frozen=True)
class GlyphPaths:
    """POSIX paths of the three themed SVG files for one ``build_qss`` call."""

    chevron: str
    check: str
    partial: str


def svg_chevron(color: str) -> str:
    """Down-chevron for ``QComboBox::down-arrow``."""
    return _stroke_svg("M4 6 L8 10 L12 6", color)


def svg_check(color: str) -> str:
    """Check mark for a checked indicator."""
    return _stroke_svg("M3 8 L6.5 11.5 L13 4.5", color)


def svg_partial(color: str) -> str:
    """Horizontal bar for a partially-checked indicator."""
    return _stroke_svg("M4 8 H12", color)


def ensure_glyphs(
    tokens: ThemeTokens, base_dir: Path | None = None
) -> GlyphPaths | None:
    """Write (or reuse) the three SVGs for ``tokens``; ``None`` on I/O failure."""
    dest = Path(base_dir) if base_dir is not None else app_data_dir() / GLYPH_DIR_NAME
    files = (
        (_stem("chevron", tokens.text3), svg_chevron(tokens.text3)),
        (_stem("check", tokens.onaccent), svg_check(tokens.onaccent)),
        (_stem("partial", tokens.onaccent), svg_partial(tokens.onaccent)),
    )
    try:
        dest.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for name, content in files:
            path = dest / name
            _write_if_changed(path, content)
            paths.append(path.resolve().as_posix())
    except OSError:
        _LOGGER.warning("could not write theme glyphs under %s", dest, exc_info=True)
        return None
    return GlyphPaths(chevron=paths[0], check=paths[1], partial=paths[2])


def _stem(kind: str, color: str) -> str:
    return f"{kind}-{color.lstrip('#')}.svg"


def _stroke_svg(path_d: str, color: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
        f'viewBox="0 0 16 16"><path d="{path_d}" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"/></svg>'
    )


def _write_if_changed(path: Path, content: str) -> None:
    """Skip the write when the file already holds ``content`` (keeps mtime)."""
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            return
    except OSError:
        pass
    path.write_text(content, encoding="utf-8")
