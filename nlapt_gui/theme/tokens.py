"""Design tokens for the NLapt GUI - exact values from the CaptionForge design.

Every color used anywhere in the GUI comes from a :class:`ThemeTokens`
instance (directly or through the generated QSS); widget modules must never
hardcode hex values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

from nlapt.core.errors import ValidationError

# Font stacks only - no bundled fonts.
FONT_STACK: tuple[str, ...] = ("IBM Plex Sans", "Segoe UI", "Microsoft YaHei", "PingFang SC")
MONO_STACK: tuple[str, ...] = ("IBM Plex Mono", "Consolas", "monospace")

DEFAULT_THEME = "雾灰"
DEFAULT_ACCENT = "#0E9384"
ACCENT_OPTIONS: tuple[str, ...] = ("#0E9384", "#4F63E7", "#D9634A", "#B58326")

THUMB_MIN_RANGE: tuple[int, int] = (72, 150)
DEFAULT_THUMB_MIN = 96
EDITOR_H_RANGE: tuple[int, int] = (170, 620)
DEFAULT_EDITOR_H = 330
# Spec 5: 滚轮在 10%–800% 间缩放 (the prototype's 40–260 could not reach a
# whole-image view for large photos; the effective floor additionally drops to
# the fit percentage so ANY image can zoom out to fully visible).
ZOOM_RANGE: tuple[int, int] = (10, 800)
ZOOM_STEP = 20
MIN_WINDOW: tuple[int, int] = (1360, 760)

# Color schemes.
SCHEME_LIGHT = "light"
SCHEME_DARK = "dark"

# Windows-native close-button hover red (title bar) - token constant so the
# integrator never hardcodes it inline.
WINDOWS_CLOSE_HOVER = "#E81123"
# Glyph color over the hovered close button (design: color #fff on hover).
WINDOWS_CLOSE_HOVER_FG = "#FFFFFF"

# color-mix percentages from the prototype.
ACCENT_SOFT_PCT = 14  # --accent-soft: 14% accent over the page background
ACCENT2_MIX_PCT = 82  # custom accent2 = 82% accent mixed with text
SELECTION_MIX_PCT = 28  # ::selection background = 28% accent over background

# --- Additive layout / elevation tokens (issue #5 aesthetics pass) ---------
# These are new constants only: they do NOT touch the ThemeTokens fields nor the
# five theme value sets (the theme tests assert those exactly). qss.py consumes
# them so radii/spacing stay consistent design-wide.
RADIUS_CARD_PX = 11  # surface cards / section frames
RADIUS_INPUT_PX = 7  # text inputs, combo boxes
RADIUS_BUTTON_PX = 6  # ghost / plain buttons
RADIUS_PILL_PX = 12  # Qt-safe approximation of the design's 999px pill
FOCUS_RING_PX = 2  # focus border thickness (Qt has no box-shadow glow)
# Elevation: the prototype's `--shadow` for cards. Qt QSS ignores box-shadow, so
# widgets that want a shadow use QGraphicsDropShadowEffect built from these.
SHADOW_CARD = "0 12px 32px rgba(20,22,26,.16)"  # reference/spec value
SHADOW_CARD_BLUR_PX = 32
SHADOW_CARD_OFFSET_Y_PX = 12
SHADOW_CARD_ALPHA = 0.16
# Spacing scale (4-pt grid) for consistent gaps/padding across widgets.
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16

_HEX_LENGTH = 7  # "#RRGGBB"


@dataclass(frozen=True)
class ThemeTokens:
    """One theme's full color set (names mirror the design's CSS variables)."""

    name: str
    scheme: str  # 'light' | 'dark'
    bg: str
    panel: str
    surface: str
    surface2: str
    bd: str
    bd2: str
    text: str
    text2: str
    text3: str
    accent: str
    accent2: str
    onaccent: str
    ok: str
    warn: str
    danger: str
    scroll: str


def _theme(name: str, scheme: str, values: str) -> ThemeTokens:
    """Build a ThemeTokens row from the contract's space-separated value list."""
    parts = values.split()
    (bg, panel, surface, surface2, bd, bd2, text, text2, text3,
     accent, accent2, onaccent, ok, warn, danger, scroll) = parts
    return ThemeTokens(
        name=name, scheme=scheme, bg=bg, panel=panel, surface=surface,
        surface2=surface2, bd=bd, bd2=bd2, text=text, text2=text2, text3=text3,
        accent=accent, accent2=accent2, onaccent=onaccent, ok=ok, warn=warn,
        danger=danger, scroll=scroll,
    )


THEMES: Mapping[str, ThemeTokens] = MappingProxyType({
    "明亮": _theme(
        "明亮", SCHEME_LIGHT,
        "#F5F5F6 #FBFBFC #FFFFFF #F0F0F2 #E4E5E9 #CACCD3 #1B1D22 #54575F "
        "#8B8E96 #0E9384 #0B7C70 #FFFFFF #188E4E #C97A10 #D4453A #CDCFD6",
    ),
    "雾灰": _theme(
        "雾灰", SCHEME_LIGHT,
        "#E7E8EA #EFEFF1 #F7F7F8 #E2E3E6 #D6D7DB #B9BBC2 #212327 #565962 "
        "#8F929B #0E9384 #0B7C70 #FFFFFF #188E4E #C97A10 #D4453A #BFC1C8",
    ),
    "石墨": _theme(
        "石墨", SCHEME_DARK,
        "#292B30 #303237 #393C42 #43464D #484B53 #5B5F69 #ECEDEF #B3B6BD "
        "#83868F #31C0AF #4FD2C3 #07211D #46C57E #E8A33D #E9705F #54575F",
    ),
    "深邃": _theme(
        "深邃", SCHEME_DARK,
        "#16171B #1C1E23 #24262C #2D3037 #32353D #464A55 #E7E8EC #A5A8B1 "
        "#6E717B #31C0AF #55D4C5 #07211D #46C57E #E8A33D #E9705F #3A3D46",
    ),
    "墨黑": _theme(
        "墨黑", SCHEME_DARK,
        "#0C0D0F #111215 #18191D #212227 #282A30 #3D4048 #E5E6E9 #9DA0A9 "
        "#63666F #38C9B7 #5CDACB #052019 #4BCB82 #EDAA45 #EE7767 #31343B",
    ),
})


def _parse_hex(color: str) -> tuple[int, int, int]:
    """Parse '#RRGGBB' into an (r, g, b) tuple; ValidationError on bad input."""
    value = color.strip()
    if len(value) != _HEX_LENGTH or not value.startswith("#"):
        raise ValidationError(f"expected a #RRGGBB color, got {color!r}")
    try:
        return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)
    except ValueError as exc:
        raise ValidationError(f"invalid hex color {color!r}") from exc


def mix(color_a: str, color_b: str, pct: float) -> str:
    """sRGB mix of two hex colors: ``pct`` percent of A over (100-pct) of B.

    Python replacement for the CSS ``color-mix(in srgb, A pct%, B)`` used
    throughout the prototype.
    """
    if not 0 <= pct <= 100:
        raise ValidationError(f"pct must be within 0..100, got {pct!r}")
    ratio = pct / 100.0
    a = _parse_hex(color_a)
    b = _parse_hex(color_b)
    mixed = tuple(round(ca * ratio + cb * (1.0 - ratio)) for ca, cb in zip(a, b))
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def accent_soft(tokens: ThemeTokens) -> str:
    """The prototype's ``--accent-soft``: 14% accent pre-mixed with the bg."""
    return mix(tokens.accent, tokens.bg, ACCENT_SOFT_PCT)


def with_accent(tokens: ThemeTokens, accent: str) -> ThemeTokens:
    """Tokens with a custom accent; accent2 = 82% accent mixed with text.

    Matches the prototype's accent override
    (``--accent2: color-mix(in srgb, accent 82%, var(--text))``).
    """
    _parse_hex(accent)  # validate early
    return replace(tokens, accent=accent, accent2=mix(accent, tokens.text, ACCENT2_MIX_PCT))
