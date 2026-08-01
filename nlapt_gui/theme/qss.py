"""Global QSS generator - one stylesheet built purely from ThemeTokens.

Widget modules opt into styles with dynamic properties instead of hardcoded
colors:

- ``variant`` on QPushButton: ``accent`` / ``ghost`` / ``outline`` /
  ``danger-ghost`` (design's button styles).
- ``panel="true"`` - column background (design ``--panel``).
- ``surfaceCard="true"`` - rounded 11px surface card (right-panel sections).
- ``pill`` - rounded pill label: ``true`` (neutral), ``accent``, ``accentSoft``,
  ``ok``, ``warn``, ``danger``.
- ``seg="true"`` (+ ``segActive="true"``) on QPushButton - segmented control
  buttons; ``segBar="true"`` on the container frame.
- ``chip="true"`` on QFrame - caption chip; ``chipActive="true"`` while its
  inline editor is open.
- ``collapsibleHeader="true"`` on QToolButton/QPushButton - section headers.
- ``toast="true"`` on QFrame - toast pill.
- ``mono="true"`` - monospace font; ``muted="true"`` / ``secondary="true"`` -
  text3 / text2 colors on QLabel.

Radii per design: sections 11px, buttons 6-8px, inputs 7px; the design's
999px pill radius is approximated with half-height radii (Qt cannot render
radii larger than half the widget size).
"""

from __future__ import annotations

from urllib.parse import quote

from nlapt_gui.theme.tokens import (
    ACCENT_SOFT_PCT,
    FONT_STACK,
    MONO_STACK,
    SELECTION_MIX_PCT,
    ThemeTokens,
    accent_soft,
    mix,
)

# Base UI font size from the design body.
BASE_FONT_PX = 13
# Pill/chip corner radius (Qt-safe approximation of the design's 999px).
PILL_RADIUS_PX = 12
# Toggle chips (Aa 区分大小写 / 作为独立标签) use the design's 6px rectangle radius,
# not the full-pill radius.
TOGGLE_CHIP_RADIUS_PX = 6
# Scrollbar thickness per design.
SCROLLBAR_PX = 10
# Combo-box drop-down button width.
COMBO_ARROW_BOX_PX = 24


def _font_family(stack: tuple[str, ...]) -> str:
    return ", ".join(f'"{name}"' for name in stack)


def _chevron_data_uri(color: str) -> str:
    """A themed down-chevron as an inline SVG ``data:`` URI for QComboBox.

    Qt QSS cannot draw a native arrow that follows arbitrary token colors, so
    the arrow glyph is generated here (inside ``theme/``) coloured with the
    given token. Self-contained: no external asset, no bundled image file.
    """
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10' "
        "viewBox='0 0 10 10'>"
        f"<path d='M2 3.6 L5 6.6 L8 3.6' fill='none' stroke='{color}' "
        "stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg, safe="")


def build_qss(tokens: ThemeTokens) -> str:
    """Build the application-wide stylesheet for one theme."""
    t = tokens
    soft = accent_soft(t)
    danger_soft = mix(t.danger, t.bg, ACCENT_SOFT_PCT)
    selection = mix(t.accent, t.bg, SELECTION_MIX_PCT)
    accent25 = mix(t.accent, t.bg, 25)
    accent40 = mix(t.accent, t.bg, 40)
    # Interaction shades: pressed = slightly toward text; scroll hover brightens
    # the thumb toward the accent so the bar highlights under the cursor.
    accent_pressed = mix(t.accent2, t.text, 88)
    surface_pressed = mix(t.surface2, t.text, 90)
    scroll_hover = mix(t.scroll, t.text, 74)
    scroll_pressed = mix(t.accent, t.scroll, 55)
    arrow_uri = _chevron_data_uri(t.text3)
    ui_font = _font_family(FONT_STACK)
    mono_font = _font_family(MONO_STACK)
    return f"""
/* ---------- base ---------- */
QWidget {{
    color: {t.text};
    font-family: {ui_font};
    font-size: {BASE_FONT_PX}px;
    selection-background-color: {selection};
    selection-color: {t.text};
}}
QMainWindow, QDialog {{ background: {t.bg}; }}
QWidget[panel="true"] {{ background: {t.panel}; }}
QWidget[surfaceCard="true"] {{
    background: {t.surface};
    border: 1px solid {t.bd};
    border-radius: 11px;
}}
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {t.text3}; }}
QLabel[secondary="true"] {{ color: {t.text2}; }}
*[mono="true"] {{ font-family: {mono_font}; }}
QToolTip {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.bd};
    border-radius: 6px;
    padding: 4px 8px;
}}
QMessageBox {{ background: {t.bg}; }}
QMessageBox QLabel {{ color: {t.text}; background: transparent; }}
QDialog QLabel {{ background: transparent; }}

/* ---------- buttons ---------- */
QPushButton, QToolButton {{
    background: transparent;
    color: {t.text2};
    border: none;
    border-radius: 6px;
    padding: 4px 10px;
}}
QPushButton:hover, QToolButton:hover {{ background: {t.surface2}; color: {t.text}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {surface_pressed}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t.text3}; }}
QPushButton[variant="accent"] {{
    background: {t.accent};
    color: {t.onaccent};
    border: none;
    border-radius: 7px;
    font-weight: 600;
    padding: 5px 13px;
}}
QPushButton[variant="accent"]:hover {{ background: {t.accent2}; }}
QPushButton[variant="accent"]:pressed {{ background: {accent_pressed}; }}
QPushButton[variant="accent"]:disabled {{ background: {t.surface2}; color: {t.text3}; }}
QPushButton[variant="outline"] {{
    background: {t.surface};
    color: {t.text2};
    border: 1px solid {t.bd};
    border-radius: 7px;
    padding: 4px 12px;
}}
QPushButton[variant="outline"]:hover {{ border-color: {t.bd2}; color: {t.text}; }}
QPushButton[variant="outline"]:pressed {{ background: {t.surface2}; border-color: {t.accent}; }}
QPushButton[variant="ghost"] {{
    background: transparent;
    color: {t.text3};
    border: none;
    border-radius: 6px;
}}
QPushButton[variant="ghost"]:hover {{ background: {t.surface2}; color: {t.text}; }}
QPushButton[variant="danger-ghost"] {{
    background: transparent;
    color: {t.text3};
    border: none;
    border-radius: 6px;
}}
QPushButton[variant="danger-ghost"]:hover {{ background: {danger_soft}; color: {t.danger}; }}

/* ---------- inputs ---------- */
QLineEdit, QPlainTextEdit, QTextEdit {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.bd};
    border-radius: 7px;
    padding: 4px 9px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1.5px solid {t.accent};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    color: {t.text3};
    background: {t.surface2};
}}
QLineEdit[editorField="true"], QPlainTextEdit[editorField="true"],
QTextEdit[editorField="true"] {{
    background: {t.bg};
    border-radius: 9px;
}}
QComboBox {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.bd};
    border-radius: 7px;
    padding: 4px 9px;
    padding-right: {COMBO_ARROW_BOX_PX + 4}px;
    min-height: 20px;
}}
QComboBox:hover {{ border-color: {t.bd2}; }}
QComboBox:focus, QComboBox:on {{ border: 1.5px solid {t.accent}; }}
QComboBox:disabled {{ color: {t.text3}; background: {t.surface2}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: {COMBO_ARROW_BOX_PX}px;
    border: none;
    border-left: 1px solid {t.bd};
    margin: 3px 0;
}}
QComboBox::down-arrow {{
    /* Quotes are load-bearing: the URI's ';utf8,' semicolon ends the
       declaration early when unquoted and Qt silently drops EVERY rule
       after this point (half the stylesheet, incl. toggleChip). */
    image: url("{arrow_uri}");
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.bd};
    border-radius: 8px;
    padding: 4px;
    outline: none;
    selection-background-color: {soft};
    selection-color: {t.accent};
}}
QComboBox QAbstractItemView::item {{
    min-height: 24px;
    padding: 4px 8px;
    border-radius: 6px;
}}
QComboBox QAbstractItemView::item:selected {{
    background: {soft};
    color: {t.accent};
}}
QComboBox QAbstractItemView::item:hover {{ background: {t.surface2}; color: {t.text}; }}
QCheckBox {{ color: {t.text2}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {t.bd2};
    border-radius: 4px;
    background: {t.surface};
}}
QCheckBox::indicator:checked {{
    background: {t.accent};
    border-color: {t.accent};
}}

/* ---------- progress bars ---------- */
QProgressBar {{
    background: {t.surface2};
    border: 1px solid {t.bd};
    border-radius: 4px;
    color: {t.text2};
    font-size: 10.5px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {t.accent};
    border-radius: 3px;
}}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: {SCROLLBAR_PX}px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: {SCROLLBAR_PX}px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {t.scroll};
    border-radius: 4px;
    min-height: 28px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {t.scroll};
    border-radius: 4px;
    min-width: 28px;
    margin: 2px;
}}
QScrollBar::handle:hover {{ background: {scroll_hover}; }}
QScrollBar::handle:pressed {{ background: {scroll_pressed}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QAbstractScrollArea {{ background: transparent; border: none; }}

/* ---------- pills ---------- */
QLabel[pill="true"] {{
    background: {t.surface2};
    color: {t.text2};
    border-radius: {PILL_RADIUS_PX}px;
    padding: 2px 9px;
    font-size: 10.5px;
}}
QLabel[pill="accent"] {{
    background: {t.accent};
    color: {t.onaccent};
    border-radius: {PILL_RADIUS_PX}px;
    padding: 2px 9px;
    font-size: 10px;
    font-weight: 700;
}}
QLabel[pill="accentSoft"] {{
    background: {soft};
    color: {t.accent};
    border-radius: {PILL_RADIUS_PX}px;
    padding: 2px 9px;
    font-size: 10.5px;
    font-weight: 600;
}}
QLabel[pill="ok"] {{
    background: {mix(t.ok, t.bg, ACCENT_SOFT_PCT)};
    color: {t.ok};
    border-radius: {PILL_RADIUS_PX}px;
    padding: 2px 9px;
    font-size: 10.5px;
    font-weight: 600;
}}
QLabel[pill="warn"] {{
    background: {mix(t.warn, t.bg, ACCENT_SOFT_PCT)};
    color: {t.warn};
    border-radius: {PILL_RADIUS_PX}px;
    padding: 2px 9px;
    font-size: 10.5px;
    font-weight: 600;
}}
QLabel[pill="danger"] {{
    background: {danger_soft};
    color: {t.danger};
    border-radius: {PILL_RADIUS_PX}px;
    padding: 2px 9px;
    font-size: 10.5px;
    font-weight: 600;
}}

/* ---------- segmented controls ---------- */
QFrame[segBar="true"] {{
    background: {t.surface2};
    border-radius: 7px;
}}
QPushButton[seg="true"] {{
    background: transparent;
    color: {t.text2};
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 11.5px;
}}
QPushButton[seg="true"]:hover {{ color: {t.text}; }}
QPushButton[seg="true"][segActive="true"] {{
    background: {t.surface};
    color: {t.accent};
    border-color: {t.bd2};
}}

/* ---------- chips ---------- */
QFrame[chip="true"] {{
    background: {t.surface};
    border: 1px solid {t.bd};
    border-radius: {PILL_RADIUS_PX + 2}px;
}}
QFrame[chip="true"]:hover {{ border-color: {t.accent}; }}
QFrame[chip="true"][chipActive="true"] {{ border: 1.5px solid {t.accent}; }}
QFrame[chip="true"][chipSelected="true"] {{
    border: 1.5px solid {t.accent};
    background: {soft};
}}
QLineEdit[chipEditor="true"] {{
    background: {t.surface};
    border: 1.5px solid {t.accent};
    border-radius: {PILL_RADIUS_PX + 2}px;
    font-family: {mono_font};
    font-size: 12.5px;
    padding: 3px 12px;
}}
QPushButton[chipAdd="true"] {{
    background: transparent;
    color: {t.text3};
    border: 1.5px dashed {t.bd2};
    border-radius: {PILL_RADIUS_PX + 2}px;
    padding: 4px 13px;
    font-size: 12px;
}}
QPushButton[chipAdd="true"]:hover {{ border-color: {t.accent}; color: {t.accent}; }}

/* ---------- collapsible section headers ---------- */
QPushButton[collapsibleHeader="true"], QToolButton[collapsibleHeader="true"] {{
    background: transparent;
    color: {t.text};
    border: none;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 12.5px;
    font-weight: 600;
    text-align: left;
}}
QPushButton[collapsibleHeader="true"]:hover,
QToolButton[collapsibleHeader="true"]:hover {{ background: {t.surface2}; }}

/* ---------- toasts ---------- */
QFrame[toast="true"] {{
    background: {t.surface};
    border: 1px solid {t.bd};
    border-radius: 17px;
}}
QFrame[toast="true"] QLabel {{ font-size: 12.5px; background: transparent; }}

/* ---------- menus ---------- */
QMenu {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.bd};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{
    padding: 5px 22px 5px 12px;
    border-radius: 6px;
    font-size: 12.5px;
}}
QMenu::item:selected {{ background: {t.surface2}; }}
QMenu::item:disabled {{ color: {t.text3}; }}
QMenu::separator {{ height: 1px; background: {t.bd}; margin: 4px 6px; }}

/* ---------- splitter / misc ---------- */
QSplitter::handle {{ background: transparent; }}
QFrame[divider="true"] {{ background: {t.bd}; border: none; }}
QWidget[accentSoftBox="true"] {{
    background: {soft};
    border: 1px solid {accent40};
    border-radius: 9px;
}}
QPushButton[toggleChip="true"] {{
    background: {t.surface2};
    color: {t.text3};
    border: 1px solid transparent;
    border-radius: {TOGGLE_CHIP_RADIUS_PX}px;
    padding: 3px 11px;
    font-size: 11px;
    font-weight: 600;
}}
QPushButton[toggleChip="true"]:hover {{
    color: {t.text2};
    border-color: {t.bd2};
}}
/* :checked mirrors the chipOn property so the flip is visible the instant
   Qt toggles the button, before the property repolish lands. */
QPushButton[toggleChip="true"]:checked,
QPushButton[toggleChip="true"][chipOn="true"] {{
    background: {soft};
    color: {t.accent};
    border-color: {t.accent};
}}
QPushButton[toggleChip="true"]:checked:hover,
QPushButton[toggleChip="true"][chipOn="true"]:hover {{
    color: {t.accent2};
    border-color: {t.accent2};
}}
QPushButton[toggleChip="true"]:pressed {{
    background: {accent25};
    color: {t.accent};
    border-color: {t.accent};
}}
"""
