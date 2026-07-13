"""Tests for nlapt_gui.theme (tokens, QSS generation, ThemeManager)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from nlapt.core.errors import ValidationError

from nlapt_gui.settings import load_ui_settings
from nlapt_gui.theme import (
    ACCENT_OPTIONS,
    DEFAULT_ACCENT,
    DEFAULT_THEME,
    THEMES,
    ThemeManager,
    ThemeTokens,
    accent_soft,
    build_qss,
    mix,
    with_accent,
)

GUI_ROOT = Path(__file__).resolve().parents[2] / "nlapt_gui"
HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}\b")

EXPECTED_BG = {
    "明亮": "#F5F5F6",
    "雾灰": "#E7E8EA",
    "石墨": "#292B30",
    "深邃": "#16171B",
    "墨黑": "#0C0D0F",
}
EXPECTED_ACCENT = {
    "明亮": "#0E9384",
    "雾灰": "#0E9384",
    "石墨": "#31C0AF",
    "深邃": "#31C0AF",
    "墨黑": "#38C9B7",
}
EXPECTED_SCHEME = {
    "明亮": "light",
    "雾灰": "light",
    "石墨": "dark",
    "深邃": "dark",
    "墨黑": "dark",
}


class TestTokens:
    def test_theme_names_and_default(self) -> None:
        assert tuple(THEMES) == ("明亮", "雾灰", "石墨", "深邃", "墨黑")
        assert DEFAULT_THEME == "雾灰"
        assert DEFAULT_ACCENT == "#0E9384"
        assert ACCENT_OPTIONS == ("#0E9384", "#4F63E7", "#D9634A", "#B58326")

    @pytest.mark.parametrize("name", list(THEMES))
    def test_exact_values(self, name: str) -> None:
        tokens = THEMES[name]
        assert isinstance(tokens, ThemeTokens)
        assert tokens.name == name
        assert tokens.bg == EXPECTED_BG[name]
        assert tokens.accent == EXPECTED_ACCENT[name]
        assert tokens.scheme == EXPECTED_SCHEME[name]

    def test_dark_theme_full_row(self) -> None:
        graphite = THEMES["石墨"]
        assert (graphite.panel, graphite.surface, graphite.surface2) == (
            "#303237", "#393C42", "#43464D",
        )
        assert (graphite.text, graphite.text2, graphite.text3) == (
            "#ECEDEF", "#B3B6BD", "#83868F",
        )
        assert graphite.onaccent == "#07211D"
        assert (graphite.ok, graphite.warn, graphite.danger) == (
            "#46C57E", "#E8A33D", "#E9705F",
        )
        assert graphite.scroll == "#54575F"

    def test_mix_endpoints_and_midpoint(self) -> None:
        assert mix("#000000", "#FFFFFF", 100) == "#000000"
        assert mix("#000000", "#FFFFFF", 0) == "#FFFFFF"
        assert mix("#000000", "#FFFFFF", 50) == "#808080"

    def test_mix_validates_input(self) -> None:
        with pytest.raises(ValidationError):
            mix("nope", "#FFFFFF", 50)
        with pytest.raises(ValidationError):
            mix("#000000", "#FFFFFF", 150)

    def test_accent_soft_is_14_pct_over_bg(self) -> None:
        tokens = THEMES["雾灰"]
        assert accent_soft(tokens) == mix(tokens.accent, tokens.bg, 14)

    def test_with_accent_recomputes_accent2(self) -> None:
        tokens = THEMES["雾灰"]
        custom = with_accent(tokens, "#4F63E7")
        assert custom.accent == "#4F63E7"
        assert custom.accent2 == mix("#4F63E7", tokens.text, 82)
        # original untouched (immutability)
        assert tokens.accent == "#0E9384"


class TestQss:
    def test_contains_core_selectors_and_colors(self) -> None:
        tokens = THEMES["深邃"]
        qss = build_qss(tokens)
        assert tokens.bg in qss
        assert tokens.accent in qss
        assert tokens.scroll in qss
        for selector in (
            'QPushButton[variant="accent"]',
            'QPushButton[variant="ghost"]',
            'QPushButton[variant="outline"]',
            'QPushButton[variant="danger-ghost"]',
            "QScrollBar:vertical",
            'QLabel[pill="true"]',
            'QPushButton[seg="true"]',
            'QFrame[chip="true"]',
            'QPushButton[collapsibleHeader="true"]',
            'QFrame[toast="true"]',
            "QLineEdit:focus",
        ):
            assert selector in qss, f"missing selector {selector}"

    def test_scrollbar_is_10px(self) -> None:
        qss = build_qss(THEMES["雾灰"])
        assert "width: 10px" in qss
        assert "height: 10px" in qss

    def test_no_unexpanded_placeholders(self) -> None:
        qss = build_qss(THEMES["明亮"])
        assert "{t." not in qss
        assert "None" not in qss

    @pytest.mark.parametrize("name", tuple(THEMES))
    def test_stylesheet_stays_valid_to_the_last_rule(self, qapp, qtbot, name) -> None:
        """Regression: an unquoted data-URI once ended parsing mid-sheet and Qt
        silently dropped every later rule (toggle chips lost all state
        feedback). Render the subject of the LAST rule block — a checked
        toggleChip — and require a visible difference from the unchecked one;
        string-containment checks cannot catch a truncated parse.
        """
        previous = qapp.styleSheet()
        qapp.setStyleSheet(build_qss(THEMES[name]))
        try:
            box = QWidget()
            qtbot.addWidget(box)
            layout = QHBoxLayout(box)
            chips: list[QPushButton] = []
            for checked in (False, True):
                chip = QPushButton("chip", box)
                chip.setProperty("toggleChip", True)
                chip.setProperty("chipOn", checked)
                chip.setCheckable(True)
                chip.setChecked(checked)
                layout.addWidget(chip)
                chips.append(chip)
            box.show()
            qapp.processEvents()
            image = box.grab().toImage()
            off = image.pixelColor(chips[0].geometry().center()).name()
            on = image.pixelColor(chips[1].geometry().center()).name()
            assert off != on, "chipOn style missing — stylesheet truncated mid-parse?"
        finally:
            qapp.setStyleSheet(previous)


class TestNoHardcodedColors:
    def test_gui_modules_outside_theme_have_no_hex_colors(self) -> None:
        offenders: list[str] = []
        for path in GUI_ROOT.rglob("*.py"):
            if "theme" in path.parts:
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if HEX_COLOR.search(line):
                    offenders.append(f"{path.name}:{line_no}: {line.strip()}")
        assert not offenders, "hardcoded colors outside nlapt_gui/theme:\n" + "\n".join(offenders)


class TestThemeManager:
    def test_apply_sets_stylesheet_palette_and_emits(self, qapp, qtbot) -> None:
        manager = ThemeManager(qapp)
        with qtbot.waitSignal(manager.theme_changed, timeout=1000) as blocker:
            tokens = manager.apply("石墨")
        assert blocker.args[0] is tokens
        assert tokens.name == "石墨"
        assert THEMES["石墨"].bg in qapp.styleSheet()
        window = qapp.palette().color(QPalette.ColorRole.Window).name().upper()
        assert window == THEMES["石墨"].bg
        assert manager.theme_name == "石墨"

    def test_apply_persists_choice(self, qapp) -> None:
        manager = ThemeManager(qapp)
        manager.apply("墨黑", "#D9634A")
        stored = load_ui_settings()
        assert stored.theme == "墨黑"
        assert stored.accent == "#D9634A"

    def test_custom_accent_recomputed(self, qapp) -> None:
        manager = ThemeManager(qapp)
        tokens = manager.apply("雾灰", "#4F63E7")
        assert tokens.accent == "#4F63E7"
        assert tokens.accent2 == mix("#4F63E7", THEMES["雾灰"].text, 82)
        assert manager.accent == "#4F63E7"

    def test_unknown_theme_falls_back_to_default(self, qapp) -> None:
        manager = ThemeManager(qapp)
        tokens = manager.apply("does-not-exist")
        assert tokens.name == DEFAULT_THEME
        assert manager.theme_name == DEFAULT_THEME
