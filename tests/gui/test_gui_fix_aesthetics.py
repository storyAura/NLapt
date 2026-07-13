"""Issue #5 - global aesthetics polish + skip-safe animation helpers.

Covers the additive work owned by the aesthetics pass:
- ``build_qss`` gains themed QComboBox (box + arrow + popup), richer scrollbars,
  button pressed states and message-box surfaces, in ALL five themes, with no
  unexpanded placeholders.
- ``nlapt_gui.anim`` helpers no-op cleanly when disabled and return a running
  animation object when enabled (jumped to the end synchronously - no sleeps).
- ``ThemeManager.apply`` still returns tokens, emits ``theme_changed`` and never
  raises, and its theme cross-fade restores full opacity without any wall clock
  wait.
"""

from __future__ import annotations

from typing import Iterator
from urllib.parse import quote

import pytest
from PySide6.QtCore import QAbstractAnimation
from PySide6.QtWidgets import QComboBox, QWidget

from nlapt_gui import anim
from nlapt_gui.theme import ThemeManager, build_qss
from nlapt_gui.theme.tokens import (
    RADIUS_CARD_PX,
    SHADOW_CARD,
    SHADOW_CARD_BLUR_PX,
    THEMES,
)

ALL_THEMES = list(THEMES)


@pytest.fixture(autouse=True)
def _restore_anim_flag() -> Iterator[None]:
    """Snapshot/restore the global animation switch around every test."""
    previous = anim.ANIMATIONS_ENABLED
    yield
    anim.set_animations_enabled(previous)


# --------------------------------------------------------------------------- #
# QSS polish
# --------------------------------------------------------------------------- #
class TestQssPolish:
    @pytest.mark.parametrize("name", ALL_THEMES)
    def test_new_selectors_present_in_every_theme(self, name: str) -> None:
        qss = build_qss(THEMES[name])
        for selector in (
            "QComboBox::drop-down",
            "QComboBox::down-arrow",
            "QComboBox QAbstractItemView",
            "QComboBox QAbstractItemView::item",
            "QScrollBar::handle:hover",
            "QScrollBar::handle:pressed",
            'QPushButton[variant="accent"]:pressed',
            "QMessageBox",
            "QMenu::item:selected",
        ):
            assert selector in qss, f"{name}: missing {selector}"

    @pytest.mark.parametrize("name", ALL_THEMES)
    def test_no_unexpanded_placeholders(self, name: str) -> None:
        qss = build_qss(THEMES[name])
        assert "{t." not in qss
        assert "None" not in qss
        # f-string literal braces must all have been consumed.
        assert "{{" not in qss and "}}" not in qss

    @pytest.mark.parametrize("name", ALL_THEMES)
    def test_combo_arrow_uses_theme_token_color(self, name: str) -> None:
        tokens = THEMES[name]
        qss = build_qss(tokens)
        # The chevron is an inline data URI; its stroke is the text3 token with
        # the leading '#' URL-encoded to %23 (proves it is not hardcoded).
        assert quote(tokens.text3, safe="") in qss
        assert "data:image/svg+xml" in qss

    def test_scrollbar_thickness_preserved(self) -> None:
        # The theme suite asserts 10px; keep that invariant after the polish.
        qss = build_qss(THEMES["雾灰"])
        assert "width: 10px" in qss
        assert "height: 10px" in qss

    def test_combo_popup_padding_and_selection(self) -> None:
        qss = build_qss(THEMES["深邃"])
        # Popup gets padding + rounded items and an accent-soft selection.
        assert "QComboBox QAbstractItemView::item:selected" in qss

    def test_qss_applies_to_real_combo_without_error(self, qapp) -> None:
        # The SVG data-uri arrow must not trip the Qt style engine.
        qapp.setStyleSheet(build_qss(THEMES["石墨"]))
        combo = QComboBox()
        combo.addItems(["a", "b"])
        combo.show()
        combo.setCurrentIndex(1)
        assert combo.currentText() == "b"
        combo.deleteLater()


class TestElevationTokens:
    def test_shadow_and_radius_constants_are_additive(self) -> None:
        assert isinstance(SHADOW_CARD, str) and SHADOW_CARD
        assert SHADOW_CARD_BLUR_PX == 32
        assert RADIUS_CARD_PX == 11


# --------------------------------------------------------------------------- #
# Animation helpers
# --------------------------------------------------------------------------- #
class TestAnimHelpers:
    def test_flag_toggle(self) -> None:
        anim.set_animations_enabled(False)
        assert anim.animations_enabled() is False
        anim.set_animations_enabled(True)
        assert anim.animations_enabled() is True

    def test_fade_in_disabled_is_noop_final_state(self, qapp) -> None:
        anim.set_animations_enabled(False)
        widget = QWidget()
        result = anim.fade_in(widget)
        assert result is None
        assert widget.graphicsEffect().opacity() == pytest.approx(1.0)
        widget.deleteLater()

    def test_fade_out_disabled_jumps_to_transparent(self, qapp) -> None:
        anim.set_animations_enabled(False)
        widget = QWidget()
        result = anim.fade_out(widget)
        assert result is None
        assert widget.graphicsEffect().opacity() == pytest.approx(0.0)
        widget.deleteLater()

    def test_pop_in_disabled_is_noop(self, qapp) -> None:
        anim.set_animations_enabled(False)
        widget = QWidget()
        widget.resize(120, 40)
        widget.show()
        result = anim.pop_in(widget)
        assert result is None
        assert widget.graphicsEffect().opacity() == pytest.approx(1.0)
        widget.deleteLater()

    def test_fade_in_enabled_returns_animation_and_ends_opaque(self, qapp) -> None:
        anim.set_animations_enabled(True)
        widget = QWidget()
        widget.show()
        animation = anim.fade_in(widget, ms=120)
        assert isinstance(animation, QAbstractAnimation)
        # Jump to the end synchronously - never wait on the clock.
        animation.setCurrentTime(animation.duration())
        assert widget.graphicsEffect().opacity() == pytest.approx(1.0)
        widget.deleteLater()

    def test_pop_in_enabled_returns_animation(self, qapp) -> None:
        anim.set_animations_enabled(True)
        widget = QWidget()
        widget.resize(140, 60)
        widget.show()
        animation = anim.pop_in(widget, ms=140)
        assert isinstance(animation, QAbstractAnimation)
        animation.setCurrentTime(animation.duration())
        assert widget.graphicsEffect().opacity() == pytest.approx(1.0)
        widget.deleteLater()


# --------------------------------------------------------------------------- #
# ThemeManager transition
# --------------------------------------------------------------------------- #
class TestThemeTransition:
    def test_apply_returns_tokens_and_emits_with_animations_off(
        self, qapp, qtbot
    ) -> None:
        anim.set_animations_enabled(False)
        manager = ThemeManager(qapp, persist=False)
        window = QWidget()
        window.show()
        with qtbot.waitSignal(manager.theme_changed, timeout=1000) as blocker:
            tokens = manager.apply("石墨")
        assert blocker.args[0] is tokens
        assert tokens.name == "石墨"
        # Skip-safe: with animations off the window opacity is never touched.
        assert window.windowOpacity() == pytest.approx(1.0)
        window.deleteLater()

    def test_apply_with_animations_dips_then_restores(self, qapp) -> None:
        anim.set_animations_enabled(True)
        manager = ThemeManager(qapp, persist=False)
        window = QWidget()
        window.show()
        manager.apply("墨黑")
        # A cross-fade was queued and the window was dipped below full opacity.
        assert manager._transitions, "expected a running cross-fade"
        assert window.windowOpacity() < 1.0
        # Jump each transition to its end - no real wait required.
        for animation in list(manager._transitions):
            animation.setCurrentTime(animation.duration())
        assert window.windowOpacity() == pytest.approx(1.0)
        window.deleteLater()

    def test_apply_with_animations_and_no_windows_does_not_raise(self, qapp) -> None:
        anim.set_animations_enabled(True)
        manager = ThemeManager(qapp, persist=False)
        # No visible windows created here; must simply not raise.
        tokens = manager.apply("深邃")
        assert tokens.name == "深邃"
