"""Tests for nlapt_gui.widgets.tools_panel (panel assembly + shared widgets)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6.QtWidgets import QLabel

from nlapt.core.errors import ValidationError

from nlapt_gui.settings import UISettings
from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES
from nlapt_gui.widgets.tools_panel import (
    HISTORY_COUNT_SUFFIX,
    PANEL_SUBTITLE,
    PANEL_TITLE,
    PANEL_WIDTH,
    SCOPE_ALL_LABEL,
    SCOPE_FOLDER_LABEL,
    SCOPE_SELECTED_LABEL,
    SECTION_FIND_REPLACE,
    SECTION_HISTORY,
    SECTION_PREFIX_SUFFIX,
    SECTION_TRANSLATE,
    ScopeSelector,
    SegmentedBar,
    ToolsPanel,
    resolve_tokens,
)

# Modules owned by this agent (no hardcoded colors allowed outside theme/).
_GUI_ROOT = Path(__file__).resolve().parents[2] / "nlapt_gui"
_OWNED_SOURCES = (
    _GUI_ROOT / "translate_bridge.py",
    _GUI_ROOT / "widgets" / "tools_panel.py",
    _GUI_ROOT / "widgets" / "collapsible.py",
    _GUI_ROOT / "widgets" / "toast.py",
    _GUI_ROOT / "widgets" / "settings_dialog.py",
    _GUI_ROOT / "widgets" / "sections" / "__init__.py",
    _GUI_ROOT / "widgets" / "sections" / "find_replace.py",
    _GUI_ROOT / "widgets" / "sections" / "prefix_suffix.py",
    _GUI_ROOT / "widgets" / "sections" / "translate.py",
    _GUI_ROOT / "widgets" / "sections" / "history.py",
    _GUI_ROOT / "widgets" / "sections" / "common.py",
    _GUI_ROOT / "widgets" / "tools_menu.py",
    _GUI_ROOT / "widgets" / "batch_scope_dialog.py",
    _GUI_ROOT / "widgets" / "flatten_alpha_dialog.py",
    _GUI_ROOT / "widgets" / "duplicate_review_dialog.py",
    _GUI_ROOT / "image_tools_bridge.py",
)
_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}\b")


class TestNoHardcodedColors:
    @pytest.mark.parametrize("source", _OWNED_SOURCES, ids=lambda p: p.name)
    def test_no_hex_literals_outside_theme(self, source: Path) -> None:
        assert source.exists(), f"missing owned module {source}"
        matches = _HEX_COLOR.findall(source.read_text(encoding="utf-8"))
        assert matches == [], f"hardcoded colors in {source.name}: {matches}"


class TestResolveTokens:
    def test_known_theme(self) -> None:
        tokens = resolve_tokens(UISettings(theme="石墨"))
        assert tokens.name == "石墨"

    def test_unknown_theme_falls_back(self) -> None:
        tokens = resolve_tokens(UISettings(theme="nope"))
        assert tokens.name == DEFAULT_THEME

    def test_custom_accent_applied(self) -> None:
        tokens = resolve_tokens(UISettings(accent="#4F63E7"))
        assert tokens.accent == "#4F63E7"

    def test_invalid_accent_ignored(self) -> None:
        tokens = resolve_tokens(UISettings(accent="not-a-color"))
        assert tokens.accent == THEMES[DEFAULT_THEME].accent


class TestSegmentedBar:
    def test_requires_options(self, qtbot) -> None:
        with pytest.raises(ValidationError):
            SegmentedBar(())

    def test_set_current_emits_changed(self, qtbot) -> None:
        bar = SegmentedBar((("a", "A"), ("b", "B")))
        qtbot.addWidget(bar)
        assert bar.current == "a"
        with qtbot.waitSignal(bar.changed, timeout=1000) as blocker:
            bar.set_current("b")
        assert blocker.args == ["b"]
        assert bar._buttons["b"].property("segActive") is True
        assert bar._buttons["a"].property("segActive") is False

    def test_unknown_option_raises(self, qtbot) -> None:
        bar = SegmentedBar((("a", "A"),))
        qtbot.addWidget(bar)
        with pytest.raises(ValidationError):
            bar.set_current("zzz")


class TestScopeSelector:
    def test_live_counts(self, qtbot, controller) -> None:
        scope = ScopeSelector(controller)
        qtbot.addWidget(scope)
        assert scope._buttons["all"].text() == SCOPE_ALL_LABEL.format(n=4)
        assert scope._buttons["folder"].text() == SCOPE_FOLDER_LABEL.format(n=4)
        assert scope._buttons["selected"].text() == SCOPE_SELECTED_LABEL.format(n=0)
        controller.toggle_selected("0001.png")
        controller.toggle_selected("0002.png")
        assert scope._buttons["selected"].text() == SCOPE_SELECTED_LABEL.format(n=2)
        controller.set_current("10_concept/0003.png")
        assert scope._buttons["folder"].text() == SCOPE_FOLDER_LABEL.format(n=2)


class TestToolsPanel:
    def test_panel_structure(self, qtbot, controller) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        assert panel.width() == PANEL_WIDTH
        titles = [label.text() for label in panel.findChildren(QLabel)]
        assert PANEL_TITLE in titles
        assert any(label.toolTip() == PANEL_SUBTITLE for label in panel.findChildren(QLabel))

    def test_sections_initial_open_states(self, qtbot, controller) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        # Design defaults: fr/hist open, ps/tr closed.
        assert panel.section(SECTION_FIND_REPLACE).is_open
        assert not panel.section(SECTION_PREFIX_SUFFIX).is_open
        assert not panel.section(SECTION_TRANSLATE).is_open
        assert panel.section(SECTION_HISTORY).is_open

    def test_unknown_section_raises(self, qtbot, controller) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        with pytest.raises(ValidationError):
            panel.section("nope")

    def test_toggle_persists_open_state(self, qtbot, controller) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        panel.section(SECTION_PREFIX_SUFFIX).set_open(True, animate=False)
        assert controller.settings.sections[SECTION_PREFIX_SUFFIX] is True
        panel.section(SECTION_FIND_REPLACE).set_open(False, animate=False)
        assert controller.settings.sections[SECTION_FIND_REPLACE] is False
        # Untouched sections keep their persisted values.
        assert controller.settings.sections[SECTION_HISTORY] is True

    def test_history_count_suffix_tracks_entries(self, qtbot, controller) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        hist_card = panel.section(SECTION_HISTORY)
        assert hist_card._suffix.text() == HISTORY_COUNT_SUFFIX.format(n=1)
        controller.set_caption("0001.png", "changed caption", "编辑")
        assert hist_card._suffix.text() == HISTORY_COUNT_SUFFIX.format(n=2)
        controller.set_current("0002.png")
        assert hist_card._suffix.text() == HISTORY_COUNT_SUFFIX.format(n=1)

    def test_translate_link_opens_settings_signal(self, qtbot, controller) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        opened: list[bool] = []
        panel.open_settings_dialog = lambda: opened.append(True)  # type: ignore[method-assign]
        panel.translate.open_settings_requested.disconnect()
        panel.translate.open_settings_requested.connect(panel.open_settings_dialog)
        panel.translate.settings_link.click()
        assert opened == [True]

    def test_sections_wired_to_same_controller(self, qtbot, controller) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        assert panel.find_replace._controller is controller
        assert panel.prefix_suffix._controller is controller
        assert panel.translate._controller is controller
        assert panel.history._controller is controller

    def test_close_button_emits(self, qtbot, controller) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        with qtbot.waitSignal(panel.close_requested, timeout=1000):
            panel.close_button.click()
