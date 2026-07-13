"""Tests for nlapt_gui.widgets.collapsible (section cards with resizable body)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from nlapt_gui.widgets.collapsible import (
    CLOSED_ROTATION_DEG,
    HEADER_HEIGHT,
    OPEN_ROTATION_DEG,
    ArrowIcon,
    CollapsibleSection,
)


def _make(qtbot, *, open_state: bool = True, duration: int = 30) -> CollapsibleSection:
    section = CollapsibleSection(
        "fr", "查找替换", open=open_state, duration_ms=duration
    )
    section.set_content(QLabel("content\nline\nline"))
    qtbot.addWidget(section)
    return section


class TestState:
    def test_initial_open(self, qtbot) -> None:
        section = _make(qtbot, open_state=True)
        assert section.is_open
        assert section.section_id == "fr"
        assert section._scroll.isVisibleTo(section)
        assert section._scroll.height() > 0 or section._scroll.maximumHeight() > 0

    def test_initial_closed(self, qtbot) -> None:
        section = _make(qtbot, open_state=False)
        assert not section.is_open
        assert not section._scroll.isVisibleTo(section)
        assert section._arrow.rotation == CLOSED_ROTATION_DEG

    def test_set_open_without_animation(self, qtbot) -> None:
        section = _make(qtbot, open_state=True)
        section.set_open(False, animate=False)
        assert not section.is_open
        assert not section._scroll.isVisibleTo(section)
        section.set_open(True, animate=False)
        assert section._scroll.isVisibleTo(section)
        assert section._arrow.rotation == OPEN_ROTATION_DEG

    def test_set_open_same_state_is_noop(self, qtbot) -> None:
        section = _make(qtbot, open_state=True)
        seen: list[tuple[str, bool]] = []
        section.toggled.connect(lambda sid, state: seen.append((sid, state)))
        section.set_open(True)
        assert seen == []

    def test_header_has_fixed_height(self, qtbot) -> None:
        # A fixed header can never be clipped by stylesheet size-hint quirks.
        section = _make(qtbot)
        assert section._header.height() == HEADER_HEIGHT
        assert section._header.minimumHeight() == HEADER_HEIGHT
        assert section._header.maximumHeight() == HEADER_HEIGHT


class TestToggle:
    def test_toggle_emits_toggled_with_id(self, qtbot) -> None:
        section = _make(qtbot, open_state=True)
        with qtbot.waitSignal(section.toggled, timeout=1000) as blocker:
            section.toggle()
        assert blocker.args == ["fr", False]
        assert not section.is_open

    def test_header_click_toggles(self, qtbot) -> None:
        section = _make(qtbot, open_state=True)
        section.show()
        with qtbot.waitSignal(section.toggled, timeout=1000) as blocker:
            qtbot.mouseClick(section._header, Qt.MouseButton.LeftButton)
        assert blocker.args == ["fr", False]

    def test_close_hides_body(self, qtbot) -> None:
        section = _make(qtbot, open_state=True, duration=30)
        section.show()
        section.set_open(False)
        assert not section._scroll.isVisibleTo(section)
        assert not section.is_open

    def test_open_shows_body(self, qtbot) -> None:
        section = _make(qtbot, open_state=False, duration=30)
        section.show()
        section.set_open(True)
        assert section._scroll.isVisibleTo(section)
        assert section.is_open
        assert section._arrow.rotation == pytest.approx(OPEN_ROTATION_DEG, abs=1.0)


class TestHeaderExtras:
    def test_suffix_label(self, qtbot) -> None:
        section = _make(qtbot)
        section.set_suffix("3 条")
        assert section._suffix.text() == "3 条"

    def test_set_content_replaces_previous(self, qtbot) -> None:
        section = _make(qtbot)
        replacement = QLabel("new")
        section.set_content(replacement)
        assert section._scroll.widget() is replacement

    def test_arrow_icon_paints(self, qtbot) -> None:
        arrow = ArrowIcon()
        qtbot.addWidget(arrow)
        arrow.rotation = -45.0
        assert arrow.rotation == -45.0
        arrow.show()
        arrow.repaint()  # exercise paintEvent offscreen
