"""Regression tests for issue #2 - collapsible section flicker, content
clipping and user-resizable section heights.

Covers:
- Instant expand/collapse toggle (zero flicker) with the chevron still
  reflecting the open (0deg) / closed (-90deg) state.
- Expanded cards showing their full content height (inner scroll-area caps
  lifted so the collapsible is the single height authority).
- The bottom drag grip changing a section's content height (clamped) and the
  value round-tripping through the owned JSON persistence file.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from nlapt_gui.widgets.collapsible import (
    CLOSED_ROTATION_DEG,
    MAX_CONTENT_HEIGHT,
    MIN_CONTENT_HEIGHT,
    OPEN_ROTATION_DEG,
    CollapsibleSection,
    ResizeGrip,
)
from nlapt_gui.widgets.tools_panel import (
    SECTION_FIND_REPLACE,
    SECTION_HISTORY,
    SECTION_TRANSLATE,
    ToolsPanel,
    load_section_heights,
    save_section_heights,
)

_BIG = 100_000  # smaller than QWIDGETSIZE_MAX, larger than any real content


def _tall_body(rows: int = 12) -> QWidget:
    body = QWidget()
    layout = QVBoxLayout(body)
    for index in range(rows):
        layout.addWidget(QLabel(f"content line {index}"))
    return body


def _scroll_body(cap: int) -> tuple[QWidget, QScrollArea]:
    body = QWidget()
    layout = QVBoxLayout(body)
    scroll = QScrollArea(body)
    scroll.setMaximumHeight(cap)
    scroll.setWidget(QLabel("rows\n" * 40))
    layout.addWidget(scroll)
    return body, scroll


class TestNoFlickerToggle:
    def test_toggle_flips_state_instantly(self, qtbot) -> None:
        section = CollapsibleSection("fr", "查找替换", open=True)
        section.set_content(QLabel("x\ny\nz"))
        qtbot.addWidget(section)
        section.show()
        # Collapse: the body snaps hidden (no animation, no post-jump).
        section.toggle()
        assert not section.is_open
        assert not section._scroll.isVisibleTo(section)
        assert section._arrow.rotation == CLOSED_ROTATION_DEG
        # Expand: body snaps back visible at its applied height, chevron 0deg.
        section.toggle()
        assert section.is_open
        assert section._scroll.isVisibleTo(section)
        assert section._scroll.maximumHeight() == section.applied_height()
        assert section._arrow.rotation == OPEN_ROTATION_DEG

    def test_header_click_toggles_without_raising(self, qtbot) -> None:
        section = CollapsibleSection("fr", "查找替换", open=True)
        section.set_content(QLabel("body"))
        qtbot.addWidget(section)
        section.show()
        with qtbot.waitSignal(section.toggled, timeout=1000) as blocker:
            qtbot.mouseClick(section._header, Qt.MouseButton.LeftButton)
        assert blocker.args == ["fr", False]

    def test_grip_visibility_follows_open_state(self, qtbot) -> None:
        section = CollapsibleSection("fr", "t", open=True)
        section.set_content(QLabel("body"))
        qtbot.addWidget(section)
        section.show()
        assert section._grip.isVisible()
        section.set_open(False)
        assert not section._grip.isVisible()


class TestShowAll:
    def test_expanded_content_not_clipped(self, qtbot) -> None:
        section = CollapsibleSection("fr", "t", open=True)
        body = _tall_body()
        section.set_content(body)
        qtbot.addWidget(section)
        section.show()
        # Following the natural height: the body viewport is at least as tall
        # as the content wants (up to the global MAX), so nothing is clipped.
        assert section.applied_height() >= min(
            body.sizeHint().height(), MAX_CONTENT_HEIGHT
        )
        assert section._scroll.maximumHeight() == section.applied_height()

    def test_inner_scroll_cap_is_lifted(self, qtbot) -> None:
        section = CollapsibleSection("tr", "t", open=True)
        body, scroll = _scroll_body(cap=218)
        section.set_content(body)
        qtbot.addWidget(section)
        # The nested scroll area no longer imposes its 218px ceiling.
        assert scroll.maximumHeight() > _BIG

    def test_real_translate_and_history_scrolls_uncapped(
        self, qtbot, controller
    ) -> None:
        panel = ToolsPanel(controller)
        qtbot.addWidget(panel)
        for section_id in (SECTION_TRANSLATE, SECTION_HISTORY):
            card = panel.section(section_id)
            scrolls = card._content.findChildren(QScrollArea)
            assert scrolls, f"{section_id} should own a scroll area"
            assert all(s.maximumHeight() > _BIG for s in scrolls)


class TestResizableHeight:
    def test_grip_drag_sets_content_height(self, qtbot) -> None:
        section = CollapsibleSection("fr", "t", open=True)
        section.set_content(_tall_body(rows=3))
        qtbot.addWidget(section)
        section.show()
        seen: list[tuple[str, int]] = []
        section.height_changed.connect(lambda sid, h: seen.append((sid, h)))
        # Simulate the user dragging the grip down by 140 px.
        section._grip.resized.emit(140)
        height = section.content_height
        assert height is not None
        assert MIN_CONTENT_HEIGHT <= height <= MAX_CONTENT_HEIGHT
        assert seen and seen[-1][0] == "fr"
        assert section._scroll.maximumHeight() == height
        # Dragging UP must also work - shrink below the content's own size
        # (the old clamp against content minimumSizeHint made this a no-op).
        section._grip.resized.emit(-(height - MIN_CONTENT_HEIGHT))
        assert section.content_height == MIN_CONTENT_HEIGHT
        assert section._scroll.maximumHeight() == MIN_CONTENT_HEIGHT

    def test_content_height_is_clamped(self, qtbot) -> None:
        section = CollapsibleSection("fr", "t", open=True)
        section.set_content(QLabel("x"))
        qtbot.addWidget(section)
        section.set_content_height(10_000)
        assert section.content_height == MAX_CONTENT_HEIGHT
        section.set_content_height(1)
        assert section.content_height == MIN_CONTENT_HEIGHT

    def test_grip_ignored_while_collapsed(self, qtbot) -> None:
        section = CollapsibleSection("fr", "t", open=False)
        section.set_content(QLabel("x"))
        qtbot.addWidget(section)
        section._grip.resized.emit(200)
        assert section.content_height is None

    def test_resize_grip_emits_deltas(self, qtbot) -> None:
        grip = ResizeGrip()
        qtbot.addWidget(grip)
        grip.show()
        grip.repaint()  # exercise paintEvent offscreen
        deltas: list[int] = []
        grip.resized.connect(deltas.append)
        # A move with no active press must not emit anything.
        assert deltas == []


class TestPersistence:
    def test_missing_file_yields_empty(self, tmp_path: Path) -> None:
        assert load_section_heights(tmp_path / "nope.json") == {}

    def test_corrupt_file_yields_empty(self, tmp_path: Path) -> None:
        target = tmp_path / "bad.json"
        target.write_text("{not valid", encoding="utf-8")
        assert load_section_heights(target) == {}

    def test_load_validates_and_clamps(self, tmp_path: Path) -> None:
        target = tmp_path / "heights.json"
        target.write_text(
            json.dumps({"fr": 99_999, "hist": 3, "bad": "x", 5: 200}),
            encoding="utf-8",
        )
        loaded = load_section_heights(target)
        assert loaded["fr"] == MAX_CONTENT_HEIGHT
        assert loaded["hist"] == MIN_CONTENT_HEIGHT
        assert "bad" not in loaded

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "heights.json"
        save_section_heights({"fr": 300, "hist": 260}, target)
        assert load_section_heights(target) == {"fr": 300, "hist": 260}

    def test_panel_persists_and_restores_height(
        self, qtbot, controller, tmp_path: Path
    ) -> None:
        target = tmp_path / "heights.json"
        panel = ToolsPanel(controller, section_heights_path=target)
        qtbot.addWidget(panel)
        card = panel.section(SECTION_FIND_REPLACE)
        card.set_content_height(340)
        # Persisted to the owned JSON file...
        assert target.exists()
        assert load_section_heights(target)[SECTION_FIND_REPLACE] == card.content_height
        # ...and restored by a fresh panel sharing the same file.
        panel2 = ToolsPanel(controller, section_heights_path=target)
        qtbot.addWidget(panel2)
        assert (
            panel2.section(SECTION_FIND_REPLACE).content_height == card.content_height
        )

    def test_grip_drag_round_trips_through_persistence(
        self, qtbot, controller, tmp_path: Path
    ) -> None:
        target = tmp_path / "heights.json"
        panel = ToolsPanel(controller, section_heights_path=target)
        qtbot.addWidget(panel)
        card = panel.section(SECTION_HISTORY)  # open by default
        card._grip.resized.emit(160)
        height = card.content_height
        assert height is not None
        assert load_section_heights(target)[SECTION_HISTORY] == height
