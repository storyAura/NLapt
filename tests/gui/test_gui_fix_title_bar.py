"""Regression tests for title-bar fixes.

Covers issue #6 (name/subtitle baseline misalignment inside the 40px bar)
and issue #4 (the 帮助 menu / 关于 dialog were too sparse). Runs offscreen
via the shared conftest; no network, no real sleeps.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt

from nlapt_gui import __version__
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME
from nlapt_gui.widgets.title_bar import (
    ACTION_ABOUT,
    ACTION_GUIDE,
    MENU_HELP,
    TITLE_BAR_HEIGHT,
    VERSION_LABEL,
    TitleBar,
)


@pytest.fixture()
def manager(qtbot) -> ThemeManager:
    mgr = ThemeManager(persist=False)
    mgr.apply(DEFAULT_THEME)
    return mgr


@pytest.fixture()
def title_bar(qtbot, controller, manager) -> TitleBar:
    bar = TitleBar(controller, manager)
    bar.resize(1360, TITLE_BAR_HEIGHT)
    qtbot.addWidget(bar)
    bar.show()
    qtbot.waitExposed(bar)
    return bar


def _bottom_in_bar(bar: TitleBar, label) -> int:
    """Bottom edge of ``label`` expressed in the title bar's coordinates."""
    top_left = label.mapTo(bar, QPoint(0, 0))
    return top_left.y() + label.height()


def _top_in_bar(bar: TitleBar, label) -> int:
    return label.mapTo(bar, QPoint(0, 0)).y()


class TestIssue6Alignment:
    def test_bar_is_exactly_40px(self, title_bar: TitleBar) -> None:
        assert title_bar.height() == TITLE_BAR_HEIGHT

    def test_name_and_version_share_layout_alignment(self, title_bar: TitleBar) -> None:
        # Both labels live in the same content-hugging container so neither is
        # staggered relative to the other (the old bug: only version_label was
        # AlignBottom).
        assert title_bar.name_label.parent() is title_bar.name_group
        assert title_bar.version_label.parent() is title_bar.name_group
        layout = title_bar.name_group.layout()
        name_flag = layout.itemAt(0).alignment()
        version_flag = layout.itemAt(1).alignment()
        assert name_flag == version_flag
        assert name_flag & Qt.AlignmentFlag.AlignBottom

    def test_labels_share_baseline(self, title_bar: TitleBar) -> None:
        # Same bottom edge (baseline approximation) -> not staggered.
        name_bottom = _bottom_in_bar(title_bar, title_bar.name_label)
        version_bottom = _bottom_in_bar(title_bar, title_bar.version_label)
        assert abs(name_bottom - version_bottom) <= 2

    def test_labels_stay_inside_the_bar(self, title_bar: TitleBar) -> None:
        for label in (title_bar.name_label, title_bar.version_label):
            top = _top_in_bar(title_bar, label)
            bottom = _bottom_in_bar(title_bar, label)
            assert top >= 0
            assert bottom <= TITLE_BAR_HEIGHT

    def test_group_is_vertically_centered(self, title_bar: TitleBar) -> None:
        # The name/version group sits near the vertical middle of the 40px bar,
        # not flush to the bottom (which would overlap the row below).
        group = title_bar.name_group
        top = group.mapTo(title_bar, QPoint(0, 0)).y()
        center = top + group.height() / 2
        assert abs(center - TITLE_BAR_HEIGHT / 2) <= 4

    def test_version_string_source_unchanged(self, title_bar: TitleBar) -> None:
        assert title_bar.version_label.text() == VERSION_LABEL
        assert __version__ in title_bar.version_label.text()


class TestIssue4HelpMenu:
    def test_help_menu_actions(self, title_bar: TitleBar) -> None:
        help_menu = title_bar.menus[MENU_HELP]
        texts = [a.text() for a in help_menu.actions() if a.text()]
        assert ACTION_GUIDE in texts
        assert ACTION_ABOUT in texts
        # 快捷键 was removed by design (spec module 3.5): the status bar at the
        # bottom of the window already lists the live shortcut hints.
        assert "快捷键" not in texts
        assert not hasattr(title_bar, "action_shortcuts")

    def test_guide_handler_builds_dialog(self, qtbot, title_bar, monkeypatch) -> None:
        captured: list[tuple[str, str]] = []
        _stub_information(monkeypatch, captured)
        title_bar.action_guide.trigger()
        assert captured
        title, body = captured[0]
        assert title == ACTION_GUIDE
        for token in ("左栏", "中栏", "右栏", "胶囊", "分句", "文本", "快照"):
            assert token in body

    def test_about_handler_is_richer(self, qtbot, title_bar, monkeypatch) -> None:
        captured: list[tuple[str, str]] = []
        _stub_information(monkeypatch, captured)
        title_bar.action_about.trigger()
        assert captured
        _title, body = captured[0]
        assert VERSION_LABEL in body
        assert __version__ in body
        assert "kohya" in body

    def test_all_help_handlers_do_not_raise(self, qtbot, title_bar, monkeypatch) -> None:
        # Swap the modal call for a no-op so triggering never blocks offscreen.
        monkeypatch.setattr(
            "nlapt_gui.widgets.title_bar.show_message", lambda *a, **k: None
        )
        title_bar.action_guide.trigger()
        title_bar.action_about.trigger()


def _stub_information(monkeypatch, captured: list[tuple[str, str]]) -> None:
    """Capture the centered fading message dialog calls (title, body)."""

    def _fake_show_message(parent, title, body, *args, **kwargs):  # noqa: ANN001, ANN202
        captured.append((title, body))
        return None

    monkeypatch.setattr(
        "nlapt_gui.widgets.title_bar.show_message", _fake_show_message
    )
