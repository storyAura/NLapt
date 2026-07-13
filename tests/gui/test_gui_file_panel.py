"""Tests for nlapt_gui.widgets.file_panel (+ thumb_cells click semantics)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

import nlapt_gui.widgets as widgets_pkg
from nlapt_gui.controller import FOLDER_ROOT_LABEL, AppController
from nlapt_gui.theme.tokens import THEMES
from nlapt_gui.widgets.file_panel import (
    TEXT_NO_MATCH,
    TEXT_SELECT_ALL,
    FilePanel,
)
from nlapt_gui.widgets.thumb_cells import ListRow, ThumbCell

K1 = "0001.png"
K2 = "0002.png"
K3 = "10_concept/0003.png"
K4 = "10_concept/0004.png"

HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}\b")
OWNED_MODULES = ("file_panel.py", "thumbnails.py", "thumb_cells.py", "preview_panel.py")


@pytest.fixture()
def panel(qtbot, controller: AppController) -> FilePanel:
    widget = FilePanel(controller)
    qtbot.addWidget(widget)
    widget.resize(300, 720)
    widget.show()
    qtbot.waitUntil(lambda: widget.cell(K1) is not None and widget.cell(K1).width() > 30)
    return widget


def _plain_click(qtbot, cell) -> None:
    qtbot.mouseClick(cell, Qt.MouseButton.LeftButton, pos=cell.rect().center())


class TestNoHardcodedColors:
    def test_owned_widget_modules_use_tokens_only(self) -> None:
        root = Path(widgets_pkg.__file__).resolve().parent
        for name in OWNED_MODULES:
            source = (root / name).read_text(encoding="utf-8")
            assert not HEX_COLOR.search(source), f"hardcoded color in {name}"


class TestHeader:
    def test_dataset_name_and_path(self, panel: FilePanel, controller: AppController) -> None:
        name, path = controller.dataset_label()
        assert panel.name_label.text() == name
        assert panel.path_label.toolTip() == path

    def test_refresh_button_rescans(self, qtbot, panel: FilePanel, controller) -> None:
        with qtbot.waitSignal(controller.dataset_opened, timeout=2000):
            qtbot.mouseClick(panel.refresh_button, Qt.MouseButton.LeftButton)

    def test_open_button_emits_request(self, qtbot, panel: FilePanel) -> None:
        with qtbot.waitSignal(panel.open_folder_requested, timeout=1000):
            qtbot.mouseClick(panel.open_button, Qt.MouseButton.LeftButton)

    def test_tooltips(self, panel: FilePanel) -> None:
        assert panel.refresh_button.toolTip() == "刷新"
        assert panel.open_button.toolTip() == "打开文件夹"


class TestFolderGroups:
    def test_groups_and_counts(self, panel: FilePanel) -> None:
        root_group = panel.folder_group(FOLDER_ROOT_LABEL)
        concept_group = panel.folder_group("10_concept")
        assert root_group is not None and concept_group is not None
        assert root_group.count_label.text() == "2 张"
        assert concept_group.count_label.text() == "2 张"
        assert root_group.name_label.text() == FOLDER_ROOT_LABEL

    def test_all_cells_are_grid_cells_in_mid_mode(self, panel: FilePanel) -> None:
        for key in (K1, K2, K3, K4):
            assert isinstance(panel.cell(key), ThumbCell)

    def test_header_click_collapses_and_persists(
        self, qtbot, panel: FilePanel, controller: AppController
    ) -> None:
        group = panel.folder_group(FOLDER_ROOT_LABEL)
        assert group.is_open
        qtbot.mouseClick(group.header, Qt.MouseButton.LeftButton)
        assert not group.is_open
        assert controller.settings.folder_open[FOLDER_ROOT_LABEL] is False
        qtbot.mouseClick(group.header, Qt.MouseButton.LeftButton)
        assert group.is_open
        assert controller.settings.folder_open[FOLDER_ROOT_LABEL] is True


class TestClickSemantics:
    def test_plain_click_sets_current_and_anchor(
        self, qtbot, panel: FilePanel, controller: AppController
    ) -> None:
        _plain_click(qtbot, panel.cell(K2))
        assert controller.current_key == K2
        # anchor moved to K2: shift range from K2..K3
        qtbot.mouseClick(
            panel.cell(K3),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ShiftModifier,
            panel.cell(K3).rect().center(),
        )
        assert controller.selected_keys() == (K2, K3)

    def test_alt_click_deselects(self, qtbot, panel: FilePanel, controller) -> None:
        controller.toggle_selected(K1)
        controller.toggle_selected(K2)
        qtbot.mouseClick(
            panel.cell(K2),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
            panel.cell(K2).rect().center(),
        )
        assert controller.selected_keys() == (K1,)

    def test_checkbox_click_toggles_without_changing_current(
        self, qtbot, panel: FilePanel, controller: AppController
    ) -> None:
        assert controller.current_key == K1
        cell = panel.cell(K2)
        qtbot.mouseClick(
            cell, Qt.MouseButton.LeftButton, pos=cell.checkbox_rect().center()
        )
        assert controller.is_selected(K2)
        assert controller.current_key == K1  # single selection keeps current
        qtbot.mouseClick(
            cell, Qt.MouseButton.LeftButton, pos=cell.checkbox_rect().center()
        )
        assert not controller.is_selected(K2)


class TestSelectionRow:
    def test_select_all_checkbox(self, qtbot, panel: FilePanel, controller) -> None:
        assert panel.select_all_box.text() == TEXT_SELECT_ALL
        qtbot.mouseClick(panel.select_all_box, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == (K1, K2, K3, K4)
        assert panel.selected_label.text() == "已选 4"
        qtbot.mouseClick(panel.select_all_box, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == ()

    def test_clear_link(self, qtbot, panel: FilePanel, controller) -> None:
        controller.select_all()
        assert panel.select_all_box.isChecked()
        qtbot.mouseClick(panel.clear_button, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == ()
        assert panel.selected_label.text() == "已选 0"
        assert not panel.select_all_box.isChecked()


class TestSearch:
    def test_placeholder(self, panel: FilePanel) -> None:
        assert panel.search_edit.placeholderText() == "搜索文件名 / 标签…"

    def test_typing_filters_and_updates_counts(
        self, qtbot, panel: FilePanel, controller: AppController
    ) -> None:
        qtbot.keyClicks(panel.search_edit, "0001")
        assert controller.filter_text == "0001"
        root_group = panel.folder_group(FOLDER_ROOT_LABEL)
        assert root_group is not None
        assert root_group.count_label.text() == "1/2 张"
        assert panel.folder_group("10_concept") is None  # no matches -> hidden
        assert panel.cell(K1) is not None
        assert panel.cell(K2) is None

    def test_filter_by_caption_content(self, panel: FilePanel, controller) -> None:
        controller.set_filter("yukata")
        assert panel.cell(K3) is not None
        assert panel.cell(K1) is None

    def test_empty_state(self, panel: FilePanel, controller: AppController) -> None:
        assert not panel.empty_label.isVisible()
        controller.set_filter("zzz-no-match")
        assert panel.empty_label.isVisible()
        assert panel.empty_label.text() == TEXT_NO_MATCH
        controller.set_filter("")
        assert not panel.empty_label.isVisible()


class TestViewModes:
    def test_switch_to_list_mode(self, qtbot, panel: FilePanel, controller) -> None:
        qtbot.mouseClick(panel.view_buttons["list"], Qt.MouseButton.LeftButton)
        assert controller.view_mode == "list"
        assert isinstance(panel.cell(K1), ListRow)
        assert panel.view_buttons["list"].property("segActive") is True
        assert panel.view_buttons["mid"].property("segActive") is False

    def test_switch_to_big_mode_keeps_grid(self, qtbot, panel: FilePanel, controller) -> None:
        qtbot.mouseClick(panel.view_buttons["big"], Qt.MouseButton.LeftButton)
        assert controller.view_mode == "big"
        assert isinstance(panel.cell(K1), ThumbCell)

    def test_list_row_click_semantics_match(self, qtbot, panel: FilePanel, controller) -> None:
        controller.set_view_mode("list")
        qtbot.waitUntil(lambda: isinstance(panel.cell(K2), ListRow))
        row = panel.cell(K2)
        qtbot.mouseClick(
            row, Qt.MouseButton.LeftButton, pos=row.checkbox_rect().center()
        )
        assert controller.is_selected(K2)
        _plain_click(qtbot, panel.cell(K3))
        assert controller.current_key == K3


class TestFooter:
    def test_stat_line_tracks_controller(self, panel: FilePanel, controller) -> None:
        assert panel.stat_label.text() == controller.stat_line()
        assert panel.stat_label.text() == "4 张图片 · 已选 0 · 未保存 0"
        controller.toggle_selected(K1)
        controller.set_caption(K2, "changed", "test")
        assert panel.stat_label.text() == "4 张图片 · 已选 1 · 未保存 1"


class TestPaintAndTheme:
    def test_panel_renders_and_survives_theme_change(
        self, panel: FilePanel, controller: AppController
    ) -> None:
        controller.set_caption(K1, "dirty text", "test")  # dirty dot path
        controller.toggle_selected(K1)  # checked checkbox path
        assert not panel.grab().isNull()
        panel.apply_tokens(THEMES["深邃"])
        assert panel.current_tokens().name == "深邃"
        assert not panel.grab().isNull()
