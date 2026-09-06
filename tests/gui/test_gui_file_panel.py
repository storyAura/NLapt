"""Tests for nlapt_gui.widgets.file_panel (+ thumb_cells click semantics)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt

import nlapt_gui.widgets as widgets_pkg
from nlapt_gui import anim
from nlapt_gui.controller import FOLDER_ROOT_LABEL, AppController
from nlapt_gui.theme.tokens import THEMES
from nlapt_gui.widgets.file_panel import (
    TEXT_NO_MATCH,
    TEXT_SELECT_ALL,
    _MAX_WIDGET_H,
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

    def test_collapse_animation_jumps_to_end(
        self, qtbot, panel: FilePanel, controller: AppController
    ) -> None:
        group = panel.folder_group(FOLDER_ROOT_LABEL)
        assert group is not None and group.is_open
        anim.set_animations_enabled(True)
        try:
            group.set_open(False, animate=True)
            motion = group._anim
            assert motion is not None
            motion.setCurrentTime(motion.duration())
            assert not group.is_open
            assert group._content.maximumHeight() == 0
            group.set_open(True, animate=True)
            motion = group._anim
            assert motion is not None
            motion.setCurrentTime(motion.duration())
            assert group.is_open
            assert group._content.maximumHeight() == _MAX_WIDGET_H
            group.set_open(False, animate=True)
            assert group._content.maximumHeight() < _MAX_WIDGET_H
            motion = group._anim
            if motion is not None:
                motion.setCurrentTime(motion.duration())
            assert not group.is_open
            assert group._content.maximumHeight() == 0
        finally:
            anim.set_animations_enabled(False)

    def test_header_click_can_collapse_after_expand(
        self, qtbot, panel: FilePanel, controller: AppController
    ) -> None:
        group = panel.folder_group(FOLDER_ROOT_LABEL)
        assert group is not None and group.is_open
        anim.set_animations_enabled(True)
        try:
            qtbot.mouseClick(group.header, Qt.MouseButton.LeftButton)
            if group._anim is not None:
                group._anim.setCurrentTime(group._anim.duration())
            assert not group.is_open
            assert group._content.maximumHeight() == 0
            qtbot.mouseClick(group.header, Qt.MouseButton.LeftButton)
            if group._anim is not None:
                group._anim.setCurrentTime(group._anim.duration())
            assert group.is_open
            qtbot.mouseClick(group.header, Qt.MouseButton.LeftButton)
            if group._anim is not None:
                group._anim.setCurrentTime(group._anim.duration())
            assert not group.is_open
            assert group._content.maximumHeight() == 0
            assert controller.settings.folder_open[FOLDER_ROOT_LABEL] is False
        finally:
            anim.set_animations_enabled(False)

    def test_retoggle_after_arrow_animation_finishes(
        self, qtbot, panel: FilePanel
    ) -> None:
        group = panel.folder_group(FOLDER_ROOT_LABEL)
        assert group is not None
        anim.set_animations_enabled(True)
        try:
            group.set_open(False, animate=True)
            if group.arrow._anim is not None:
                group.arrow._anim.setCurrentTime(group.arrow._anim.duration())
            if group._anim is not None:
                group._anim.setCurrentTime(group._anim.duration())
            group.set_open(True, animate=True)
            if group.arrow._anim is not None:
                group.arrow._anim.setCurrentTime(group.arrow._anim.duration())
            if group._anim is not None:
                group._anim.setCurrentTime(group._anim.duration())
            group.set_open(False, animate=True)
            assert not group.is_open
        finally:
            anim.set_animations_enabled(False)

    def test_expand_animation_starts_at_zero_and_keeps_rows_natural(
        self, panel: FilePanel
    ) -> None:
        group = panel.folder_group(FOLDER_ROOT_LABEL)
        assert group is not None
        anim.set_animations_enabled(True)
        try:
            group.set_open(False, animate=False)
            natural = group._content.sizeHint().height()
            group.set_open(True, animate=True)
            assert group._anim is not None
            assert group._anim.startValue() == 0
            assert group._anim.endValue() == natural
            group._anim.setCurrentTime(group._anim.duration() // 2)
            assert 0 < group._content.maximumHeight() < natural
            body = group._content.layout().itemAt(0).widget()
            assert body is not None
            assert body.height() >= natural - 12
        finally:
            anim.set_animations_enabled(False)


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
    def test_select_all_uses_only_matching_results(self, qtbot, panel, controller) -> None:
        controller.set_filter("0003")
        qtbot.mouseClick(panel.select_all_box, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == (K3,)
        assert panel.select_all_box.checkState() == Qt.CheckState.Checked
        assert panel.selected_label.text() == "已选 1"
        assert panel.all_row.check.checkState() == Qt.CheckState.PartiallyChecked
        qtbot.mouseClick(panel.select_all_box, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == ()

    def test_filter_removes_hidden_selection_and_updates_partial_state(
        self, qtbot, panel, controller
    ) -> None:
        controller.toggle_selected(K1)
        controller.toggle_selected(K3)
        controller.set_filter("hair")
        assert controller.selected_keys() == (K1,)
        assert panel.select_all_box.checkState() == Qt.CheckState.PartiallyChecked
        qtbot.mouseClick(panel.select_all_box, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == (K1, K2)
        assert panel.select_all_box.checkState() == Qt.CheckState.Checked

    def test_no_results_disables_filtered_selection_but_all_stays_global(
        self, qtbot, panel, controller
    ) -> None:
        controller.set_filter("no-matching-files")
        assert not panel.select_all_box.isEnabled()
        assert panel.select_all_box.checkState() == Qt.CheckState.Unchecked
        qtbot.mouseClick(panel.all_row.check, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == (K1, K2, K3, K4)
        assert panel.all_row.check.checkState() == Qt.CheckState.Checked
        assert not panel.select_all_box.isEnabled()
        qtbot.mouseClick(panel.clear_button, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == ()

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


class TestFolderMultiSelect:
    def test_folder_check_uses_results_but_menu_keeps_whole_folder(
        self, qtbot, qapp, panel, controller
    ) -> None:
        controller.set_filter("0003")
        qapp.processEvents()
        group = panel.folder_group("10_concept")
        qtbot.mouseClick(group.check, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == (K3,)
        assert group.check.checkState() == Qt.CheckState.Checked
        assert controller.folder_keys("10_concept") == (K3, K4)
        labels = [label for label, _action in panel.infer_menu_actions("10_concept")]
        assert "用 LLM 推理此文件夹(2 张)" in labels
        qtbot.mouseClick(group.check, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == ()

    def test_folder_checkbox_selects_whole_folder(
        self, qtbot, panel: FilePanel, controller: AppController
    ) -> None:
        group = panel.folder_group("10_concept")
        assert group is not None
        qtbot.mouseClick(group.check, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == (K3, K4)
        assert group.check.checkState() == Qt.CheckState.Checked
        qtbot.mouseClick(group.check, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == ()
        assert group.check.checkState() == Qt.CheckState.Unchecked

    def test_partial_folder_shows_tristate(
        self, panel: FilePanel, controller: AppController
    ) -> None:
        controller.toggle_selected(K3)
        group = panel.folder_group("10_concept")
        assert group.check.checkState() == Qt.CheckState.PartiallyChecked

    def test_all_row_selects_everything(
        self, qtbot, panel: FilePanel, controller: AppController
    ) -> None:
        assert panel.all_row.isVisibleTo(panel)
        assert panel.all_row.name_label.text() == "ALL"
        assert panel.all_row.count_label.text() == "4 张"
        qtbot.mouseClick(panel.all_row.check, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == (K1, K2, K3, K4)
        assert panel.all_row.check.checkState() == Qt.CheckState.Checked
        qtbot.mouseClick(panel.all_row.check, Qt.MouseButton.LeftButton)
        assert controller.selected_keys() == ()

    def test_controller_folder_helpers(self, controller: AppController) -> None:
        assert controller.folder_keys("10_concept") == (K3, K4)
        assert controller.folder_selection_state("10_concept") == "none"
        controller.set_folder_selected("10_concept", True)
        assert controller.folder_selection_state("10_concept") == "all"
        controller.set_folder_selected("10_concept", False)
        assert controller.folder_selection_state("10_concept") == "none"


class TestInferMenu:
    def test_folder_menu_lists_folder_unlabeled_and_all_scopes(
        self, panel: FilePanel, controller: AppController
    ) -> None:
        controller.toggle_selected(K1)  # 已选 never shows on folder menus
        labels = [label for label, _run in panel.infer_menu_actions("10_concept")]
        assert "用 LLM 推理此文件夹(2 张)" in labels
        assert "用本地模型推理此文件夹(2 张)" in labels
        assert "CHA标注此文件夹(2 张)" in labels
        assert "CHA标注此文件夹未标注(1 张)" in labels
        assert "CHA标注全部(4 张)" in labels
        assert "用 LLM 推理此文件夹未标注(1 张)" in labels
        assert "用本地模型推理此文件夹未标注(1 张)" in labels
        assert "用 LLM 推理全部(4 张)" in labels
        assert "用本地模型推理全部(4 张)" in labels
        assert not any("已选" in label for label in labels)

    def test_all_row_menu_hides_selection_and_folder_scopes(
        self, panel: FilePanel, controller: AppController
    ) -> None:
        controller.toggle_selected(K1)
        labels = [label for label, _run in panel.infer_menu_actions(None)]
        assert not any("已选" in label for label in labels)
        assert not any("此文件夹" in label for label in labels)
        assert "用 LLM 推理全部(4 张)" in labels
        assert "用 LLM 推理全部未标注(1 张)" in labels
        assert "用本地模型推理全部未标注(1 张)" in labels

    def test_root_group_menu_matches_all_row(
        self, panel: FilePanel, controller: AppController
    ) -> None:
        controller.toggle_selected(K1)
        labels = [label for label, _run in panel.infer_menu_actions(FOLDER_ROOT_LABEL)]
        assert not any("此文件夹" in label for label in labels)
        assert not any("已选" in label for label in labels)
        assert "用 LLM 推理全部(4 张)" in labels
        assert "用 LLM 推理全部未标注(1 张)" in labels

    def test_image_menu_single_image(
        self, panel: FilePanel, controller: AppController
    ) -> None:
        labels = [label for label, _run in panel.infer_menu_actions(None, image=K1)]
        assert labels == [
            "用 LLM 推理这张图片",
            "用本地模型推理这张图片",
            "CHA标注这张图片",
        ]

    def test_image_menu_multiselect_adds_selected_and_all(
        self, panel: FilePanel, controller: AppController
    ) -> None:
        controller.toggle_selected(K1)
        controller.toggle_selected(K2)
        labels = [label for label, _run in panel.infer_menu_actions(None, image=K1)]
        assert "用 LLM 推理已选(2 张)" in labels
        assert "用本地模型推理已选(2 张)" in labels
        assert "用 LLM 推理全部(4 张)" in labels
        assert "用本地模型推理全部(4 张)" in labels
        assert not any("这张图片" in label for label in labels)
        assert not any("此文件夹" in label for label in labels)

    def test_image_menu_unselected_image_ignores_selection(
        self, panel: FilePanel, controller: AppController
    ) -> None:
        controller.toggle_selected(K1)
        controller.toggle_selected(K2)
        labels = [label for label, _run in panel.infer_menu_actions(None, image=K3)]
        assert labels == [
            "用 LLM 推理这张图片",
            "用本地模型推理这张图片",
            "CHA标注这张图片",
        ]

    def test_image_action_infers_single_key(
        self, qtbot, panel: FilePanel, controller: AppController, monkeypatch
    ) -> None:
        import nlapt_gui.widgets.file_panel as fp_module

        monkeypatch.setattr(fp_module, "ask_confirm", lambda *a, **k: True)
        received: list[tuple[tuple[str, ...], str]] = []
        panel.infer_requested.connect(
            lambda keys, engine: received.append((tuple(keys), engine))
        )
        dict(panel.infer_menu_actions(None, image=K2))["用 LLM 推理这张图片"]()
        assert received == [((K2,), "llm")]

    def test_layered_action_emits_without_confirm(
        self, qtbot, panel: FilePanel, controller: AppController, monkeypatch
    ) -> None:
        import nlapt_gui.widgets.file_panel as fp_module

        confirms: list[object] = []
        monkeypatch.setattr(
            fp_module, "ask_confirm", lambda *a, **k: confirms.append(True) or True
        )
        received: list[tuple[str, ...]] = []
        panel.layered_infer_requested.connect(
            lambda keys: received.append(tuple(keys))
        )
        dict(panel.infer_menu_actions(None, image=K2))["CHA标注这张图片"]()
        assert received == [(K2,)]
        assert confirms == []

    def test_unlabeled_action_targets_only_unlabeled(
        self, qtbot, panel: FilePanel, controller: AppController, monkeypatch
    ) -> None:
        import nlapt_gui.widgets.file_panel as fp_module

        monkeypatch.setattr(fp_module, "ask_confirm", lambda *a, **k: True)
        received: list[tuple[tuple[str, ...], str]] = []
        panel.infer_requested.connect(
            lambda keys, engine: received.append((tuple(keys), engine))
        )
        actions = dict(panel.infer_menu_actions("10_concept"))
        actions["用本地模型推理此文件夹未标注(1 张)"]()
        assert received == [((K4,), "local")]

    def test_cell_right_click_opens_image_menu(
        self, panel: FilePanel, controller: AppController, monkeypatch
    ) -> None:
        calls: list[tuple[str | None, str | None]] = []
        monkeypatch.setattr(
            panel,
            "show_infer_menu",
            lambda folder, _w, _pos, image=None: calls.append((folder, image)),
        )
        panel.cell(K1).customContextMenuRequested.emit(QPoint(3, 3))
        assert calls == [(None, K1)]

    def test_menu_offers_cancel_while_batch_runs(
        self, panel: FilePanel, controller: AppController, monkeypatch
    ) -> None:
        monkeypatch.setattr(controller, "batch_running", lambda: True)
        cancels: list[bool] = []
        monkeypatch.setattr(controller, "cancel_batch", lambda: cancels.append(True))
        actions = panel.infer_menu_actions("10_concept")
        assert [label for label, _run in actions] == ["取消当前推标"]
        actions[0][1]()
        assert cancels == [True]

    def test_confirmed_action_emits_infer_requested(
        self, qtbot, panel: FilePanel, controller: AppController, monkeypatch
    ) -> None:
        import nlapt_gui.widgets.file_panel as fp_module

        monkeypatch.setattr(fp_module, "ask_confirm", lambda *a, **k: True)
        received: list[tuple[tuple[str, ...], str]] = []
        panel.infer_requested.connect(
            lambda keys, engine: received.append((tuple(keys), engine))
        )
        actions = dict(panel.infer_menu_actions("10_concept"))
        actions["用本地模型推理此文件夹(2 张)"]()
        assert received == [((K3, K4), "local")]

    def test_cancelled_confirm_emits_nothing(
        self, qtbot, panel: FilePanel, controller: AppController, monkeypatch
    ) -> None:
        import nlapt_gui.widgets.file_panel as fp_module

        monkeypatch.setattr(fp_module, "ask_confirm", lambda *a, **k: False)
        received: list[object] = []
        panel.infer_requested.connect(lambda keys, engine: received.append(keys))
        actions = dict(panel.infer_menu_actions(None))
        actions["用 LLM 推理全部(4 张)"]()
        assert received == []
