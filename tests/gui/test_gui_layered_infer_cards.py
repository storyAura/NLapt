"""Tests for the CHA标注 reference-image strip pieces."""

from __future__ import annotations

from PySide6.QtCore import Qt

from nlapt_gui.controller import FOLDER_ROOT_LABEL
from nlapt_gui.widgets.layered_infer_cards import (
    GRID_SPACING,
    REF_FOLDER_ALL,
    THUMB_W,
    RefPickerGrid,
    RefPickerPanel,
)
from nlapt_gui.widgets.thumb_cells import tokens_for_settings


class TestRefPickerGrid:
    def test_set_keys_rebuilds_and_keeps_highlight(self, qtbot, controller) -> None:
        tokens = tokens_for_settings(controller.settings)
        grid = RefPickerGrid(("0001.png", "0002.png"), controller, None, tokens)
        qtbot.addWidget(grid)
        picked: list[str] = []
        grid.picked.connect(picked.append)
        grid.select("0002.png")
        assert picked == ["0002.png"]

        grid.set_keys(("0002.png", "10_concept/0003.png"))
        assert grid.keys() == ("0002.png", "10_concept/0003.png")
        assert grid.thumb("0001.png") is None
        assert grid.selected_key() == "0002.png"
        assert "2px" in grid.thumb("0002.png").styleSheet()
        # Restoring the highlight on rebuild never re-emits ``picked``.
        assert picked == ["0002.png"]

        grid.set_keys(("0001.png",))
        assert grid.selected_key() == "0002.png"  # remembered, just hidden
        assert grid.thumb("0002.png") is None

    def test_grid_wraps_to_viewport_width(self, qtbot, controller) -> None:
        tokens = tokens_for_settings(controller.settings)
        grid = RefPickerGrid(controller.keys(), controller, None, tokens)
        qtbot.addWidget(grid)
        grid.show()
        grid.resize(THUMB_W + 20, 400)
        qtbot.waitExposed(grid)
        assert grid.columns() == 1
        first, second = grid.thumb("0001.png"), grid.thumb("0002.png")
        # Layouts settle lazily; wait for the geometry to reflect the reflow.
        qtbot.waitUntil(lambda: first.geometry().top() < second.geometry().top())

        grid.resize(3 * (THUMB_W + GRID_SPACING) + 30, 400)
        qtbot.waitUntil(lambda: grid.columns() == 3)
        qtbot.waitUntil(lambda: first.geometry().top() == second.geometry().top())
        assert first.geometry().right() < second.geometry().left()
        # Cells stay clickable after a reflow.
        qtbot.mouseClick(grid.thumb("10_concept/0004.png"), Qt.MouseButton.LeftButton)
        assert grid.selected_key() == "10_concept/0004.png"


class TestRefPickerPanel:
    def test_filter_only_narrows_the_given_keys(self, qtbot, controller) -> None:
        tokens = tokens_for_settings(controller.settings)
        root_only = controller.folder_keys(FOLDER_ROOT_LABEL)
        panel = RefPickerPanel(root_only, controller, None, tokens)
        qtbot.addWidget(panel)
        # The combo mirrors the dataset's folders (全部 + 根目录 + 10_concept).
        assert panel.folder_combo.count() == 3
        assert panel.keys() == root_only
        panel.set_folder("10_concept")
        assert panel.keys() == ()  # keys outside the given set are not invented
        panel.set_folder(REF_FOLDER_ALL)
        assert panel.keys() == root_only

    def test_select_unknown_key_is_ignored(self, qtbot, controller) -> None:
        tokens = tokens_for_settings(controller.settings)
        panel = RefPickerPanel(controller.keys(), controller, None, tokens)
        qtbot.addWidget(panel)
        panel.select("missing.png", notify=False)
        assert panel.selected_key() == ""
        panel.select("10_concept/0004.png")
        assert panel.selected_key() == "10_concept/0004.png"
