"""Folder transitions keep their painted origin and viewport width stable."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest
from PySide6.QtCore import Qt

from nlapt_gui import anim
from nlapt_gui.controller import FOLDER_ROOT_LABEL, AppController
from nlapt_gui.widgets.file_panel import FilePanel


@pytest.fixture(params=("list", "mid", "big"))
def overflowing_panel(
    request, qtbot, qapp, controller: AppController, demo_dataset: Path
) -> FilePanel:
    """Use real images and enough rows to cross the scroll-range boundary."""
    for index in range(27):
        copyfile(demo_dataset / "0001.png", demo_dataset / f"motion-{index:02d}.png")
    with qtbot.waitSignal(controller.dataset_opened, timeout=2000):
        controller.refresh()
    controller.set_view_mode(request.param)
    panel = FilePanel(controller)
    qtbot.addWidget(panel)
    panel.resize(300, 720)
    panel.show()
    qtbot.waitUntil(lambda: panel._scroll.verticalScrollBar().maximum() > 0)
    qapp.processEvents()
    return panel


def test_toggle_keeps_header_and_content_top_aligned(
    qtbot, qapp, overflowing_panel: FilePanel
) -> None:
    group = overflowing_panel.folder_group(FOLDER_ROOT_LABEL)
    assert group is not None
    content_top = group._content.y()
    content_height = group._content.height()
    body_height = group._body.height()
    first_row = overflowing_panel.cell("0001.png")
    first_geometry = first_row.geometry()
    anim.set_animations_enabled(True)
    try:
        # Reuse the animation as in the recording's second collapse.
        for _cycle in range(2):
            for open_ in (False, True):
                qtbot.mouseClick(group.header, Qt.MouseButton.LeftButton)
                motion = group._anim
                assert motion is not None
                for elapsed in (16, 40, 80, 120, 180, motion.duration()):
                    motion.setCurrentTime(elapsed)
                    # The child layout runs before its parent receives LayoutRequest.
                    group.layout().activate()
                    assert group.header.y() == 0
                    assert group._content.y() == content_top
                    assert group._body.height() == body_height
                    assert first_row.geometry() == first_geometry
                    qapp.processEvents()
                target = content_height if open_ else 0
                qtbot.waitUntil(lambda: group._content.height() == target)
    finally:
        anim.set_animations_enabled(False)


def test_collapse_keeps_viewport_width_when_scroll_range_disappears(
    qtbot, qapp, overflowing_panel: FilePanel
) -> None:
    panel = overflowing_panel
    for group in panel._groups:
        if group.folder != FOLDER_ROOT_LABEL:
            group.set_open(False, animate=False)
    qapp.processEvents()
    group = panel.folder_group(FOLDER_ROOT_LABEL)
    assert group is not None
    width = panel._scroll.viewport().width()
    first_row = panel.cell("0001.png")
    row_width = first_row.width()
    anim.set_animations_enabled(True)
    try:
        qtbot.mouseClick(group.header, Qt.MouseButton.LeftButton)
        motion = group._anim
        assert motion is not None
        for elapsed in (16, 40, 80, 120, 180, motion.duration()):
            motion.setCurrentTime(elapsed)
            qapp.processEvents()
            assert panel._scroll.viewport().width() == width
            assert first_row.width() == row_width
        assert panel._scroll.verticalScrollBar().maximum() == 0
    finally:
        anim.set_animations_enabled(False)
