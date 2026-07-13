"""Regression: multi-select current editor block must not highlight its chips.

Issue #1 - ``EditorBlock.refresh_meta`` applied an UNSCOPED stylesheet
(``background:...; border: 1.5px solid palette(highlight);...``) to the block
frame, so Qt cascaded the accent border onto every descendant chip. The fix
scopes the rule to the ``EditorBlock`` frame via a type#id selector so only the
frame gains the accent border; the chips keep their own per-chip borders and
the 编辑中 badge still marks the current block.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

from nlapt_gui.widgets.chips_editor import ChipsEditor
from nlapt_gui.widgets.editor_panel import (
    BADGE_EDITING,
    _BLOCK_OBJECT_NAME,
    EditorPanel,
)

from tests.gui.conftest import DEMO_KEYS


def _enter_multi_chips(controller) -> EditorPanel:
    controller.set_mode("chips")
    controller.toggle_selected(DEMO_KEYS[0])
    controller.toggle_selected(DEMO_KEYS[1])
    panel = EditorPanel(controller)
    return panel


def test_current_block_style_is_scoped_to_the_frame(qtbot, controller) -> None:
    """The current-block accent QSS targets EditorBlock only, not its chips."""
    panel = _enter_multi_chips(controller)
    qtbot.addWidget(panel)

    assert controller.multi_mode() is True
    current_block = panel.block_for(controller.current_key)
    assert current_block is not None

    sheet = current_block.styleSheet()
    # Scoped (has a selector + braces) - an unscoped rule would cascade to chips.
    assert "{" in sheet and "}" in sheet
    assert f"EditorBlock#{_BLOCK_OBJECT_NAME}" in sheet
    assert current_block.objectName() == _BLOCK_OBJECT_NAME


def test_chips_of_current_block_get_no_forced_border(qtbot, controller) -> None:
    """Individual chips carry no inline stylesheet from the block highlight."""
    panel = _enter_multi_chips(controller)
    qtbot.addWidget(panel)

    current_block = panel.block_for(controller.current_key)
    assert current_block is not None
    editor = current_block.editor
    assert isinstance(editor, ChipsEditor)
    chips = editor.chips()
    assert chips  # 0001.png has several comma segments
    # No chip carries its own accent border - the block rule cannot match them.
    for chip in chips:
        assert chip.styleSheet() == ""


def test_non_current_block_has_no_accent_border(qtbot, controller) -> None:
    """Only the current block is styled; the sibling block stays neutral."""
    panel = _enter_multi_chips(controller)
    qtbot.addWidget(panel)

    other_key = DEMO_KEYS[1]
    assert controller.current_key != other_key
    other_block = panel.block_for(other_key)
    assert other_block is not None
    assert other_block.styleSheet() == ""


def test_current_block_shows_editing_badge(qtbot, controller) -> None:
    """The 编辑中 badge marks the current block (and only it) in multi mode."""
    panel = _enter_multi_chips(controller)
    qtbot.addWidget(panel)

    current_block = panel.block_for(controller.current_key)
    other_block = panel.block_for(DEMO_KEYS[1])
    assert current_block is not None and other_block is not None

    def _badge(block) -> QLabel:  # noqa: ANN001
        for label in block.findChildren(QLabel):
            if label.text() == BADGE_EDITING:
                return label
        raise AssertionError("编辑中 badge not found")

    assert _badge(current_block).isHidden() is False
    assert _badge(other_block).isHidden() is True
