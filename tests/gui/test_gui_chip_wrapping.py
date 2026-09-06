"""Real Qt layout checks for wrapped chips and bounded inline editing."""

from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAccessible, QFocusEvent
from PySide6.QtWidgets import QPushButton

from nlapt_gui.controller import AppController
from nlapt_gui.theme.qss import build_qss
from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES
from nlapt_gui.widgets.chip_widgets import InlineChipField
from nlapt_gui.widgets.chips_editor import ChipsEditor
from nlapt_gui.widgets.editor_panel import EditorPanel

KEY = "0001.png"
LONG_TEXTS = (
    "A long caption with details about the subject and its surroundings. " * 12,
    "long_unbroken_caption_" * 45,
    "这是需要完整显示的长段落而且不应该撑宽编辑区" * 25,
    "First paragraph with literal <b>plain text</b>\n" * 15 + "Last paragraph",
)


def make_panel(qtbot, controller: AppController, text: str) -> EditorPanel:
    controller.set_caption(KEY, text.strip(), "Fixture caption")
    panel = EditorPanel(controller)
    panel.setStyleSheet(build_qss(THEMES[DEFAULT_THEME]))
    qtbot.addWidget(panel)
    panel.resize(460, 520)
    panel.show()
    qtbot.waitUntil(lambda: panel.blocks()[0].editor.isVisible())
    return panel


@pytest.mark.parametrize("text", LONG_TEXTS)
def test_wrapped_chips_fit_scroll_viewport_and_paint_all_lines(
    qtbot, qapp, controller: AppController, text: str
) -> None:
    panel = make_panel(qtbot, controller, text)
    editor = panel.blocks()[0].editor
    assert isinstance(editor, ChipsEditor)
    before = controller.record(KEY).text
    history_size = len(controller.history.entries(KEY))
    narrow_height = 0
    for width in (460, 1080, 460):
        panel.resize(width, 520)
        qapp.processEvents()
        chip = editor.chips()[0]
        assert chip.geometry().right() < editor.width()
        assert panel._scroll.horizontalScrollBar().maximum() == 0
        label = chip._label
        accessible = QAccessible.queryAccessibleInterface(label)
        assert accessible is not None
        assert accessible.text(QAccessible.Text.Name) == text.strip()
        image = label.grab().toImage()
        document = label._document
        assert document.toPlainText() == text.strip()
        assert math.ceil(document.size().height()) <= label.height()
        block = document.begin()
        while block.isValid():
            layout = block.layout()
            assert sum(layout.lineAt(i).textLength() for i in range(layout.lineCount())) == len(
                block.text()
            )
            assert layout.boundingRect().width() <= label.width() + 1
            block = block.next()
        # The last line must contain painted glyphs, not just unallocated text.
        last = document.lastBlock().layout()
        line = last.lineAt(last.lineCount() - 1)
        first_y = int(last.position().y() + line.y())
        colors = {
            image.pixel(x, y)
            for y in range(first_y, min(image.height(), math.ceil(first_y + line.height())))
            for x in range(image.width())
        }
        assert len(colors) > 1
        remove = chip.findChild(QPushButton, "chipDelete")
        assert remove is not None
        assert chip.rect().contains(remove.geometry())
        assert remove.y() <= 7
        if width == 460:
            narrow_height = chip.height()
        else:
            assert chip.height() <= narrow_height
    assert controller.record(KEY).text == before
    assert len(controller.history.entries(KEY)) == history_size


def test_short_chips_remain_compact(qtbot, qapp, controller: AppController) -> None:
    panel = make_panel(qtbot, controller, "short, tiny, compact")
    qapp.processEvents()
    editor = panel.blocks()[0].editor
    chips = editor.chips()
    assert all(chip.width() < editor.width() // 2 for chip in chips)
    assert len({chip.y() for chip in chips}) == 1


@pytest.mark.parametrize("text", LONG_TEXTS)
def test_inline_resize_preserves_full_text_cursor_and_selection(
    qtbot, qapp, controller: AppController, text: str
) -> None:
    panel = make_panel(qtbot, controller, text)
    editor = panel.blocks()[0].editor
    editor.start_edit(0)
    qtbot.waitUntil(lambda: editor._field.isVisible())
    field = editor._field
    assert isinstance(field, InlineChipField)
    field.setSelection(9, 22)
    cursor = field.cursorPosition()
    selected = field.selectedText()
    for width in (920, 460, 680):
        panel.resize(width, 520)
        qapp.processEvents()
        assert editor._field is field
        assert field.text() == text.strip()
        assert field.cursorPosition() == cursor
        assert field.selectionStart() == 9
        assert field.selectedText() == selected
        assert field.geometry().right() < editor.width()
        assert panel._scroll.horizontalScrollBar().maximum() == 0
    editor.set_editor_text("temporary replacement")
    qtbot.keyClick(field, Qt.Key.Key_Escape)
    assert controller.segments(KEY) == (text.strip(),)


@pytest.mark.parametrize("commit", ("enter", "blur"))
def test_long_inline_edit_commits_full_text(
    qtbot, qapp, controller: AppController, commit: str
) -> None:
    panel = make_panel(qtbot, controller, "initial")
    editor = panel.blocks()[0].editor
    editor.start_edit(0)
    qtbot.waitUntil(lambda: editor._field.isVisible())
    text = LONG_TEXTS[1] + "final suffix"
    editor.set_editor_text(text)
    qapp.processEvents()
    assert editor._field.geometry().right() < editor.width()
    if commit == "enter":
        qtbot.keyClick(editor._field, Qt.Key.Key_Return)
    else:
        editor._field.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
    assert controller.segments(KEY) == (text,)
    assert not editor.has_active()
