"""Tests for nlapt_gui.widgets.text_editor (free-text mode + selection bar)."""

from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtGui import QFocusEvent, QTextCursor

from nlapt_gui.widgets.text_editor import (
    AREA_HEIGHT_MULTI,
    AREA_HEIGHT_SINGLE,
    TEXT_PLACEHOLDER,
    TOAST_DELETED_SEL,
    TOAST_REPLACED_SEL,
    TOAST_TIDIED_SEL,
    TextEditor,
    delete_selection_text,
    tidy_selection_text,
)

KEY = "0002.png"
CAPTION = "1girl, school uniform, short hair, best quality"


def make_editor(qtbot, controller, *, multi: bool = False) -> TextEditor:
    editor = TextEditor(controller, KEY, multi=multi)
    qtbot.addWidget(editor)
    editor.resize(520, 320)
    editor.show()
    return editor


def select(editor: TextEditor, start: int, end: int) -> None:
    cursor = editor.area.textCursor()
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
    editor.area.setTextCursor(cursor)


class TestPureRules:
    def test_delete_collapses_double_commas(self) -> None:
        caption = "1girl, solo, long hair"
        start = caption.index("solo")
        end = start + len("solo")
        assert delete_selection_text(caption, start, end) == "1girl, long hair"

    def test_delete_strips_leading_comma(self) -> None:
        caption = "1girl, solo"
        assert delete_selection_text(caption, 0, 5) == "solo"

    def test_tidy_normalizes_commas_and_spaces(self) -> None:
        caption = "a ,b,   c , d"
        assert tidy_selection_text(caption, 0, len(caption)) == "a, b, c, d"

    def test_tidy_drops_empty_fragments(self) -> None:
        caption = "x,, y,"
        assert tidy_selection_text(caption, 0, len(caption)) == "x, y"


class TestArea:
    def test_shows_caption_and_placeholder(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        assert editor.area.toPlainText() == CAPTION
        assert editor.area.placeholderText() == TEXT_PLACEHOLDER
        assert editor.area.height() == AREA_HEIGHT_SINGLE

    def test_multi_mode_uses_compact_height(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller, multi=True)
        assert editor.area.height() == AREA_HEIGHT_MULTI

    def test_typing_updates_core_without_history(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        history_before = len(controller.history.entries(KEY))
        cursor = editor.area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        editor.area.setTextCursor(cursor)
        editor.area.insertPlainText(", extra")
        assert controller.record(KEY).text == CAPTION + ", extra"
        assert controller.record(KEY).dirty
        assert len(controller.history.entries(KEY)) == history_before

    def test_blur_pushes_free_edit_history(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.area.insertPlainText("x")
        editor.area.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
        assert controller.history.entries(KEY)[0].label == "自由编辑"

    def test_blur_without_change_pushes_nothing(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        history_before = len(controller.history.entries(KEY))
        editor.area.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
        assert len(controller.history.entries(KEY)) == history_before

    def test_refresh_syncs_external_change(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        controller.set_caption(KEY, "brand new caption", "测试")
        editor.refresh()
        assert editor.area.toPlainText() == "brand new caption"


class TestSelectionBar:
    def test_selection_shows_bar_with_count(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        assert not editor.selection_bar_visible()
        select(editor, 0, 5)
        assert editor.selection_bar_visible()
        assert editor.selection() == (0, 5)
        assert editor._sel_info.text() == "已选中 5 字符"

    def test_collapse_hides_bar(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        select(editor, 0, 5)
        select(editor, 3, 3)
        assert not editor.selection_bar_visible()
        assert editor.selection() is None

    def test_replace_selection(self, qtbot, controller, toasts) -> None:
        editor = make_editor(qtbot, controller)
        select(editor, 0, 5)  # "1girl"
        editor.set_replacement_text("1boy")
        editor.replace_selection()
        assert controller.record(KEY).text == "1boy" + CAPTION[5:]
        assert controller.history.entries(KEY)[0].label == "替换选中片段"
        assert (TOAST_REPLACED_SEL, "ok") in toasts
        assert editor._replace_input.text() == ""
        assert not editor.selection_bar_visible()

    def test_delete_selection_applies_comma_rules(self, qtbot, controller, toasts) -> None:
        editor = make_editor(qtbot, controller)
        start = CAPTION.index("school uniform")
        select(editor, start, start + len("school uniform"))
        editor.delete_selection()
        assert controller.record(KEY).text == "1girl, short hair, best quality"
        assert controller.history.entries(KEY)[0].label == "删除选中片段"
        assert (TOAST_DELETED_SEL, "ok") in toasts

    def test_delete_leading_selection_strips_comma(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        select(editor, 0, 5)  # "1girl"
        editor.delete_selection()
        assert controller.record(KEY).text == CAPTION[7:]  # no leading ", "

    def test_tidy_selection(self, qtbot, controller, toasts) -> None:
        controller.set_caption(KEY, "a ,b,   c", "测试")
        editor = make_editor(qtbot, controller)
        select(editor, 0, 9)
        editor.tidy_selection()
        assert controller.record(KEY).text == "a, b, c"
        assert controller.history.entries(KEY)[0].label == "润色选中片段"
        assert (TOAST_TIDIED_SEL, "info") in toasts

    def test_actions_noop_without_selection(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        before = controller.record(KEY).text
        editor.replace_selection()
        editor.delete_selection()
        editor.tidy_selection()
        assert controller.record(KEY).text == before
