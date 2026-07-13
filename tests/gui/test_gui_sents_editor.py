"""Tests for nlapt_gui.widgets.sents_editor (sentences mode rows)."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import QLabel, QPushButton

from nlapt_gui.widgets.sents_editor import (
    ADD_SENT_TEXT,
    ActiveSentRow,
    SentsEditor,
    pad2,
)

KEY = "0001.png"
SEGS = ("1girl", "solo", "long hair", "少女站在樱花树下。masterpiece")


def make_editor(qtbot, controller, key: str = KEY) -> SentsEditor:
    editor = SentsEditor(controller, key)
    qtbot.addWidget(editor)
    editor.resize(520, 400)
    editor.show()
    return editor


def blur(area) -> None:
    area.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))


class TestRows:
    def test_pad2(self) -> None:
        assert pad2(1) == "01"
        assert pad2(12) == "12"

    def test_rows_show_comma_segments_with_numbers(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        rows = editor.rows()
        assert tuple(row.segment_text() for row in rows) == SEGS
        badges = [row.findChild(QLabel, "segBadge").text() for row in rows]
        assert badges == ["01", "02", "03", "04"]

    def test_first_click_selects_second_click_edits(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        label = editor.rows()[1].findChild(QLabel, "sentText")
        qtbot.waitUntil(lambda: label.width() > 10, timeout=2000)
        # First click: pure selection (spec module 2) — no inline editor yet.
        qtbot.mouseClick(label, Qt.MouseButton.LeftButton)
        assert not editor.has_active()
        assert editor.selected_index() == 1
        assert editor.active_text() == "solo"
        selected_label = editor.rows()[1].findChild(QLabel, "sentText")
        assert selected_label.property("selected") == "true"
        # Second click on the selected row opens the multi-line editor.
        qtbot.waitUntil(lambda: selected_label.width() > 10, timeout=2000)
        qtbot.mouseClick(selected_label, Qt.MouseButton.LeftButton)
        assert editor.has_active()
        assert editor.edit_index() == 1
        assert editor.editor_text() == "solo"
        assert isinstance(editor._field, ActiveSentRow)
        badge = editor._field.findChild(QLabel, "segBadgeActive")
        assert badge.text() == "02"

    def test_insert_row_badge_is_plus(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_insert_end()
        badge = editor._field.findChild(QLabel, "segBadgeActive")
        assert badge.text() == "+"


class TestEditing:
    def test_enter_commits(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(0)
        editor.set_editor_text("1boy")
        qtbot.keyClick(editor._field.area, Qt.Key.Key_Return)
        assert controller.segments(KEY)[0] == "1boy"
        assert controller.history.entries(KEY)[0].label == "编辑分段「1girl」"
        assert not editor.has_active()

    def test_shift_enter_inserts_newline(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(0)
        qtbot.keyClick(
            editor._field.area, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier
        )
        assert editor.has_active()  # still editing
        assert "\n" in editor.editor_text()

    def test_escape_cancels(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        before = controller.record(KEY).text
        editor.start_edit(0)
        editor.set_editor_text("nope")
        qtbot.keyClick(editor._field.area, Qt.Key.Key_Escape)
        assert controller.record(KEY).text == before
        assert not editor.has_active()

    def test_blur_commits(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(2)
        editor.set_editor_text("very long hair")
        blur(editor._field.area)
        assert controller.segments(KEY)[2] == "very long hair"

    def test_delete_button_removes_row(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        row = editor.rows()[1]
        button = row.findChild(QPushButton)
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        assert "solo" not in controller.segments(KEY)
        assert controller.history.entries(KEY)[0].label == "删除分段「solo」"

    def test_add_sent_button_inserts_at_end(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        add = next(b for b in editor.findChildren(QPushButton) if b.text() == ADD_SENT_TEXT)
        qtbot.mouseClick(add, Qt.MouseButton.LeftButton)
        assert editor.is_inserting()
        editor.set_editor_text("closing sentence")
        editor.commit_active()
        assert controller.segments(KEY)[-1] == "closing sentence"
        assert controller.history.entries(KEY)[0].label == "插入片段「closing sent…」"

    def test_area_autosizes_with_content(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(0)
        area = editor._field.area
        short_height = area.height()
        editor.set_editor_text("line1\nline2\nline3\nline4\nline5")
        assert area.height() > short_height


class TestReorderAndToolbar:
    def test_reorder_shares_chip_semantics(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.reorder(0, 2)
        assert controller.segments(KEY)[:3] == ("solo", "1girl", "long hair")
        assert controller.history.entries(KEY)[0].label == "拖拽排序"

    def test_reorder_to_end(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.reorder(0, len(SEGS))
        assert controller.segments(KEY)[-1] == "1girl"

    def test_toolbar_split_uses_textarea_cursor(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(2)  # "long hair"
        cursor = editor._field.area.textCursor()
        cursor.setPosition(4)
        editor._field.area.setTextCursor(cursor)
        editor.toolbar_split()
        assert controller.segments(KEY)[2:4] == ("long", "hair")
        assert controller.history.entries(KEY)[0].label == "分段"

    def test_toolbar_delete_active_row(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(1)
        editor.toolbar_delete()
        assert "solo" not in controller.segments(KEY)
        assert not editor.has_active()

    def test_toolbar_insert_opens_after_current(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(1)
        editor.toolbar_insert()
        assert editor.is_inserting()
        assert editor.insert_pos() == 2
        assert editor.active_label() == "插入新段 · 第 3 位"
