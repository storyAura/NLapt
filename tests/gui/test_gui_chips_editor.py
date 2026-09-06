"""Tests for nlapt_gui.widgets.chips_editor (chips mode + shared base)."""

from __future__ import annotations

from PySide6.QtCore import (
    QAbstractAnimation,
    QEvent,
    QMimeData,
    QParallelAnimationGroup,
    QPoint,
    QPointF,
    QRect,
    Qt,
)
from PySide6.QtGui import QDrag, QDropEvent, QFocusEvent
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from nlapt_gui import anim
from nlapt_gui.widgets.chips_editor import (
    ADD_CHIP_TEXT,
    EMPTY_STATE_TEXT,
    INSERT_PLACEHOLDER,
    MIME_SEGMENT,
    TOAST_FINISH_INSERT_FIRST,
    TOAST_PUT_CURSOR,
    TOAST_SPLIT_DONE,
    ChipsEditor,
    short_label,
    split_segment_text,
)

KEY = "0001.png"
# demo caption: "1girl, solo, long hair, 少女站在樱花树下。masterpiece"
SEGS = ("1girl", "solo", "long hair", "少女站在樱花树下。masterpiece")
EMPTY_KEY = "10_concept/0004.png"
# No commas: join_chips keeps this as a single wrapping chip.
LONG_CHIP = (
    "and a curved pink ahoge protruding from the top of her head. "
    "Her eyes feature red irises with round dark pupils. A dark intricate "
    "mark is located on her upper chest. Her outfit includes a dark navy "
    "headband with a side bow"
)
LONG_CAPTION = (
    f"side locks, {LONG_CHIP}, light blue drop earrings, "
    "choker, ribbon"
)


def make_editor(qtbot, controller, key: str = KEY) -> ChipsEditor:
    editor = ChipsEditor(controller, key)
    qtbot.addWidget(editor)
    editor.resize(500, 240)
    editor.show()
    return editor


def blur(field) -> None:
    field.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))


class TestHelpers:
    def test_short_label_truncates_to_12_chars(self) -> None:
        assert short_label("long hair") == "long hair"
        assert short_label("123456789012") == "123456789012"
        assert short_label("1234567890123") == "123456789012…"

    def test_split_segment_text_strips_commas(self) -> None:
        assert split_segment_text("long hair", 4) == ("long", "hair")
        assert split_segment_text("a, b", 2) == ("a", "b")
        assert split_segment_text("a，b", 2) == ("a", "b")  # full-width comma
        assert split_segment_text("ab", 0) is None
        assert split_segment_text("ab", 2) is None
        assert split_segment_text(", b", 1) is None  # left side empty


class TestChipsView:
    def test_builds_chips_from_segments(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        assert tuple(chip.chip_text for chip in editor.chips()) == SEGS
        assert not editor.has_active()

    def test_empty_state_label(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller, EMPTY_KEY)
        assert editor.chips() == ()
        labels = [lbl.text() for lbl in editor.findChildren(QLabel)]
        assert EMPTY_STATE_TEXT in labels
        buttons = [b.text() for b in editor.findChildren(QPushButton)]
        assert ADD_CHIP_TEXT in buttons

    def test_first_click_selects_second_click_edits(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        chip = editor.chips()[1]
        qtbot.waitUntil(lambda: chip.width() > 20, timeout=2000)
        # First click: pure selection (spec module 2) — no inline editor yet.
        qtbot.mouseClick(chip, Qt.MouseButton.LeftButton, pos=QPoint(14, chip.height() // 2))
        assert not editor.has_active()
        assert editor.selected_index() == 1
        assert editor.has_selection()
        assert editor.active_label() == "第 2 段"
        assert editor.active_text() == "solo"
        assert controller.current_key == KEY
        # The rebuilt chip carries the selected visual state.
        selected_chip = editor.chips()[1]
        assert selected_chip.property("chipSelected") == "true"
        # Second click on the selected chip opens the inline editor.
        qtbot.waitUntil(lambda: selected_chip.width() > 20, timeout=2000)
        qtbot.mouseClick(
            selected_chip,
            Qt.MouseButton.LeftButton,
            pos=QPoint(14, selected_chip.height() // 2),
        )
        assert editor.has_active()
        assert editor.edit_index() == 1
        assert editor.editor_text() == "solo"
        assert editor.selected_index() is None

    def test_inline_editor_auto_width_grows(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(0)
        qtbot.waitUntil(lambda: editor._field.isVisible())
        narrow = editor._field.width()
        editor.set_editor_text("a much longer chip text value")
        qtbot.waitUntil(lambda: editor._field.width() > narrow)


class TestEditCommit:
    def test_enter_commits_with_edit_label(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(1)
        editor.set_editor_text("short hair")
        qtbot.keyClick(editor._field, Qt.Key.Key_Return)
        assert controller.segments(KEY)[1] == "short hair"
        assert controller.history.entries(KEY)[0].label == "编辑分段「solo」"
        assert not editor.has_active()
        assert editor._field is None

    def test_commit_with_commas_splits_into_chips(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(0)
        editor.set_editor_text("red, blue")
        editor.commit_active()
        assert controller.segments(KEY)[:2] == ("red", "blue")
        assert len(controller.segments(KEY)) == len(SEGS) + 1

    def test_escape_cancels_edit(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        before = controller.record(KEY).text
        editor.start_edit(0)
        editor.set_editor_text("changed")
        qtbot.keyClick(editor._field, Qt.Key.Key_Escape)
        assert controller.record(KEY).text == before
        assert not editor.has_active()

    def test_blur_commits(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(0)
        editor.set_editor_text("1girl v2")
        blur(editor._field)
        assert controller.segments(KEY)[0] == "1girl v2"

    def test_empty_commit_deletes_segment(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(1)
        editor.set_editor_text("   ")
        editor.commit_active()
        assert "solo" not in controller.segments(KEY)
        assert controller.history.entries(KEY)[0].label == "删除分段"

    def test_unchanged_commit_pushes_no_history(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        history_before = len(controller.history.entries(KEY))
        editor.start_edit(0)
        editor.commit_active()
        assert len(controller.history.entries(KEY)) == history_before


class TestRemoveAndInsert:
    def test_remove_button_deletes_with_label(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        chip = editor.chips()[3]  # the long CJK segment
        button = chip.findChild(QPushButton)
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        assert len(controller.segments(KEY)) == 3
        # 「x」 truncated to 12 chars + …
        assert controller.history.entries(KEY)[0].label == "删除分段「少女站在樱花树下。mas…」"

    def test_add_chip_opens_insert_editor(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        add = next(b for b in editor.findChildren(QPushButton) if b.text() == ADD_CHIP_TEXT)
        qtbot.mouseClick(add, Qt.MouseButton.LeftButton)
        assert editor.is_inserting()
        assert editor.insert_pos() == len(SEGS)
        assert editor.active_label() == f"插入新段 · 第 {len(SEGS) + 1} 位"
        assert editor._field.placeholderText() == INSERT_PLACEHOLDER
        # add button hidden while inserting
        assert not any(
            b.text() == ADD_CHIP_TEXT for b in editor.findChildren(QPushButton) if b.isVisible()
        )

    def test_insert_commit_appends_segment(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_insert_end()
        editor.set_editor_text("new tag")
        qtbot.keyClick(editor._field, Qt.Key.Key_Return)
        assert controller.segments(KEY)[-1] == "new tag"
        assert controller.history.entries(KEY)[0].label == "插入片段「new tag」"
        assert not editor.has_active()

    def test_insert_empty_commit_is_cancel(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        before = controller.record(KEY).text
        editor.start_insert_end()
        editor.commit_active()
        assert controller.record(KEY).text == before
        assert not editor.has_active()


class TestReorder:
    def test_reorder_moves_before_target(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.reorder(0, 2)  # drop chip 0 onto chip 2
        assert controller.segments(KEY)[:3] == ("solo", "1girl", "long hair")
        assert controller.history.entries(KEY)[0].label == "拖拽排序"

    def test_reorder_backwards(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.reorder(2, 0)
        assert controller.segments(KEY)[:3] == ("long hair", "1girl", "solo")

    def test_reorder_to_container_end(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.reorder(0, len(SEGS))
        assert controller.segments(KEY)[-1] == "1girl"

    def test_reorder_noops(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        before = controller.record(KEY).text
        editor.reorder(1, 1)
        editor.reorder(None, 2)
        editor.reorder(99, 0)
        assert controller.record(KEY).text == before

    def test_reorder_flip_lands_on_layout(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        anim.set_animations_enabled(True)
        try:
            editor.reorder(0, 2)
            editor.refresh()
            motion = editor._reflow_anim
            assert isinstance(motion, QAbstractAnimation)
            motion.setCurrentTime(motion.duration())
            assert [chip.chip_text for chip in editor.chips()] == list(
                controller.segments(KEY)
            )
            if editor.layout() is not None:
                editor.layout().activate()
            finals = [chip.geometry() for chip in editor.chips()]
            assert all(not rect.isEmpty() for rect in finals)
        finally:
            anim.set_animations_enabled(False)

    def test_drop_defers_commit_until_exec_returns(
        self, qtbot, controller, monkeypatch
    ) -> None:
        editor = make_editor(qtbot, controller)
        qtbot.waitUntil(lambda: editor.chips()[0].width() > 20, timeout=2000)
        source = editor.chips()[0]
        source_id = id(source)
        during: dict[str, object] = {}

        def fake_exec(_self: QDrag, *_args: object, **_kwargs: object) -> Qt.DropAction:
            target = editor.chips()[2]
            mime = QMimeData()
            mime.setData(MIME_SEGMENT, b"0")
            event = QDropEvent(
                QPointF(target.geometry().center()),
                Qt.DropAction.MoveAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            editor.dropEvent(event)
            during["segs"] = controller.segments(KEY)
            during["source_alive"] = source.parent() is editor
            during["same_widget"] = any(id(chip) == source_id for chip in editor.chips())
            return Qt.DropAction.MoveAction

        monkeypatch.setattr(QDrag, "exec", fake_exec)
        editor.begin_drag(0, source)
        assert during["segs"] == SEGS
        assert during["source_alive"] is True
        assert during["same_widget"] is True
        assert controller.segments(KEY)[:3] == ("solo", "1girl", "long hair")
        assert controller.history.entries(KEY)[0].label == "拖拽排序"

    def test_refresh_during_drag_is_deferred(
        self, qtbot, controller, monkeypatch
    ) -> None:
        editor = make_editor(qtbot, controller)
        qtbot.waitUntil(lambda: editor.chips()[0].width() > 20, timeout=2000)
        source = editor.chips()[0]

        def fake_exec(_self: QDrag, *_args: object, **_kwargs: object) -> Qt.DropAction:
            controller.set_caption(KEY, "a, b", "测试")
            editor.refresh()
            assert tuple(chip.chip_text for chip in editor.chips()) == SEGS
            assert editor._refresh_pending is True
            return Qt.DropAction.IgnoreAction

        monkeypatch.setattr(QDrag, "exec", fake_exec)
        editor.begin_drag(0, source)
        assert tuple(chip.chip_text for chip in editor.chips()) == ("a", "b")
        assert editor._refresh_pending is False
        assert editor._drag_index is None

    def test_cancelled_drag_leaves_no_artifacts(
        self, qtbot, controller, monkeypatch
    ) -> None:
        editor = make_editor(qtbot, controller)
        qtbot.waitUntil(lambda: editor.chips()[0].width() > 20, timeout=2000)
        history_before = len(controller.history.entries(KEY))
        before = controller.record(KEY).text

        def fake_exec(_self: QDrag, *_args: object, **_kwargs: object) -> Qt.DropAction:
            return Qt.DropAction.IgnoreAction

        monkeypatch.setattr(QDrag, "exec", fake_exec)
        editor.begin_drag(0, editor.chips()[0])
        assert controller.record(KEY).text == before
        assert len(controller.history.entries(KEY)) == history_before
        assert editor._drag_index is None
        assert editor._pending_drop is None
        for chip in editor.chips():
            assert chip.graphicsEffect() is None
        assert tuple(chip.chip_text for chip in editor.chips()) == SEGS

    def test_reorder_leaves_no_overlap_or_effects(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        qtbot.waitUntil(lambda: editor.chips()[0].width() > 20, timeout=2000)
        anim.set_animations_enabled(True)
        try:
            editor.reorder(0, 2)
            editor.refresh()
            anim.finish_animation(editor._reflow_anim)
            for chip in editor.chips():
                assert chip.graphicsEffect() is None
        finally:
            anim.set_animations_enabled(False)
        editor.refresh()
        editor.resize(500, 240)
        qtbot.waitUntil(
            lambda: editor.chips()[0].geometry().topLeft()
            != editor.chips()[1].geometry().topLeft(),
            timeout=2000,
        )
        chips = editor.chips()
        assert [chip.chip_text for chip in chips] == list(controller.segments(KEY))
        for i, left in enumerate(chips):
            for right in chips[i + 1 :]:
                assert not left.geometry().intersects(right.geometry())

    def test_rebuild_lays_out_chips_without_event_loop(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.resize(900, 400)
        controller.set_caption(KEY, LONG_CAPTION, "测试")
        editor._rebuild()
        chips = editor.chips()
        assert [chip.chip_text for chip in chips] == list(controller.segments(KEY))
        assert all(chip.width() != 100 for chip in chips)
        long_chip = next(chip for chip in chips if chip.chip_text == LONG_CHIP)
        assert long_chip.width() == editor.width()
        assert long_chip.height() > 26

    def test_drop_flip_uses_laid_out_end_rects(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.resize(900, 400)
        controller.set_caption(KEY, LONG_CAPTION, "测试")
        editor.refresh()
        controller.caption_changed.connect(editor.refresh)
        anim.set_animations_enabled(True)
        try:
            editor._drag_index = 1
            editor._pending_drop = (1, 3)
            editor._finish_drag()
            motion = editor._reflow_anim
            assert isinstance(motion, QParallelAnimationGroup)
            ends: list[tuple[QWidget, QRect]] = []
            for i in range(motion.animationCount()):
                child = motion.animationAt(i)
                target = child.targetObject()
                end = QRect(child.endValue())
                assert end != QRect(0, 0, 100, 30)
                ends.append((target, end))
            anim.finish_animation(motion)
            for widget, end in ends:
                assert widget.geometry() == end
        finally:
            anim.set_animations_enabled(False)

    def test_pop_in_keeps_width(self, qtbot) -> None:
        anim.set_animations_enabled(True)
        try:
            widget = QWidget()
            qtbot.addWidget(widget)
            widget.setGeometry(10, 10, 400, 80)
            widget.show()
            motion = anim.pop_in(widget, ms=140)
            assert isinstance(motion, QAbstractAnimation)
            assert motion.startValue().width() == motion.endValue().width() == 400
            assert motion.startValue().height() < motion.endValue().height()
        finally:
            anim.set_animations_enabled(False)


class TestToolbarActions:
    def test_split_at_cursor(self, qtbot, controller, toasts) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(2)  # "long hair"
        editor._field.setCursorPosition(4)
        editor.toolbar_split()
        assert controller.segments(KEY)[2:4] == ("long", "hair")
        assert controller.history.entries(KEY)[0].label == "分段"
        assert (TOAST_SPLIT_DONE, "ok") in toasts
        assert not editor.has_active()

    def test_split_at_edge_warns(self, qtbot, controller, toasts) -> None:
        editor = make_editor(qtbot, controller)
        before = controller.record(KEY).text
        editor.start_edit(0)
        editor._field.setCursorPosition(0)
        editor.toolbar_split()
        assert controller.record(KEY).text == before
        assert (TOAST_PUT_CURSOR, "warn") in toasts
        assert editor.has_active()  # edit stays open

    def test_split_during_insert_warns(self, qtbot, controller, toasts) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_insert_end()
        editor.toolbar_split()
        assert (TOAST_FINISH_INSERT_FIRST, "warn") in toasts
        assert editor.is_inserting()

    def test_toolbar_insert_commits_then_opens_insert(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(0)
        editor.set_editor_text("1girl v2")
        editor.toolbar_insert()
        assert controller.segments(KEY)[0] == "1girl v2"
        assert editor.is_inserting()
        assert editor.insert_pos() == 1
        assert editor.active_label() == "插入新段 · 第 2 位"

    def test_toolbar_delete_removes_active_segment(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(1)
        editor.toolbar_delete()
        assert "solo" not in controller.segments(KEY)
        assert controller.history.entries(KEY)[0].label == "删除分段「solo」"
        assert not editor.has_active()

    def test_toolbar_delete_cancels_insert(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        before = controller.record(KEY).text
        editor.start_insert_end()
        editor.set_editor_text("pending")
        editor.toolbar_delete()
        assert controller.record(KEY).text == before
        assert not editor.has_active()


class TestExternalRefresh:
    def test_refresh_rebuilds_from_controller(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        controller.set_caption(KEY, "a, b", "测试")
        editor.refresh()
        assert tuple(chip.chip_text for chip in editor.chips()) == ("a", "b")

    def test_refresh_closes_out_of_range_edit(self, qtbot, controller) -> None:
        editor = make_editor(qtbot, controller)
        editor.start_edit(3)
        controller.set_caption(KEY, "only one", "测试")
        editor.refresh()
        assert not editor.has_active()
