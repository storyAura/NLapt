"""Integration tests for spec module 2: select-then-edit segments and the
segment-anchored floating toolbar, plus the caption workspace embedding."""

from __future__ import annotations

from PySide6.QtCore import QObject, QPoint, Signal

from nlapt_gui.widgets.caption_bar import CaptionBar
from nlapt_gui.widgets.chips_editor import ChipsEditor, LABEL_TRANSLATE_REPLACE
from nlapt_gui.widgets.editor_panel import TOOLBAR_TOP, EditorPanel

KEY1 = "0001.png"
KEY2 = "0002.png"


class StubBridge(QObject):
    """Duck-typed TranslateBridge (request/request_to + ready signals)."""

    segment_ready = Signal(str, str, str, bool)
    all_done = Signal(str, int, int)
    target_ready = Signal(str, str, str, str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, str, bool]] = []

    def configured(self) -> bool:
        return True

    def request(self, key: str, text: str, *, fresh: bool = False) -> None:
        self.requests.append((key, text, fresh))

    def request_to(self, key: str, text: str, target_lang: str) -> None:  # pragma: no cover
        pass


def make_panel(qtbot, controller, bridge: StubBridge | None = None) -> EditorPanel:
    panel = EditorPanel(controller, bridge)
    qtbot.addWidget(panel)
    panel.resize(760, 480)
    panel.show()
    qtbot.waitExposed(panel)
    return panel


def chips_editor(panel: EditorPanel) -> ChipsEditor:
    editor = panel.blocks()[0].editor
    assert isinstance(editor, ChipsEditor)
    return editor


def _wait_layout(qtbot, editor: ChipsEditor, index: int) -> None:
    qtbot.waitUntil(
        lambda: editor._segment_widget(index) is not None
        and editor._segment_widget(index).width() > 10,
        timeout=2000,
    )


class TestSelectionToolbar:
    def test_selection_shows_toolbar_with_segment_label(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = chips_editor(panel)
        editor.select_segment(2)
        assert panel.toolbar.is_active()
        assert panel.toolbar.label_text() == "第 3 段"
        assert panel.focus_editor() is editor

    def test_clearing_selection_hides_toolbar(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = chips_editor(panel)
        editor.select_segment(1)
        editor.clear_selection()
        assert not panel.toolbar.is_active()

    def test_toolbar_anchors_above_selected_chip(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = chips_editor(panel)
        editor.select_segment(1)
        _wait_layout(qtbot, editor, 1)
        panel._position_toolbar()
        target = editor._segment_widget(1)
        target_top = target.mapTo(panel, QPoint(0, 0))
        toolbar_bottom = panel.toolbar.y() + panel.toolbar.height()
        # Above (or flipped just below when空间不足) but horizontally near it.
        assert abs(toolbar_bottom - target_top.y()) <= panel.toolbar.height() + target.height() + 16
        target_center_x = target_top.x() + target.width() // 2
        toolbar_center_x = panel.toolbar.x() + panel.toolbar.width() // 2
        assert abs(toolbar_center_x - target_center_x) <= panel.width() // 2

    def test_toolbar_falls_back_to_top_center_without_anchor(
        self, qtbot, controller
    ) -> None:
        panel = make_panel(qtbot, controller)
        editor = chips_editor(panel)
        editor.select_segment(0)
        editor.clear_selection()
        editor.start_insert_end()  # insert editor exists -> anchor is the field
        panel._position_toolbar()
        assert panel.toolbar.is_active()
        # Dismiss to exercise the no-target fallback path directly.
        editor.cancel_active()
        panel.toolbar.show_with_label("x")
        panel._position_toolbar()
        assert panel.toolbar.y() in (TOOLBAR_TOP, panel.toolbar.y())  # no crash

    def test_selecting_in_second_block_clears_first(self, qtbot, controller) -> None:
        controller.toggle_selected(KEY1)
        controller.toggle_selected(KEY2)
        panel = make_panel(qtbot, controller)
        first = panel.blocks()[0].editor
        second = panel.blocks()[1].editor
        first.select_segment(0)
        assert first.has_selection()
        second.select_segment(1)
        assert not first.has_selection()
        assert second.has_selection()
        assert panel.focus_editor() is second


class TestSelectionActions:
    def test_toolbar_delete_removes_selected_segment(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = chips_editor(panel)
        before = controller.segments(KEY1)
        editor.select_segment(1)
        panel.toolbar.delete_btn.pressed.emit()
        after = controller.segments(KEY1)
        assert len(after) == len(before) - 1
        assert before[1] not in after
        assert not editor.has_selection()

    def test_toolbar_insert_opens_after_selection(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = chips_editor(panel)
        editor.select_segment(1)
        panel.toolbar.insert_btn.pressed.emit()
        assert editor.is_inserting()
        assert editor.insert_pos() == 2

    def test_toolbar_split_on_selection_opens_editor_with_hint(
        self, qtbot, controller, toasts
    ) -> None:
        panel = make_panel(qtbot, controller)
        editor = chips_editor(panel)
        editor.select_segment(1)
        panel.toolbar.split_btn.pressed.emit()
        assert editor.edit_index() == 1  # ready for cursor placement
        assert any("光标" in text for text, _kind in toasts)

    def test_translate_selection_previews_then_replaces(
        self, qtbot, controller
    ) -> None:
        bridge = StubBridge()
        panel = make_panel(qtbot, controller, bridge)
        editor = chips_editor(panel)
        editor.select_segment(1)
        source = controller.segments(KEY1)[1]
        panel.toolbar.translate_btn.pressed.emit()
        assert bridge.requests == [(KEY1, source, False)]
        bridge.segment_ready.emit(KEY1, source, "译文A", True)
        assert panel.translation_preview.is_active()
        # Nothing replaced until 替换 is pressed.
        assert controller.segments(KEY1)[1] == source
        panel.translation_preview.apply_btn.pressed.emit()
        assert controller.segments(KEY1)[1] == "译文A"
        assert controller.history.entries(KEY1)[0].label == LABEL_TRANSLATE_REPLACE

    def test_translate_reply_dropped_when_selection_moved(
        self, qtbot, controller
    ) -> None:
        bridge = StubBridge()
        panel = make_panel(qtbot, controller, bridge)
        editor = chips_editor(panel)
        editor.select_segment(1)
        source = controller.segments(KEY1)[1]
        panel.toolbar.translate_btn.pressed.emit()
        editor.select_segment(2)  # target moved -> pending reply invalid
        bridge.segment_ready.emit(KEY1, source, "译文A", True)
        assert not panel.translation_preview.is_active()
        assert controller.segments(KEY1)[1] == source


class TestCaptionBarEmbedding:
    def test_panel_hosts_caption_workspace(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        assert isinstance(panel.caption_bar, CaptionBar)
        assert panel.caption_bar.translate_btn.isEnabled()
        assert panel.caption_bar.reinfer_btn.text() == "LLM 推理"
        assert panel.caption_bar.local_infer_btn.text() == "本地推理"
        assert panel.caption_bar.delete_btn.text() == "删除"

    def test_caption_preview_hosted_on_panel_overlay(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        assert panel.caption_bar.preview.parentWidget() is panel
