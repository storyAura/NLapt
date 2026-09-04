"""Regression tests for the resize-crash purge and the real status indicators.

Covers:
- No ``QGraphicsOpacityEffect`` construction remains in widgets that live
  through a window resize (the native-crash source: effect buffers re-render
  during resizes). Drag-time effects in the chip/sentence editors are exempt
  (the mouse is captured during a drag - no resize can happen).
- The bottom-left live clock (system year + time).
- The real save-state indicator (spec 2.3: 已保存 / 保存中 / 保存失败 + 未保存 n).
- The file-panel footer 未保存 indicator is live (was a prototype legend).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nlapt_gui.widgets.file_panel import FilePanel
from nlapt_gui.widgets.status_bar import (
    SAVE_STATE_FAILED,
    SAVE_STATE_SAVED,
    SAVE_STATE_SAVING,
    StatusBar,
)

K1 = "0001.png"

_WIDGETS_DIR = Path(__file__).resolve().parents[2] / "nlapt_gui" / "widgets"
# Widgets that exist while the user resizes the window: a graphics effect on
# any of them re-renders through an effect buffer and hard-crashes Qt.
_RESIZE_SENSITIVE = (
    "main_window.py",
    "preview_panel.py",
    "editor_panel.py",
    "toast.py",
    "collapsible.py",
    "status_bar.py",
    "file_panel.py",
    "tools_panel.py",
    "title_bar.py",
    "toolbar_rail.py",
    "window_chrome.py",
)


class TestNoResizeCrashEffects:
    @pytest.mark.parametrize("name", _RESIZE_SENSITIVE)
    def test_no_graphics_effect_constructed(self, name: str) -> None:
        source = (_WIDGETS_DIR / name).read_text(encoding="utf-8")
        assert "QGraphicsOpacityEffect(" not in source, (
            f"{name} constructs a QGraphicsOpacityEffect - this crashes Qt "
            "during native window resizes; use windowOpacity or painter "
            "opacity instead"
        )

    def test_window_fade_uses_window_opacity(self, qtbot, controller) -> None:
        from nlapt_gui.theme.manager import ThemeManager
        from nlapt_gui.theme.tokens import DEFAULT_THEME
        from nlapt_gui.widgets.main_window import MainWindow

        manager = ThemeManager(persist=False)
        manager.apply(DEFAULT_THEME)
        window = MainWindow(controller, manager)
        qtbot.addWidget(window)
        window.show()
        window.finish_fade_in()
        assert window.windowOpacity() == pytest.approx(1.0)
        assert window.graphicsEffect() is None

    def test_preview_fade_is_paint_level(self, qtbot, controller) -> None:
        from nlapt_gui.widgets.preview_panel import PreviewPanel

        panel = PreviewPanel(controller)
        qtbot.addWidget(panel)
        view = panel.single_view
        view.finish_image_fade()
        assert view._fade_alpha == pytest.approx(1.0)
        assert view.graphicsEffect() is None

    def test_toolbar_shows_and_hides_without_effect(self, qtbot, controller) -> None:
        from nlapt_gui.widgets.editor_panel import SegmentToolbar

        toolbar = SegmentToolbar()
        qtbot.addWidget(toolbar)
        assert toolbar.graphicsEffect() is None
        toolbar.show_with_label("第 1 段")
        assert toolbar.is_active()
        toolbar.dismiss()
        assert not toolbar.is_active()


class TestClock:
    def test_clock_shows_year_and_time(self, qtbot, controller) -> None:
        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        bar.tick_clock()
        text = bar.clock_label.text()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text), text

    def test_clock_timer_running(self, qtbot, controller) -> None:
        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        assert bar._clock_timer.isActive()


class TestSaveStateIndicator:
    def test_clean_dataset_shows_saved(self, qtbot, controller) -> None:
        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        assert bar.save_state_text() == SAVE_STATE_SAVED

    def test_dirty_count_and_save_round_trip(self, qtbot, controller) -> None:
        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        controller.set_current(K1)
        controller.set_caption(K1, "edited text", "编辑")
        assert bar.save_state_text() == "未保存 1"
        with qtbot.waitSignal(controller.files_saved, timeout=2000):
            controller.save_current()
        assert bar.save_state_text() == SAVE_STATE_SAVED

    def test_saving_and_failed_states(self, qtbot, controller) -> None:
        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        controller.save_state_changed.emit("saving")
        assert bar.save_state_text() == SAVE_STATE_SAVING
        controller.save_state_changed.emit("failed")
        assert bar.save_state_text() == SAVE_STATE_FAILED
        controller.save_state_changed.emit("saved")
        assert bar.save_state_text() == SAVE_STATE_SAVED

    def test_real_save_failure_reports_failed(self, qtbot, controller, monkeypatch) -> None:
        from nlapt.core.errors import StorageError

        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        controller.set_current(K1)
        controller.set_caption(K1, "will fail", "编辑")

        def boom(_key: str) -> None:
            raise StorageError("disk full")

        monkeypatch.setattr(controller.app, "save", boom)
        with qtbot.waitSignal(controller.save_state_changed, timeout=2000):
            controller.save_current()
        # first emission is "saving"; wait until the failure lands
        qtbot.waitUntil(lambda: bar.save_state_text() == SAVE_STATE_FAILED, timeout=2000)


class TestFilePanelUnsavedIndicator:
    def test_hidden_when_clean_and_live_when_dirty(self, qtbot, controller) -> None:
        panel = FilePanel(controller)
        qtbot.addWidget(panel)
        panel.show()
        assert not panel.unsaved_label.isVisibleTo(panel)
        controller.set_caption(K1, "dirty now", "编辑")
        assert panel.unsaved_label.isVisibleTo(panel)
        assert panel.unsaved_label.text() == "未保存 1"
        with qtbot.waitSignal(controller.files_saved, timeout=2000):
            controller.set_current(K1)
            controller.save_current()
        assert not panel.unsaved_label.isVisibleTo(panel)


class TestBatchProgressLabel:
    """v1.7: the status bar surfaces live 推标 progress."""

    def test_progress_shows_and_finish_hides(self, qtbot, controller) -> None:
        from nlapt_gui.widgets.status_bar import StatusBar

        bar = StatusBar(controller)
        qtbot.addWidget(bar)
        assert not bar.batch_label.isVisibleTo(bar)
        controller.batch_progress.emit("推标(LLM) · 2 张", 1, 2)
        assert bar.batch_label.isVisibleTo(bar)
        assert bar.batch_label.text() == "推标(LLM) · 2 张 1/2"
        controller.batch_finished.emit("推标(LLM) · 2 张", None)
        assert not bar.batch_label.isVisibleTo(bar)
