"""BatchProgressDialog: show/update/close lifecycle, speed & ETA math, cancel.

The dialog is driven purely by AppController signals, so most tests emit
them directly with an injected fake clock (no sleeps); one end-to-end test
runs a real ``run_caption_batch``.
"""

from __future__ import annotations

import pytest

from nlapt_gui.widgets.batch_progress_dialog import (
    BTN_CANCEL,
    BTN_CANCELLING,
    VALUE_PLACEHOLDER,
    BatchProgressDialog,
    format_duration,
)

DESC = "推标(LLM) · 4 张"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def dialog(qtbot, controller) -> tuple[BatchProgressDialog, FakeClock]:
    clock = FakeClock()
    dlg = BatchProgressDialog(controller, clock=clock)
    qtbot.addWidget(dlg)
    return dlg, clock


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "00:00"),
            (5, "00:05"),
            (65, "01:05"),
            (59.6, "01:00"),
            (3600, "1:00:00"),
            (3661, "1:01:01"),
            (-3, "00:00"),
        ],
    )
    def test_formats(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected


class TestLifecycle:
    def test_shows_on_started_with_reset_state(self, dialog, controller) -> None:
        dlg, _clock = dialog
        assert not dlg.isVisible()
        controller.batch_started.emit(DESC, 4)
        assert dlg.isVisible()
        assert dlg.description_label.text() == DESC
        assert dlg.progress_bar.maximum() == 4
        assert dlg.progress_bar.value() == 0
        assert dlg.percent_label.text() == "0%"
        assert dlg.progress_value.text() == "0 / 4"
        assert dlg.speed_value.text() == VALUE_PLACEHOLDER
        assert dlg.eta_value.text() == VALUE_PLACEHOLDER
        assert dlg.cancel_button.isEnabled()
        assert dlg.cancel_button.text() == BTN_CANCEL

    def test_progress_updates_bar_percent_speed_eta(self, dialog, controller) -> None:
        dlg, clock = dialog
        controller.batch_started.emit(DESC, 4)
        clock.now += 30
        controller.batch_progress.emit(DESC, 1, 4)
        assert dlg.progress_bar.value() == 1
        assert dlg.percent_label.text() == "25%"
        assert dlg.progress_value.text() == "1 / 4"
        assert dlg.elapsed_value.text() == "00:30"
        # 1 item / 30s -> 2.0 张/分; 3 remaining -> 90s.
        assert dlg.speed_value.text() == "2.0 张/分"
        assert dlg.eta_value.text() == "01:30"

    def test_elapsed_ticks_without_progress(self, dialog, controller) -> None:
        dlg, clock = dialog
        controller.batch_started.emit(DESC, 4)
        assert dlg._ticker.isActive()
        clock.now += 61
        dlg.refresh_stats()  # what the 1s ticker calls
        assert dlg.elapsed_value.text() == "01:01"
        assert dlg.speed_value.text() == VALUE_PLACEHOLDER

    def test_closes_on_finish_and_stops_ticker(self, dialog, controller) -> None:
        dlg, _clock = dialog
        controller.batch_started.emit(DESC, 4)
        controller.batch_finished.emit(DESC, None)
        assert not dlg.isVisible()
        assert not dlg._ticker.isActive()

    def test_ignores_text_op_batches(self, dialog, controller) -> None:
        dlg, _clock = dialog
        controller.batch_finished.emit("查找替换", None)
        assert not dlg.isVisible()
        untouched = dlg.progress_bar.value()  # Qt's pristine value (-1)
        controller.batch_progress.emit("查找替换", 1, 2)
        assert dlg.progress_bar.value() == untouched
        assert dlg.progress_value.text() == VALUE_PLACEHOLDER

    def test_background_close_keeps_updating_and_reshows(
        self, dialog, controller
    ) -> None:
        dlg, _clock = dialog
        controller.batch_started.emit(DESC, 4)
        dlg.background_button.click()
        assert not dlg.isVisible()
        controller.batch_progress.emit(DESC, 1, 4)  # batch keeps running
        assert not dlg.isVisible()
        assert dlg.progress_value.text() == "1 / 4"
        controller.batch_finished.emit(DESC, None)
        controller.batch_started.emit("推标(本地模型) · 2 张", 2)
        assert dlg.isVisible()
        assert dlg.progress_bar.maximum() == 2


class TestCancel:
    def test_cancel_delegates_and_locks_button(
        self, dialog, controller, monkeypatch
    ) -> None:
        dlg, _clock = dialog
        calls: list[bool] = []
        monkeypatch.setattr(controller, "cancel_batch", lambda: calls.append(True))
        controller.batch_started.emit(DESC, 4)
        dlg.cancel_button.click()
        assert calls == [True]
        assert dlg.cancel_button.text() == BTN_CANCELLING
        assert not dlg.cancel_button.isEnabled()
        # A fresh batch re-arms the button.
        controller.batch_started.emit(DESC, 4)
        assert dlg.cancel_button.isEnabled()
        assert dlg.cancel_button.text() == BTN_CANCEL


class TestEndToEnd:
    def test_real_caption_batch_drives_dialog(self, qtbot, dialog, controller) -> None:
        dlg, _clock = dialog
        seen: list[tuple[str, int, bool]] = []
        # Connected after the dialog, so the dialog's slot has already run.
        controller.batch_started.connect(
            lambda d, t: seen.append((d, t, dlg.isVisible()))
        )
        with qtbot.waitSignal(controller.batch_finished, timeout=4000):
            assert controller.run_caption_batch(
                ("0001.png", "0002.png"),
                lambda key, path: f"cap {key}",
                description="推标(LLM) · 2 张",
                history_label="推标(LLM)",
                engine="llm",
                concurrency=2,
            )
        assert seen == [("推标(LLM) · 2 张", 2, True)]
        assert not dlg.isVisible()  # closed itself when the batch finished
        assert dlg.progress_bar.value() == 2


def test_qss_styles_progress_bar() -> None:
    from nlapt_gui.theme.qss import build_qss
    from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES

    qss = build_qss(THEMES[DEFAULT_THEME])
    assert "QProgressBar" in qss
    assert "QProgressBar::chunk" in qss
