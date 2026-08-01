"""推标进度窗口 — live progress window for caption batches (LLM / 本地).

A non-modal :class:`CenteredDialog` that pops up whenever a caption batch
starts and closes itself when the batch ends. Driven purely by
:class:`AppController` signals:

- ``batch_started(description, total)`` — reset + show (each batch opens
  centered with the standard dialog fade, like every other sub-window);
- ``batch_progress(description, done, total)`` — bar / count / percent,
  speed (张/分, overall average) and estimated remaining time;
- ``batch_finished`` — close. Text-operation batches never emit
  ``batch_started``, so an ``_active`` flag keeps their ``batch_finished``
  (and late queued progress events) from touching the window.

后台运行 (or Esc / the title-bar close) merely hides the window — the
batch keeps running and the status bar keeps showing progress. 取消推标
delegates to ``controller.cancel_batch()`` (cooperative: completed items
are kept). Wall time comes from an injectable monotonic ``clock`` and the
1s elapsed/ETA tick is a plain QTimer; tests call :meth:`refresh_stats`
directly instead of waiting.
"""

from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt_gui.controller import AppController
from nlapt_gui.widgets.dialogs import CenteredDialog

WINDOW_TITLE = "推标进度"

# Stat captions (top grid row) and value formats.
CAPTION_PROGRESS = "进度"
CAPTION_SPEED = "速度"
CAPTION_ELAPSED = "已用时间"
CAPTION_ETA = "预计剩余"
PROGRESS_FMT = "{done} / {total}"
SPEED_FMT = "{rate:.1f} 张/分"
PERCENT_FMT = "{pct}%"
# Shown until the first item lands (no speed/ETA can be computed yet).
VALUE_PLACEHOLDER = "—"

BTN_BACKGROUND = "后台运行"
BTN_CANCEL = "取消推标"
BTN_CANCELLING = "正在取消…"

DIALOG_MIN_W = 400
# Slim themed bar (percent lives in its own label, not inside the bar).
BAR_HEIGHT_PX = 8
# Elapsed / ETA labels tick once per second while a batch runs.
TICK_INTERVAL_MS = 1000

Clock = Callable[[], float]


def format_duration(seconds: float) -> str:
    """``mm:ss`` under an hour, ``h:mm:ss`` from one hour up (floor at 0)."""
    total = max(0, int(round(seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class BatchProgressDialog(CenteredDialog):
    """Floating 推标 progress: bar + count + speed + elapsed + ETA + cancel."""

    def __init__(
        self,
        controller: AppController,
        parent: QWidget | None = None,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._clock = clock
        self._active = False
        self._started_at = 0.0
        self._done = 0
        self._total = 0

        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(False)
        self.setMinimumWidth(DIALOG_MIN_W)

        # Header: batch description + right-aligned live percent.
        self.description_label = QLabel(self)
        self.description_label.setProperty("secondary", "true")
        self.description_label.setWordWrap(True)
        self.percent_label = QLabel(VALUE_PLACEHOLDER, self)
        self.percent_label.setProperty("mono", "true")
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(self.description_label, 1)
        header.addWidget(self.percent_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(BAR_HEIGHT_PX)

        # Stat grid: muted captions over mono values.
        self.progress_value = QLabel(VALUE_PLACEHOLDER, self)
        self.speed_value = QLabel(VALUE_PLACEHOLDER, self)
        self.elapsed_value = QLabel(VALUE_PLACEHOLDER, self)
        self.eta_value = QLabel(VALUE_PLACEHOLDER, self)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        captions = (CAPTION_PROGRESS, CAPTION_SPEED, CAPTION_ELAPSED, CAPTION_ETA)
        values = (
            self.progress_value,
            self.speed_value,
            self.elapsed_value,
            self.eta_value,
        )
        for column, (caption, value) in enumerate(zip(captions, values)):
            caption_label = QLabel(caption, self)
            caption_label.setProperty("muted", "true")
            value.setProperty("mono", "true")
            grid.addWidget(caption_label, 0, column)
            grid.addWidget(value, 1, column)
            grid.setColumnStretch(column, 1)

        self.background_button = QPushButton(BTN_BACKGROUND, self)
        self.background_button.setProperty("variant", "ghost")
        self.cancel_button = QPushButton(BTN_CANCEL, self)
        self.cancel_button.setProperty("variant", "outline")
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        buttons.addWidget(self.background_button)
        buttons.addWidget(self.cancel_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 14)
        root.setSpacing(10)
        root.addLayout(header)
        root.addWidget(self.progress_bar)
        root.addLayout(grid)
        root.addLayout(buttons)

        self._ticker = QTimer(self)
        self._ticker.setInterval(TICK_INTERVAL_MS)
        self._ticker.timeout.connect(self.refresh_stats)

        controller.batch_started.connect(self._on_started)
        controller.batch_progress.connect(self._on_progress)
        controller.batch_finished.connect(self._on_finished)
        self.background_button.clicked.connect(self.close)
        self.cancel_button.clicked.connect(self._on_cancel)

    # -- controller signal handlers ------------------------------------------------------
    def _on_started(self, description: str, total: int) -> None:
        self._active = True
        self._started_at = self._clock()
        self._done = 0
        self._total = max(0, total)
        self.description_label.setText(description)
        self.progress_bar.setRange(0, max(1, self._total))
        self.progress_bar.setValue(0)
        self.cancel_button.setText(BTN_CANCEL)
        self.cancel_button.setEnabled(True)
        self.refresh_stats()
        self._ticker.start()
        self.prepare_reshow()
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_progress(self, _description: str, done: int, total: int) -> None:
        if not self._active:
            return
        self._done = max(0, done)
        if total > 0:
            self._total = total
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(min(self._done, self.progress_bar.maximum()))
        self.refresh_stats()

    def _on_finished(self, _description: str, _report: object) -> None:
        if not self._active:
            return  # text-op batches never opened this window
        self._active = False
        self._ticker.stop()
        self.close()

    def _on_cancel(self) -> None:
        self._controller.cancel_batch()
        self.cancel_button.setText(BTN_CANCELLING)
        self.cancel_button.setEnabled(False)

    # -- stats ---------------------------------------------------------------------------
    def refresh_stats(self) -> None:
        """Recompute count / percent / speed / elapsed / ETA labels.

        Public: the 1s ticker calls it live; tests call it directly after
        advancing the injected clock.
        """
        if not self._active:
            return
        self.progress_value.setText(
            PROGRESS_FMT.format(done=self._done, total=self._total)
        )
        self.percent_label.setText(
            PERCENT_FMT.format(pct=round(self._done * 100 / max(1, self._total)))
        )
        elapsed = max(0.0, self._clock() - self._started_at)
        self.elapsed_value.setText(format_duration(elapsed))
        if self._done > 0 and elapsed > 0:
            per_second = self._done / elapsed
            self.speed_value.setText(SPEED_FMT.format(rate=per_second * 60))
            remaining = max(0, self._total - self._done)
            self.eta_value.setText(format_duration(remaining / per_second))
        else:
            self.speed_value.setText(VALUE_PLACEHOLDER)
            self.eta_value.setText(VALUE_PLACEHOLDER)
