"""独立弹窗: download / delete Hy-MT2 translation GGUF files.

Opened from 设置 ▸ 翻译服务 ▸ 管理模型. Downloads go through the shared
caption-model download slot so only one transfer runs at a time.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger
from nlapt.local.hardware import format_bytes
from nlapt.local.mt_catalog import ALL_MT_MODELS, MTModel, find_mt_model

from nlapt_gui.local_bridge import (
    DOWNLOAD_CANCELLED,
    DOWNLOAD_ERROR,
    DOWNLOAD_OK,
    active_download,
    get_download_hub,
)
from nlapt_gui.mt_bridge import (
    MT_DOWNLOAD_PREFIX,
    cancel_mt_download,
    delete_mt_model,
    is_tier_downloaded,
    start_mt_download,
)
from nlapt_gui.widgets.dialogs import CenteredDialog, ask_confirm

_LOGGER = get_logger(__name__)

WINDOW_TITLE = "本地翻译模型"
HINT_TEXT = "Hy-MT2 由腾讯开源,在本机运行,不经过网络翻译接口。"
BUTTON_DOWNLOAD = "下载"
BUTTON_CANCEL = "取消下载"
BUTTON_DELETE = "删除"
BUTTON_CLOSE = "关闭"
CONFIRM_DELETE_TITLE = "删除模型"
CONFIRM_DELETE = "删除 {name}？已下载的文件将从磁盘移除。"
STATUS_READY = "已下载"
STATUS_MISSING = "未下载"
STATUS_BUSY = "下载中…"
PROGRESS_IDLE = "没有正在进行的下载"
DIALOG_WIDTH = 520
TIER_LABELS: dict[str, str] = {
    "fast": "快速",
    "balanced": "均衡",
    "quality": "高质量",
}


class MTModelsDialog(CenteredDialog):
    """List the three Hy-MT2 tiers with download / delete actions."""

    def __init__(
        self,
        *,
        models_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._models_dir = models_dir
        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(True)
        self.setMinimumWidth(DIALOG_WIDTH)

        column = QVBoxLayout(self)
        hint = QLabel(HINT_TEXT, self)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        column.addWidget(hint)

        self._status_labels: dict[str, QLabel] = {}
        self._download_buttons: dict[str, QPushButton] = {}
        self._delete_buttons: dict[str, QPushButton] = {}
        for model in ALL_MT_MODELS:
            column.addLayout(self._build_row(model))

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        column.addWidget(self.progress)
        self.progress_note = QLabel(PROGRESS_IDLE, self)
        self.progress_note.setProperty("muted", True)
        column.addWidget(self.progress_note)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton(BUTTON_CANCEL, self)
        self.cancel_button.setProperty("variant", "outline")
        self.cancel_button.clicked.connect(cancel_mt_download)
        close_button = QPushButton(BUTTON_CLOSE, self)
        close_button.clicked.connect(self.accept)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        column.addLayout(buttons)

        hub = get_download_hub()
        hub.progress.connect(self._on_progress)
        hub.finished.connect(self._on_finished)
        self._refresh()
        self._reattach()

    def _build_row(self, model: MTModel) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel(
            f"{TIER_LABELS[model.tier]} · {model.name}  ({format_bytes(model.size_bytes)})",
            self,
        )
        title.setWordWrap(True)
        status = QLabel(STATUS_MISSING, self)
        status.setProperty("muted", True)
        download = QPushButton(BUTTON_DOWNLOAD, self)
        download.setProperty("variant", "outline")
        download.clicked.connect(lambda *_a, tier=model.tier: self._start(tier))
        delete = QPushButton(BUTTON_DELETE, self)
        delete.setProperty("variant", "outline")
        delete.clicked.connect(lambda *_a, tier=model.tier: self._delete(tier))
        self._status_labels[model.tier] = status
        self._download_buttons[model.tier] = download
        self._delete_buttons[model.tier] = delete
        row.addWidget(title, 1)
        row.addWidget(status)
        row.addWidget(download)
        row.addWidget(delete)
        return row

    def _start(self, tier: str) -> None:
        start_mt_download(tier, models_dir=self._models_dir)
        self._refresh()

    def _delete(self, tier: str) -> None:
        model = find_mt_model(tier)
        if not ask_confirm(
            self,
            CONFIRM_DELETE_TITLE,
            CONFIRM_DELETE.format(name=model.name),
        ):
            return
        delete_mt_model(tier, models_dir=self._models_dir)
        self._refresh()

    def _on_progress(
        self, family_id: str, _quant: str, done: object, total: object
    ) -> None:
        if not str(family_id).startswith(MT_DOWNLOAD_PREFIX):
            return
        done_n = int(done) if isinstance(done, int) else 0
        total_n = int(total) if isinstance(total, int) and total else 0
        if total_n > 0:
            self.progress.setValue(min(100, int(done_n * 100 / total_n)))
        self.progress_note.setText(f"{format_bytes(done_n)} / {format_bytes(total_n)}")

    def _on_finished(
        self, family_id: str, _quant: str, status: str, message: str
    ) -> None:
        if not str(family_id).startswith(MT_DOWNLOAD_PREFIX):
            return
        self.progress.setValue(100 if status == DOWNLOAD_OK else 0)
        if status == DOWNLOAD_ERROR and message:
            self.progress_note.setText(message)
        elif status == DOWNLOAD_CANCELLED:
            self.progress_note.setText(BUTTON_CANCEL)
        else:
            self.progress_note.setText(PROGRESS_IDLE)
        self._refresh()

    def _reattach(self) -> None:
        snapshot = active_download()
        if snapshot is None:
            return
        family_id, _quant, done, total = snapshot
        if not family_id.startswith(MT_DOWNLOAD_PREFIX):
            return
        self._on_progress(family_id, "", done, total)
        self._refresh()

    def _refresh(self) -> None:
        snapshot = active_download()
        busy_tier = ""
        if snapshot is not None and snapshot[0].startswith(MT_DOWNLOAD_PREFIX):
            busy_tier = snapshot[0][len(MT_DOWNLOAD_PREFIX) :]
        self.cancel_button.setEnabled(bool(busy_tier))
        for model in ALL_MT_MODELS:
            ready = is_tier_downloaded(model.tier, models_dir=self._models_dir)
            if model.tier == busy_tier:
                self._status_labels[model.tier].setText(STATUS_BUSY)
            elif ready:
                self._status_labels[model.tier].setText(STATUS_READY)
            else:
                self._status_labels[model.tier].setText(STATUS_MISSING)
            self._download_buttons[model.tier].setEnabled(
                not ready and not busy_tier
            )
            self._delete_buttons[model.tier].setEnabled(ready and not busy_tier)
