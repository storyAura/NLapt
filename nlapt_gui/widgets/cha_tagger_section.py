"""设置 ▸ CHA标注 · optional CL Tagger download + Hugging Face token."""

from __future__ import annotations

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices, QShowEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.local.catalog import TAGGER_FAMILY_ID
from nlapt.local.hardware import format_bytes

from nlapt_gui.api_config import HuggingFaceAuth, load_api_config, update_api_config
from nlapt_gui.controller import TOAST_ERR, TOAST_OK, TOAST_WARN
from nlapt_gui.download_hub import (
    DOWNLOAD_CANCELLED,
    DOWNLOAD_OK,
    active_download,
    cancel_active_download,
    get_download_hub,
)
from nlapt_gui.tagger_bridge import (
    MSG_NEED_HF_TOKEN,
    is_tagger_ready,
    start_tagger_download,
)

TITLE = "角色识别（CL Tagger v2.01a，可选）"
HINT_TAGGER = (
    "用本地 CL Tagger 从参考图识别角色名 / 作品名。"
    "该模型为受限仓库：请先在网页同意条款，并填写 Hugging Face Token"
    "（保存在文档目录，后续下载其他模型也可复用）。"
    "许可证禁止再分发，请从官方仓库自行下载。"
)
LABEL_TOKEN = "Hugging Face Token"
BUTTON_GET_TOKEN = "获取 Token"
BUTTON_LICENSE = "打开模型页"
BUTTON_DOWNLOAD = "下载 CL Tagger"
BUTTON_CANCEL = "取消下载"
STATUS_READY = "已就绪"
STATUS_MISSING = "未下载"
STATUS_DOWNLOADING = "下载中 {percent}%"
STATUS_DOWNLOADING_INDETERMINATE = "下载中…"
LINK_TOKEN = "https://huggingface.co/settings/tokens"
LINK_MODEL = "https://huggingface.co/cella110n/cl_tagger_v2"
TOAST_DOWNLOAD_OK = "CL Tagger 已就绪"
TOAST_DOWNLOAD_CANCELLED = "已取消下载(已下载部分保留,可续传)"
TOAST_DOWNLOAD_FAIL = "下载失败: {message}"
TOAST_DOWNLOAD_BUSY = "已有下载任务正在进行"
PROGRESS_BAR_MAX = 1000


class TaggerSection(QWidget):
    """HF token + download/status for the optional CHA character tagger."""

    toast_requested = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        title = QLabel(TITLE, self)
        title.setProperty("sectionTitle", True)
        hint = QLabel(HINT_TAGGER, self)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)

        self.token_edit = QLineEdit(self)
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText(LABEL_TOKEN)
        self.get_token_button = QPushButton(BUTTON_GET_TOKEN, self)
        self.get_token_button.setProperty("variant", "ghost")
        self.get_token_button.clicked.connect(self._open_token_page)
        self.license_button = QPushButton(BUTTON_LICENSE, self)
        self.license_button.setProperty("variant", "ghost")
        self.license_button.clicked.connect(self._open_model_page)
        token_row = QHBoxLayout()
        token_row.setContentsMargins(0, 0, 0, 0)
        token_row.addWidget(self.token_edit, 1)
        token_row.addWidget(self.get_token_button)
        token_row.addWidget(self.license_button)

        self.status_label = QLabel(STATUS_MISSING, self)
        self.status_label.setProperty("muted", True)
        self.download_button = QPushButton(BUTTON_DOWNLOAD, self)
        self.download_button.setProperty("variant", "outline")
        self.download_button.clicked.connect(self._on_download_clicked)
        self.progress = QProgressBar(self)
        self.progress.setMaximum(PROGRESS_BAR_MAX)
        self.progress.hide()
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.addWidget(self.status_label, 1)
        action_row.addWidget(self.download_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addLayout(token_row)
        layout.addLayout(action_row)
        layout.addWidget(self.progress)

        hub = get_download_hub()
        hub.progress.connect(self._on_progress)
        hub.finished.connect(self._on_finished)
        self.prefill_token()
        self.refresh_status()

    def token(self) -> str:
        return self.token_edit.text().strip()

    def prefill_token(self, token: str | None = None) -> None:
        """Load the token field from ``api.json`` when ``token`` is omitted."""
        text = token if token is not None else load_api_config().huggingface.token
        self.token_edit.setText(text)

    def persist_token(self) -> None:
        """Write the current token into ``Documents/NLapt/api.json``."""
        update_api_config(huggingface=HuggingFaceAuth(token=self.token()))

    def refresh_status(self) -> None:
        """Update the status label and download button for the current disk state."""
        active = active_download()
        if active is not None and active[0] == TAGGER_FAMILY_ID:
            self.download_button.setText(BUTTON_CANCEL)
            self.progress.setVisible(True)
            self._on_progress(active[0], active[1], active[2], active[3])
            return
        self.download_button.setText(BUTTON_DOWNLOAD)
        self.progress.hide()
        self.status_label.setText(STATUS_READY if is_tagger_ready() else STATUS_MISSING)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.refresh_status()

    def _on_download_clicked(self) -> None:
        active = active_download()
        if active is not None and active[0] == TAGGER_FAMILY_ID:
            cancel_active_download()
            return
        if active is not None:
            self.toast_requested.emit(TOAST_DOWNLOAD_BUSY, TOAST_WARN)
            return
        token = self.token()
        self.persist_token()
        if not token:
            self.toast_requested.emit(MSG_NEED_HF_TOKEN, TOAST_WARN)
            return
        if not start_tagger_download(token):
            self.toast_requested.emit(TOAST_DOWNLOAD_BUSY, TOAST_WARN)
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.download_button.setText(BUTTON_CANCEL)
        self.status_label.setText(STATUS_DOWNLOADING_INDETERMINATE)

    def _on_progress(
        self, family_id: str, _quant: str, done: object, total: object
    ) -> None:
        if family_id != TAGGER_FAMILY_ID or not isinstance(done, int):
            return
        self.progress.setVisible(True)
        if isinstance(total, int) and total > 0:
            self.progress.setRange(0, PROGRESS_BAR_MAX)
            self.progress.setValue(
                min(PROGRESS_BAR_MAX, round(done / total * PROGRESS_BAR_MAX))
            )
            percent = min(100, round(done / total * 100))
            self.status_label.setText(STATUS_DOWNLOADING.format(percent=percent))
            self.progress.setFormat(f"{format_bytes(done)} / {format_bytes(total)}")
        else:
            self.progress.setRange(0, 0)
            self.status_label.setText(STATUS_DOWNLOADING_INDETERMINATE)

    def _on_finished(
        self, family_id: str, _quant: str, status: str, message: str
    ) -> None:
        if family_id != TAGGER_FAMILY_ID:
            return
        self.progress.hide()
        self.download_button.setText(BUTTON_DOWNLOAD)
        if status == DOWNLOAD_OK:
            self.status_label.setText(STATUS_READY)
            self.toast_requested.emit(TOAST_DOWNLOAD_OK, TOAST_OK)
        elif status == DOWNLOAD_CANCELLED:
            self.status_label.setText(
                STATUS_READY if is_tagger_ready() else STATUS_MISSING
            )
            self.toast_requested.emit(TOAST_DOWNLOAD_CANCELLED, TOAST_WARN)
        else:
            self.status_label.setText(STATUS_MISSING)
            self.toast_requested.emit(
                TOAST_DOWNLOAD_FAIL.format(message=message), TOAST_ERR
            )

    @staticmethod
    def _open_token_page() -> None:
        QDesktopServices.openUrl(QUrl(LINK_TOKEN))

    @staticmethod
    def _open_model_page() -> None:
        QDesktopServices.openUrl(QUrl(LINK_MODEL))
