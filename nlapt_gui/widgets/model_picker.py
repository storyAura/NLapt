"""获取模型 result picker (设置 ▸ LLM: model list fetched from the API).

Shows every model id the endpoint reported; ids that *look* multimodal (name
heuristic — listing APIs do not expose modality) carry a 视觉 tag. The user
assigns the highlighted id to the 文本模型 / 视觉模型 field (one shared 模型
button in 统一 mode).
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger
from nlapt.llm.model_list import looks_vision_capable

from nlapt_gui.widgets.dialogs import CenteredDialog

_LOGGER = get_logger(__name__)

WINDOW_TITLE = "获取模型"
VISION_TAG = " · 视觉"
COUNT_NOTE = "共 {n} 个模型 ·「视觉」为按名称推测的多模态模型"
EMPTY_NOTE = "接口没有返回任何模型"
BUTTON_SET_TEXT = "设为文本模型"
BUTTON_SET_VISION = "设为视觉模型"
BUTTON_SET_UNIFIED = "设为模型"
BUTTON_CLOSE = "关闭"
LIST_MIN_HEIGHT = 260
DIALOG_WIDTH = 420


class ModelPickerDialog(CenteredDialog):
    """List the fetched model ids; assign the selection to a model field."""

    text_model_picked = Signal(str)
    vision_model_picked = Signal(str)

    def __init__(
        self,
        models: Sequence[str],
        *,
        unified: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(True)
        self.setMinimumWidth(DIALOG_WIDTH)
        self._models = tuple(models)

        column = QVBoxLayout(self)
        column.setSpacing(8)
        self.list = QListWidget(self)
        self.list.setMinimumHeight(LIST_MIN_HEIGHT)
        for model_id in self._models:
            label = model_id + (VISION_TAG if looks_vision_capable(model_id) else "")
            item = QListWidgetItem(label, self.list)
            item.setData(Qt.ItemDataRole.UserRole, model_id)
        column.addWidget(self.list, 1)

        self.note = QLabel(
            COUNT_NOTE.format(n=len(self._models)) if self._models else EMPTY_NOTE,
            self,
        )
        self.note.setProperty("muted", True)
        self.note.setWordWrap(True)
        column.addWidget(self.note)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        if unified:
            self.unified_button = QPushButton(BUTTON_SET_UNIFIED, self)
            self.unified_button.setProperty("variant", "accent")
            self.unified_button.clicked.connect(self._pick_unified)
            buttons.addWidget(self.unified_button)
        else:
            self.text_button = QPushButton(BUTTON_SET_TEXT, self)
            self.text_button.setProperty("variant", "accent")
            self.text_button.clicked.connect(self._pick_text)
            buttons.addWidget(self.text_button)
            self.vision_button = QPushButton(BUTTON_SET_VISION, self)
            self.vision_button.setProperty("variant", "outline")
            self.vision_button.clicked.connect(self._pick_vision)
            buttons.addWidget(self.vision_button)
        buttons.addStretch(1)
        self.close_button = QPushButton(BUTTON_CLOSE, self)
        self.close_button.setProperty("variant", "outline")
        self.close_button.clicked.connect(self.reject)
        buttons.addWidget(self.close_button)
        column.addLayout(buttons)

        if self._models:
            self.list.setCurrentRow(0)

    # -- selection --------------------------------------------------------------------
    def selected_model(self) -> str:
        item = self.list.currentItem()
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _pick_text(self) -> None:
        model = self.selected_model()
        if model:
            self.text_model_picked.emit(model)

    def _pick_vision(self) -> None:
        model = self.selected_model()
        if model:
            self.vision_model_picked.emit(model)

    def _pick_unified(self) -> None:
        model = self.selected_model()
        if model:
            self.text_model_picked.emit(model)
            self.vision_model_picked.emit(model)
