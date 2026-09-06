"""Always-visible Hy-MT2 tier list for 设置 ▸ 翻译服务."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.local.mt_catalog import ALL_MT_MODELS, DEFAULT_TIER

from nlapt_gui.mt_bridge import is_tier_downloaded

LABEL_TITLE = "本地翻译模型"
HINT_TIER = (
    "首选或备选为「本地模型」时使用这里选中的档位。"
    "未下载的请先点「管理模型」。"
)
BUTTON_MANAGE = "管理模型"
STATUS_READY = "已下载"
STATUS_MISSING = "未下载"
TIER_LABELS: dict[str, str] = {
    "fast": "快速",
    "balanced": "均衡",
    "quality": "高质量",
}
LIST_HEIGHT = 92


class MtTierPicker(QWidget):
    """Three-row list of Hy-MT2 tiers plus a 管理模型 button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = QLabel(LABEL_TITLE, self)
        self.hint = QLabel(HINT_TIER, self)
        self.hint.setProperty("muted", True)
        self.hint.setWordWrap(True)
        self.list = QListWidget(self)
        self.list.setFixedHeight(LIST_HEIGHT)
        self.manage_button = QPushButton(BUTTON_MANAGE, self)
        self.manage_button.setProperty("variant", "outline")

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.title, 1)
        head.addWidget(self.manage_button)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addLayout(head)
        column.addWidget(self.hint)
        column.addWidget(self.list)
        self.refresh()

    def currentData(self) -> str | None:
        """Selected tier id, or None when the list is empty."""
        item = self.list.currentItem()
        if item is None:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        return str(data) if data is not None else None

    def findData(self, tier: str) -> int:
        """Row index of ``tier``, or -1."""
        for index in range(self.list.count()):
            item = self.list.item(index)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == tier:
                return index
        return -1

    def setCurrentIndex(self, index: int) -> None:
        """Select ``index`` (combo-box compatible)."""
        if 0 <= index < self.list.count():
            self.list.setCurrentRow(index)

    def refresh(self, selected: str | None = None) -> None:
        """Rebuild rows and restore ``selected`` (or the current tier)."""
        current = selected if selected is not None else self.currentData()
        if not current:
            current = DEFAULT_TIER
        self.list.clear()
        for model in ALL_MT_MODELS:
            mark = (
                STATUS_READY if is_tier_downloaded(model.tier) else STATUS_MISSING
            )
            label = f"{TIER_LABELS[model.tier]} · {model.name}（{mark}）"
            item = QListWidgetItem(label, self.list)
            item.setData(Qt.ItemDataRole.UserRole, model.tier)
        index = self.findData(current)
        self.setCurrentIndex(index if index >= 0 else 0)
