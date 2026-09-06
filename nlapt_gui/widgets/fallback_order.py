"""Reorderable 备选顺序 list for 设置 ▸ 翻译服务."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt_gui.translate_config import (
    KNOWN_PROVIDERS,
    PROVIDER_LABELS,
    SUGGESTED_FALLBACK_ORDER,
    normalize_fallback_order,
)

LABEL_FALLBACK = "备选顺序"
HINT_FALLBACK = (
    "首选失败时按此顺序尝试，未配置的服务会跳过。"
    "例如 DeepLX → Google → 百度 → 本地模型。"
)
BUTTON_UP = "上移"
BUTTON_DOWN = "下移"
BUTTON_REMOVE = "移除"
BUTTON_ADD = "添加"
BUTTON_SUGGEST = "填入推荐"
LIST_HEIGHT = 112


class FallbackOrderEditor(QWidget):
    """List of fallback provider ids with add / remove / reorder."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._primary = ""
        self.hint = QLabel(HINT_FALLBACK, self)
        self.hint.setProperty("muted", True)
        self.hint.setWordWrap(True)
        self.list = QListWidget(self)
        self.list.setFixedHeight(LIST_HEIGHT)
        self.up_button = QPushButton(BUTTON_UP, self)
        self.up_button.setProperty("variant", "ghost")
        self.up_button.clicked.connect(self.move_up)
        self.down_button = QPushButton(BUTTON_DOWN, self)
        self.down_button.setProperty("variant", "ghost")
        self.down_button.clicked.connect(self.move_down)
        self.remove_button = QPushButton(BUTTON_REMOVE, self)
        self.remove_button.setProperty("variant", "ghost")
        self.remove_button.clicked.connect(self.remove_selected)
        self.add_combo = QComboBox(self)
        self.add_button = QPushButton(BUTTON_ADD, self)
        self.add_button.setProperty("variant", "outline")
        self.add_button.clicked.connect(self.add_selected)
        self.suggest_button = QPushButton(BUTTON_SUGGEST, self)
        self.suggest_button.setProperty("variant", "outline")
        self.suggest_button.clicked.connect(self.apply_suggested)

        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(4)
        side.addWidget(self.up_button)
        side.addWidget(self.down_button)
        side.addWidget(self.remove_button)
        side.addStretch(1)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.list, 1)
        row.addLayout(side)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.addWidget(self.add_combo, 1)
        add_row.addWidget(self.add_button)
        add_row.addWidget(self.suggest_button)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(self.hint)
        column.addLayout(row)
        column.addLayout(add_row)
        self._refresh_add_combo()

    def set_primary(self, provider_id: str) -> None:
        """Drop ``provider_id`` from the list (it is the 首选)."""
        self._primary = provider_id
        self.set_order([item for item in self.order() if item != provider_id])

    def set_order(self, fallbacks: Sequence[str]) -> None:
        """Replace the list with a normalized fallback sequence."""
        self.list.clear()
        for provider_id in normalize_fallback_order(self._primary, fallbacks):
            item = QListWidgetItem(
                PROVIDER_LABELS.get(provider_id, provider_id), self.list
            )
            item.setData(Qt.ItemDataRole.UserRole, provider_id)
        self._refresh_add_combo()

    def order(self) -> tuple[str, ...]:
        """Current fallback ids, top to bottom."""
        ids: list[str] = []
        for index in range(self.list.count()):
            item = self.list.item(index)
            if item is None:
                continue
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, str):
                ids.append(data)
        return tuple(ids)

    def apply_suggested(self) -> None:
        """Fill Google → 百度 → 本地 (minus the current 首选)."""
        self.set_order(SUGGESTED_FALLBACK_ORDER)

    def add_selected(self) -> None:
        """Append the combo's provider if it is not already in the chain."""
        data = self.add_combo.currentData()
        if not isinstance(data, str) or not data:
            return
        self.set_order((*self.order(), data))

    def remove_selected(self) -> None:
        """Remove the highlighted row."""
        row = self.list.currentRow()
        if row < 0:
            return
        ids = list(self.order())
        del ids[row]
        self.set_order(ids)
        if self.list.count():
            self.list.setCurrentRow(min(row, self.list.count() - 1))

    def move_up(self) -> None:
        """Swap the highlighted row with the one above."""
        self._move(-1)

    def move_down(self) -> None:
        """Swap the highlighted row with the one below."""
        self._move(1)

    def _move(self, delta: int) -> None:
        row = self.list.currentRow()
        target = row + delta
        ids = list(self.order())
        if row < 0 or target < 0 or target >= len(ids):
            return
        ids[row], ids[target] = ids[target], ids[row]
        self.set_order(ids)
        self.list.setCurrentRow(target)

    def _refresh_add_combo(self) -> None:
        taken = {self._primary, *self.order()}
        self.add_combo.clear()
        for provider_id in KNOWN_PROVIDERS:
            if provider_id in taken:
                continue
            self.add_combo.addItem(
                PROVIDER_LABELS.get(provider_id, provider_id), provider_id
            )
        usable = self.add_combo.count() > 0
        self.add_combo.setEnabled(usable)
        self.add_button.setEnabled(usable)
