"""Checkable list of the model pool grouped by API provider.

Header rows (one per provider) are not selectable; model rows carry a
``ModelRef`` and a check box. Used by 设置 ▸ 多对比推标 to pick the models
that run side by side.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget

from nlapt.core.config import ModelRef

from nlapt_gui.model_targets import ModelChoice

PoolGroups = Sequence[tuple[str, Sequence[ModelChoice]]]

REF_ROLE = Qt.ItemDataRole.UserRole
GROUP_HEADER_FMT = "── {name} ──"
VISION_TAG = " · 视觉"
EMPTY_HINT = "模型池为空 — 请先在 LLM 设置中启用模型"


class PoolModelList(QListWidget):
    """Provider-grouped, checkable pool list."""

    changed = Signal()
    REF_ROLE = REF_ROLE

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self.itemChanged.connect(self._on_item_changed)

    def set_groups(self, groups: PoolGroups, checked: Iterable[ModelRef]) -> None:
        """Rebuild rows from ``groups``; refs in ``checked`` start checked."""
        wanted = set(checked)
        self._syncing = True
        try:
            self.clear()
            if not groups:
                hint = QListWidgetItem(EMPTY_HINT, self)
                hint.setFlags(Qt.ItemFlag.NoItemFlags)
            for name, choices in groups:
                header = QListWidgetItem(GROUP_HEADER_FMT.format(name=name), self)
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                for choice in choices:
                    label = choice.label + (VISION_TAG if choice.vision_hint else "")
                    item = QListWidgetItem(label, self)
                    item.setData(REF_ROLE, choice.ref)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if choice.ref in wanted
                        else Qt.CheckState.Unchecked
                    )
        finally:
            self._syncing = False
        self.changed.emit()

    def checked_refs(self) -> tuple[ModelRef, ...]:
        """Checked model refs in display order."""
        refs: list[ModelRef] = []
        for index in range(self.count()):
            item = self.item(index)
            data = item.data(REF_ROLE)
            if isinstance(data, ModelRef) and item.checkState() == Qt.CheckState.Checked:
                refs.append(data)
        return tuple(refs)

    def set_checked(self, refs: Iterable[ModelRef]) -> None:
        """Check exactly ``refs`` (rows not in the list are ignored)."""
        wanted = set(refs)
        self._syncing = True
        try:
            for index in range(self.count()):
                item = self.item(index)
                data = item.data(REF_ROLE)
                if isinstance(data, ModelRef):
                    item.setCheckState(
                        Qt.CheckState.Checked if data in wanted else Qt.CheckState.Unchecked
                    )
        finally:
            self._syncing = False
        self.changed.emit()

    def _on_item_changed(self, _item: QListWidgetItem) -> None:
        if not self._syncing:
            self.changed.emit()
