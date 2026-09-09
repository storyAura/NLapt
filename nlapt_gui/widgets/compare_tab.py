"""设置 ▸ 多对比推标 tab: pick the pool models that caption side by side."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from nlapt.core.config import ModelRef

from nlapt_gui.compare_config import MIN_COMPARE_MODELS, CompareSettings
from nlapt_gui.widgets.pool_model_list import PoolGroups, PoolModelList

PoolProvider = Callable[[], PoolGroups]

SECTION_GAP = 10
LIST_MIN_H = 200

HINT_COMPARE = (
    "多对比推标:同一批图片分别交给下面勾选的模型推标,逐张并排对照后再选一份写入。"
    "模型来自 LLM 设置里已启用的模型池,可跨 API 供应商;至少勾选 {n} 个。"
)
LABEL_COUNT_FMT = "已勾选 {n} 个模型"
LABEL_COUNT_TOO_FEW = "已勾选 {n} 个模型 — 至少需要 {min} 个才能开始"


class CompareTab(QWidget):
    """Settings page holding the compare-model checklist."""

    def __init__(
        self,
        *,
        pool_provider: PoolProvider,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pool_provider = pool_provider
        self._checked: tuple[ModelRef, ...] = ()

        hint = QLabel(HINT_COMPARE.format(n=MIN_COMPARE_MODELS), self)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        self.model_list = PoolModelList(self)
        self.model_list.setMinimumHeight(LIST_MIN_H)
        self.model_list.changed.connect(self._on_changed)
        self.count_label = QLabel(self)
        self.count_label.setProperty("muted", True)

        column = QVBoxLayout(self)
        column.setSpacing(SECTION_GAP)
        column.addWidget(hint)
        column.addWidget(self.model_list, 1)
        column.addWidget(self.count_label)
        self.refresh_pool()

    def current_settings(self) -> CompareSettings:
        """Settings built from the checked rows (stale refs are dropped)."""
        return CompareSettings(models=self.model_list.checked_refs())

    def prefill(self, settings: CompareSettings) -> None:
        """Check the persisted refs (those still in the pool)."""
        self._checked = settings.models
        self.model_list.set_checked(self._checked)

    def refresh_pool(self) -> None:
        """Rebuild the list from the pool provider, keeping current checks.

        Checked refs whose provider is currently switched off are remembered
        (not saved) so they come back when the provider is re-enabled.
        """
        self._remember_checks()
        self.model_list.set_groups(self._pool_provider(), self._checked)

    def _listed_refs(self) -> set[ModelRef]:
        refs: set[ModelRef] = set()
        for index in range(self.model_list.count()):
            data = self.model_list.item(index).data(self.model_list.REF_ROLE)
            if isinstance(data, ModelRef):
                refs.add(data)
        return refs

    def _remember_checks(self) -> None:
        listed = self._listed_refs()
        hidden = tuple(ref for ref in self._checked if ref not in listed)
        self._checked = tuple(dict.fromkeys((*self.model_list.checked_refs(), *hidden)))

    def _on_changed(self) -> None:
        self._remember_checks()
        n = len(self.model_list.checked_refs())
        if n < MIN_COMPARE_MODELS:
            self.count_label.setText(LABEL_COUNT_TOO_FEW.format(n=n, min=MIN_COMPARE_MODELS))
        else:
            self.count_label.setText(LABEL_COUNT_FMT.format(n=n))
