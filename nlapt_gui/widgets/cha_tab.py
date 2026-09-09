"""设置 ▸ CHA标注 tab: own API or sync, plus per-scheme vision models.

The four model pickers list the WHOLE model pool grouped by API provider
(every switched-on model of every profile), so a candidate card may run on
a different API than the base endpoint. Each picker is an editable combo:
picking a row stores a pool ``ModelRef(profile, model)``; typing a free id
stores a bare ``ModelRef("", id)`` used on the base endpoint; 留空 keeps the
base 视觉模型. 获取模型 lists the base endpoint's ids under 「当前接口」.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.config import LLMProfile, ModelRef
from nlapt.diagnostics import get_logger
from nlapt.llm.base import create_client
from nlapt.llm.model_list import list_models

from nlapt_gui.cha_config import (
    API_MODE_OWN,
    API_MODE_SYNC,
    CARD_SLOTS,
    CHASettings,
    ProfileLookup,
    resolve_base_profile,
    resolve_batch_profile,
    resolve_card_profile,
)
from nlapt_gui.controller import TOAST_ERR, TOAST_OK, TOAST_WARN
from nlapt_gui.model_targets import ModelChoice, choice_label
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

PoolGroups = Sequence[tuple[str, Sequence[ModelChoice]]]
PoolProvider = Callable[[], PoolGroups]

DEFAULT_API_TYPES: tuple[str, ...] = ("openai", "anthropic", "ollama")
FORM_LABEL_MIN_W = 88
SECTION_GAP = 10
FORM_V_GAP = 8
# QComboBox defaults to a Preferred width that collapses to its (empty)
# text under ExpandingFieldsGrow; force it to fill the form row.
MODEL_COMBO_MIN_W = 240
REF_ROLE = Qt.ItemDataRole.UserRole

HINT_CHA = (
    "组合分层推标（Combined Hierarchical Annotation，日常简称 CHA标注）。"
    "三个候选方案与整批画面段可分别指定模型;下拉按 API 供应商列出所有已启用的模型,"
    "可跨供应商选择,也可直接输入模型 ID;留空沿用当前视觉模型。"
)
LABEL_SYNC = "同步 LLM 设置中的接口"
LABEL_API_TYPE = "接口类型"
LABEL_BASE_URL = "Base URL"
LABEL_API_KEY = "API Key"
LABEL_CARD_FMT = "候选 {n} 模型"
LABEL_BATCH_MODEL = "整批画面模型"
PLACEHOLDER_MODEL = "留空沿用视觉模型"
GROUP_HEADER_FMT = "── {name} ──"
GROUP_BASE_ENDPOINT = "当前接口"
GROUP_STALE = "已失效"
BUTTON_FETCH_MODELS = "获取模型"
BUTTON_TEST = "测试连接"
TOAST_NEED_BASE_URL = "请填写 Base URL"
TOAST_NEED_VISION_MODEL = "请填写模型或在 LLM 设置中配置视觉模型"
TOAST_STALE_REF = "模型「{label}」已不在模型池中,请重新选择"
TOAST_TEST_OK = "连接成功({seconds:.1f} 秒)"
TOAST_TEST_FAIL = "连接失败: {message}"
TOAST_FETCH_FAILED = "获取模型失败: {message}"
TOAST_FETCH_EMPTY = "接口没有返回任何模型"


def _no_pool() -> PoolGroups:
    return ()


class CHATab(QWidget):
    """CHA标注 settings: sync/own API plus four pool-aware model pickers."""

    toast_requested = Signal(str, str)

    def __init__(
        self,
        *,
        main_profile_provider: Callable[[], LLMProfile | None],
        pool_provider: PoolProvider = _no_pool,
        profile_lookup: ProfileLookup | None = None,
        api_types: Sequence[str] = DEFAULT_API_TYPES,
        pool: QThreadPool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_profile_provider = main_profile_provider
        self._pool_provider = pool_provider
        self._profile_lookup = profile_lookup
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._groups: tuple[tuple[str, tuple[ModelChoice, ...]], ...] = ()
        self._fetched_models: tuple[str, ...] = ()

        hint = QLabel(HINT_CHA, self)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)

        self.sync_check = QCheckBox(LABEL_SYNC, self)
        self.sync_check.setChecked(True)
        self.sync_check.toggled.connect(self._sync_api_rows)

        self.api_type = QComboBox(self)
        self.api_type.addItems(list(api_types))
        self.api_type.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.base_url = QLineEdit(self)
        self.api_key = QLineEdit(self)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)

        api_form = QFormLayout()
        self._api_form = api_form
        api_form.addRow(LABEL_API_TYPE, self.api_type)
        api_form.addRow(LABEL_BASE_URL, self.base_url)
        api_form.addRow(LABEL_API_KEY, self.api_key)
        self._tune_form(api_form)

        self.card_combos: list[QComboBox] = []
        model_form = QFormLayout()
        for index in range(CARD_SLOTS):
            combo = self._model_combo()
            self.card_combos.append(combo)
            model_form.addRow(LABEL_CARD_FMT.format(n=index + 1), combo)
        self.batch_combo = self._model_combo()
        model_form.addRow(LABEL_BATCH_MODEL, self.batch_combo)
        self._tune_form(model_form)

        self.fetch_models_button = QPushButton(BUTTON_FETCH_MODELS, self)
        self.fetch_models_button.setProperty("variant", "outline")
        self.fetch_models_button.clicked.connect(self.fetch_models)
        self.test_button = QPushButton(BUTTON_TEST, self)
        self.test_button.setProperty("variant", "outline")
        self.test_button.clicked.connect(self.test_connection)
        probe_row = QHBoxLayout()
        probe_row.addWidget(self.fetch_models_button)
        probe_row.addWidget(self.test_button)
        probe_row.addStretch(1)

        column = QVBoxLayout(self)
        column.setSpacing(SECTION_GAP)
        column.addWidget(hint)
        column.addWidget(self.sync_check)
        column.addLayout(api_form)
        column.addLayout(model_form)
        column.addLayout(probe_row)
        column.addStretch(1)
        self._sync_api_rows()
        self.refresh_pool_models()

    # -- settings <-> form -------------------------------------------------------------
    def current_settings(self) -> CHASettings:
        """CHASettings built from the current field values."""
        return CHASettings(
            api_mode=API_MODE_SYNC if self.sync_check.isChecked() else API_MODE_OWN,
            api_type=self.api_type.currentText(),
            base_url=self.base_url.text().strip(),
            api_key=self.api_key.text().strip(),
            card_models=tuple(self._combo_ref(combo) for combo in self.card_combos),
            batch_model=self._combo_ref(self.batch_combo),
        )

    def prefill(self, settings: CHASettings) -> None:
        """Load persisted values into the form."""
        self.sync_check.setChecked(settings.api_mode != API_MODE_OWN)
        index = self.api_type.findText(settings.api_type)
        if index >= 0:
            self.api_type.setCurrentIndex(index)
        self.base_url.setText(settings.base_url)
        self.api_key.setText(settings.api_key)
        for combo, ref in zip(self.card_combos, settings.card_models, strict=True):
            self._set_combo_ref(combo, ref)
        self._set_combo_ref(self.batch_combo, settings.batch_model)
        self._sync_api_rows()

    def refresh_pool_models(self) -> None:
        """Re-read the provider-grouped pool and rebuild every picker."""
        self._groups = tuple(
            (str(name), tuple(choices)) for name, choices in self._pool_provider()
        )
        self._rebuild_combo_items()

    # -- probes ----------------------------------------------------------------------
    def fetch_models(self) -> None:
        """获取模型: async list of the BASE endpoint, shown under 「当前接口」."""
        profile = resolve_base_profile(self.current_settings(), self._main_profile_provider())
        if profile is None or not profile.base_url:
            self.toast_requested.emit(TOAST_NEED_BASE_URL, TOAST_WARN)
            return
        self.fetch_models_button.setEnabled(False)

        def probe() -> tuple[str, ...]:
            return list_models(profile)

        def done(models: object) -> None:
            self.fetch_models_button.setEnabled(True)
            names = tuple(models) if isinstance(models, tuple) else ()
            if not names:
                self.toast_requested.emit(TOAST_FETCH_EMPTY, TOAST_WARN)
                return
            self._fetched_models = names
            self._rebuild_combo_items()

        def failed(message: str) -> None:
            self.fetch_models_button.setEnabled(True)
            self.toast_requested.emit(TOAST_FETCH_FAILED.format(message=message), TOAST_ERR)

        run_async(self._pool, probe, on_done=done, on_error=failed)

    def test_connection(self) -> None:
        """Async connectivity probe with the first filled picker's model + API."""
        profile = self._probe_profile()
        if profile is None:
            return
        self.test_button.setEnabled(False)
        model = profile.vision_model

        def probe() -> float:
            started = time.monotonic()
            create_client(profile).test_connection(model)
            return time.monotonic() - started

        def done(elapsed: object) -> None:
            self.test_button.setEnabled(True)
            seconds = elapsed if isinstance(elapsed, float) else 0.0
            self.toast_requested.emit(TOAST_TEST_OK.format(seconds=seconds), TOAST_OK)

        def failed(message: str) -> None:
            self.test_button.setEnabled(True)
            self.toast_requested.emit(TOAST_TEST_FAIL.format(message=message), TOAST_ERR)

        run_async(self._pool, probe, on_done=done, on_error=failed)

    def _probe_profile(self) -> LLMProfile | None:
        """Profile the 测试连接 probe should hit, toasting the reason when none."""
        settings = self.current_settings()
        active = self._main_profile_provider()
        refs = (*settings.card_models, settings.batch_model)
        first = next((ref for ref in refs if ref.model), None)
        if first is not None and first.profile:
            resolved = self._profile_lookup(first) if self._profile_lookup else None
            if resolved is None:
                self.toast_requested.emit(
                    TOAST_STALE_REF.format(label=choice_label(first)), TOAST_WARN
                )
            return resolved
        base = resolve_base_profile(settings, active)
        if base is None or not base.base_url:
            self.toast_requested.emit(TOAST_NEED_BASE_URL, TOAST_WARN)
            return None
        model = first.model if first is not None else base.vision_model
        if not model:
            self.toast_requested.emit(TOAST_NEED_VISION_MODEL, TOAST_WARN)
            return None
        if first is None:
            return resolve_card_profile(settings, active, 0)
        return resolve_batch_profile(settings.with_changes(batch_model=first), active)

    # -- pickers ---------------------------------------------------------------------
    def _model_combo(self) -> QComboBox:
        combo = QComboBox(self)
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        combo.setMinimumWidth(MODEL_COMBO_MIN_W)
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        edit = combo.lineEdit()
        if edit is not None:
            edit.setPlaceholderText(PLACEHOLDER_MODEL)
        return combo

    def _model_combos(self) -> tuple[QComboBox, ...]:
        return (*self.card_combos, self.batch_combo)

    def _rebuild_combo_items(self) -> None:
        for combo in self._model_combos():
            current = self._combo_ref(combo)
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItem("", ModelRef())
                for name, choices in self._groups:
                    self._add_header(combo, name)
                    for choice in choices:
                        combo.addItem(choice.label, choice.ref)
                if self._fetched_models:
                    self._add_header(combo, GROUP_BASE_ENDPOINT)
                    for model in self._fetched_models:
                        combo.addItem(model, ModelRef("", model))
            finally:
                combo.blockSignals(False)
            self._set_combo_ref(combo, current)

    @staticmethod
    def _add_header(combo: QComboBox, name: str) -> None:
        combo.addItem(GROUP_HEADER_FMT.format(name=name), None)
        item = combo.model().item(combo.count() - 1)
        if item is not None:
            item.setFlags(Qt.ItemFlag.NoItemFlags)

    @staticmethod
    def _combo_ref(combo: QComboBox) -> ModelRef:
        """Pool ref when a listed row is selected verbatim; else a bare typed id."""
        text = combo.currentText().strip()
        index = combo.currentIndex()
        if index >= 0 and combo.itemText(index).strip() == text:
            data = combo.itemData(index, REF_ROLE)
            if isinstance(data, ModelRef):
                return data
        return ModelRef("", text)

    def _set_combo_ref(self, combo: QComboBox, ref: ModelRef) -> None:
        if ref.profile:
            for index in range(combo.count()):
                if combo.itemData(index, REF_ROLE) == ref:
                    combo.setCurrentIndex(index)
                    return
            # Persisted ref that left the pool: keep it selectable (and visibly stale).
            self._add_header(combo, GROUP_STALE)
            combo.addItem(choice_label(ref), ref)
            combo.setCurrentIndex(combo.count() - 1)
            return
        if ref.model:
            combo.setCurrentText(ref.model)
            return
        combo.setCurrentIndex(0)

    def _sync_api_rows(self) -> None:
        own = not self.sync_check.isChecked()
        self._api_form.setRowVisible(self.api_type, own)
        self._api_form.setRowVisible(self.base_url, own)
        self._api_form.setRowVisible(self.api_key, own)

    @staticmethod
    def _tune_form(form: QFormLayout) -> None:
        form.setHorizontalSpacing(SECTION_GAP)
        form.setVerticalSpacing(FORM_V_GAP)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            widget = item.widget() if item is not None else None
            if isinstance(widget, QLabel):
                widget.setMinimumWidth(FORM_LABEL_MIN_W)
