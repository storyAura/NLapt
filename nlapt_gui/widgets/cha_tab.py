"""设置 ▸ CHA标注 tab: own API or sync, plus per-scheme vision models."""

from __future__ import annotations

import time
from typing import Callable, Sequence

from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.config import LLMProfile
from nlapt.diagnostics import get_logger
from nlapt.llm.base import create_client
from nlapt.llm.model_list import list_models

from nlapt_gui.cha_config import (
    API_MODE_OWN,
    API_MODE_SYNC,
    CARD_SLOTS,
    CHASettings,
    resolve_base_profile,
)
from nlapt_gui.controller import TOAST_ERR, TOAST_OK, TOAST_WARN
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

DEFAULT_API_TYPES: tuple[str, ...] = ("openai", "anthropic", "ollama")
FORM_LABEL_MIN_W = 88
SECTION_GAP = 10
FORM_V_GAP = 8

HINT_CHA = (
    "组合分层推标（Combined Hierarchical Annotation，日常简称 CHA标注）。"
    "三个候选方案与整批画面段可分别指定模型；留空沿用 LLM 设置中的视觉模型。"
)
LABEL_SYNC = "同步 LLM 设置中的接口"
LABEL_API_TYPE = "接口类型"
LABEL_BASE_URL = "Base URL"
LABEL_API_KEY = "API Key"
LABEL_CARD_FMT = "候选 {n} 模型"
LABEL_BATCH_MODEL = "整批画面模型"
PLACEHOLDER_MODEL = "留空沿用视觉模型"
BUTTON_FETCH_MODELS = "获取模型"
BUTTON_TEST = "测试连接"
TOAST_NEED_BASE_URL = "请填写 Base URL"
TOAST_NEED_VISION_MODEL = "请填写模型或在 LLM 设置中配置视觉模型"
TOAST_TEST_OK = "连接成功({seconds:.1f} 秒)"
TOAST_TEST_FAIL = "连接失败: {message}"
TOAST_FETCH_FAILED = "获取模型失败: {message}"
TOAST_FETCH_EMPTY = "接口没有返回任何模型"


class CHATab(QWidget):
    """CHA标注 settings: sync/own API plus four model fields."""

    toast_requested = Signal(str, str)

    def __init__(
        self,
        *,
        main_profile_provider: Callable[[], LLMProfile],
        api_types: Sequence[str] = DEFAULT_API_TYPES,
        pool: QThreadPool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_profile_provider = main_profile_provider
        self._pool = pool if pool is not None else QThreadPool.globalInstance()

        hint = QLabel(HINT_CHA, self)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)

        self.sync_check = QCheckBox(LABEL_SYNC, self)
        self.sync_check.setChecked(True)
        self.sync_check.toggled.connect(self._sync_api_rows)

        self.api_type = QComboBox(self)
        self.api_type.addItems(list(api_types))
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

    def current_settings(self) -> CHASettings:
        """CHASettings built from the current field values."""
        return CHASettings(
            api_mode=API_MODE_SYNC if self.sync_check.isChecked() else API_MODE_OWN,
            api_type=self.api_type.currentText(),
            base_url=self.base_url.text().strip(),
            api_key=self.api_key.text().strip(),
            card_models=tuple(combo.currentText().strip() for combo in self.card_combos),
            batch_model=self.batch_combo.currentText().strip(),
        )

    def prefill(self, settings: CHASettings) -> None:
        """Load persisted values into the form."""
        self.sync_check.setChecked(settings.api_mode != API_MODE_OWN)
        index = self.api_type.findText(settings.api_type)
        if index >= 0:
            self.api_type.setCurrentIndex(index)
        self.base_url.setText(settings.base_url)
        self.api_key.setText(settings.api_key)
        for combo, model in zip(self.card_combos, settings.card_models, strict=True):
            combo.setCurrentText(model)
        self.batch_combo.setCurrentText(settings.batch_model)
        self._sync_api_rows()

    def fetch_models(self) -> None:
        """获取模型: async list, then fill every model combo."""
        profile = self._probe_profile()
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
            self._fill_model_combos(names)

        def failed(message: str) -> None:
            self.fetch_models_button.setEnabled(True)
            self.toast_requested.emit(TOAST_FETCH_FAILED.format(message=message), TOAST_ERR)

        run_async(self._pool, probe, on_done=done, on_error=failed)

    def test_connection(self) -> None:
        """Async connectivity probe using the first filled model."""
        profile = self._probe_profile()
        if profile is None or not profile.base_url:
            self.toast_requested.emit(TOAST_NEED_BASE_URL, TOAST_WARN)
            return
        model = self._probe_model(profile)
        if not model:
            self.toast_requested.emit(TOAST_NEED_VISION_MODEL, TOAST_WARN)
            return
        self.test_button.setEnabled(False)

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

    def _model_combo(self) -> QComboBox:
        combo = QComboBox(self)
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        edit = combo.lineEdit()
        if edit is not None:
            edit.setPlaceholderText(PLACEHOLDER_MODEL)
        return combo

    def _model_combos(self) -> tuple[QComboBox, ...]:
        return (*self.card_combos, self.batch_combo)

    def _fill_model_combos(self, names: tuple[str, ...]) -> None:
        for combo in self._model_combos():
            current = combo.currentText()
            combo.clear()
            combo.addItem("")
            combo.addItems(list(names))
            combo.setCurrentText(current)

    def _sync_api_rows(self) -> None:
        own = not self.sync_check.isChecked()
        self._api_form.setRowVisible(self.api_type, own)
        self._api_form.setRowVisible(self.base_url, own)
        self._api_form.setRowVisible(self.api_key, own)

    def _probe_profile(self) -> LLMProfile | None:
        return resolve_base_profile(self.current_settings(), self._main_profile_provider())

    def _probe_model(self, profile: LLMProfile) -> str:
        settings = self.current_settings()
        for model in (*settings.card_models, settings.batch_model):
            if model:
                return model
        return profile.vision_model

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
