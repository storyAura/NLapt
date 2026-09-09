"""设置 ▸ LLM 设置: several API profiles, per-model switches, two pool targets.

Left: the profile list (新增 / 删除). Right: the selected profile's endpoint
form, 获取模型 / 测试连接 probes, and a checklist of the models known on that
endpoint — checked = switched on for the shared pool. Below: 当前文本模型 /
当前视觉模型 combos over the pool, and the request-concurrency spin.

Edits are kept in immutable ``LLMProfile`` drafts (one per list row); the
dialog reads ``build_config`` on 保存. Probes run on the worker pool through
``run_async`` and never block the UI.
"""

from __future__ import annotations

import time
from dataclasses import replace as _dc_replace
from typing import Sequence

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.config import (
    ROLE_TEXT,
    ROLE_VISION,
    AppConfig,
    LLMProfile,
    ModelRef,
    bind_model_ref,
    resolve_vision_profile,
)
from nlapt.diagnostics import get_logger
from nlapt.llm.base import create_client
from nlapt.llm.model_list import list_models, looks_vision_capable

from nlapt_gui.controller import TOAST_ERR, TOAST_OK, TOAST_WARN
from nlapt_gui.model_targets import ModelChoice, grouped_pool_choices, pool_choices
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

DEFAULT_API_TYPES: tuple[str, ...] = ("openai", "anthropic", "ollama")
NEW_PROFILE_NAME = "API {n}"
SECTION_GAP = 10
FORM_LABEL_MIN_W = 88
FORM_V_GAP = 8
DIVIDER_H = 1
PROFILE_LIST_W = 128
MODEL_LIST_MIN_H = 120
CONCURRENCY_RANGE = (1, 32)
CHOICE_REF_ROLE = Qt.ItemDataRole.UserRole

# UI strings.
LABEL_PROFILES = "API 档案"
BUTTON_ADD_PROFILE = "新增"
BUTTON_REMOVE_PROFILE = "删除"
LABEL_NAME = "名称"
LABEL_API_TYPE = "接口类型"
LABEL_BASE_URL = "Base URL"
LABEL_API_KEY = "API Key"
BUTTON_FETCH_MODELS = "获取模型"
BUTTON_TEST = "测试连接"
LABEL_MODELS = "模型开关(勾选后加入模型池)"
LABEL_MODELS_FILTERED = "模型开关 · 匹配 {n} / {total}"
PLACEHOLDER_MODEL_FILTER = "搜索模型…"
BUTTON_ENABLE_ALL = "全选"
BUTTON_DISABLE_ALL = "全不选"
BUTTON_ADD_MODEL = "添加"
PLACEHOLDER_NEW_MODEL = "手动输入模型 ID(接口不支持列表时)"
VISION_TAG = " · 视觉"
LABEL_TEXT_TARGET = "当前文本模型"
LABEL_VISION_TARGET = "当前视觉模型"
LABEL_UNSET_CHOICE = "未设置"
HINT_TARGETS = (
    "文本模型用于翻译 / 改写,视觉模型用于推标 — 两者可来自不同 API;"
    "视觉需选择具备多模态能力的模型。"
)
LABEL_CONCURRENCY = "并发请求数"
EMPTY_HINT = "尚未添加 API。点击「新增」填写接口地址,再「获取模型」勾选要使用的模型。"
TOAST_TEST_OK = "连接成功({seconds:.1f} 秒)"
TOAST_TEST_FAIL = "连接失败: {message}"
TOAST_NEED_BASE_URL = "请填写 Base URL"
TOAST_NEED_PROFILE_URL = "档案「{name}」缺少 Base URL"
TOAST_NEED_NAME = "档案名称不能为空"
TOAST_DUPLICATE_NAME = "档案名称「{name}」重复"
TOAST_NEED_TEXT_MODEL = "请选择当前文本模型"
TOAST_NEED_MODEL = "请先获取或手动添加模型"
TOAST_FETCH_FAILED = "获取模型失败: {message}"
TOAST_FETCH_EMPTY = "接口没有返回任何模型"
TOAST_FETCHED = "已获取 {n} 个模型"


class LLMProvidersTab(QWidget):
    """Multi-profile LLM settings page (see module docstring)."""

    toast_requested = Signal(str, str)
    pool_changed = Signal()

    def __init__(
        self,
        existing: AppConfig | None,
        *,
        api_types: Sequence[str] = DEFAULT_API_TYPES,
        pool: QThreadPool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        base = existing if existing is not None else AppConfig()
        self._drafts: list[LLMProfile] = list(base.profiles)
        self._text_target = base.text_target
        self._vision_target = base.vision_target
        self._syncing = False

        self._build(api_types, base)
        self._rebuild_profile_list()
        initial = self._index_of(self._text_target.profile)
        self.profile_list.setCurrentRow(initial if initial >= 0 else 0)
        self._load_form()
        self._refresh_targets()

    # -- construction -------------------------------------------------------------------
    def _build(self, api_types: Sequence[str], base: AppConfig) -> None:
        self.profile_list = QListWidget(self)
        self.profile_list.setFixedWidth(PROFILE_LIST_W)
        self.profile_list.currentRowChanged.connect(self._on_profile_selected)
        self.add_button = QPushButton(BUTTON_ADD_PROFILE, self)
        self.add_button.setProperty("variant", "outline")
        self.add_button.clicked.connect(self.add_profile)
        self.remove_button = QPushButton(BUTTON_REMOVE_PROFILE, self)
        self.remove_button.setProperty("variant", "outline")
        self.remove_button.clicked.connect(self.remove_profile)
        list_buttons = QHBoxLayout()
        list_buttons.addWidget(self.add_button)
        list_buttons.addWidget(self.remove_button)
        left = QVBoxLayout()
        left.setSpacing(FORM_V_GAP)
        left.addWidget(QLabel(LABEL_PROFILES, self))
        left.addWidget(self.profile_list, 1)
        left.addLayout(list_buttons)

        self.name_edit = QLineEdit(self)
        self.api_type = QComboBox(self)
        self.api_type.addItems(list(api_types))
        self.base_url = QLineEdit(self)
        self.api_key = QLineEdit(self)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        for edit in (self.name_edit, self.base_url, self.api_key):
            edit.textChanged.connect(self._on_form_edited)
        self.api_type.currentIndexChanged.connect(self._on_form_edited)
        form = QFormLayout()
        form.addRow(LABEL_NAME, self.name_edit)
        form.addRow(LABEL_API_TYPE, self.api_type)
        form.addRow(LABEL_BASE_URL, self.base_url)
        form.addRow(LABEL_API_KEY, self.api_key)
        self._tune_form(form)

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

        self.models_label = QLabel(LABEL_MODELS, self)
        self.models_label.setProperty("muted", True)
        self.model_list = QListWidget(self)
        self.model_list.setMinimumHeight(MODEL_LIST_MIN_H)
        self.model_list.itemChanged.connect(self._on_model_toggled)
        self.enable_all_button = QPushButton(BUTTON_ENABLE_ALL, self)
        self.enable_all_button.setProperty("variant", "outline")
        self.enable_all_button.clicked.connect(lambda: self._set_all_models(True))
        self.disable_all_button = QPushButton(BUTTON_DISABLE_ALL, self)
        self.disable_all_button.setProperty("variant", "outline")
        self.disable_all_button.clicked.connect(lambda: self._set_all_models(False))
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.models_label, 1)
        toggle_row.addWidget(self.enable_all_button)
        toggle_row.addWidget(self.disable_all_button)
        self.model_filter = QLineEdit(self)
        self.model_filter.setPlaceholderText(PLACEHOLDER_MODEL_FILTER)
        self.model_filter.setClearButtonEnabled(True)
        self.model_filter.textChanged.connect(self._apply_model_filter)
        self.new_model_edit = QLineEdit(self)
        self.new_model_edit.setPlaceholderText(PLACEHOLDER_NEW_MODEL)
        self.new_model_edit.returnPressed.connect(self.add_model)
        self.add_model_button = QPushButton(BUTTON_ADD_MODEL, self)
        self.add_model_button.setProperty("variant", "outline")
        self.add_model_button.clicked.connect(self.add_model)
        add_row = QHBoxLayout()
        add_row.addWidget(self.new_model_edit, 1)
        add_row.addWidget(self.add_model_button)

        self.form_panel = QWidget(self)
        right = QVBoxLayout(self.form_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(FORM_V_GAP)
        right.addLayout(form)
        right.addLayout(probe_row)
        right.addLayout(toggle_row)
        right.addWidget(self.model_filter)
        right.addWidget(self.model_list, 1)
        right.addLayout(add_row)
        self.empty_hint = QLabel(EMPTY_HINT, self)
        self.empty_hint.setProperty("muted", True)
        self.empty_hint.setWordWrap(True)

        split = QHBoxLayout()
        split.setSpacing(SECTION_GAP)
        split.addLayout(left)
        split.addWidget(self.form_panel, 1)

        self.text_combo = QComboBox(self)
        self.vision_combo = QComboBox(self)
        self.text_combo.currentIndexChanged.connect(self._on_target_changed)
        self.vision_combo.currentIndexChanged.connect(self._on_target_changed)
        targets_form = QFormLayout()
        targets_form.addRow(LABEL_TEXT_TARGET, self.text_combo)
        targets_form.addRow(LABEL_VISION_TARGET, self.vision_combo)
        self._tune_form(targets_form)
        self.targets_hint = QLabel(HINT_TARGETS, self)
        self.targets_hint.setProperty("muted", True)
        self.targets_hint.setWordWrap(True)

        self.divider = QFrame(self)
        self.divider.setObjectName("settingsDivider")
        self.divider.setProperty("divider", True)
        self.divider.setFixedHeight(DIVIDER_H)
        self.divider.setFrameShape(QFrame.Shape.NoFrame)
        self.concurrency_spin = QSpinBox(self)
        self.concurrency_spin.setRange(*CONCURRENCY_RANGE)
        self.concurrency_spin.setValue(base.request.concurrency)
        request_form = QFormLayout()
        request_form.addRow(LABEL_CONCURRENCY, self.concurrency_spin)
        self._tune_form(request_form)

        column = QVBoxLayout(self)
        column.setSpacing(SECTION_GAP)
        column.addLayout(split, 1)
        column.addWidget(self.empty_hint)
        column.addLayout(targets_form)
        column.addWidget(self.targets_hint)
        column.addWidget(self.divider)
        column.addLayout(request_form)

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

    # -- public accessors -------------------------------------------------------------
    def profiles(self) -> tuple[LLMProfile, ...]:
        """Every draft profile, fields stripped."""
        return tuple(self._drafts)

    def text_target(self) -> ModelRef:
        return self._text_target

    def vision_target(self) -> ModelRef:
        return self._vision_target

    def concurrency(self) -> int:
        return self.concurrency_spin.value()

    def current_profile(self) -> LLMProfile | None:
        """The draft selected in the list, or None when the list is empty."""
        row = self.profile_list.currentRow()
        if 0 <= row < len(self._drafts):
            return self._drafts[row]
        return None

    def build_config(self, base: AppConfig) -> AppConfig:
        """``base`` with this tab's profiles / targets / concurrency applied.

        ``active_profile`` is cleared: the pool targets replace it, and a
        blank value keeps a deliberately cleared target from being re-derived.
        """
        return _dc_replace(
            base,
            profiles=self.profiles(),
            active_profile="",
            text_target=self._text_target,
            vision_target=self._vision_target,
            request=_dc_replace(base.request, concurrency=self.concurrency()),
        )

    def vision_profile(self) -> LLMProfile | None:
        """Draft profile bound to the 当前视觉模型 (CHA标注 sync source)."""
        return resolve_vision_profile(self.build_config(AppConfig()))

    def pool_groups(self) -> tuple[tuple[str, tuple[ModelChoice, ...]], ...]:
        """The draft pool grouped by provider (CHA标注 / 多对比推标 pickers)."""
        return grouped_pool_choices(self.build_config(AppConfig()))

    def profile_for_ref(self, ref: ModelRef) -> LLMProfile | None:
        """Draft pool profile for ``ref`` with ``vision_model`` bound (None if stale)."""
        return bind_model_ref(self.build_config(AppConfig()), ref, ROLE_VISION)

    def validate(self, *, require_text: bool) -> str | None:
        """Chinese error for the first problem, or None when the tab can be saved."""
        seen: set[str] = set()
        for profile in self._drafts:
            if not profile.name:
                return TOAST_NEED_NAME
            if profile.name in seen:
                return TOAST_DUPLICATE_NAME.format(name=profile.name)
            seen.add(profile.name)
            if not profile.base_url:
                return TOAST_NEED_PROFILE_URL.format(name=profile.name)
        if require_text and not self._text_target.is_set():
            return TOAST_NEED_TEXT_MODEL
        return None

    # -- profile list ----------------------------------------------------------------
    def add_profile(self) -> None:
        """Append an empty profile with a unique default name and select it."""
        names = {p.name for p in self._drafts}
        n = len(self._drafts) + 1
        while NEW_PROFILE_NAME.format(n=n) in names:
            n += 1
        self._drafts.append(
            LLMProfile(
                name=NEW_PROFILE_NAME.format(n=n),
                api_type=self.api_type.itemText(0) if self.api_type.count() else "",
                base_url="",
            )
        )
        self._rebuild_profile_list()
        self.profile_list.setCurrentRow(len(self._drafts) - 1)
        self._load_form()
        self._refresh_targets()
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def remove_profile(self) -> None:
        """Drop the selected profile; targets pointing at it become 未设置."""
        row = self.profile_list.currentRow()
        if not 0 <= row < len(self._drafts):
            return
        removed = self._drafts.pop(row)
        if self._text_target.profile == removed.name:
            self._text_target = ModelRef()
        if self._vision_target.profile == removed.name:
            self._vision_target = ModelRef()
        self._rebuild_profile_list()
        self.profile_list.setCurrentRow(min(row, len(self._drafts) - 1))
        self._load_form()
        self._refresh_targets()

    def select_profile(self, name: str) -> bool:
        """Select the row named ``name``; False when absent."""
        index = self._index_of(name)
        if index < 0:
            return False
        self.profile_list.setCurrentRow(index)
        return True

    def _index_of(self, name: str) -> int:
        for index, profile in enumerate(self._drafts):
            if profile.name == name:
                return index
        return -1

    def _rebuild_profile_list(self) -> None:
        self._syncing = True
        try:
            self.profile_list.clear()
            for profile in self._drafts:
                self.profile_list.addItem(profile.name)
        finally:
            self._syncing = False
        has_rows = bool(self._drafts)
        self.form_panel.setVisible(has_rows)
        self.empty_hint.setVisible(not has_rows)
        self.remove_button.setEnabled(has_rows)

    def _on_profile_selected(self, _row: int) -> None:
        if not self._syncing:
            self._load_form()

    # -- endpoint form ---------------------------------------------------------------
    def _load_form(self) -> None:
        profile = self.current_profile()
        self._syncing = True
        try:
            if profile is None:
                self.name_edit.clear()
                self.base_url.clear()
                self.api_key.clear()
                self.model_list.clear()
                return
            self.name_edit.setText(profile.name)
            index = self.api_type.findText(profile.api_type)
            if index >= 0:
                self.api_type.setCurrentIndex(index)
            self.base_url.setText(profile.base_url)
            self.api_key.setText(profile.api_key)
            self._load_model_list(profile)
        finally:
            self._syncing = False

    def _load_model_list(self, profile: LLMProfile) -> None:
        self.model_list.clear()
        enabled = set(profile.enabled_models)
        for model in profile.models:
            label = model + (VISION_TAG if looks_vision_capable(model) else "")
            item = QListWidgetItem(label, self.model_list)
            item.setData(CHOICE_REF_ROLE, model)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if model in enabled else Qt.CheckState.Unchecked
            )
        self._apply_model_filter()

    def _apply_model_filter(self, *_args: object) -> None:
        """Hide catalog rows whose id does not contain the filter (case-insensitive)."""
        needle = self.model_filter.text().strip().lower()
        total = self.model_list.count()
        shown = 0
        for i in range(total):
            item = self.model_list.item(i)
            model = str(item.data(CHOICE_REF_ROLE) or "")
            match = not needle or needle in model.lower()
            item.setHidden(not match)
            shown += int(match)
        self.models_label.setText(
            LABEL_MODELS_FILTERED.format(n=shown, total=total) if needle else LABEL_MODELS
        )

    def _visible_models(self) -> tuple[str, ...]:
        return tuple(
            str(self.model_list.item(i).data(CHOICE_REF_ROLE) or "")
            for i in range(self.model_list.count())
            if not self.model_list.item(i).isHidden()
        )

    def _on_form_edited(self, *_args: object) -> None:
        if self._syncing:
            return
        row = self.profile_list.currentRow()
        if not 0 <= row < len(self._drafts):
            return
        old = self._drafts[row]
        new_name = self.name_edit.text().strip()
        updated = _dc_replace(
            old,
            name=new_name,
            api_type=self.api_type.currentText(),
            base_url=self.base_url.text().strip(),
            api_key=self.api_key.text().strip(),
        )
        self._drafts[row] = updated
        if new_name != old.name:
            self._rename_targets(old.name, new_name)
            item = self.profile_list.item(row)
            if item is not None:
                item.setText(new_name)
            self._refresh_targets()

    def _rename_targets(self, old: str, new: str) -> None:
        if self._text_target.profile == old:
            self._text_target = _dc_replace(self._text_target, profile=new)
        if self._vision_target.profile == old:
            self._vision_target = _dc_replace(self._vision_target, profile=new)

    # -- model switches --------------------------------------------------------------
    def _update_profile(self, **changes: object) -> None:
        row = self.profile_list.currentRow()
        if 0 <= row < len(self._drafts):
            self._drafts[row] = _dc_replace(self._drafts[row], **changes)  # type: ignore[arg-type]

    def _on_model_toggled(self, _item: QListWidgetItem) -> None:
        if self._syncing:
            return
        enabled = tuple(
            str(self.model_list.item(i).data(CHOICE_REF_ROLE))
            for i in range(self.model_list.count())
            if self.model_list.item(i).checkState() == Qt.CheckState.Checked
        )
        self._update_profile(enabled_models=enabled)
        self._refresh_targets()

    def _set_all_models(self, on: bool) -> None:
        """全选 / 全不选 — scoped to the rows the filter currently shows."""
        profile = self.current_profile()
        if profile is None:
            return
        affected = set(self._visible_models())
        enabled = set(profile.enabled_models)
        enabled = enabled | affected if on else enabled - affected
        self._update_profile(
            enabled_models=tuple(m for m in profile.models if m in enabled)
        )
        self._syncing = True
        try:
            self._load_model_list(self._drafts[self.profile_list.currentRow()])
        finally:
            self._syncing = False
        self._refresh_targets()

    def add_model(self) -> None:
        """Add the typed model id to the catalog and switch it on."""
        model = self.new_model_edit.text().strip()
        profile = self.current_profile()
        if not model or profile is None:
            return
        self._merge_models(profile, (model,), enable=True)
        self.new_model_edit.clear()

    def _merge_models(
        self, profile: LLMProfile, names: Sequence[str], *, enable: bool
    ) -> None:
        models = list(profile.models)
        for name in names:
            if name and name not in models:
                models.append(name)
        enabled = list(profile.enabled_models)
        if enable:
            for name in names:
                if name and name not in enabled:
                    enabled.append(name)
        ordered_enabled = tuple(m for m in models if m in enabled)
        self._update_profile(models=tuple(models), enabled_models=ordered_enabled)
        self._syncing = True
        try:
            self._load_model_list(self._drafts[self.profile_list.currentRow()])
        finally:
            self._syncing = False
        self._refresh_targets()

    # -- targets -----------------------------------------------------------------------
    def _refresh_targets(self) -> None:
        config = self.build_config(AppConfig())
        choices = pool_choices(config)
        valid = {choice.ref for choice in choices}
        if self._text_target.is_set() and self._text_target not in valid:
            self._text_target = ModelRef()
        if self._vision_target.is_set() and self._vision_target not in valid:
            self._vision_target = ModelRef()
        self._syncing = True
        try:
            self._fill_combo(self.text_combo, choices, self._text_target)
            self._fill_combo(self.vision_combo, choices, self._vision_target)
        finally:
            self._syncing = False
        self.pool_changed.emit()

    @staticmethod
    def _fill_combo(
        combo: QComboBox, choices: Sequence[ModelChoice], current: ModelRef
    ) -> None:
        combo.clear()
        combo.addItem(LABEL_UNSET_CHOICE, ModelRef())
        selected = 0
        for position, choice in enumerate(choices, start=1):
            label = choice.label + (VISION_TAG if choice.vision_hint else "")
            combo.addItem(label, choice.ref)
            if choice.ref == current:
                selected = position
        combo.setCurrentIndex(selected)

    def _on_target_changed(self, *_args: object) -> None:
        if self._syncing:
            return
        text = self.text_combo.currentData()
        vision = self.vision_combo.currentData()
        self._text_target = text if isinstance(text, ModelRef) else ModelRef()
        self._vision_target = vision if isinstance(vision, ModelRef) else ModelRef()

    def set_target(self, role: str, ref: ModelRef) -> None:
        """Programmatic target pick (tests / 本地推理 apply); unknown refs -> 未设置."""
        if role == ROLE_TEXT:
            self._text_target = ref
        elif role == ROLE_VISION:
            self._vision_target = ref
        self._refresh_targets()

    # -- async probes -----------------------------------------------------------------
    def test_connection(self) -> None:
        """Async connectivity probe against the first enabled model; toasts result."""
        profile = self.current_profile()
        if profile is None or not profile.base_url:
            self._toast(TOAST_NEED_BASE_URL, TOAST_WARN)
            return
        model = next(iter(profile.enabled_models), next(iter(profile.models), ""))
        if not model:
            self._toast(TOAST_NEED_MODEL, TOAST_WARN)
            return
        self.test_button.setEnabled(False)

        def probe() -> float:
            started = time.monotonic()
            create_client(profile).test_connection(model)
            return time.monotonic() - started

        def done(elapsed: object) -> None:
            self.test_button.setEnabled(True)
            seconds = elapsed if isinstance(elapsed, float) else 0.0
            self._toast(TOAST_TEST_OK.format(seconds=seconds), TOAST_OK)

        def failed(message: str) -> None:
            self.test_button.setEnabled(True)
            self._toast(TOAST_TEST_FAIL.format(message=message), TOAST_ERR)

        run_async(self._pool, probe, on_done=done, on_error=failed)

    def fetch_models(self) -> None:
        """获取模型: async listing merged into the selected profile's catalog."""
        profile = self.current_profile()
        if profile is None or not profile.base_url:
            self._toast(TOAST_NEED_BASE_URL, TOAST_WARN)
            return
        self.fetch_models_button.setEnabled(False)
        target_name = profile.name

        def probe() -> tuple[str, ...]:
            return list_models(profile)

        def done(models: object) -> None:
            self.fetch_models_button.setEnabled(True)
            names = tuple(models) if isinstance(models, tuple) else ()
            if not names:
                self._toast(TOAST_FETCH_EMPTY, TOAST_WARN)
                return
            self._apply_fetched(target_name, names)

        def failed(message: str) -> None:
            self.fetch_models_button.setEnabled(True)
            self._toast(TOAST_FETCH_FAILED.format(message=message), TOAST_ERR)

        run_async(self._pool, probe, on_done=done, on_error=failed)

    def _apply_fetched(self, name: str, names: tuple[str, ...]) -> None:
        # The user may have switched rows while the probe ran: merge into the
        # profile that was probed, then re-select it so the checklist shows it.
        index = self._index_of(name)
        if index < 0:
            return
        if self.profile_list.currentRow() != index:
            self.profile_list.setCurrentRow(index)
        self._merge_models(self._drafts[index], names, enable=False)
        self._toast(TOAST_FETCHED.format(n=len(names)), TOAST_OK)

    def _toast(self, text: str, kind: str) -> None:
        self.toast_requested.emit(text, kind)
