"""设置 dialog - 翻译服务 / LLM 设置 / 提示词 / 本地推理 four-tab window.

Spec module 3: the translation-service selection and the LLM profile are two
separate tabs; the LLM tab supports a 统一 (one multimodal model for text +
vision) or split text/vision model configuration, a request-concurrency
setting and a 获取模型 probe that lists what the endpoint offers; the 提示词
tab manages the custom system/user prompts used before image inference; the
本地推理 tab reserves the future in-app local-model module.

Edits a single ``default`` profile of the core :class:`AppConfig`. 测试连接
runs an async connectivity probe; 保存 persists ``app_data_dir()/config.json``
via the core ``save_config`` and asks the controller to rebuild its translator.

This dialog is the sanctioned exception to the "widgets only talk to the
controller" rule: the migration contract routes config IO and the connection
test through the core config/client APIs directly.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.config import AppConfig, LLMProfile, load_config, save_config
from nlapt.core.errors import LLMError, NLaptError, StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.llm.base import create_client
from nlapt.llm.model_list import list_models
from nlapt.llm.translate import Direction, Translator, detect_direction
from nlapt.llm.web_translate import (
    PROVIDER_BAIDU,
    PROVIDER_DEEPL,
    PROVIDER_LLM,
    REGISTRATION_INFO,
    create_provider,
)

from nlapt_gui.controller import AppController, TOAST_ERR, TOAST_OK, TOAST_WARN
from nlapt_gui.prompt_store import load_vision_prompts, save_vision_prompts
from nlapt_gui.resources import app_data_dir
from nlapt_gui.translate_config import (
    KNOWN_PROVIDERS,
    TranslationConfig,
    load_translation_config,
    save_translation_config,
)
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.model_picker import ModelPickerDialog
from nlapt_gui.widgets.prompt_editor import PromptsTab
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

CONFIG_FILE_NAME = "config.json"
DEFAULT_PROFILE_NAME = "default"
DEFAULT_API_TYPES: tuple[str, ...] = ("openai", "anthropic", "ollama")
DIALOG_WIDTH = 480
DIALOG_MIN_HEIGHT = 460
SECTION_GAP = 10
DIVIDER_H = 1
CONCURRENCY_RANGE = (1, 32)

# UI strings.
WINDOW_TITLE = "设置"
TAB_TRANSLATE = "翻译服务"
TAB_LLM = "LLM 设置"
TAB_PROMPTS = "提示词"
TAB_LOCAL = "本地推理"
LABEL_API_TYPE = "接口类型"
LABEL_BASE_URL = "Base URL"
LABEL_API_KEY = "API Key"
LABEL_TEXT_MODEL = "文本模型"
LABEL_VISION_MODEL = "视觉模型"
LABEL_MODEL_UNIFIED = "模型"
LABEL_UNIFIED = "统一文本 / 视觉模型"
HINT_UNIFIED = "统一模式下该模型同时用于文本与图片推理 — 需选择具备多模态(视觉)能力的模型。"
HINT_SEPARATE = "视觉模型用于图片推理(重译);留空则无法使用重译。"
LABEL_CONCURRENCY = "并发请求数"
BUTTON_FETCH_MODELS = "获取模型"
BUTTON_TEST = "测试连接"
BUTTON_SAVE = "保存"
BUTTON_CANCEL = "取消"
TOAST_TEST_OK = "连接成功"
TOAST_TEST_FAIL = "连接失败: {message}"
TOAST_SAVED = "设置已保存"
TOAST_NEED_BASE_URL = "请填写 Base URL"
TOAST_NEED_TEXT_MODEL = "请填写文本模型"
TOAST_LOAD_FAILED = "无法读取现有配置,已取消保存以避免覆盖其它设置"
TOAST_FETCH_FAILED = "获取模型失败: {message}"
TOAST_FETCH_EMPTY = "接口没有返回任何模型"

# Translation-provider tab.
LABEL_TR_PROVIDER = "翻译服务"
LABEL_BAIDU_APPID = "百度 APPID"
LABEL_BAIDU_KEY = "百度密钥"
LABEL_DEEPL_KEY = "DeepL API Key"
BUTTON_TEST_TRANSLATE = "测试翻译"
TR_NOTE = "翻译服务用于工作区的 翻译 / 重译 与右栏的翻译对照;选择「大模型」时使用下方 LLM 设置中的档案。"
TR_TEST_SAMPLE = "你好世界"
TOAST_TR_TEST_OK = "翻译测试成功: {result}"
TOAST_TR_TEST_FAIL = "翻译测试失败: {message}"
TOAST_TR_NEED_KEY = "请先填写所选翻译服务所需的密钥"

# 本地推理 placeholder tab.
LOCAL_TITLE = "本地推理模型"
LOCAL_BADGE = "规划中"
LOCAL_DESC = (
    "该模块用于在软件内下载并管理本地推理模型,离线为图片生成标注"
    "(无需配置在线 API)。当前版本为占位界面,模型下载与推理将在后续版本开放。"
)
LOCAL_BUTTON = "下载模型(即将推出)"

# Human-readable dropdown labels per provider id (UI stays Chinese).
PROVIDER_LABELS: dict[str, str] = {
    "llm": "大模型 (LLM)",
    "google": "Google 翻译 (免费)",
    "baidu": "百度翻译",
    "deepl": "DeepL",
}


def config_path() -> Path:
    """Location of the persisted core config."""
    return app_data_dir() / CONFIG_FILE_NAME


class SettingsDialog(CenteredDialog):
    """Modal 设置 dialog: four tabs, centered on the app with fade in/out."""

    saved = Signal(object)  # AppConfig just persisted

    def __init__(
        self,
        controller: AppController,
        *,
        api_types: Sequence[str] = DEFAULT_API_TYPES,
        pool: QThreadPool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(True)
        self.setMinimumWidth(DIALOG_WIDTH)
        self.setMinimumHeight(DIALOG_MIN_HEIGHT)

        # Read the existing config ONCE. A genuine read/parse failure is kept
        # (not masked as defaults) so save() can refuse to overwrite and destroy
        # the user's other profiles/templates.
        self._existing: AppConfig | None
        self._load_error: str | None
        try:
            self._existing = load_config(config_path())
            self._load_error = None
        except (ValidationError, StorageError) as exc:
            self._existing = None
            self._load_error = str(exc)
            _LOGGER.error("could not read existing config: %s", exc)
        self._translation_config = load_translation_config()

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_translate_tab(), TAB_TRANSLATE)
        self.tabs.addTab(self._build_llm_tab(api_types), TAB_LLM)
        self.prompts_tab = PromptsTab(load_vision_prompts(), self)
        self.prompts_tab.toast_requested.connect(self._controller.toast_requested.emit)
        self.tabs.addTab(self.prompts_tab, TAB_PROMPTS)
        self.tabs.addTab(self._build_local_tab(), TAB_LOCAL)

        self.cancel_button = QPushButton(BUTTON_CANCEL, self)
        self.cancel_button.setProperty("variant", "outline")
        self.cancel_button.clicked.connect(self.reject)
        self.save_button = QPushButton(BUTTON_SAVE, self)
        self.save_button.setProperty("variant", "accent")
        self.save_button.clicked.connect(self.save)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, 1)
        layout.addLayout(buttons)

        self._prefill()
        self._prefill_translation()
        self._on_provider_changed()
        self._on_unified_toggled()

    # -- tab construction --------------------------------------------------------------
    def _build_translate_tab(self) -> QWidget:
        tab = QWidget(self)
        self.provider = QComboBox(tab)
        for provider_id in KNOWN_PROVIDERS:
            self.provider.addItem(PROVIDER_LABELS.get(provider_id, provider_id), provider_id)
        self.provider.currentIndexChanged.connect(self._on_provider_changed)
        self.baidu_appid = QLineEdit(tab)
        self.baidu_key = QLineEdit(tab)
        self.baidu_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepl_key = QLineEdit(tab)
        self.deepl_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.registration_note = QLabel(tab)
        self.registration_note.setWordWrap(True)
        self.registration_note.setProperty("muted", True)
        self.registration_note.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.registration_note.setOpenExternalLinks(True)
        self.test_translate_button = QPushButton(BUTTON_TEST_TRANSLATE, tab)
        self.test_translate_button.setProperty("variant", "outline")
        self.test_translate_button.clicked.connect(self.test_translate)

        tr_form = QFormLayout()
        self._tr_form = tr_form
        tr_form.addRow(LABEL_TR_PROVIDER, self.provider)
        self._baidu_appid_label = QLabel(LABEL_BAIDU_APPID, tab)
        self._baidu_key_label = QLabel(LABEL_BAIDU_KEY, tab)
        self._deepl_key_label = QLabel(LABEL_DEEPL_KEY, tab)
        tr_form.addRow(self._baidu_appid_label, self.baidu_appid)
        tr_form.addRow(self._baidu_key_label, self.baidu_key)
        tr_form.addRow(self._deepl_key_label, self.deepl_key)

        note = QLabel(TR_NOTE, tab)
        note.setProperty("muted", True)
        note.setWordWrap(True)

        button_row = QHBoxLayout()
        button_row.addWidget(self.test_translate_button)
        button_row.addStretch(1)

        column = QVBoxLayout(tab)
        column.setSpacing(SECTION_GAP)
        column.addLayout(tr_form)
        column.addWidget(self.registration_note)
        column.addLayout(button_row)
        column.addWidget(note)
        column.addStretch(1)
        return tab

    def _build_llm_tab(self, api_types: Sequence[str]) -> QWidget:
        tab = QWidget(self)
        self.api_type = QComboBox(tab)
        self.api_type.addItems(list(api_types))
        self.base_url = QLineEdit(tab)
        self.api_key = QLineEdit(tab)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.unified_check = QCheckBox(LABEL_UNIFIED, tab)
        self.unified_check.toggled.connect(self._on_unified_toggled)
        self.text_model = QLineEdit(tab)
        self.vision_model = QLineEdit(tab)

        form = QFormLayout()
        self._llm_form = form
        form.addRow(LABEL_API_TYPE, self.api_type)
        form.addRow(LABEL_BASE_URL, self.base_url)
        form.addRow(LABEL_API_KEY, self.api_key)
        form.addRow("", self.unified_check)
        self._text_model_label = QLabel(LABEL_TEXT_MODEL, tab)
        self._vision_model_label = QLabel(LABEL_VISION_MODEL, tab)
        form.addRow(self._text_model_label, self.text_model)
        form.addRow(self._vision_model_label, self.vision_model)

        self.model_hint = QLabel(HINT_SEPARATE, tab)
        self.model_hint.setProperty("muted", True)
        self.model_hint.setWordWrap(True)

        # Themed 1px divider between the model form and the request controls.
        self.divider = QFrame(tab)
        self.divider.setObjectName("settingsDivider")
        self.divider.setProperty("divider", True)
        self.divider.setFixedHeight(DIVIDER_H)
        self.divider.setFrameShape(QFrame.Shape.NoFrame)

        self.concurrency_spin = QSpinBox(tab)
        self.concurrency_spin.setRange(*CONCURRENCY_RANGE)
        existing = self._existing
        self.concurrency_spin.setValue(
            existing.request.concurrency if existing is not None else AppConfig().request.concurrency
        )
        request_form = QFormLayout()
        request_form.addRow(LABEL_CONCURRENCY, self.concurrency_spin)

        self.fetch_models_button = QPushButton(BUTTON_FETCH_MODELS, tab)
        self.fetch_models_button.setProperty("variant", "outline")
        self.fetch_models_button.clicked.connect(self.fetch_models)
        self.test_button = QPushButton(BUTTON_TEST, tab)
        self.test_button.setProperty("variant", "outline")
        self.test_button.clicked.connect(self.test_connection)
        probe_row = QHBoxLayout()
        probe_row.addWidget(self.fetch_models_button)
        probe_row.addWidget(self.test_button)
        probe_row.addStretch(1)

        column = QVBoxLayout(tab)
        column.setSpacing(SECTION_GAP)
        column.addLayout(form)
        column.addWidget(self.model_hint)
        column.addWidget(self.divider)
        column.addLayout(request_form)
        column.addLayout(probe_row)
        column.addStretch(1)
        return tab

    def _build_local_tab(self) -> QWidget:
        tab = QWidget(self)
        title_row = QHBoxLayout()
        title = QLabel(LOCAL_TITLE, tab)
        title.setStyleSheet("font-weight: 700; font-size: 13.5px;")
        title_row.addWidget(title)
        badge = QLabel(LOCAL_BADGE, tab)
        badge.setProperty("pill", "accentSoft")
        title_row.addWidget(badge)
        title_row.addStretch(1)
        description = QLabel(LOCAL_DESC, tab)
        description.setProperty("muted", True)
        description.setWordWrap(True)
        self.local_download_button = QPushButton(LOCAL_BUTTON, tab)
        self.local_download_button.setProperty("variant", "outline")
        self.local_download_button.setEnabled(False)
        column = QVBoxLayout(tab)
        column.setSpacing(SECTION_GAP)
        column.addLayout(title_row)
        column.addWidget(description)
        column.addWidget(self.local_download_button, 0, Qt.AlignmentFlag.AlignLeft)
        column.addStretch(1)
        return tab

    # -- public accessors ---------------------------------------------------------------
    def selected_provider(self) -> str:
        """Provider id chosen in the 翻译服务 dropdown."""
        data = self.provider.currentData()
        return str(data) if data is not None else PROVIDER_LLM

    def translation_config(self) -> TranslationConfig:
        """TranslationConfig built from the current 翻译服务 field values."""
        return TranslationConfig(
            provider=self.selected_provider(),
            baidu_appid=self.baidu_appid.text().strip(),
            baidu_key=self.baidu_key.text().strip(),
            deepl_key=self.deepl_key.text().strip(),
        )

    def is_unified(self) -> bool:
        """Whether one multimodal model backs both text and vision requests."""
        return self.unified_check.isChecked()

    def current_profile(self) -> LLMProfile:
        """Profile built from the current field values (统一 → shared model)."""
        text_model = self.text_model.text().strip()
        vision_model = text_model if self.is_unified() else self.vision_model.text().strip()
        return LLMProfile(
            name=DEFAULT_PROFILE_NAME,
            api_type=self.api_type.currentText(),
            base_url=self.base_url.text().strip(),
            api_key=self.api_key.text().strip(),
            text_model=text_model,
            vision_model=vision_model,
        )

    # -- async probes ---------------------------------------------------------------------
    def test_connection(self) -> None:
        """Async connectivity probe with a result toast."""
        profile = self.current_profile()
        if not self._validate(profile):
            return
        self.test_button.setEnabled(False)

        def probe() -> bool:
            return create_client(profile).test_connection(profile.text_model)

        def done(_ok: object) -> None:
            self.test_button.setEnabled(True)
            self._toast(TOAST_TEST_OK, TOAST_OK)

        def failed(message: str) -> None:
            self.test_button.setEnabled(True)
            self._toast(TOAST_TEST_FAIL.format(message=message), TOAST_ERR)

        run_async(self._pool, probe, on_done=done, on_error=failed)

    def fetch_models(self) -> None:
        """获取模型: async model-list probe, then the assignment picker."""
        profile = self.current_profile()
        if not profile.base_url:
            self._toast(TOAST_NEED_BASE_URL, TOAST_WARN)
            return
        self.fetch_models_button.setEnabled(False)

        def probe() -> tuple[str, ...]:
            return list_models(profile)

        def done(models: object) -> None:
            self.fetch_models_button.setEnabled(True)
            names = tuple(models) if isinstance(models, tuple) else ()
            if not names:
                self._toast(TOAST_FETCH_EMPTY, TOAST_WARN)
                return
            self._open_model_picker(names)

        def failed(message: str) -> None:
            self.fetch_models_button.setEnabled(True)
            self._toast(TOAST_FETCH_FAILED.format(message=message), TOAST_ERR)

        run_async(self._pool, probe, on_done=done, on_error=failed)

    def _open_model_picker(self, models: tuple[str, ...]) -> None:
        picker = ModelPickerDialog(models, unified=self.is_unified(), parent=self)
        picker.text_model_picked.connect(self.text_model.setText)
        picker.vision_model_picked.connect(self.vision_model.setText)
        picker.exec()

    def test_translate(self) -> None:
        """Translate a sample string via the selected provider; toast result."""
        provider_id = self.selected_provider()
        direction = detect_direction(TR_TEST_SAMPLE)
        try:
            translate = self._build_test_translator(provider_id, direction)
        except LLMError as exc:
            self._toast(TOAST_TR_TEST_FAIL.format(message=exc.message), TOAST_ERR)
            return
        if translate is None:
            return
        self.test_translate_button.setEnabled(False)

        def done(result: object) -> None:
            self.test_translate_button.setEnabled(True)
            self._toast(TOAST_TR_TEST_OK.format(result=result), TOAST_OK)

        def failed(message: str) -> None:
            self.test_translate_button.setEnabled(True)
            self._toast(TOAST_TR_TEST_FAIL.format(message=message), TOAST_ERR)

        run_async(self._pool, translate, on_done=done, on_error=failed)

    # -- save --------------------------------------------------------------------------
    def save(self) -> None:
        """Persist translation + LLM + prompt settings, reload the translator."""
        provider_id = self.selected_provider()
        write_llm = provider_id == PROVIDER_LLM or self._llm_fields_present()
        # Provider-specific validation. The LLM path keeps the original
        # requirement (Base URL + 文本模型); web providers only need their key.
        if provider_id == PROVIDER_LLM:
            if not self._validate(self.current_profile()):
                return
        elif not self._validate_web(provider_id):
            return
        # Refuse to overwrite an unreadable core config BEFORE writing anything.
        if write_llm and (self._load_error is not None or self._existing is None):
            self._toast(TOAST_LOAD_FAILED, TOAST_ERR)
            return
        save_translation_config(self.translation_config())
        try:
            save_vision_prompts(self.prompts_tab.current_prompts())
        except NLaptError as exc:  # prompts must not block the LLM/provider save
            _LOGGER.exception("could not persist vision prompts")
            self._toast(str(exc), TOAST_ERR)
        saved_config: AppConfig | None = None
        if write_llm:
            profile = self.current_profile()
            existing = self._existing
            assert existing is not None  # guarded above
            others = tuple(
                p for p in existing.profiles if p.name != DEFAULT_PROFILE_NAME
            )
            request = _dc_replace(
                existing.request, concurrency=self.concurrency_spin.value()
            )
            config = AppConfig(
                profiles=(profile, *others),
                active_profile=DEFAULT_PROFILE_NAME,
                request=request,
                image_max_edge=existing.image_max_edge,
                revert_confirmed_on_edit=existing.revert_confirmed_on_edit,
                snapshot_retention=existing.snapshot_retention,
                prompt_orphan_cleanup=existing.prompt_orphan_cleanup,
                custom_templates=existing.custom_templates,
                trigger_presets=existing.trigger_presets,
            )
            save_config(config_path(), config)
            self._controller.reload_config(config)
            saved_config = config
        else:
            # No LLM write: still drop the cached translator so a provider
            # switch away from/back to LLM re-resolves cleanly.
            self._controller.invalidate_translator()
        _LOGGER.info("settings saved (translation provider=%s)", provider_id)
        self._toast(TOAST_SAVED, TOAST_OK)
        self.saved.emit(saved_config if saved_config is not None else self._existing)
        self.accept()

    # -- internals -------------------------------------------------------------------
    def _llm_fields_present(self) -> bool:
        return bool(self.base_url.text().strip()) and bool(self.text_model.text().strip())

    def _validate(self, profile: LLMProfile) -> bool:
        if not profile.base_url:
            self._toast(TOAST_NEED_BASE_URL, TOAST_WARN)
            return False
        if not profile.text_model:
            self._toast(TOAST_NEED_TEXT_MODEL, TOAST_WARN)
            return False
        return True

    def _validate_web(self, provider_id: str) -> bool:
        """Ensure the selected web provider has its required credentials."""
        info = REGISTRATION_INFO.get(provider_id, {})
        if not info.get("needs_key"):
            return True
        try:
            create_provider(provider_id, self.translation_config().credentials())
        except LLMError:
            self._toast(TOAST_TR_NEED_KEY, TOAST_WARN)
            return False
        return True

    def _build_test_translator(
        self, provider_id: str, direction: Direction
    ) -> Callable[[], str] | None:
        """Return a zero-arg callable translating the sample, or None on invalid input."""
        if provider_id == PROVIDER_LLM:
            profile = self.current_profile()
            if not self._validate(profile):
                return None
            translator = Translator(create_client(profile), profile)
            return lambda: translator.translate(
                "settings-test", TR_TEST_SAMPLE, direction
            ).translated
        provider = create_provider(
            provider_id, self.translation_config().credentials()
        )
        return lambda: provider.translate(TR_TEST_SAMPLE, direction)

    def _on_provider_changed(self, *_args: object) -> None:
        """Reflect the selected provider: show ONLY that provider's key rows.

        Uses ``QFormLayout.setRowVisible`` so a hidden row collapses entirely -
        both its label and field are removed from the layout, leaving no stray
        empty row, gap, or bare line. Switching back to a provider fully
        restores its rows. LLM and Google have no key rows at all.
        """
        provider_id = self.selected_provider()
        info = REGISTRATION_INFO.get(provider_id, {})
        self.registration_note.setText(str(info.get("note_zh", "")))
        is_baidu = provider_id == PROVIDER_BAIDU
        is_deepl = provider_id == PROVIDER_DEEPL
        self._tr_form.setRowVisible(self.baidu_appid, is_baidu)
        self._tr_form.setRowVisible(self.baidu_key, is_baidu)
        self._tr_form.setRowVisible(self.deepl_key, is_deepl)

    def _on_unified_toggled(self, *_args: object) -> None:
        """统一模式: one shared model row + multimodal hint; otherwise split rows."""
        unified = self.is_unified()
        self._text_model_label.setText(LABEL_MODEL_UNIFIED if unified else LABEL_TEXT_MODEL)
        self._llm_form.setRowVisible(self.vision_model, not unified)
        self.model_hint.setText(HINT_UNIFIED if unified else HINT_SEPARATE)

    def _toast(self, text: str, kind: str) -> None:
        self._controller.toast_requested.emit(text, kind)

    def _prefill_translation(self) -> None:
        config = self._translation_config
        index = self.provider.findData(config.provider)
        if index >= 0:
            self.provider.setCurrentIndex(index)
        self.baidu_appid.setText(config.baidu_appid)
        self.baidu_key.setText(config.baidu_key)
        self.deepl_key.setText(config.deepl_key)

    def _prefill(self) -> None:
        config = self._existing
        if config is None:
            return
        profile = next(
            (p for p in config.profiles if p.name == config.active_profile), None
        )
        if profile is None and config.profiles:
            profile = config.profiles[0]
        if profile is None:
            return
        index = self.api_type.findText(profile.api_type)
        if index >= 0:
            self.api_type.setCurrentIndex(index)
        self.base_url.setText(profile.base_url)
        self.api_key.setText(profile.api_key)
        self.text_model.setText(profile.text_model)
        self.vision_model.setText(profile.vision_model)
        unified = bool(profile.text_model) and profile.text_model == profile.vision_model
        self.unified_check.setChecked(unified)
