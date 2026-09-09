"""设置 dialog - 翻译服务 / LLM 设置 / 提示词 / 本地推理 / CHA标注 five-tab window.

Spec module 3: the translation-service selection and the LLM APIs are two
separate tabs; the LLM tab (:class:`LLMProvidersTab`) stores several API
profiles with per-model switches and picks the 当前文本模型 / 当前视觉模型
from the resulting pool; the 提示词 tab manages the custom system/user prompts
used before image inference; the 本地推理 tab hosts the in-app local-model
module.

保存 writes LLM profiles + pool targets to ``Documents/NLapt/api.json`` and
non-interface fields to ``config.json``.

This dialog is the sanctioned exception to the "widgets only talk to the
controller" rule: the migration contract routes config IO and the connection
test through the core config/client APIs directly.
"""

from __future__ import annotations

import time
from dataclasses import replace as _dc_replace
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.config import AppConfig, resolve_text_profile
from nlapt.core.errors import LLMError, NLaptError, StorageError, ValidationError
from nlapt.diagnostics import configure_logging, get_logger
from nlapt.llm.base import create_client
from nlapt.llm.translate import Direction, Translator, detect_direction
from nlapt.llm.web_translate import (
    PROVIDER_BAIDU,
    PROVIDER_CUSTOM,
    PROVIDER_DEEPL,
    PROVIDER_DEEPLX,
    PROVIDER_LLM,
    PROVIDER_LOCAL_MT,
    REGISTRATION_INFO,
    create_provider,
)
from nlapt.local.mt_catalog import DEFAULT_TIER

from nlapt_gui.api_config import load_app_config, save_app_config
from nlapt_gui.cha_config import API_MODE_OWN, load_cha_settings, save_cha_settings
from nlapt_gui.compare_config import load_compare_settings, save_compare_settings
from nlapt_gui.controller import AppController, TOAST_ERR, TOAST_OK, TOAST_WARN
from nlapt_gui.mt_bridge import LocalMTProvider, is_tier_downloaded
from nlapt_gui.prompt_store import load_vision_prompts, save_vision_prompts
from nlapt_gui.resources import app_data_dir, config_path
from nlapt_gui.settings import load_ui_settings, save_ui_settings
from nlapt_gui.translate_config import (
    KNOWN_PROVIDERS,
    PROVIDER_LABELS,
    TranslationConfig,
    load_translation_config,
    save_translation_config,
)
from nlapt_gui.widgets.cha_tab import CHATab
from nlapt_gui.widgets.compare_tab import CompareTab
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.fallback_order import LABEL_FALLBACK, FallbackOrderEditor
from nlapt_gui.widgets.llm_providers_tab import (
    DEFAULT_API_TYPES,
    TOAST_NEED_BASE_URL,
    TOAST_NEED_TEXT_MODEL,
    LLMProvidersTab,
)
from nlapt_gui.widgets.local_tab import LocalTab
from nlapt_gui.widgets.mt_models_dialog import MTModelsDialog
from nlapt_gui.widgets.mt_tier_picker import MtTierPicker
from nlapt_gui.widgets.prompt_editor import PromptsTab
from nlapt_gui.workers import run_async, set_debug

# Re-export so existing tests keep importing config_path from this module.
_ = config_path

_LOGGER = get_logger(__name__)

DIALOG_WIDTH = 480
DIALOG_MIN_HEIGHT = 620
SECTION_GAP = 10
FORM_LABEL_MIN_W = 88
FORM_V_GAP = 8

# Re-exported so callers keep one import site for the LLM-tab toasts.
_ = (TOAST_NEED_BASE_URL, TOAST_NEED_TEXT_MODEL)

# UI strings.
WINDOW_TITLE = "设置"
TAB_TRANSLATE = "翻译服务"
TAB_LLM = "LLM 设置"
TAB_PROMPTS = "提示词"
TAB_LOCAL = "本地推理"
TAB_CHA = "CHA标注"
TAB_COMPARE = "多对比推标"
BUTTON_SAVE = "保存"
BUTTON_CANCEL = "取消"
TOAST_SAVED = "设置已保存"
TOAST_LOAD_FAILED = "无法读取现有配置,已取消保存以避免覆盖其它设置"

# Translation-provider tab.
LABEL_TR_PROVIDER = "翻译服务"
LABEL_BAIDU_APPID = "百度 APPID"
LABEL_BAIDU_KEY = "百度密钥"
LABEL_DEEPL_KEY = "DeepL API Key"
BUTTON_TEST_TRANSLATE = "测试翻译"
TR_NOTE = (
    "翻译服务用于工作区的 翻译 / 重译 与右栏的翻译对照;"
    "选择「大模型」时使用 LLM 设置中的当前文本模型。"
    "首选失败时按备选顺序继续尝试。"
)
LABEL_DEBUG = "调试模式"
HINT_DEBUG = "打开后在启动器控制台打印详细日志（含异常堆栈）"
LOG_DIR_NAME = "logs"
TR_TEST_SAMPLE = "你好世界"
TOAST_TR_TEST_OK = "翻译测试成功: {result}"
TOAST_TR_TEST_FAIL = "翻译测试失败: {message}"
TOAST_TR_NEED_KEY = "请先填写所选翻译服务所需的密钥"
TOAST_TR_NEED_CUSTOM = "请填写自定义翻译的 Base URL 与模型"
TOAST_TR_NEED_DEEPLX = "请填写 DeepLX 接口地址"
TOAST_TR_NEED_LOCAL_MT = "请先下载所选档位的本地翻译模型"
LABEL_DEEPLX_URL = "DeepLX 地址"
LABEL_DEEPLX_TOKEN = "访问令牌"
HINT_DEEPLX_URL = "http://127.0.0.1:1188"
LABEL_CUSTOM_URL = "Base URL"
LABEL_CUSTOM_KEY = "API Key"
LABEL_CUSTOM_MODEL = "模型"


class SettingsDialog(CenteredDialog):
    """Modal 设置 dialog: five tabs, centered on the app with fade in/out."""

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
            self._existing = load_app_config()
            self._load_error = None
        except (ValidationError, StorageError) as exc:
            self._existing = None
            self._load_error = str(exc)
            _LOGGER.error("could not read existing config: %s", exc)
        self._translation_config = load_translation_config()

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_translate_tab(), TAB_TRANSLATE)
        self.llm_tab = LLMProvidersTab(
            self._existing, api_types=api_types, pool=self._pool, parent=self
        )
        self.llm_tab.toast_requested.connect(self._controller.toast_requested.emit)
        self.tabs.addTab(self.llm_tab, TAB_LLM)
        self.prompts_tab = PromptsTab(load_vision_prompts(), self)
        self.prompts_tab.toast_requested.connect(self._controller.toast_requested.emit)
        self.tabs.addTab(self.prompts_tab, TAB_PROMPTS)
        self.local_tab = LocalTab(controller, pool=self._pool, parent=self)
        self.tabs.addTab(self.local_tab, TAB_LOCAL)
        self.cha_tab = CHATab(
            main_profile_provider=self.llm_tab.vision_profile,
            pool_provider=self.llm_tab.pool_groups,
            profile_lookup=self.llm_tab.profile_for_ref,
            api_types=api_types,
            pool=self._pool,
            parent=self,
        )
        self.cha_tab.toast_requested.connect(self._controller.toast_requested.emit)
        # Toggling a model switch on the LLM tab updates the CHA dropdowns live.
        self.llm_tab.pool_changed.connect(self.cha_tab.refresh_pool_models)
        self.tabs.addTab(self.cha_tab, TAB_CHA)
        self.compare_tab = CompareTab(pool_provider=self.llm_tab.pool_groups, parent=self)
        self.compare_tab.prefill(load_compare_settings())
        self.llm_tab.pool_changed.connect(self.compare_tab.refresh_pool)
        self.tabs.addTab(self.compare_tab, TAB_COMPARE)

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

        self.cha_tab.prefill(load_cha_settings())
        self.cha_tab.refresh_pool_models()
        self._prefill_translation()
        self._on_provider_changed()

    @property
    def divider(self) -> QWidget:
        """The themed 1px divider on the LLM tab (design: no bare HLine)."""
        return self.llm_tab.divider

    @property
    def concurrency_spin(self) -> QWidget:
        return self.llm_tab.concurrency_spin

    # -- tab construction --------------------------------------------------------------
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
        self.deeplx_url = QLineEdit(tab)
        self.deeplx_url.setPlaceholderText(HINT_DEEPLX_URL)
        self.deeplx_token = QLineEdit(tab)
        self.deeplx_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.custom_base_url = QLineEdit(tab)
        self.custom_api_key = QLineEdit(tab)
        self.custom_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.custom_model = QLineEdit(tab)
        self.local_mt_tier = MtTierPicker(tab)
        self.local_mt_tier.manage_button.clicked.connect(self._open_mt_models)
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
        self._deeplx_url_label = QLabel(LABEL_DEEPLX_URL, tab)
        self._deeplx_token_label = QLabel(LABEL_DEEPLX_TOKEN, tab)
        self._custom_url_label = QLabel(LABEL_CUSTOM_URL, tab)
        self._custom_key_label = QLabel(LABEL_CUSTOM_KEY, tab)
        self._custom_model_label = QLabel(LABEL_CUSTOM_MODEL, tab)
        tr_form.addRow(self._baidu_appid_label, self.baidu_appid)
        tr_form.addRow(self._baidu_key_label, self.baidu_key)
        tr_form.addRow(self._deepl_key_label, self.deepl_key)
        tr_form.addRow(self._deeplx_url_label, self.deeplx_url)
        tr_form.addRow(self._deeplx_token_label, self.deeplx_token)
        tr_form.addRow(self._custom_url_label, self.custom_base_url)
        tr_form.addRow(self._custom_key_label, self.custom_api_key)
        tr_form.addRow(self._custom_model_label, self.custom_model)
        self._tune_form(tr_form)

        note = QLabel(TR_NOTE, tab)
        note.setProperty("muted", True)
        note.setWordWrap(True)
        self.debug_check = QCheckBox(LABEL_DEBUG, tab)
        debug_hint = QLabel(HINT_DEBUG, tab)
        debug_hint.setProperty("muted", True)
        debug_hint.setWordWrap(True)

        self.fallback_editor = FallbackOrderEditor(tab)
        fallback_label = QLabel(LABEL_FALLBACK, tab)

        button_row = QHBoxLayout()
        button_row.addWidget(self.test_translate_button)
        button_row.addStretch(1)

        column = QVBoxLayout(tab)
        column.setSpacing(SECTION_GAP)
        column.addLayout(tr_form)
        column.addWidget(self.registration_note)
        column.addWidget(self.local_mt_tier)
        column.addWidget(fallback_label)
        column.addWidget(self.fallback_editor)
        column.addLayout(button_row)
        column.addWidget(note)
        column.addWidget(self.debug_check)
        column.addWidget(debug_hint)
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
            deeplx_url=self.deeplx_url.text().strip(),
            deeplx_token=self.deeplx_token.text().strip(),
            custom_base_url=self.custom_base_url.text().strip(),
            custom_api_key=self.custom_api_key.text().strip(),
            custom_model=self.custom_model.text().strip(),
            local_mt_tier=self._selected_mt_tier(),
            fallback_order=self.fallback_editor.order(),
        )

    def draft_config(self) -> AppConfig:
        """AppConfig as 保存 would write it (LLM tab applied to the loaded base)."""
        base = self._existing if self._existing is not None else AppConfig()
        return self.llm_tab.build_config(base)

    # -- async probes ---------------------------------------------------------------------
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
        # Every profile needs a name + Base URL; the 大模型 translation
        # provider additionally needs a 当前文本模型. Web providers need their key.
        error = self.llm_tab.validate(require_text=provider_id == PROVIDER_LLM)
        if error is not None:
            self._toast(error, TOAST_WARN)
            return
        if provider_id != PROVIDER_LLM and not self._validate_web(provider_id):
            return
        cha = self.cha_tab.current_settings()
        if cha.api_mode == API_MODE_OWN and not cha.base_url:
            self._toast(TOAST_NEED_BASE_URL, TOAST_WARN)
            return
        # Refuse to overwrite an unreadable core config BEFORE writing anything:
        # the LLM tab always rewrites profiles, so a failed read would clobber them.
        if self._load_error is not None or self._existing is None:
            self._toast(TOAST_LOAD_FAILED, TOAST_ERR)
            return
        save_translation_config(self.translation_config())
        try:
            save_vision_prompts(self.prompts_tab.current_prompts())
        except NLaptError as exc:  # prompts must not block the LLM/provider save
            _LOGGER.exception("could not persist vision prompts")
            self._toast(str(exc), TOAST_ERR)
        # Local-inference tab persists its own file (errors toast internally).
        self.local_tab.persist()
        try:
            save_cha_settings(self.cha_tab.current_settings())
        except NLaptError as exc:
            _LOGGER.exception("could not persist CHA settings")
            self._toast(str(exc), TOAST_ERR)
        try:
            save_compare_settings(self.compare_tab.current_settings())
        except NLaptError as exc:
            _LOGGER.exception("could not persist compare settings")
            self._toast(str(exc), TOAST_ERR)
        config = self.draft_config()
        save_app_config(config)
        # reload_config also drops the cached translator, so a provider switch
        # away from / back to LLM re-resolves cleanly.
        self._controller.reload_config(config)
        self._persist_debug()
        _LOGGER.info(
            "settings saved (translation provider=%s, profiles=%s)",
            provider_id,
            len(config.profiles),
        )
        self._toast(TOAST_SAVED, TOAST_OK)
        self.saved.emit(config)
        self.accept()

    # -- internals -------------------------------------------------------------------
    def _validate_web(self, provider_id: str) -> bool:
        """Ensure the selected web provider has its required credentials."""
        if provider_id == PROVIDER_LOCAL_MT:
            if not is_tier_downloaded(self._selected_mt_tier()):
                self._toast(TOAST_TR_NEED_LOCAL_MT, TOAST_WARN)
                return False
            return True
        info = REGISTRATION_INFO.get(provider_id, {})
        if not info.get("needs_key"):
            return True
        try:
            create_provider(provider_id, self.translation_config().credentials())
        except LLMError:
            if provider_id == PROVIDER_CUSTOM:
                self._toast(TOAST_TR_NEED_CUSTOM, TOAST_WARN)
            elif provider_id == PROVIDER_DEEPLX:
                self._toast(TOAST_TR_NEED_DEEPLX, TOAST_WARN)
            else:
                self._toast(TOAST_TR_NEED_KEY, TOAST_WARN)
            return False
        return True

    def _build_test_translator(
        self, provider_id: str, direction: Direction
    ) -> Callable[[], str] | None:
        """Return a zero-arg callable translating the sample, or None on invalid input."""
        if provider_id == PROVIDER_LLM:
            error = self.llm_tab.validate(require_text=True)
            if error is not None:
                self._toast(error, TOAST_WARN)
                return None
            profile = resolve_text_profile(self.draft_config())
            if profile is None:
                self._toast(TOAST_NEED_TEXT_MODEL, TOAST_WARN)
                return None
            translator = Translator(create_client(profile), profile)
            return lambda: translator.translate(
                "settings-test", TR_TEST_SAMPLE, direction
            ).translated
        if provider_id == PROVIDER_LOCAL_MT:
            if not self._validate_web(PROVIDER_LOCAL_MT):
                return None
            provider = LocalMTProvider(self._selected_mt_tier(), retry_sleep=time.sleep)
            return lambda: provider.translate(TR_TEST_SAMPLE, direction)
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
        is_deeplx = provider_id == PROVIDER_DEEPLX
        is_custom = provider_id == PROVIDER_CUSTOM
        self._tr_form.setRowVisible(self.baidu_appid, is_baidu)
        self._tr_form.setRowVisible(self.baidu_key, is_baidu)
        self._tr_form.setRowVisible(self.deepl_key, is_deepl)
        self._tr_form.setRowVisible(self.deeplx_url, is_deeplx)
        self._tr_form.setRowVisible(self.deeplx_token, is_deeplx)
        self._tr_form.setRowVisible(self.custom_base_url, is_custom)
        self._tr_form.setRowVisible(self.custom_api_key, is_custom)
        self._tr_form.setRowVisible(self.custom_model, is_custom)
        self.fallback_editor.set_primary(provider_id)
        self.local_mt_tier.refresh()

    def _toast(self, text: str, kind: str) -> None:
        self._controller.toast_requested.emit(text, kind)

    def _persist_debug(self) -> None:
        """Write UISettings.debug and reconfigure console / worker stacks now."""
        enabled = self.debug_check.isChecked()
        self._controller.update_settings(debug=enabled)
        try:
            disk = load_ui_settings()
            merged = _dc_replace(
                self._controller.settings, theme=disk.theme, accent=disk.accent
            )
            save_ui_settings(merged)
        except NLaptError as exc:
            _LOGGER.warning("could not persist debug setting: %s", exc)
        configure_logging(app_data_dir() / LOG_DIR_NAME, console=enabled)
        set_debug(enabled)

    def _selected_mt_tier(self) -> str:
        data = self.local_mt_tier.currentData()
        return str(data) if data is not None else DEFAULT_TIER

    def _refresh_mt_tier_labels(self, selected: str | None = None) -> None:
        self.local_mt_tier.refresh(selected)

    def _open_mt_models(self) -> None:
        dialog = MTModelsDialog(parent=self)
        dialog.exec()
        self.local_mt_tier.refresh()

    def _prefill_translation(self) -> None:
        config = self._translation_config
        index = self.provider.findData(config.provider)
        if index >= 0:
            self.provider.setCurrentIndex(index)
        self.baidu_appid.setText(config.baidu_appid)
        self.baidu_key.setText(config.baidu_key)
        self.deepl_key.setText(config.deepl_key)
        self.deeplx_url.setText(config.deeplx_url)
        self.deeplx_token.setText(config.deeplx_token)
        self.custom_base_url.setText(config.custom_base_url)
        self.custom_api_key.setText(config.custom_api_key)
        self.custom_model.setText(config.custom_model)
        self._refresh_mt_tier_labels(config.local_mt_tier)
        self.fallback_editor.set_primary(config.provider)
        self.fallback_editor.set_order(config.fallback_order)
        self.debug_check.setChecked(self._controller.settings.debug)
