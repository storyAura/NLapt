"""分层推标 wizard: lock a character card, then batch-caption pose/scene."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from nlapt.core.config import LLMProfile, ModelRef
from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger

from nlapt_gui.cha_config import (
    CHASettings,
    load_cha_settings,
    resolve_batch_profile,
    resolve_card_profile,
)
from nlapt_gui.controller import AppController
from nlapt_gui.model_targets import choice_label
from nlapt_gui.layered_prompts import (
    LABEL_CARD_APPEARANCE,
    LABEL_CARD_OUTFIT,
    TIP_FLORENCE_NO_LAYERED,
    WARN_SPLIT_FAILED,
    build_card_prompt,
    build_scene_prompt,
    format_card_stats,
    split_card,
)
from nlapt_gui.layered_store import LayeredMemory, load_layered_memory, save_layered_memory
from nlapt_gui.prompt_store import ENGINE_LLM, ENGINE_LOCAL, load_vision_prompts
from nlapt_gui.translate_bridge import TranslateBridge
from nlapt_gui.vision_bridge import VisionBridge, local_engine_is_florence
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.layered_infer_cards import (
    PLACEHOLDER_ZH,
    CandidateCard,
    CardsScrollArea,
    PreviewPane,
    RefPickerGrid,
)
from nlapt_gui.widgets.thumb_cells import tokens_for_settings
from nlapt_gui.widgets.thumbnails import ThumbnailLoader, bucket_height

_LOGGER = get_logger(__name__)

WINDOW_TITLE = "CHA标注 · 组合分层推标 (Combined Hierarchical Annotation)"
LABEL_NAME = "角色名"
LABEL_SERIES = "作品名"
LABEL_ENGINE = "推理引擎"
LABEL_REF = "参考图"
HINT_NAME = "必填，将锁定为全程主语"
HINT_SERIES = "选填，写入人物卡首句 name from series"
HINT_CARDS = "审阅并编辑英文；中文仅供对照。选定一套后用于整批。"
HINT_CONFIRM = (
    "确认后为每张图生成画面段，并拼接「人物卡 + 空行 + 画面段」。"
    "每张图会同时核对服装：与官方服装不同时保留固定外貌、改写服装段。"
)
LABEL_BATCH_MODEL_FMT = "整批画面模型: {model}"
HINT_REF = "点击下方缩略图选择参考图"
ENGINE_LLM_TEXT = "LLM"
ENGINE_LOCAL_TEXT = "本地模型"
BUTTON_BACK = "上一步"
BUTTON_GENERATE = "生成人物卡"
BUTTON_NEXT = "下一步"
BUTTON_CONFIRM = "开始CHA标注"
STATUS_NEED_NAME = "请先填写角色名"
STATUS_GENERATING = "正在生成人物卡…"
STATUS_TRANSLATING = "正在翻译对照…"
STATUS_PICK = "请选定一套人物卡"
STATUS_READY = "将写入 {n} 张标注"
STATUS_ENGINE = "当前引擎未配置"
TRANSLATE_FAILED_FMT = "翻译失败: {message}"
TRANSLATE_FAILED_UNKNOWN = "未知错误"
PREVIEW_SCENE = "画面提示词预览"
LABEL_CHOSEN_CARD = "选定人物卡"
CARD_COUNT = 3
PREVIEW_DECODE_H = 480
DIALOG_MIN_W = 720
DIALOG_MIN_H = 680
SCREEN_MARGIN = 24
PAGE_ROLE = 0
PAGE_CARDS = 1
PAGE_CONFIRM = 2
TARGET_ZH = "zh"


class LayeredInferDialog(CenteredDialog):
    """Three-page wizard: role → three card candidates → confirm batch."""

    def __init__(
        self,
        controller: AppController,
        vision_bridge: VisionBridge,
        keys: Sequence[str],
        *,
        loader: ThumbnailLoader | None = None,
        translate_bridge: TranslateBridge | None = None,
        memory: LayeredMemory | None = None,
        florence: bool | None = None,
        cha_settings: CHASettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._vision = vision_bridge
        self._keys = tuple(keys)
        self._loader = loader
        self._translate = (
            translate_bridge
            if translate_bridge is not None
            else TranslateBridge(controller, parent=self)
        )
        self._memory = memory if memory is not None else load_layered_memory()
        self._cha = cha_settings if cha_settings is not None else load_cha_settings()
        self._florence = local_engine_is_florence() if florence is None else florence
        self._pending: set[str] = set()
        self._translating: set[str] = set()
        self._zh_queue: deque[tuple[str, str]] = deque()
        self._zh_inflight: str | None = None

        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(True)
        self.setMinimumSize(DIALOG_MIN_W, DIALOG_MIN_H)
        self._cap_to_screen()

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._build_role_page())
        self._stack.addWidget(self._build_cards_page())
        self._stack.addWidget(self._build_confirm_page())

        self.status_label = QLabel(self)
        self.status_label.setProperty("muted", True)
        self.status_label.setWordWrap(True)
        self.back_button = QPushButton(BUTTON_BACK, self)
        self.next_button = QPushButton(BUTTON_GENERATE, self)
        self.back_button.clicked.connect(self._on_back)
        self.next_button.clicked.connect(self._on_next)

        buttons = QHBoxLayout()
        buttons.addWidget(self.status_label, 1)
        buttons.addWidget(self.back_button)
        buttons.addWidget(self.next_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addWidget(self._stack, 1)
        root.addLayout(buttons)

        self._vision.custom_ready.connect(self._on_custom_ready)
        self._translate.target_ready.connect(self._on_target_ready)
        if self._loader is not None:
            self._loader.ready.connect(self._on_thumb_ready)

        self._apply_memory()
        self._refresh_ref_preview()
        self._sync_chrome()

    def selected_card_text(self) -> str:
        """English body of the checked candidate (empty when none)."""
        for card in self.cards:
            if card.radio.isChecked():
                return card.english_text()
        return self.cards[0].english_text() if self.cards else ""

    def current_engine(self) -> str:
        """Engine radio value (LLM unless 本地 is checked and enabled)."""
        if self.engine_local.isChecked() and self.engine_local.isEnabled():
            return ENGINE_LOCAL
        return ENGINE_LLM

    def reference_key(self) -> str:
        """Image key used for character-card generation."""
        selected = self.ref_picker.selected_key()
        if selected:
            return selected
        return self._keys[0] if self._keys else ""

    # -- pages ----------------------------------------------------------------
    def _build_role_page(self) -> QWidget:
        page = QWidget(self)
        self.name_edit = QLineEdit(page)
        self.name_edit.setPlaceholderText(HINT_NAME)
        self.name_edit.textChanged.connect(lambda _text: self._sync_chrome())
        self.series_edit = QLineEdit(page)
        self.series_edit.setPlaceholderText(HINT_SERIES)

        self.engine_llm = QRadioButton(ENGINE_LLM_TEXT, page)
        self.engine_local = QRadioButton(ENGINE_LOCAL_TEXT, page)
        engines = QButtonGroup(page)
        engines.addButton(self.engine_llm)
        engines.addButton(self.engine_local)
        self.engine_llm.setChecked(True)
        self.florence_tip = QLabel(TIP_FLORENCE_NO_LAYERED, page)
        self.florence_tip.setProperty("muted", True)
        self.florence_tip.setWordWrap(True)
        if self._florence:
            self.engine_local.setEnabled(False)
            self.florence_tip.setVisible(True)
        else:
            self.florence_tip.setVisible(False)

        engine_row = QHBoxLayout()
        engine_row.setContentsMargins(0, 0, 0, 0)
        engine_row.addWidget(self.engine_llm)
        engine_row.addWidget(self.engine_local)
        engine_row.addStretch(1)

        tokens = tokens_for_settings(self._controller.settings)
        self.ref_picker = RefPickerGrid(
            self._keys, self._controller, self._loader, tokens, page
        )
        current = self._controller.current_key
        initial = current if current in self._keys else (self._keys[0] if self._keys else "")
        if initial:
            self.ref_picker.select(initial, notify=False)
        self.ref_picker.picked.connect(lambda _key: self._refresh_ref_preview())

        ref_label = QLabel(LABEL_REF, page)
        ref_hint = QLabel(HINT_REF, page)
        ref_hint.setProperty("muted", True)
        self.preview = PreviewPane(page)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow(LABEL_NAME, self.name_edit)
        form.addRow(LABEL_SERIES, self.series_edit)
        form.addRow(LABEL_ENGINE, engine_row)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(form)
        layout.addWidget(self.florence_tip)
        layout.addWidget(ref_label)
        layout.addWidget(ref_hint)
        layout.addWidget(self.ref_picker)
        layout.addWidget(self.preview, 1)
        return page

    def _build_cards_page(self) -> QWidget:
        page = QWidget(self)
        hint = QLabel(HINT_CARDS, page)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        inner = QWidget(page)
        self.cards: list[CandidateCard] = []
        self._card_group = QButtonGroup(inner)
        cards_layout = QVBoxLayout(inner)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)
        for index in range(CARD_COUNT):
            card = CandidateCard(index, inner)
            card.regen_requested.connect(self._regen_one)
            self._card_group.addButton(card.radio, index)
            self.cards.append(card)
            cards_layout.addWidget(card)
        self.cards[0].radio.setChecked(True)
        self.cards_scroll = CardsScrollArea(page)
        self.cards_scroll.setWidget(inner)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(hint)
        layout.addWidget(self.cards_scroll, 1)
        return page

    def _build_confirm_page(self) -> QWidget:
        page = QWidget(self)
        hint = QLabel(HINT_CONFIRM, page)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        self.confirm_model = QLabel("", page)
        self.confirm_model.setProperty("muted", True)
        self.confirm_card = QPlainTextEdit(page)
        self.confirm_card.setReadOnly(True)
        self.confirm_card_stats = QLabel(format_card_stats(""), page)
        self.confirm_card_stats.setProperty("muted", True)
        self.confirm_appearance = QPlainTextEdit(page)
        self.confirm_appearance.setReadOnly(True)
        self.confirm_outfit = QPlainTextEdit(page)
        self.confirm_outfit.setReadOnly(True)
        self.confirm_split_warning = QLabel(WARN_SPLIT_FAILED, page)
        self.confirm_split_warning.setProperty("muted", True)
        self.confirm_split_warning.hide()
        self.confirm_scene = QPlainTextEdit(page)
        self.confirm_scene.setReadOnly(True)
        self.confirm_scene_stats = QLabel(format_card_stats(""), page)
        self.confirm_scene_stats.setProperty("muted", True)
        card_title = QLabel(LABEL_CHOSEN_CARD, page)
        card_title.setProperty("sectionTitle", True)
        appearance_title = QLabel(LABEL_CARD_APPEARANCE, page)
        appearance_title.setProperty("sectionTitle", True)
        outfit_title = QLabel(LABEL_CARD_OUTFIT, page)
        outfit_title.setProperty("sectionTitle", True)
        scene_title = QLabel(PREVIEW_SCENE, page)
        scene_title.setProperty("sectionTitle", True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(hint)
        layout.addWidget(self.confirm_model)
        layout.addWidget(card_title)
        layout.addWidget(self.confirm_card, 1)
        layout.addWidget(self.confirm_card_stats)
        layout.addWidget(appearance_title)
        layout.addWidget(self.confirm_appearance, 1)
        layout.addWidget(outfit_title)
        layout.addWidget(self.confirm_outfit, 1)
        layout.addWidget(self.confirm_split_warning)
        layout.addWidget(scene_title)
        layout.addWidget(self.confirm_scene, 1)
        layout.addWidget(self.confirm_scene_stats)
        return page

    # -- navigation ------------------------------------------------------------
    def _on_next(self) -> None:
        page = self._stack.currentIndex()
        if page == PAGE_ROLE:
            if not self.name_edit.text().strip():
                self.status_label.setText(STATUS_NEED_NAME)
                return
            self._generate_all()
            self._stack.setCurrentIndex(PAGE_CARDS)
        elif page == PAGE_CARDS:
            if not self.selected_card_text() or self._pending:
                self.status_label.setText(STATUS_PICK)
                return
            self._fill_confirm()
            self._stack.setCurrentIndex(PAGE_CONFIRM)
        else:
            self._start_batch()
        self._sync_chrome()

    def _on_back(self) -> None:
        page = self._stack.currentIndex()
        if page > PAGE_ROLE:
            self._stack.setCurrentIndex(page - 1)
        self._sync_chrome()

    def _sync_chrome(self) -> None:
        page = self._stack.currentIndex()
        self.back_button.setEnabled(page > PAGE_ROLE)
        if page == PAGE_ROLE:
            self.next_button.setText(BUTTON_GENERATE)
            self.next_button.setEnabled(bool(self.name_edit.text().strip()) and bool(self._keys))
            if not self.name_edit.text().strip():
                self.status_label.setText(STATUS_NEED_NAME)
            elif self._pending:
                self.status_label.setText(STATUS_GENERATING)
            else:
                self.status_label.setText(STATUS_READY.format(n=len(self._keys)))
        elif page == PAGE_CARDS:
            self.next_button.setText(BUTTON_NEXT)
            self.next_button.setEnabled(bool(self.selected_card_text()) and not self._pending)
            if self._pending:
                self.status_label.setText(STATUS_GENERATING)
            elif self._translating:
                self.status_label.setText(STATUS_TRANSLATING)
            else:
                self.status_label.setText(STATUS_PICK)
        else:
            self.next_button.setText(BUTTON_CONFIRM)
            self.next_button.setEnabled(bool(self.selected_card_text()))
            self.status_label.setText(STATUS_READY.format(n=len(self._keys)))

    # -- generate / translate --------------------------------------------------
    def _generate_all(self) -> None:
        for index in range(CARD_COUNT):
            self._request_card(index)

    def _regen_one(self, index: int) -> None:
        self._request_card(index)

    def _request_card(self, index: int) -> None:
        name = self.name_edit.text().strip()
        if not name or not self._keys:
            return
        engine = self.current_engine()
        profile = self._card_profile(engine, index)
        if not self._vision.configured(engine, profile=profile):
            self.status_label.setText(STATUS_ENGINE)
            return
        request_id = f"card-{index}"
        self._drop_zh(request_id)
        self._pending.add(request_id)
        self.cards[index].set_english("")
        self.cards[index].set_busy(True)
        self.cards[index].set_chinese(PLACEHOLDER_ZH)
        self.cards[index].set_model_name(
            self._model_label(engine, profile, self._card_ref(index))
        )
        ok = self._vision.request_custom(
            request_id,
            self.reference_key(),
            engine,
            system="",
            user_prompt=build_card_prompt(
                name, self.series_edit.text(), variant=index
            ),
            profile=profile,
        )
        if not ok and request_id in self._pending:
            # Failure already emitted custom_ready (sync) or will not run.
            pass
        self._sync_chrome()

    def _on_custom_ready(self, request_id: str, text: str, ok: bool) -> None:
        if not isValid(self) or not request_id.startswith("card-"):
            return
        try:
            index = int(request_id.split("-", 1)[1])
        except ValueError:
            return
        if index < 0 or index >= len(self.cards):
            return
        self._pending.discard(request_id)
        self.cards[index].set_busy(False)
        if ok and text.strip():
            self.cards[index].set_english(text.strip())
            self._enqueue_zh(request_id, text.strip())
        else:
            self.cards[index].set_english("")
            self.cards[index].set_chinese(text or PLACEHOLDER_ZH)
        self._sync_chrome()

    def _enqueue_zh(self, request_id: str, english: str) -> None:
        self._translating.add(request_id)
        self._zh_queue.append((request_id, english))
        self._pump_zh_queue()

    def _drop_zh(self, request_id: str) -> None:
        self._zh_queue = deque(
            item for item in self._zh_queue if item[0] != request_id
        )
        self._translating.discard(request_id)

    def _pump_zh_queue(self) -> None:
        if self._zh_inflight is not None:
            return
        while self._zh_queue:
            request_id, english = self._zh_queue.popleft()
            if request_id not in self._translating:
                continue
            self._zh_inflight = request_id
            self._translate.request_to(request_id, english, TARGET_ZH)
            return

    def _on_target_ready(
        self, key: str, _source: str, lang: str, result: str, ok: bool
    ) -> None:
        if not isValid(self) or lang != TARGET_ZH or not key.startswith("card-"):
            return
        try:
            index = int(key.split("-", 1)[1])
        except ValueError:
            return
        if index < 0 or index >= len(self.cards):
            return
        if key == self._zh_inflight:
            self._zh_inflight = None
        if key in self._pending:
            self._translating.discard(key)
            self._pump_zh_queue()
            self._sync_chrome()
            return
        self._translating.discard(key)
        if ok and result.strip():
            self.cards[index].set_chinese(result)
        else:
            message = result.strip() or TRANSLATE_FAILED_UNKNOWN
            self.cards[index].set_chinese(
                TRANSLATE_FAILED_FMT.format(message=message)
            )
        self._pump_zh_queue()
        self._sync_chrome()

    def _fill_confirm(self) -> None:
        name = self.name_edit.text().strip()
        card = self.selected_card_text()
        parts = split_card(card, name)
        scene = build_scene_prompt(name, official_outfit=parts.outfit)
        engine = self.current_engine()
        profile = self._batch_profile(engine)
        model = self._model_label(engine, profile, self._cha.batch_model)
        self.confirm_card.setPlainText(card)
        self.confirm_card_stats.setText(format_card_stats(card))
        self.confirm_appearance.setPlainText(parts.appearance)
        self.confirm_outfit.setPlainText(parts.outfit)
        self.confirm_split_warning.setVisible(bool(card) and not parts.outfit)
        self.confirm_scene.setPlainText(scene)
        self.confirm_scene_stats.setText(format_card_stats(scene))
        self.confirm_model.setText(
            LABEL_BATCH_MODEL_FMT.format(model=model) if model else ""
        )
        self.confirm_model.setVisible(bool(model))

    def _start_batch(self) -> None:
        card = self.selected_card_text()
        name = self.name_edit.text().strip()
        if not card or not name or not self._keys:
            return
        engine = self.current_engine()
        prompts = load_vision_prompts()
        parts = split_card(card, name)
        started = self._vision.request_layered_batch(
            self._keys,
            engine,
            card_text=card,
            scene_system=prompts.system_text_for(engine),
            scene_user=build_scene_prompt(name, official_outfit=parts.outfit),
            profile=self._batch_profile(engine),
            card_parts=parts,
        )
        if not started:
            return
        try:
            save_layered_memory(
                LayeredMemory(
                    name=name,
                    series=self.series_edit.text().strip(),
                    card_text=card,
                )
            )
        except NLaptError as exc:
            _LOGGER.warning("could not persist layered infer memory: %s", exc)
        self.accept()

    def _card_profile(self, engine: str, index: int) -> LLMProfile | None:
        if engine != ENGINE_LLM:
            return None
        return resolve_card_profile(
            self._cha,
            self._controller.vision_profile(),
            index,
            lookup=self._controller.profile_for_ref,
        )

    def _batch_profile(self, engine: str) -> LLMProfile | None:
        if engine != ENGINE_LLM:
            return None
        return resolve_batch_profile(
            self._cha,
            self._controller.vision_profile(),
            lookup=self._controller.profile_for_ref,
        )

    def _model_label(
        self, engine: str, profile: LLMProfile | None, ref: ModelRef = ModelRef()
    ) -> str:
        if engine == ENGINE_LOCAL:
            return ENGINE_LOCAL_TEXT
        if profile is not None and profile.vision_model:
            # A pool pick on another API shows its provider too.
            if ref.profile:
                return choice_label(ModelRef(profile.name, profile.vision_model))
            return profile.vision_model
        active = self._controller.vision_profile()
        return active.vision_model if active is not None else ""

    def _card_ref(self, index: int) -> ModelRef:
        models = self._cha.card_models
        return models[index] if 0 <= index < len(models) else ModelRef()

    # -- helpers ---------------------------------------------------------------
    def _cap_to_screen(self) -> None:
        """Keep the wizard inside the available desktop so it stays draggable."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.setMaximumWidth(max(DIALOG_MIN_W, geo.width() - SCREEN_MARGIN))
        self.setMaximumHeight(max(DIALOG_MIN_H, geo.height() - SCREEN_MARGIN))

    def _apply_memory(self) -> None:
        if self._memory.name:
            self.name_edit.setText(self._memory.name)
        if self._memory.series:
            self.series_edit.setText(self._memory.series)

    def _refresh_ref_preview(self) -> None:
        if self._loader is None:
            return
        key = self.reference_key()
        if not key:
            return
        try:
            path = self._controller.image_path(key)
        except NLaptError:
            return
        pixmap = self._loader.request(key, path, PREVIEW_DECODE_H)
        if pixmap is not None:
            self.preview.set_source(pixmap)

    def _on_thumb_ready(self, key: str, pixmap: QPixmap) -> None:
        if not isValid(self) or key != self.reference_key():
            return
        # Ignore the strip's small-bucket decode; only the 480px request.
        if pixmap.height() < bucket_height(PREVIEW_DECODE_H):
            return
        self.preview.set_source(pixmap)
