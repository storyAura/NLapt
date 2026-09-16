"""CHA标注 wizard: lock 1..MAX_CARDS character cards, then batch-caption pose/scene."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTabWidget,
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
from nlapt_gui.layered_prompts import (
    MAX_CARDS,
    CardRoster,
    CharacterCard,
    build_card_prompt,
    build_scene_prompt,
    effective_card_template,
    effective_scene_template,
    format_card_stats,
)
from nlapt_gui.layered_store import (
    LayeredMemory,
    SlotMemory,
    load_layered_memory,
    save_layered_memory,
)
from nlapt_gui.model_targets import choice_label
from nlapt_gui.prompt_store import ENGINE_LLM, ENGINE_LOCAL, load_vision_prompts
from nlapt_gui.tagger_bridge import TaggerBridge, release_tagger_engine
from nlapt_gui.translate_bridge import TranslateBridge
from nlapt_gui.vision_bridge import VisionBridge, local_engine_is_florence
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.layered_infer_cards import (
    PLACEHOLDER_ZH,
    CandidateCard,
    CardsScrollArea,
    PreviewPane,
    RefPickerPanel,
)
from nlapt_gui.widgets.layered_infer_identify import IdentifyCoordinator
from nlapt_gui.widgets.layered_infer_role_page import (
    ENGINE_LOCAL_TEXT,
    RolePage,
)
from nlapt_gui.widgets.layered_infer_slots import (
    CARD_COUNT,
    SlotCardsPage,
    SlotConfirmView,
    SlotRoleForm,
    SlotWidgets,
)
from nlapt_gui.widgets.thumb_cells import tokens_for_settings
from nlapt_gui.widgets.thumbnails import ThumbnailLoader, bucket_height

_LOGGER = get_logger(__name__)

WINDOW_TITLE = "CHA标注 · 组合分层推标 (Combined Hierarchical Annotation)"
HINT_CARDS = "审阅并编辑英文；中文仅供对照。每张角色卡选定一套后用于整批。"
HINT_CONFIRM = (
    "确认后为每张图判定属于哪几张角色卡，并拼接「匹配的人物卡 + 空行 + 画面段」；"
    "都不属于的图会跳过不写。每张图同时核对服装：与官方服装不同时保留固定外貌、改写服装段。"
)
LABEL_BATCH_MODEL_FMT = "整批画面模型: {model}"
BUTTON_BACK = "上一步"
BUTTON_GENERATE = "生成人物卡"
BUTTON_NEXT = "下一步"
BUTTON_CONFIRM = "开始CHA标注"
STATUS_NEED_NAME = "请先填写每张角色卡的角色名"
STATUS_GENERATING = "正在生成人物卡…"
STATUS_TRANSLATING = "正在翻译对照…"
STATUS_PICK = "请为每张角色卡选定一套人物卡"
STATUS_READY = "将写入 {n} 张标注"
STATUS_ENGINE = "当前引擎未配置"
TRANSLATE_FAILED_FMT = "翻译失败: {message}"
TRANSLATE_FAILED_UNKNOWN = "未知错误"
PREVIEW_SCENE = "画面提示词预览"
REQUEST_PREFIX = "card"
PREVIEW_DECODE_H = 480
DIALOG_MIN_W = 1180
DIALOG_MIN_H = 720
SCREEN_MARGIN = 24
PAGE_ROLE = 0
PAGE_CARDS = 1
PAGE_CONFIRM = 2
TARGET_ZH = "zh"


class LayeredInferDialog(CenteredDialog):
    """Three-page wizard: roles (1..MAX_CARDS slots) → candidates per slot → confirm."""

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
        tagger_bridge: TaggerBridge | None = None,
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
        self._tagger = (
            tagger_bridge if tagger_bridge is not None else TaggerBridge(parent=self)
        )
        self._identify = IdentifyCoordinator(controller, self._tagger, parent=self)
        self._pending: set[str] = set()
        self._translating: set[str] = set()
        self._zh_queue: deque[tuple[str, str]] = deque()
        self._zh_inflight: str | None = None
        self.slots: list[SlotWidgets] = []
        self._next_uid = 0

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

    # -- public accessors (current slot) ----------------------------------------
    @property
    def current_slot(self) -> SlotWidgets:
        """Slot shown on the role tabs (the one the reference strip edits)."""
        index = max(0, self.role_tabs.currentIndex())
        return self.slots[min(index, len(self.slots) - 1)]

    @property
    def name_edit(self) -> QLineEdit:
        return self.current_slot.role.name_edit

    @property
    def series_edit(self) -> QLineEdit:
        return self.current_slot.role.series_edit

    @property
    def cards(self) -> list[CandidateCard]:
        """Candidate cards of the current slot."""
        return self.current_slot.cards_page.cards

    @property
    def cards_scroll(self) -> CardsScrollArea:
        return self.current_slot.cards_page.cards_scroll

    @property
    def confirm_card(self) -> QPlainTextEdit:
        return self.current_slot.confirm.confirm_card

    @property
    def confirm_card_stats(self) -> QLabel:
        return self.current_slot.confirm.confirm_card_stats

    @property
    def confirm_appearance(self) -> QPlainTextEdit:
        return self.current_slot.confirm.confirm_appearance

    @property
    def confirm_outfit(self) -> QPlainTextEdit:
        return self.current_slot.confirm.confirm_outfit

    @property
    def confirm_split_warning(self) -> QLabel:
        return self.current_slot.confirm.confirm_split_warning

    def selected_card_text(self) -> str:
        """English body of the current slot's checked candidate."""
        return self.current_slot.cards_page.selected_card_text()

    def current_engine(self) -> str:
        """Engine radio value (LLM unless 本地 is checked and enabled)."""
        if self.engine_local.isChecked() and self.engine_local.isEnabled():
            return ENGINE_LOCAL
        return ENGINE_LLM

    def reference_key(self) -> str:
        """Image key used for the current slot's character-card generation."""
        return self._slot_ref(self.current_slot)

    def roster(self) -> CardRoster | None:
        """Locked cards of every slot; None while a slot has no card yet."""
        cards = []
        for slot in self.slots:
            text = slot.cards_page.selected_card_text()
            name = slot.role.name()
            if not text or not name:
                return None
            cards.append(CharacterCard.from_text(name, slot.role.series(), text))
        return CardRoster(tuple(cards))

    # -- pages ----------------------------------------------------------------
    def _build_role_page(self) -> QWidget:
        tokens = tokens_for_settings(self._controller.settings)
        # Reference images come from the WHOLE dataset; ``self._keys`` is only
        # the batch that gets written.
        self.role_page = RolePage(
            self._controller.keys(),
            self._controller,
            self._loader,
            tokens,
            florence=self._florence,
            parent=self,
        )
        page = self.role_page
        page.role_tabs.currentChanged.connect(self._on_role_tab_changed)
        page.add_slot_button.clicked.connect(lambda: self.add_slot())
        page.remove_slot_button.clicked.connect(self.remove_current_slot)
        page.ref_picker.picked.connect(self._on_ref_picked)
        return page

    # Page-1 controls live on ``role_page``; keep the historical names.
    @property
    def role_tabs(self) -> QTabWidget:
        return self.role_page.role_tabs

    @property
    def add_slot_button(self) -> QPushButton:
        return self.role_page.add_slot_button

    @property
    def remove_slot_button(self) -> QPushButton:
        return self.role_page.remove_slot_button

    @property
    def engine_llm(self) -> QRadioButton:
        return self.role_page.engine_llm

    @property
    def engine_local(self) -> QRadioButton:
        return self.role_page.engine_local

    @property
    def florence_tip(self) -> QLabel:
        return self.role_page.florence_tip

    @property
    def ref_picker(self) -> RefPickerPanel:
        return self.role_page.ref_picker

    @property
    def preview(self) -> PreviewPane:
        return self.role_page.preview

    def _build_cards_page(self) -> QWidget:
        page = QWidget(self)
        hint = QLabel(HINT_CARDS, page)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        self.cards_tabs = QTabWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(hint)
        layout.addWidget(self.cards_tabs, 1)
        return page

    def _build_confirm_page(self) -> QWidget:
        page = QWidget(self)
        hint = QLabel(HINT_CONFIRM, page)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        self.confirm_model = QLabel("", page)
        self.confirm_model.setProperty("muted", True)
        self.confirm_tabs = QTabWidget(page)
        self.confirm_scene = QPlainTextEdit(page)
        self.confirm_scene.setReadOnly(True)
        self.confirm_scene_stats = QLabel(format_card_stats(""), page)
        self.confirm_scene_stats.setProperty("muted", True)
        scene_title = QLabel(PREVIEW_SCENE, page)
        scene_title.setProperty("sectionTitle", True)
        scene_column = QVBoxLayout()
        scene_column.setContentsMargins(0, 0, 0, 0)
        scene_column.setSpacing(8)
        scene_column.addWidget(scene_title)
        scene_column.addWidget(self.confirm_scene, 1)
        scene_column.addWidget(self.confirm_scene_stats)
        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(12)
        columns.addWidget(self.confirm_tabs, 1)
        columns.addLayout(scene_column, 1)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(hint)
        layout.addWidget(self.confirm_model)
        layout.addLayout(columns, 1)
        return page

    # -- slots ------------------------------------------------------------------
    def add_slot(self, memory: SlotMemory | None = None) -> SlotWidgets | None:
        """Append a card slot (up to MAX_CARDS) and make it current."""
        if len(self.slots) >= MAX_CARDS:
            return None
        uid = self._next_uid
        self._next_uid += 1
        slot = SlotWidgets(
            uid=uid,
            role=SlotRoleForm(self),
            cards_page=SlotCardsPage(self),
            confirm=SlotConfirmView(self),
        )
        current = self._controller.current_key
        slot.ref_key = current if current in self._controller.keys() else self._default_ref()
        if memory is not None:
            slot.role.name_edit.setText(memory.name)
            slot.role.series_edit.setText(memory.series)
        slot.role.name_changed.connect(self._on_slot_name_changed)
        self._identify.bind_slot(slot)
        slot.cards_page.regen_requested.connect(
            lambda index, uid=uid: self._request_card(uid, index)
        )
        self.slots.append(slot)
        position = len(self.slots) - 1
        title = slot.tab_title(position)
        self.role_tabs.addTab(slot.role, title)
        self.cards_tabs.addTab(slot.cards_page, title)
        self.confirm_tabs.addTab(slot.confirm, title)
        self.role_tabs.setCurrentIndex(position)
        self._sync_slot_buttons()
        self._sync_chrome()
        return slot

    def remove_current_slot(self) -> None:
        """Drop the current slot (at least one slot always remains)."""
        if len(self.slots) <= 1:
            return
        position = self.role_tabs.currentIndex()
        slot = self.slots.pop(position)
        for tabs in (self.role_tabs, self.cards_tabs, self.confirm_tabs):
            tabs.removeTab(position)
        prefix = self._request_prefix(slot.uid)
        for request_id in [rid for rid in self._pending if rid.startswith(prefix)]:
            self._pending.discard(request_id)
            self._drop_zh(request_id)
        slot.role.deleteLater()
        slot.cards_page.deleteLater()
        slot.confirm.deleteLater()
        self._retitle_tabs()
        self._sync_slot_buttons()
        self._sync_chrome()

    def _slot_by_uid(self, uid: int) -> SlotWidgets | None:
        for slot in self.slots:
            if slot.uid == uid:
                return slot
        return None

    def _default_ref(self) -> str:
        """First batch key: the fallback reference when a slot has none."""
        return self._keys[0] if self._keys else ""

    def _slot_ref(self, slot: SlotWidgets) -> str:
        if slot.ref_key in self._controller.keys():
            return slot.ref_key
        return self._default_ref()

    def _sync_slot_buttons(self) -> None:
        self.add_slot_button.setEnabled(len(self.slots) < MAX_CARDS)
        self.remove_slot_button.setEnabled(len(self.slots) > 1)

    def _retitle_tabs(self) -> None:
        for position, slot in enumerate(self.slots):
            title = slot.tab_title(position)
            for tabs in (self.role_tabs, self.cards_tabs, self.confirm_tabs):
                tabs.setTabText(position, title)

    def _on_slot_name_changed(self) -> None:
        self._retitle_tabs()
        self._sync_chrome()

    def _on_role_tab_changed(self, _index: int) -> None:
        if not self.slots:
            return
        key = self.reference_key()
        if key:
            self.ref_picker.select(key, notify=False)
        self._refresh_ref_preview()
        self._sync_chrome()

    def _on_ref_picked(self, key: str) -> None:
        self.current_slot.ref_key = key
        self._refresh_ref_preview()

    def _all_named(self) -> bool:
        return bool(self.slots) and all(slot.role.name() for slot in self.slots)

    def _all_picked(self) -> bool:
        return bool(self.slots) and all(
            slot.cards_page.selected_card_text() for slot in self.slots
        )

    # -- navigation ------------------------------------------------------------
    def _on_next(self) -> None:
        page = self._stack.currentIndex()
        if page == PAGE_ROLE:
            if not self._all_named():
                self._focus_first_unnamed()
                self.status_label.setText(STATUS_NEED_NAME)
                return
            self._generate_all()
            self._stack.setCurrentIndex(PAGE_CARDS)
        elif page == PAGE_CARDS:
            if not self._all_picked() or self._pending:
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

    def _focus_first_unnamed(self) -> None:
        for position, slot in enumerate(self.slots):
            if not slot.role.name():
                self.role_tabs.setCurrentIndex(position)
                slot.role.name_edit.setFocus()
                return

    def _sync_chrome(self) -> None:
        if not hasattr(self, "next_button"):
            return
        page = self._stack.currentIndex()
        self.back_button.setEnabled(page > PAGE_ROLE)
        if page == PAGE_ROLE:
            self.next_button.setText(BUTTON_GENERATE)
            self.next_button.setEnabled(self._all_named() and bool(self._keys))
            if not self._all_named():
                self.status_label.setText(STATUS_NEED_NAME)
            elif self._pending:
                self.status_label.setText(STATUS_GENERATING)
            else:
                self.status_label.setText(STATUS_READY.format(n=len(self._keys)))
        elif page == PAGE_CARDS:
            self.next_button.setText(BUTTON_NEXT)
            self.next_button.setEnabled(self._all_picked() and not self._pending)
            if self._pending:
                self.status_label.setText(STATUS_GENERATING)
            elif self._translating:
                self.status_label.setText(STATUS_TRANSLATING)
            else:
                self.status_label.setText(STATUS_PICK)
        else:
            self.next_button.setText(BUTTON_CONFIRM)
            self.next_button.setEnabled(self._all_picked())
            self.status_label.setText(STATUS_READY.format(n=len(self._keys)))

    # -- generate / translate --------------------------------------------------
    @staticmethod
    def _request_prefix(uid: int) -> str:
        return f"{REQUEST_PREFIX}-{uid}-"

    @staticmethod
    def _parse_request_id(request_id: str) -> tuple[int, int] | None:
        """``card-{uid}-{index}`` → ``(uid, index)``; None for foreign ids."""
        parts = request_id.split("-")
        if len(parts) != 3 or parts[0] != REQUEST_PREFIX:
            return None
        try:
            return int(parts[1]), int(parts[2])
        except ValueError:
            return None

    def _generate_all(self) -> None:
        for slot in self.slots:
            for index in range(CARD_COUNT):
                self._request_card(slot.uid, index)

    def _request_card(self, uid: int, index: int) -> None:
        slot = self._slot_by_uid(uid)
        if slot is None:
            return
        name = slot.role.name()
        if not name or not self._keys:
            return
        engine = self.current_engine()
        profile = self._card_profile(engine, index)
        if not self._vision.configured(engine, profile=profile):
            self.status_label.setText(STATUS_ENGINE)
            return
        request_id = f"{self._request_prefix(uid)}{index}"
        self._drop_zh(request_id)
        self._pending.add(request_id)
        card = slot.cards_page.cards[index]
        card.set_english("")
        card.set_busy(True)
        card.set_chinese(PLACEHOLDER_ZH)
        card.set_model_name(self._model_label(engine, profile, self._card_ref(index)))
        self._vision.request_custom(
            request_id,
            self._slot_ref(slot),
            engine,
            system="",
            user_prompt=build_card_prompt(
                name,
                slot.role.series(),
                variant=index,
                template=effective_card_template(self._cha.card_prompt),
            ),
            profile=profile,
        )
        self._sync_chrome()

    def _card_for(self, request_id: str) -> CandidateCard | None:
        parsed = self._parse_request_id(request_id)
        if parsed is None:
            return None
        uid, index = parsed
        slot = self._slot_by_uid(uid)
        if slot is None or not 0 <= index < len(slot.cards_page.cards):
            return None
        return slot.cards_page.cards[index]

    def _on_custom_ready(self, request_id: str, text: str, ok: bool) -> None:
        if not isValid(self):
            return
        card = self._card_for(request_id)
        if card is None:
            self._pending.discard(request_id)
            return
        self._pending.discard(request_id)
        card.set_busy(False)
        if ok and text.strip():
            card.set_english(text.strip())
            self._enqueue_zh(request_id, text.strip())
        else:
            card.set_english("")
            card.set_chinese(text or PLACEHOLDER_ZH)
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
        if not isValid(self) or lang != TARGET_ZH or self._parse_request_id(key) is None:
            return
        if key == self._zh_inflight:
            self._zh_inflight = None
        card = self._card_for(key)
        if card is None or key in self._pending:
            self._translating.discard(key)
            self._pump_zh_queue()
            self._sync_chrome()
            return
        self._translating.discard(key)
        if ok and result.strip():
            card.set_chinese(result)
        else:
            message = result.strip() or TRANSLATE_FAILED_UNKNOWN
            card.set_chinese(TRANSLATE_FAILED_FMT.format(message=message))
        self._pump_zh_queue()
        self._sync_chrome()

    def _fill_confirm(self) -> None:
        roster = self.roster()
        for slot in self.slots:
            slot.confirm.show_card(slot.cards_page.selected_card_text(), slot.role.name())
        scene = (
            build_scene_prompt(
                roster, template=effective_scene_template(self._cha.scene_prompt)
            )
            if roster is not None
            else ""
        )
        engine = self.current_engine()
        profile = self._batch_profile(engine)
        model = self._model_label(engine, profile, self._cha.batch_model)
        self.confirm_scene.setPlainText(scene)
        self.confirm_scene_stats.setText(format_card_stats(scene))
        self.confirm_model.setText(
            LABEL_BATCH_MODEL_FMT.format(model=model) if model else ""
        )
        self.confirm_model.setVisible(bool(model))
        self.confirm_tabs.setCurrentIndex(self.role_tabs.currentIndex())

    def _start_batch(self) -> None:
        roster = self.roster()
        if roster is None or not self._keys:
            return
        engine = self.current_engine()
        prompts = load_vision_prompts()
        started = self._vision.request_layered_batch(
            self._keys,
            engine,
            roster=roster,
            scene_system=prompts.system_text_for(engine),
            scene_user=build_scene_prompt(
                roster, template=effective_scene_template(self._cha.scene_prompt)
            ),
            profile=self._batch_profile(engine),
        )
        if not started:
            return
        try:
            save_layered_memory(
                LayeredMemory(
                    slots=tuple(
                        SlotMemory(name=card.name, series=card.series, card_text=card.text)
                        for card in roster.cards
                    )
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
        """One slot per remembered slot (names / series only), at least one."""
        remembered = self._memory.slots[:MAX_CARDS]
        for slot_memory in remembered:
            self.add_slot(slot_memory)
        if not self.slots:
            self.add_slot()
        self.role_tabs.setCurrentIndex(0)
        key = self.reference_key()
        if key:
            self.ref_picker.select(key, notify=False)

    def _refresh_ref_preview(self) -> None:
        if self._loader is None or not self.slots:
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

    def done(self, result: int) -> None:  # noqa: A003 - Qt override
        release_tagger_engine()
        super().done(result)

    def _on_thumb_ready(self, key: str, pixmap: QPixmap) -> None:
        if not isValid(self) or not self.slots or key != self.reference_key():
            return
        # Ignore the strip's small-bucket decode; only the 480px request.
        if pixmap.height() < bucket_height(PREVIEW_DECODE_H):
            return
        self.preview.set_source(pixmap)
