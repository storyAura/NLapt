"""分层推标 wizard: lock a character card, then batch-caption pose/scene."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtGui import QPixmap
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

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController
from nlapt_gui.layered_prompts import (
    TIP_FLORENCE_NO_LAYERED,
    build_card_prompt,
    build_scene_prompt,
    format_card_stats,
)
from nlapt_gui.layered_store import LayeredMemory, load_layered_memory, save_layered_memory
from nlapt_gui.prompt_store import ENGINE_LLM, ENGINE_LOCAL, load_vision_prompts
from nlapt_gui.translate_bridge import TranslateBridge
from nlapt_gui.vision_bridge import VisionBridge, local_engine_is_florence
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.layered_infer_cards import (
    PLACEHOLDER_ZH,
    CandidateCard,
    PreviewPane,
    RefPickerGrid,
)
from nlapt_gui.widgets.thumb_cells import tokens_for_settings
from nlapt_gui.widgets.thumbnails import ThumbnailLoader, bucket_height

_LOGGER = get_logger(__name__)

WINDOW_TITLE = "分层推标"
LABEL_NAME = "角色名"
LABEL_SERIES = "作品名"
LABEL_ENGINE = "推理引擎"
LABEL_REF = "参考图"
HINT_NAME = "必填，将锁定为全程主语"
HINT_SERIES = "选填，写入人物卡首句 name from series"
HINT_CARDS = "审阅并编辑英文；中文仅供对照。选定一套后用于整批。"
HINT_CONFIRM = "确认后为每张图生成画面段，并拼接「人物卡 + 空行 + 画面段」。"
HINT_REF = "点击下方缩略图选择参考图"
ENGINE_LLM_TEXT = "LLM"
ENGINE_LOCAL_TEXT = "本地模型"
BUTTON_BACK = "上一步"
BUTTON_GENERATE = "生成人物卡"
BUTTON_NEXT = "下一步"
BUTTON_CONFIRM = "开始分层推标"
STATUS_NEED_NAME = "请先填写角色名"
STATUS_GENERATING = "正在生成人物卡…"
STATUS_TRANSLATING = "正在翻译对照…"
STATUS_PICK = "请选定一套人物卡"
STATUS_READY = "将写入 {n} 张标注"
STATUS_ENGINE = "当前引擎未配置"
PREVIEW_SCENE = "画面提示词预览"
LABEL_CHOSEN_CARD = "选定人物卡"
CARD_COUNT = 3
PREVIEW_DECODE_H = 480
DIALOG_MIN_W = 720
DIALOG_MIN_H = 680
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
        self._florence = local_engine_is_florence() if florence is None else florence
        self._pending: set[str] = set()
        self._translating: set[str] = set()

        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(True)
        self.setMinimumSize(DIALOG_MIN_W, DIALOG_MIN_H)

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
        self.cards: list[CandidateCard] = []
        self._card_group = QButtonGroup(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(hint)
        for index in range(CARD_COUNT):
            card = CandidateCard(index, page)
            card.regen_requested.connect(self._regen_one)
            self._card_group.addButton(card.radio, index)
            self.cards.append(card)
            layout.addWidget(card, 1)
        self.cards[0].radio.setChecked(True)
        return page

    def _build_confirm_page(self) -> QWidget:
        page = QWidget(self)
        hint = QLabel(HINT_CONFIRM, page)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        self.confirm_card = QPlainTextEdit(page)
        self.confirm_card.setReadOnly(True)
        self.confirm_card_stats = QLabel(format_card_stats(""), page)
        self.confirm_card_stats.setProperty("muted", True)
        self.confirm_scene = QPlainTextEdit(page)
        self.confirm_scene.setReadOnly(True)
        self.confirm_scene_stats = QLabel(format_card_stats(""), page)
        self.confirm_scene_stats.setProperty("muted", True)
        card_title = QLabel(LABEL_CHOSEN_CARD, page)
        card_title.setProperty("sectionTitle", True)
        scene_title = QLabel(PREVIEW_SCENE, page)
        scene_title.setProperty("sectionTitle", True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(hint)
        layout.addWidget(card_title)
        layout.addWidget(self.confirm_card, 1)
        layout.addWidget(self.confirm_card_stats)
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
        if not self._vision.configured(engine):
            self.status_label.setText(STATUS_ENGINE)
            return
        request_id = f"card-{index}"
        self._pending.add(request_id)
        self.cards[index].set_busy(True)
        self.cards[index].set_chinese(PLACEHOLDER_ZH)
        ok = self._vision.request_custom(
            request_id,
            self.reference_key(),
            engine,
            system="",
            user_prompt=build_card_prompt(
                name, self.series_edit.text(), variant=index
            ),
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
            self._translating.add(request_id)
            self._translate.request_to(request_id, text.strip(), TARGET_ZH)
        else:
            self.cards[index].set_english("")
            self.cards[index].set_chinese(text or PLACEHOLDER_ZH)
        self._sync_chrome()

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
        self._translating.discard(key)
        self.cards[index].set_chinese(result if ok and result.strip() else PLACEHOLDER_ZH)
        self._sync_chrome()

    def _fill_confirm(self) -> None:
        name = self.name_edit.text().strip()
        card = self.selected_card_text()
        scene = build_scene_prompt(name)
        self.confirm_card.setPlainText(card)
        self.confirm_card_stats.setText(format_card_stats(card))
        self.confirm_scene.setPlainText(scene)
        self.confirm_scene_stats.setText(format_card_stats(scene))

    def _start_batch(self) -> None:
        card = self.selected_card_text()
        name = self.name_edit.text().strip()
        if not card or not name or not self._keys:
            return
        engine = self.current_engine()
        prompts = load_vision_prompts()
        started = self._vision.request_layered_batch(
            self._keys,
            engine,
            card_text=card,
            scene_system=prompts.system_text_for(engine),
            scene_user=build_scene_prompt(name),
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

    # -- helpers ---------------------------------------------------------------
    def _apply_memory(self) -> None:
        if self._memory.name:
            self.name_edit.setText(self._memory.name)
        if self._memory.series:
            self.series_edit.setText(self._memory.series)
        if self._memory.card_text:
            self.cards[0].set_english(self._memory.card_text)

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
