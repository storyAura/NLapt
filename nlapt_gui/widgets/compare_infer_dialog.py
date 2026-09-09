"""多对比推标 review window: several models caption the same images, you pick.

The dialog starts a :class:`CompareRunner` on open; results stream into one
card per model while you browse the images on the left. Pick a card per
image (or 「全部采用」 one model), edit the text if you like, then 「写入已选」
hands the chosen captions to ``AppController.run_caption_batch`` — the same
snapshot + oplog + history path as LLM 推标, so the write is rollback-able.
Images you leave unpicked are never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from nlapt.core.config import ModelRef
from nlapt.diagnostics import get_logger

from nlapt_gui.compare_bridge import CompareRunner
from nlapt_gui.compare_config import load_compare_settings
from nlapt_gui.controller import AppController, TOAST_WARN
from nlapt_gui.model_targets import choice_label
from nlapt_gui.prompt_store import ENGINE_LLM, VisionPrompts, load_vision_prompts
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.layered_infer_cards import CardsScrollArea, PreviewPane
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

_LOGGER = get_logger(__name__)

WINDOW_TITLE = "多对比推标"
STATUS_RUNNING_FMT = "已完成 {done} / {total}"
STATUS_DONE = "全部完成 — 逐张选择要写入的标注"
STATUS_CANCELLED = "已取消 — 已返回的结果仍可选择写入"
BUTTON_CANCEL_RUN = "停止推理"
BUTTON_APPLY_ALL = "应用"
LABEL_APPLY_ALL = "全部采用"
BUTTON_WRITE_FMT = "写入已选 ({n})"
BUTTON_CLOSE = "关闭"
CARD_MODEL_FMT = "模型: {model}"
CARD_WAITING = "生成中…"
CARD_FAILED_FMT = "失败: {message}"
CHOSEN_MARK = "✓ "
BATCH_DESCRIPTION_FMT = "多对比推标 · {n} 张"
BATCH_HISTORY_LABEL = "多对比推标"
TOAST_NOTHING_CHOSEN = "尚未为任何图片选择标注"

DIALOG_MIN_W = 900
DIALOG_MIN_H = 620
KEY_LIST_W = 200
PREVIEW_H = 220
CARD_TEXT_MIN_H = 72
ROLE_KEY = Qt.ItemDataRole.UserRole

ResultKey = tuple[str, ModelRef]


class CompareCard(QFrame):
    """One model's caption for the current image: radio + label + editable text."""

    def __init__(self, ref: ModelRef, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ref = ref
        self.setProperty("surfaceCard", True)
        self.radio = QRadioButton(self)
        self.model_label = QLabel(CARD_MODEL_FMT.format(model=choice_label(ref)), self)
        self.model_label.setProperty("muted", True)
        self.status = QLabel("", self)
        self.status.setProperty("muted", True)
        self.status.setWordWrap(True)
        self.text = QPlainTextEdit(self)
        self.text.setMinimumHeight(CARD_TEXT_MIN_H)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.radio)
        head.addWidget(self.model_label, 1)
        body = QVBoxLayout(self)
        body.setContentsMargins(10, 8, 10, 8)
        body.setSpacing(6)
        body.addLayout(head)
        body.addWidget(self.text)
        body.addWidget(self.status)
        self.set_waiting()

    def set_waiting(self) -> None:
        self.text.setPlainText("")
        self.text.setPlaceholderText(CARD_WAITING)
        self.text.setReadOnly(True)
        self.status.setText("")
        self.status.hide()
        self.radio.setEnabled(False)

    def set_result(self, text: str, ok: bool) -> None:
        if ok:
            self.text.setPlainText(text)
            self.text.setReadOnly(False)
            self.status.hide()
            self.radio.setEnabled(True)
        else:
            self.text.setPlainText("")
            self.text.setReadOnly(True)
            self.status.setText(CARD_FAILED_FMT.format(message=text))
            self.status.show()
            self.radio.setEnabled(False)

    def caption(self) -> str:
        return self.text.toPlainText().strip()


class CompareInferDialog(CenteredDialog):
    """Side-by-side review of several models' captions (see module docstring)."""

    def __init__(
        self,
        controller: AppController,
        keys: Sequence[str],
        refs: Sequence[ModelRef],
        *,
        loader: ThumbnailLoader | None = None,
        runner: CompareRunner | None = None,
        prompts: VisionPrompts | None = None,
        pool: QThreadPool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._keys = tuple(dict.fromkeys(keys))
        self._refs = tuple(dict.fromkeys(refs))
        self._loader = loader
        self._prompts = prompts if prompts is not None else load_vision_prompts()
        self._runner = runner if runner is not None else CompareRunner(controller, pool=pool)
        self._results: dict[ResultKey, tuple[str, bool]] = {}
        self._edits: dict[ResultKey, str] = {}
        self._choice: dict[str, ModelRef] = {}
        self._current = ""
        self._loading = False
        self._finished = False

        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(True)
        self.setMinimumSize(DIALOG_MIN_W, DIALOG_MIN_H)
        self._build()
        self._runner.item_ready.connect(self._on_item_ready)
        self._runner.progress.connect(self._on_progress)
        self._runner.finished.connect(self._on_finished)
        if loader is not None:
            loader.ready.connect(self._on_thumb_ready)
        if self._keys:
            self.key_list.setCurrentRow(0)

    # -- construction ---------------------------------------------------------------
    def _build(self) -> None:
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.status_label = QLabel(STATUS_RUNNING_FMT.format(done=0, total=0), self)
        self.status_label.setProperty("muted", True)
        self.cancel_run_button = QPushButton(BUTTON_CANCEL_RUN, self)
        self.cancel_run_button.setProperty("variant", "outline")
        self.cancel_run_button.clicked.connect(self._cancel_run)
        header = QHBoxLayout()
        header.addWidget(self.progress_bar, 1)
        header.addWidget(self.status_label)
        header.addWidget(self.cancel_run_button)

        self.key_list = QListWidget(self)
        self.key_list.setFixedWidth(KEY_LIST_W)
        for key in self._keys:
            item = QListWidgetItem(key, self.key_list)
            item.setData(ROLE_KEY, key)
        self.key_list.currentRowChanged.connect(self._on_key_row)

        self.preview = PreviewPane(self)
        self.preview.setMinimumHeight(PREVIEW_H)
        self.preview.setMaximumHeight(PREVIEW_H)
        self.cards: list[CompareCard] = []
        # Radios live in separate frames, so exclusivity needs an explicit group.
        self._radio_group = QButtonGroup(self)
        cards_host = QWidget(self)
        cards_column = QVBoxLayout(cards_host)
        cards_column.setContentsMargins(0, 0, 0, 0)
        cards_column.setSpacing(8)
        for ref in self._refs:
            card = CompareCard(ref, cards_host)
            self._radio_group.addButton(card.radio)
            card.radio.toggled.connect(
                lambda checked, r=ref: self._on_radio(r, checked)
            )
            card.text.textChanged.connect(lambda r=ref: self._on_text_edited(r))
            self.cards.append(card)
            cards_column.addWidget(card)
        cards_column.addStretch(1)
        self.cards_scroll = CardsScrollArea(self)
        self.cards_scroll.setWidget(cards_host)
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(self.preview)
        right.addWidget(self.cards_scroll, 1)
        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self.key_list)
        body.addLayout(right, 1)

        self.apply_all_combo = QComboBox(self)
        for ref in self._refs:
            self.apply_all_combo.addItem(choice_label(ref), ref)
        self.apply_all_button = QPushButton(BUTTON_APPLY_ALL, self)
        self.apply_all_button.setProperty("variant", "outline")
        self.apply_all_button.clicked.connect(self.apply_all)
        self.write_button = QPushButton(BUTTON_WRITE_FMT.format(n=0), self)
        self.write_button.setProperty("variant", "accent")
        self.write_button.setEnabled(False)
        self.write_button.clicked.connect(self.write_chosen)
        self.close_button = QPushButton(BUTTON_CLOSE, self)
        self.close_button.setProperty("variant", "outline")
        self.close_button.clicked.connect(self.reject)
        footer = QHBoxLayout()
        footer.addWidget(QLabel(LABEL_APPLY_ALL, self))
        footer.addWidget(self.apply_all_combo)
        footer.addWidget(self.apply_all_button)
        footer.addStretch(1)
        footer.addWidget(self.write_button)
        footer.addWidget(self.close_button)

        column = QVBoxLayout(self)
        column.setSpacing(10)
        column.addLayout(header)
        column.addLayout(body, 1)
        column.addLayout(footer)

    # -- run -------------------------------------------------------------------------
    def start(self) -> str | None:
        """Kick off the runner; returns the Chinese error when it cannot start."""
        return self._runner.start(
            self._keys,
            self._refs,
            system=self._prompts.system_text_for(ENGINE_LLM),
            user_prompt=self._prompts.user_prompt_for(ENGINE_LLM),
            concurrency=self._controller.request_concurrency(),
        )

    def _cancel_run(self) -> None:
        self._runner.cancel()
        self.cancel_run_button.setEnabled(False)

    def _on_item_ready(self, key: str, ref: object, text: str, ok: bool) -> None:
        if not isValid(self) or not isinstance(ref, ModelRef):
            return
        self._results[(key, ref)] = (text, ok)
        if key == self._current:
            self._refresh_card(ref)
            self._sync_radios()

    def _on_progress(self, done: int, total: int) -> None:
        if not isValid(self):
            return
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        self.status_label.setText(STATUS_RUNNING_FMT.format(done=done, total=total))

    def _on_finished(self) -> None:
        if not isValid(self):
            return
        self._finished = True
        self.cancel_run_button.setEnabled(False)
        complete = len(self._results) >= len(self._keys) * len(self._refs)
        self.status_label.setText(STATUS_DONE if complete else STATUS_CANCELLED)
        self._update_write_button()

    # -- browsing --------------------------------------------------------------------
    def current_key(self) -> str:
        return self._current

    def _on_key_row(self, row: int) -> None:
        item = self.key_list.item(row)
        if item is None:
            return
        self._show_key(str(item.data(ROLE_KEY)))

    def _show_key(self, key: str) -> None:
        self._current = key
        self._load_preview(key)
        for card in self.cards:
            self._refresh_card(card.ref)
        self._sync_radios()

    def _sync_radios(self) -> None:
        """Check the card picked for the current image (none when unpicked)."""
        chosen = self._choice.get(self._current)
        self._loading = True
        self._radio_group.setExclusive(False)
        try:
            for card in self.cards:
                card.radio.setChecked(card.ref == chosen and card.radio.isEnabled())
        finally:
            self._radio_group.setExclusive(True)
            self._loading = False

    def _load_preview(self, key: str) -> None:
        if self._loader is None:
            return
        pixmap = self._loader.request(key, self._controller.image_path(key), PREVIEW_H)
        self.preview.set_source(pixmap if pixmap is not None else QPixmap())

    def _on_thumb_ready(self, key: str, pixmap: QPixmap) -> None:
        if isValid(self) and key == self._current:
            self.preview.set_source(pixmap)

    def _refresh_card(self, ref: ModelRef) -> None:
        card = self._card_for(ref)
        if card is None:
            return
        slot = (self._current, ref)
        self._loading = True
        try:
            result = self._results.get(slot)
            if result is None:
                card.set_waiting()
            else:
                text, ok = result
                card.set_result(self._edits.get(slot, text), ok)
        finally:
            self._loading = False

    def _card_for(self, ref: ModelRef) -> CompareCard | None:
        for card in self.cards:
            if card.ref == ref:
                return card
        return None

    # -- choosing --------------------------------------------------------------------
    def choice_for(self, key: str) -> ModelRef | None:
        return self._choice.get(key)

    def chosen_caption(self, key: str) -> str | None:
        """The (possibly edited) caption picked for ``key``; None when unpicked."""
        ref = self._choice.get(key)
        if ref is None:
            return None
        result = self._results.get((key, ref))
        if result is None or not result[1]:
            return None
        return self._edits.get((key, ref), result[0]).strip()

    def _on_radio(self, ref: ModelRef, checked: bool) -> None:
        if self._loading or not checked or not self._current:
            return
        self._choice[self._current] = ref
        self._mark_key(self._current)
        self._update_write_button()

    def _on_text_edited(self, ref: ModelRef) -> None:
        if self._loading or not self._current:
            return
        card = self._card_for(ref)
        slot = (self._current, ref)
        if card is not None and slot in self._results and self._results[slot][1]:
            self._edits[slot] = card.caption()

    def apply_all(self) -> None:
        """Pick the selected model for every image where it succeeded."""
        ref = self.apply_all_combo.currentData()
        if not isinstance(ref, ModelRef):
            return
        for key in self._keys:
            result = self._results.get((key, ref))
            if result is not None and result[1]:
                self._choice[key] = ref
                self._mark_key(key)
        self._sync_radios()
        self._update_write_button()

    def _mark_key(self, key: str) -> None:
        for row in range(self.key_list.count()):
            item = self.key_list.item(row)
            if item.data(ROLE_KEY) == key:
                chosen = self.chosen_caption(key) is not None
                item.setText((CHOSEN_MARK if chosen else "") + key)
                return

    def _chosen_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self._keys if self.chosen_caption(key) is not None)

    def _update_write_button(self) -> None:
        n = len(self._chosen_keys())
        self.write_button.setText(BUTTON_WRITE_FMT.format(n=n))
        self.write_button.setEnabled(self._finished and n > 0)

    # -- writing ---------------------------------------------------------------------
    def write_chosen(self) -> bool:
        """Write every picked caption through the rollback-able batch path."""
        keys = self._chosen_keys()
        if not keys:
            self._controller.toast_requested.emit(TOAST_NOTHING_CHOSEN, TOAST_WARN)
            return False
        captions = {key: self.chosen_caption(key) or "" for key in keys}

        def caption_one(key: str, _path: Path) -> str:
            return captions[key]

        started = self._controller.run_caption_batch(
            keys,
            caption_one,
            description=BATCH_DESCRIPTION_FMT.format(n=len(keys)),
            history_label=BATCH_HISTORY_LABEL,
            engine=ENGINE_LLM,
        )
        if started:
            _LOGGER.info("compare review wrote %s captions", len(keys))
            self.accept()
        return started

    def reject(self) -> None:  # noqa: D102 - QDialog override
        self._runner.cancel()
        super().reject()


def open_compare_infer(
    controller: AppController,
    keys: Sequence[str],
    loader: ThumbnailLoader | None,
    parent: QWidget | None = None,
) -> CompareInferDialog | None:
    """Launch 多对比推标 for ``keys`` with the models chosen in 设置; None if it cannot start."""
    refs = load_compare_settings().models
    dialog = CompareInferDialog(controller, keys, refs, loader=loader, parent=parent)
    error = dialog.start()
    if error is not None:
        controller.toast_requested.emit(error, TOAST_WARN)
        dialog.deleteLater()
        return None
    dialog.exec()
    return dialog
