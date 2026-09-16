"""CHA标注 wizard page 1 in a wide two-column layout.

Left column: slot tabs (角色名 / 作品名), add / remove slot, engine radios and
the whole-dataset reference-image grid. Right column: the large preview of
the current slot's reference. The dialog owns all behaviour; this widget only
builds and exposes the controls.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nlapt_gui.controller import AppController
from nlapt_gui.layered_prompts import MAX_CARDS, TIP_FLORENCE_NO_LAYERED
from nlapt_gui.theme.tokens import ThemeTokens
from nlapt_gui.widgets.layered_infer_cards import PreviewPane, RefPickerPanel
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

LABEL_ENGINE = "推理引擎"
LABEL_REF = "参考图（当前角色卡）"
HINT_SLOTS = f"最多 {MAX_CARDS} 张角色卡：多套服装或多个角色各占一张，每张图会判定属于哪几张卡。"
HINT_REF = "点击缩略图为当前角色卡选择参考图；右侧为大图预览"
ENGINE_LLM_TEXT = "LLM"
ENGINE_LOCAL_TEXT = "本地模型"
BUTTON_ADD_SLOT = "添加角色卡"
BUTTON_REMOVE_SLOT = "删除此卡"
LEFT_MIN_W = 520
PREVIEW_MIN_W = 360
# Initial splitter split: controls 45% / preview 55%.
SPLIT_RATIO = (45, 55)


class RolePage(QWidget):
    """Two-column first page: controls + reference grid | preview."""

    def __init__(
        self,
        keys: Sequence[str],
        controller: AppController,
        loader: ThumbnailLoader | None,
        tokens: ThemeTokens,
        *,
        florence: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        left = QWidget(self)
        left.setMinimumWidth(LEFT_MIN_W)

        hint = QLabel(HINT_SLOTS, left)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        self.role_tabs = QTabWidget(left)
        self.add_slot_button = QPushButton(BUTTON_ADD_SLOT, left)
        self.remove_slot_button = QPushButton(BUTTON_REMOVE_SLOT, left)
        slot_row = QHBoxLayout()
        slot_row.setContentsMargins(0, 0, 0, 0)
        slot_row.addWidget(self.add_slot_button)
        slot_row.addWidget(self.remove_slot_button)
        slot_row.addStretch(1)

        self.engine_llm = QRadioButton(ENGINE_LLM_TEXT, left)
        self.engine_local = QRadioButton(ENGINE_LOCAL_TEXT, left)
        engines = QButtonGroup(left)
        engines.addButton(self.engine_llm)
        engines.addButton(self.engine_local)
        self.engine_llm.setChecked(True)
        self.florence_tip = QLabel(TIP_FLORENCE_NO_LAYERED, left)
        self.florence_tip.setProperty("muted", True)
        self.florence_tip.setWordWrap(True)
        self.engine_local.setEnabled(not florence)
        self.florence_tip.setVisible(florence)
        engine_row = QHBoxLayout()
        engine_row.setContentsMargins(0, 0, 0, 0)
        engine_row.addWidget(QLabel(LABEL_ENGINE, left))
        engine_row.addWidget(self.engine_llm)
        engine_row.addWidget(self.engine_local)
        engine_row.addStretch(1)

        ref_label = QLabel(LABEL_REF, left)
        ref_hint = QLabel(HINT_REF, left)
        ref_hint.setProperty("muted", True)
        ref_hint.setWordWrap(True)
        self.ref_picker = RefPickerPanel(keys, controller, loader, tokens, left)

        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(hint)
        left_layout.addWidget(self.role_tabs)
        left_layout.addLayout(slot_row)
        left_layout.addLayout(engine_row)
        left_layout.addWidget(self.florence_tip)
        left_layout.addWidget(ref_label)
        left_layout.addWidget(ref_hint)
        left_layout.addWidget(self.ref_picker, 1)

        self.preview = PreviewPane(self)
        self.preview.setMinimumWidth(PREVIEW_MIN_W)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.preview)
        self.splitter.setStretchFactor(0, SPLIT_RATIO[0])
        self.splitter.setStretchFactor(1, SPLIT_RATIO[1])

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
