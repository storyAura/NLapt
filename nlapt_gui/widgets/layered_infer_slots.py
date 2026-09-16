"""CHA标注 wizard per-slot widgets: role form, candidate page, confirm view.

One :class:`SlotWidgets` bundle exists per character card slot (1..MAX_CARDS)
and is shown as one tab on each wizard page. The dialog owns the bundles and
the shared reference-image strip; these widgets never touch the filesystem
or bridges.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt_gui.layered_prompts import (
    LABEL_CARD_APPEARANCE,
    LABEL_CARD_OUTFIT,
    WARN_SPLIT_FAILED,
    format_card_stats,
    split_card,
)
from nlapt_gui.widgets.layered_infer_cards import CandidateCard, CardsScrollArea

LABEL_NAME = "角色名"
LABEL_SERIES = "作品名"
HINT_NAME = "必填，将锁定为该卡主语"
HINT_SERIES = "选填，写入人物卡首句 name from series"
LABEL_CHOSEN_CARD = "选定人物卡"
SLOT_TAB_FMT = "角色卡 {n}"
SLOT_TAB_NAMED_FMT = "角色卡 {n} · {name}"
CARD_COUNT = 3
BUTTON_IDENTIFY = "识别角色"
BUTTON_IDENTIFYING = "识别中…"
TIP_NEED_TAGGER = "需在 设置▸CHA标注 下载 CL Tagger 模型"


class SlotRoleForm(QWidget):
    """角色名 / 作品名 for one slot; the reference key is held by the dialog."""

    name_changed = Signal()
    identify_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText(HINT_NAME)
        self.name_edit.textChanged.connect(lambda _text: self.name_changed.emit())
        self.identify_button = QPushButton(BUTTON_IDENTIFY, self)
        self.identify_button.setProperty("variant", "outline")
        self.identify_button.clicked.connect(self.identify_requested.emit)
        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.addWidget(self.name_edit, 1)
        name_row.addWidget(self.identify_button)
        self.series_edit = QLineEdit(self)
        self.series_edit.setPlaceholderText(HINT_SERIES)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow(LABEL_NAME, name_row)
        form.addRow(LABEL_SERIES, self.series_edit)
        self.set_identify_available(False)

    def set_identify_available(self, enabled: bool, tip: str = TIP_NEED_TAGGER) -> None:
        """Enable 识别角色 when the CL Tagger files are on disk."""
        self.identify_button.setEnabled(enabled)
        self.identify_button.setToolTip("" if enabled else tip)

    def name(self) -> str:
        return self.name_edit.text().strip()

    def series(self) -> str:
        return self.series_edit.text().strip()


class SlotCardsPage(QWidget):
    """Three selectable candidate cards for one slot, side by side."""

    regen_requested = Signal(int)  # candidate index

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        inner = QWidget(self)
        self.cards: list[CandidateCard] = []
        self._group = QButtonGroup(inner)
        cards_layout = QHBoxLayout(inner)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(10)
        for index in range(CARD_COUNT):
            card = CandidateCard(index, inner)
            card.regen_requested.connect(self.regen_requested)
            self._group.addButton(card.radio, index)
            self.cards.append(card)
            cards_layout.addWidget(card, 1)
        self.cards[0].radio.setChecked(True)
        self.cards_scroll = CardsScrollArea(self)
        self.cards_scroll.setWidget(inner)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.cards_scroll, 1)

    def selected_card_text(self) -> str:
        """English body of the checked candidate (first card when none)."""
        for card in self.cards:
            if card.radio.isChecked():
                return card.english_text()
        return self.cards[0].english_text() if self.cards else ""


class SlotConfirmView(QWidget):
    """Read-only view of one locked card and its appearance / outfit split."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.confirm_card = QPlainTextEdit(self)
        self.confirm_card.setReadOnly(True)
        self.confirm_card_stats = QLabel(format_card_stats(""), self)
        self.confirm_card_stats.setProperty("muted", True)
        self.confirm_appearance = QPlainTextEdit(self)
        self.confirm_appearance.setReadOnly(True)
        self.confirm_outfit = QPlainTextEdit(self)
        self.confirm_outfit.setReadOnly(True)
        self.confirm_split_warning = QLabel(WARN_SPLIT_FAILED, self)
        self.confirm_split_warning.setProperty("muted", True)
        self.confirm_split_warning.hide()
        card_title = QLabel(LABEL_CHOSEN_CARD, self)
        card_title.setProperty("sectionTitle", True)
        appearance_title = QLabel(LABEL_CARD_APPEARANCE, self)
        appearance_title.setProperty("sectionTitle", True)
        outfit_title = QLabel(LABEL_CARD_OUTFIT, self)
        outfit_title.setProperty("sectionTitle", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(card_title)
        layout.addWidget(self.confirm_card, 1)
        layout.addWidget(self.confirm_card_stats)
        layout.addWidget(appearance_title)
        layout.addWidget(self.confirm_appearance, 1)
        layout.addWidget(outfit_title)
        layout.addWidget(self.confirm_outfit, 1)
        layout.addWidget(self.confirm_split_warning)

    def show_card(self, card: str, name: str) -> None:
        """Fill the boxes from ``card``; warn when no clothing block is found."""
        parts = split_card(card, name)
        self.confirm_card.setPlainText(card)
        self.confirm_card_stats.setText(format_card_stats(card))
        self.confirm_appearance.setPlainText(parts.appearance)
        self.confirm_outfit.setPlainText(parts.outfit)
        self.confirm_split_warning.setVisible(bool(card) and not parts.outfit)


@dataclass
class SlotWidgets:
    """The three per-slot widgets plus a stable id (request ids use it)."""

    uid: int
    role: SlotRoleForm
    cards_page: SlotCardsPage
    confirm: SlotConfirmView
    ref_key: str = ""

    def tab_title(self, position: int) -> str:
        """Tab label: ``角色卡 n`` or ``角色卡 n · name``."""
        name = self.role.name()
        if name:
            return SLOT_TAB_NAMED_FMT.format(n=position + 1, name=name)
        return SLOT_TAB_FMT.format(n=position + 1)
