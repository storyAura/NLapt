"""CHA标注 closing dialog: the locked character cards, ready to copy.

Opened by :class:`MainWindow` when ``VisionBridge.layered_finished`` fires.
Shows every roster card with its match count and a copy button, a
copy-all button, the images that matched no card (skipped) and the failed
ones. The skipped images are listed as rows with a thumbnail and a 移出
checkbox; the dialog only *asks* for the quarantine (``quarantine_requested``)
— the main window confirms and runs it. Pure presentation: no filesystem,
no bridges.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QGuiApplication, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from nlapt.core.errors import NLaptError

from nlapt_gui.layered_prompts import LayeredSummary, format_card_stats
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

WINDOW_TITLE = "CHA标注完成 · 角色卡"
WINDOW_TITLE_CANCELLED = "CHA标注已取消 · 角色卡"
HINT = "以下是本次锁定的角色卡，可直接复制到其它工具或数据集。"
CARD_TITLE_FMT = "卡 {n} · {name}"
CARD_SERIES_FMT = "作品：{series}"
CARD_MATCHED_FMT = "匹配 {n} 张"
BUTTON_COPY = "复制"
BUTTON_COPY_ALL = "复制全部"
BUTTON_CLOSE = "关闭"
BUTTON_QUARANTINE_FMT = "移出选中 ({n})"
CHECK_QUARANTINE = "移出"
COPIED_FMT = "已复制卡 {n}"
COPIED_ALL = "已复制全部角色卡"
NONE_TITLE_FMT = "未归类 {n} 张（不属于任何角色卡，已跳过未写）"
NONE_HINT = "勾选后可移出数据集（连同同名 txt，可用「撤销上次图像操作」恢复）；双击行在主窗口查看。"
NONE_QUARANTINED_FMT = "已移出 {n} 张"
FAILED_TITLE_FMT = "失败 {n} 张（回复无法解析或请求出错，原文未改）"
CANCELLED_NOTE = "批次被取消：已完成的图片已写入，其余未处理。"
KEY_SEPARATOR = "\n"
CARD_JOINER = "\n\n"
CARD_MIN_H = 96
LIST_MAX_H = 120
ROWS_MAX_H = 220
ROW_THUMB = 56
DIALOG_MIN_W = 640
DIALOG_MIN_H = 520


class SummaryCard(QFrame):
    """One roster card: title, count, read-only text, copy button."""

    def __init__(
        self, index: int, name: str, series: str, text: str, matched: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.index = index
        self.text = text
        self.setProperty("surfaceCard", True)
        title = QLabel(CARD_TITLE_FMT.format(n=index + 1, name=name), self)
        title.setProperty("sectionTitle", True)
        self.matched_label = QLabel(CARD_MATCHED_FMT.format(n=matched), self)
        self.matched_label.setProperty("muted", True)
        self.copy_button = QPushButton(BUTTON_COPY, self)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(title)
        head.addWidget(self.matched_label, 1)
        head.addWidget(self.copy_button)
        self.body = QPlainTextEdit(self)
        self.body.setReadOnly(True)
        self.body.setPlainText(text)
        self.body.setMinimumHeight(CARD_MIN_H)
        stats = QLabel(format_card_stats(text), self)
        stats.setProperty("muted", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addLayout(head)
        if series:
            series_label = QLabel(CARD_SERIES_FMT.format(series=series), self)
            series_label.setProperty("muted", True)
            layout.addWidget(series_label)
        layout.addWidget(self.body)
        layout.addWidget(stats)


class NoneKeyRow(QFrame):
    """One skipped image: thumbnail, key, and a 移出 checkbox (default keep)."""

    activated = Signal(str)

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.quarantined = False
        self.setProperty("surfaceCard", True)
        self.thumb = QLabel(self)
        self.thumb.setFixedSize(ROW_THUMB, ROW_THUMB)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label = QLabel(key, self)
        self.name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.check = QCheckBox(CHECK_QUARANTINE, self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)
        layout.addWidget(self.thumb)
        layout.addWidget(self.name_label, 1)
        layout.addWidget(self.check)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Fit the decoded thumbnail into the row cell."""
        self.thumb.setPixmap(
            pixmap.scaled(
                QSize(ROW_THUMB, ROW_THUMB),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_quarantined(self) -> None:
        """Grey the row out once the file has left the dataset."""
        self.quarantined = True
        self.check.setChecked(False)
        self.check.setEnabled(False)
        self.name_label.setProperty("muted", True)
        self.setEnabled(False)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and not self.quarantined:
            self.activated.emit(self.key)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class LayeredSummaryDialog(CenteredDialog):
    """Recap of a finished CHA标注 batch with copyable cards and skipped rows."""

    # Keys the user ticked for removal (the main window confirms + runs it).
    quarantine_requested = Signal(object)
    # Double-clicked skipped key (the main window makes it current).
    key_activated = Signal(str)

    def __init__(
        self,
        summary: LayeredSummary,
        parent: QWidget | None = None,
        *,
        loader: ThumbnailLoader | None = None,
        resolve_path: Callable[[str], Path] | None = None,
    ) -> None:
        super().__init__(parent)
        self._summary = summary
        self._loader = loader
        self._resolve_path = resolve_path
        self.setWindowTitle(WINDOW_TITLE_CANCELLED if summary.cancelled else WINDOW_TITLE)
        # Non-modal on purpose: double-clicking a skipped row shows that image
        # in the main window so the user can judge 移出 / 保留.
        self.setModal(False)
        self.setMinimumSize(DIALOG_MIN_W, DIALOG_MIN_H)

        hint = QLabel(HINT, self)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)

        inner = QWidget(self)
        cards_layout = QVBoxLayout(inner)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)
        self.cards: list[SummaryCard] = []
        for index, card in enumerate(summary.roster.cards):
            matched = (
                summary.matched_counts[index] if index < len(summary.matched_counts) else 0
            )
            widget = SummaryCard(index, card.name, card.series, card.text, matched, inner)
            widget.copy_button.clicked.connect(lambda _=False, i=index: self.copy_card(i))
            self.cards.append(widget)
            cards_layout.addWidget(widget)
        cards_layout.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setWidget(inner)

        self.cancelled_note = QLabel(CANCELLED_NOTE, self)
        self.cancelled_note.setProperty("muted", True)
        self.cancelled_note.setWordWrap(True)
        self.cancelled_note.setVisible(summary.cancelled)
        self._build_none_rows(summary.none_keys)
        self.failed_title, self.failed_list = self._key_list(
            FAILED_TITLE_FMT, summary.failed_keys
        )

        self.status_label = QLabel("", self)
        self.status_label.setProperty("muted", True)
        self.quarantine_button = QPushButton(BUTTON_QUARANTINE_FMT.format(n=0), self)
        self.quarantine_button.setEnabled(False)
        self.quarantine_button.setVisible(bool(summary.none_keys))
        self.quarantine_button.clicked.connect(self._request_quarantine)
        self.copy_all_button = QPushButton(BUTTON_COPY_ALL, self)
        self.copy_all_button.clicked.connect(self.copy_all)
        self.close_button = QPushButton(BUTTON_CLOSE, self)
        self.close_button.setDefault(True)
        self.close_button.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self.status_label, 1)
        buttons.addWidget(self.quarantine_button)
        buttons.addWidget(self.copy_all_button)
        buttons.addWidget(self.close_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)
        root.addWidget(hint)
        root.addWidget(scroll, 1)
        root.addWidget(self.cancelled_note)
        root.addWidget(self.none_title)
        root.addWidget(self.none_hint)
        root.addWidget(self.none_scroll)
        root.addWidget(self.failed_title)
        root.addWidget(self.failed_list)
        root.addLayout(buttons)
        if loader is not None:
            loader.ready.connect(self._on_thumb_ready)

    # -- skipped rows -----------------------------------------------------------
    def _build_none_rows(self, keys: tuple[str, ...]) -> None:
        self.none_title = QLabel(NONE_TITLE_FMT.format(n=len(keys)), self)
        self.none_title.setProperty("sectionTitle", True)
        self.none_hint = QLabel(NONE_HINT, self)
        self.none_hint.setProperty("muted", True)
        self.none_hint.setWordWrap(True)
        inner = QWidget(self)
        rows = QVBoxLayout(inner)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(4)
        self.none_rows: list[NoneKeyRow] = []
        for key in keys:
            row = NoneKeyRow(key, inner)
            row.check.toggled.connect(lambda _c: self._refresh_quarantine_button())
            row.activated.connect(self.key_activated)
            self.none_rows.append(row)
            rows.addWidget(row)
            self._request_thumb(row)
        rows.addStretch(1)
        self.none_scroll = QScrollArea(self)
        self.none_scroll.setWidgetResizable(True)
        self.none_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.none_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.none_scroll.setMaximumHeight(ROWS_MAX_H)
        self.none_scroll.setWidget(inner)
        visible = bool(keys)
        self.none_title.setVisible(visible)
        self.none_hint.setVisible(visible)
        self.none_scroll.setVisible(visible)

    def _request_thumb(self, row: NoneKeyRow) -> None:
        if self._loader is None or self._resolve_path is None:
            return
        try:
            path = self._resolve_path(row.key)
        except NLaptError:
            return
        pixmap = self._loader.request(row.key, path, ROW_THUMB)
        if pixmap is not None:
            row.set_pixmap(pixmap)

    def _on_thumb_ready(self, key: str, pixmap: QPixmap) -> None:
        if not isValid(self):
            return
        for row in self.none_rows:
            if row.key == key:
                row.set_pixmap(pixmap)

    def checked_none_keys(self) -> tuple[str, ...]:
        """Skipped keys currently ticked for removal."""
        return tuple(
            row.key for row in self.none_rows if row.check.isChecked() and not row.quarantined
        )

    def mark_quarantined(self, keys: tuple[str, ...]) -> None:
        """Grey out rows whose files have left the dataset."""
        wanted = set(keys)
        for row in self.none_rows:
            if row.key in wanted:
                row.set_quarantined()
        done = sum(1 for row in self.none_rows if row.quarantined)
        if done:
            self.status_label.setText(NONE_QUARANTINED_FMT.format(n=done))
        self._refresh_quarantine_button()

    def _refresh_quarantine_button(self) -> None:
        count = len(self.checked_none_keys())
        self.quarantine_button.setText(BUTTON_QUARANTINE_FMT.format(n=count))
        self.quarantine_button.setEnabled(count > 0)

    def _request_quarantine(self) -> None:
        keys = self.checked_none_keys()
        if keys:
            self.quarantine_requested.emit(keys)

    # -- failed list / copy -----------------------------------------------------
    def _key_list(self, title_fmt: str, keys: tuple[str, ...]) -> tuple[QLabel, QPlainTextEdit]:
        title = QLabel(title_fmt.format(n=len(keys)), self)
        title.setProperty("sectionTitle", True)
        box = QPlainTextEdit(self)
        box.setReadOnly(True)
        box.setMaximumHeight(LIST_MAX_H)
        box.setPlainText(KEY_SEPARATOR.join(keys))
        visible = bool(keys)
        title.setVisible(visible)
        box.setVisible(visible)
        return title, box

    def all_cards_text(self) -> str:
        """Every card's text joined by blank lines (the copy-all payload)."""
        return CARD_JOINER.join(card.text for card in self._summary.roster.cards)

    def copy_card(self, index: int) -> None:
        """Put card ``index`` on the clipboard."""
        if not 0 <= index < len(self.cards):
            return
        QGuiApplication.clipboard().setText(self.cards[index].text)
        self.status_label.setText(COPIED_FMT.format(n=index + 1))

    def copy_all(self) -> None:
        """Put all cards on the clipboard."""
        QGuiApplication.clipboard().setText(self.all_cards_text())
        self.status_label.setText(COPIED_ALL)
