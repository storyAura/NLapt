"""分层推标 wizard pieces: candidate cards, ref-image picker, preview pane."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from nlapt.core.errors import NLaptError

from nlapt_gui.controller import AppController
from nlapt_gui.layered_prompts import format_card_stats
from nlapt_gui.theme.tokens import ThemeTokens
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

CARD_TITLE_FMT = "候选 {n}"
CARD_MODEL_FMT = "模型: {model}"
BUTTON_REGEN_ONE = "重新生成"
PLACEHOLDER_CARD = "生成中…"
PLACEHOLDER_ZH = "中文对照将显示在这里"
REF_FOLDER_ALL_FMT = "全部 {n}"
REF_FOLDER_FMT = "{folder} {n}"
# Combo item data for the "every folder" entry (folder names are never empty).
REF_FOLDER_ALL = ""
ENGLISH_MIN_H = 88
THUMB_H = 110
THUMB_W = 110
GRID_SPACING = 6
GRID_MIN_H = 2 * THUMB_H + GRID_SPACING
PREVIEW_MIN_H = 200
# Scroll viewport stays this size so long 中文对照 cannot inflate the wizard.
CARDS_SCROLL_HINT = QSize(1040, 380)
CARDS_SCROLL_MIN = QSize(480, 200)


class CandidateCard(QFrame):
    """One selectable card: editable English, stats, read-only Chinese, regen."""

    regen_requested = Signal(int)

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        self.setProperty("surfaceCard", True)
        self.radio = QRadioButton(CARD_TITLE_FMT.format(n=index + 1), self)
        self.model_label = QLabel("", self)
        self.model_label.setProperty("muted", True)
        self.model_label.hide()
        self.english = QPlainTextEdit(self)
        self.english.setPlaceholderText(PLACEHOLDER_CARD)
        self.english.setMinimumHeight(ENGLISH_MIN_H)
        self.stats = QLabel(format_card_stats(""), self)
        self.stats.setProperty("muted", True)
        self.chinese = QLabel(PLACEHOLDER_ZH, self)
        self.chinese.setProperty("muted", True)
        self.chinese.setWordWrap(True)
        self.chinese.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.regen = QPushButton(BUTTON_REGEN_ONE, self)
        self.regen.clicked.connect(lambda: self.regen_requested.emit(self.index))
        self.english.textChanged.connect(self._refresh_stats)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.radio)
        head.addWidget(self.model_label, 1)
        head.addWidget(self.regen)

        body = QVBoxLayout(self)
        body.setContentsMargins(10, 10, 10, 10)
        body.setSpacing(6)
        body.addLayout(head)
        body.addWidget(self.english)
        body.addWidget(self.stats)
        body.addWidget(self.chinese)

    def english_text(self) -> str:
        """Current English card body."""
        return self.english.toPlainText().strip()

    def set_english(self, text: str) -> None:
        """Replace the English body (used when a generate call returns)."""
        self.english.setPlainText(text)

    def set_chinese(self, text: str) -> None:
        """Replace the review-only Chinese gloss."""
        self.chinese.setText(text if text.strip() else PLACEHOLDER_ZH)

    def set_model_name(self, model: str) -> None:
        """Show which model this slot calls; hide the label when empty."""
        name = model.strip()
        self.model_label.setText(CARD_MODEL_FMT.format(model=name) if name else "")
        self.model_label.setVisible(bool(name))

    def set_busy(self, busy: bool) -> None:
        """Disable editing while this slot is in flight."""
        self.english.setReadOnly(busy)
        self.regen.setEnabled(not busy)
        if busy:
            self.english.setPlaceholderText(PLACEHOLDER_CARD)

    def _refresh_stats(self) -> None:
        self.stats.setText(format_card_stats(self.english.toPlainText()))


class CardsScrollArea(QScrollArea):
    """Candidate-card viewport with a stable size hint.

    A wrapping ``QLabel`` for 中文对照 grows with the text. Without a
    capped scroll area that sizeHint bubbles to the dialog, the window
    jumps to ``y=0`` and the title bar goes off-screen.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: N802
        return CARDS_SCROLL_HINT

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return CARDS_SCROLL_MIN


class PreviewPane(QLabel):
    """Scales a source pixmap into the current widget rect (KeepAspectRatio)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(PREVIEW_MIN_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setProperty("muted", True)

    def set_source(self, pixmap: QPixmap) -> None:
        """Keep the original and redraw to the current size."""
        self._source = pixmap
        self._rescale()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source.isNull():
            return
        box = self.contentsRect().size()
        if box.width() < 8 or box.height() < 8:
            return
        self.setPixmap(
            self._source.scaled(
                box,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class RefThumb(QFrame):
    """One clickable thumbnail cell in the reference strip."""

    clicked = Signal(str)

    def __init__(
        self, key: str, tokens: ThemeTokens, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.key = key
        self._tokens = tokens
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(THUMB_W, THUMB_H)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._image = QLabel(self)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setFixedSize(THUMB_W - 6, THUMB_H - 6)
        self._image.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.addWidget(self._image)
        self.set_selected(False)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Fit ``pixmap`` inside the cell."""
        self._image.setPixmap(
            pixmap.scaled(
                QSize(THUMB_W - 6, THUMB_H - 6),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_selected(self, selected: bool) -> None:
        """Accent border when this cell is the reference image."""
        tokens = self._tokens
        border = tokens.accent if selected else tokens.bd
        width = "2px" if selected else "1px"
        self.setStyleSheet(
            f"RefThumb {{ border: {width} solid {border}; border-radius: 7px; }}"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.key)
            event.accept()
            return
        super().mousePressEvent(event)


class RefPickerGrid(QScrollArea):
    """Wrapping grid of clickable thumbs; ``picked`` fires on selection.

    Cells reflow to as many columns as the viewport width allows (at least
    one) and the grid scrolls vertically, so a whole dataset stays browsable.
    """

    picked = Signal(str)

    def __init__(
        self,
        keys: Sequence[str],
        controller: AppController,
        loader: ThumbnailLoader | None,
        tokens: ThemeTokens,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._loader = loader
        self._tokens = tokens
        self._selected = ""
        self._thumbs: dict[str, RefThumb] = {}
        self._columns = 1
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumHeight(GRID_MIN_H)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._inner = QWidget(self)
        self._grid = QGridLayout(self._inner)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(GRID_SPACING)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setWidget(self._inner)
        self.set_keys(keys)
        if loader is not None:
            loader.ready.connect(self._on_thumb_ready)

    def keys(self) -> tuple[str, ...]:
        """Keys currently shown in the grid, in display order."""
        return tuple(self._thumbs)

    def columns(self) -> int:
        """Columns the grid currently wraps at (≥ 1)."""
        return self._columns

    def set_keys(self, keys: Sequence[str]) -> None:
        """Rebuild the grid for ``keys``; the highlight survives when still listed."""
        for cell in self._thumbs.values():
            self._grid.removeWidget(cell)
            cell.deleteLater()
        self._thumbs = {}
        for key in keys:
            thumb = RefThumb(key, self._tokens, self._inner)
            thumb.clicked.connect(self.select)
            self._thumbs[key] = thumb
            self._request_thumb(key, thumb)
        self._reflow()
        if self._selected in self._thumbs:
            self.select(self._selected, notify=False)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._columns_for_width() != self._columns:
            self._reflow()

    def _columns_for_width(self) -> int:
        width = self.viewport().width()
        return max(1, (width + GRID_SPACING) // (THUMB_W + GRID_SPACING))

    def _reflow(self) -> None:
        self._columns = self._columns_for_width()
        for cell in self._thumbs.values():
            self._grid.removeWidget(cell)
        for index, cell in enumerate(self._thumbs.values()):
            self._grid.addWidget(cell, index // self._columns, index % self._columns)

    def selected_key(self) -> str:
        """Currently highlighted reference key (empty when none)."""
        return self._selected

    def thumb(self, key: str) -> RefThumb | None:
        """Cell widget for tests / ensure-visible."""
        return self._thumbs.get(key)

    def select(self, key: str, *, notify: bool = True) -> None:
        """Mark ``key`` as the reference image."""
        if key not in self._thumbs:
            return
        self._selected = key
        for name, cell in self._thumbs.items():
            cell.set_selected(name == key)
        cell = self._thumbs[key]
        self.ensureWidgetVisible(cell)
        if notify:
            self.picked.emit(key)

    def _request_thumb(self, key: str, thumb: RefThumb) -> None:
        if self._loader is None:
            return
        try:
            path = self._controller.image_path(key)
        except NLaptError:
            return
        pixmap = self._loader.request(key, path, THUMB_H)
        if pixmap is not None:
            thumb.set_pixmap(pixmap)

    def _on_thumb_ready(self, key: str, pixmap: QPixmap) -> None:
        if not isValid(self):
            return
        thumb = self._thumbs.get(key)
        if thumb is not None:
            thumb.set_pixmap(pixmap)


class RefPickerPanel(QWidget):
    """Folder filter + :class:`RefPickerGrid` over the WHOLE dataset.

    The reference image for a character card may live in any folder, not
    only in the batch being captioned, so the strip lists every key and the
    combo narrows it per folder (hidden when the dataset has one folder).
    """

    picked = Signal(str)

    def __init__(
        self,
        keys: Sequence[str],
        controller: AppController,
        loader: ThumbnailLoader | None,
        tokens: ThemeTokens,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._all_keys = tuple(keys)
        self.folder_combo = QComboBox(self)
        self.folder_combo.addItem(
            REF_FOLDER_ALL_FMT.format(n=len(self._all_keys)), REF_FOLDER_ALL
        )
        folders = controller.folders()
        for folder in folders:
            count = len(controller.folder_keys(folder))
            self.folder_combo.addItem(REF_FOLDER_FMT.format(folder=folder, n=count), folder)
        self.folder_combo.setVisible(len(folders) > 1)
        self.folder_combo.currentIndexChanged.connect(self._on_folder_changed)

        self.grid = RefPickerGrid(self._all_keys, controller, loader, tokens, self)
        self.grid.picked.connect(self.picked)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.addWidget(self.folder_combo)
        filter_row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(filter_row)
        layout.addWidget(self.grid)

    def current_folder(self) -> str:
        """Folder the strip is narrowed to (``REF_FOLDER_ALL`` for everything)."""
        return str(self.folder_combo.currentData() or REF_FOLDER_ALL)

    def keys(self) -> tuple[str, ...]:
        """Keys currently visible in the strip."""
        return self.grid.keys()

    def selected_key(self) -> str:
        return self.grid.selected_key()

    def thumb(self, key: str) -> RefThumb | None:
        return self.grid.thumb(key)

    def set_folder(self, folder: str) -> None:
        """Narrow the strip to ``folder`` (``REF_FOLDER_ALL`` shows every key)."""
        index = self.folder_combo.findData(folder)
        if index < 0:
            index = 0
        if index == self.folder_combo.currentIndex():
            return
        self.folder_combo.setCurrentIndex(index)

    def select(self, key: str, *, notify: bool = True) -> None:
        """Highlight ``key``, widening the filter to its folder when hidden."""
        if key not in self._all_keys:
            return
        if key not in self.grid.keys():
            self.set_folder(self._controller.folder_of(key))
        self.grid.select(key, notify=notify)

    def _on_folder_changed(self, _index: int) -> None:
        folder = self.current_folder()
        if folder == REF_FOLDER_ALL:
            keys: Sequence[str] = self._all_keys
        else:
            keys = tuple(k for k in self._all_keys if self._controller.folder_of(k) == folder)
        self.grid.set_keys(keys)
