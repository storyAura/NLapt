"""Sub-window: hash-based near-duplicate review and quarantine."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from nlapt.images.hashing import DuplicateGroup, ImageFingerprint, suggest_keep

from nlapt_gui.controller import AppController
from nlapt_gui.image_tools_bridge import OP_DUPLICATES, ImageToolsBridge
from nlapt_gui.widgets.dialogs import ask_confirm, CenteredDialog
from nlapt_gui.widgets.sections.common import ScopeRow
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

WINDOW_TITLE = "查找雷同图片"
HINT = "用感知哈希找出范围内视觉相近或完全相同的图片，确认后再移出数据集。"
LABEL_DISTANCE = "汉明距离 {n}（0 = 完全相同）"
BUTTON_SCAN = "开始检索"
BUTTON_CLOSE = "关闭"
BUTTON_QUARANTINE = "将未保留的 {n} 张移到 .backups/duplicates/"
CONFIRM_TITLE = "移出雷同图片"
CONFIRM_TEXT = "将把 {n} 张图片（及同名 txt）移到 .backups/duplicates/，可从「撤销上次图像操作」恢复。"
EMPTY_GROUPS = "还没有结果。选择范围后点「开始检索」。"
KEEP_LABEL = "保留"
META_FMT = "{w} × {h} · {size}"
GROUP_TITLE = "第 {n} 组 · {kind}"
KIND_EXACT = "完全相同"
KIND_SIMILAR = "视觉相近"
THUMB_H = 88
DIALOG_W = 720
DIALOG_H = 560
DEFAULT_DISTANCE = 6
MAX_DISTANCE = 16


def _size_label(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes / (1024 * 1024):.1f} MB"


class DuplicateReviewDialog(CenteredDialog):
    """Scan a scope for near-duplicates, then quarantine unchecked members."""

    def __init__(
        self,
        controller: AppController,
        bridge: ImageToolsBridge,
        loader: ThumbnailLoader,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._bridge = bridge
        self._loader = loader
        self._groups: tuple[DuplicateGroup, ...] = ()
        self._keep: dict[str, QCheckBox] = {}
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(DIALOG_W, DIALOG_H)
        self.setModal(True)

        hint = QLabel(HINT, self)
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        self.scope_row = ScopeRow(controller, self)

        self.distance_label = QLabel(LABEL_DISTANCE.format(n=DEFAULT_DISTANCE), self)
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, MAX_DISTANCE)
        self.slider.setValue(DEFAULT_DISTANCE)
        self.slider.valueChanged.connect(self._on_distance)

        self.scan_button = QPushButton(BUTTON_SCAN, self)
        self.scan_button.setProperty("variant", "accent")
        self.scan_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.scan_button.clicked.connect(self._scan)

        self.progress = QProgressBar(self)
        self.progress.hide()

        self.empty_label = QLabel(EMPTY_GROUPS, self)
        self.empty_label.setProperty("muted", True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._stack = QWidget(self)
        self._stack_layout = QVBoxLayout(self._stack)
        self._stack_layout.setContentsMargins(0, 0, 0, 0)
        self._stack_layout.setSpacing(10)
        self._stack_layout.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._stack)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.close_button = QPushButton(BUTTON_CLOSE, self)
        self.close_button.setProperty("variant", "outline")
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.clicked.connect(self.reject)
        self.quarantine_button = QPushButton(BUTTON_QUARANTINE.format(n=0), self)
        self.quarantine_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.quarantine_button.setEnabled(False)
        self.quarantine_button.clicked.connect(self._quarantine)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        layout.addWidget(hint)
        layout.addWidget(self.scope_row)
        layout.addWidget(self.distance_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.scan_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.progress)
        layout.addWidget(self.empty_label)
        layout.addWidget(scroll, 1)
        actions = QHBoxLayout()
        actions.addWidget(self.close_button)
        actions.addStretch(1)
        actions.addWidget(self.quarantine_button)
        layout.addLayout(actions)

        loader.ready.connect(self._on_thumb)
        bridge.progress.connect(self._on_progress)
        bridge.groups_ready.connect(self._on_groups)
        bridge.quarantine_finished.connect(self._on_quarantined)
        self._thumbs: dict[str, QLabel] = {}

    def _on_distance(self, value: int) -> None:
        self.distance_label.setText(LABEL_DISTANCE.format(n=value))

    def _scan(self) -> None:
        keys = self._controller.scope_keys(self.scope_row.scope)
        self.scan_button.setEnabled(False)
        self.progress.setRange(0, max(1, len(keys)))
        self.progress.setValue(0)
        self.progress.show()
        self._bridge.scan_duplicates(keys, self.slider.value())

    def _on_progress(self, description: str, done: int, total: int) -> None:
        if description != OP_DUPLICATES:
            return
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _on_groups(self, groups: object) -> None:
        self.scan_button.setEnabled(True)
        self.progress.hide()
        self._groups = groups if isinstance(groups, tuple) else ()
        self._rebuild()

    def _rebuild(self) -> None:
        while self._stack_layout.count() > 1:
            item = self._stack_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._keep.clear()
        self._thumbs.clear()
        self.empty_label.setVisible(not self._groups)
        for index, group in enumerate(self._groups, start=1):
            self._stack_layout.insertWidget(self._stack_layout.count() - 1, self._group_card(index, group))
        self._refresh_quarantine()

    def _group_card(self, index: int, group: DuplicateGroup) -> QFrame:
        card = QFrame(self._stack)
        card.setProperty("surfaceCard", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        kind = KIND_EXACT if group.exact else KIND_SIMILAR
        title = QLabel(GROUP_TITLE.format(n=index, kind=kind), card)
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        row = QHBoxLayout()
        row.setSpacing(10)
        keep_key = suggest_keep(group).key
        for member in group.members:
            row.addWidget(self._member_cell(member, keep=member.key == keep_key))
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _member_cell(self, member: ImageFingerprint, *, keep: bool) -> QWidget:
        cell = QWidget(self._stack)
        cell.setFixedWidth(140)
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        thumb = QLabel(cell)
        thumb.setFixedSize(140, THUMB_H)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setProperty("muted", True)
        self._thumbs[member.key] = thumb
        pix = self._loader.request(member.key, member.path, THUMB_H)
        if pix is not None:
            thumb.setPixmap(pix.scaled(
                thumb.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        name = QLabel(member.key, cell)
        name.setWordWrap(True)
        name.setStyleSheet("font-size: 11px;")
        meta = QLabel(
            META_FMT.format(w=member.width, h=member.height, size=_size_label(member.size)),
            cell,
        )
        meta.setProperty("muted", True)
        box = QCheckBox(KEEP_LABEL, cell)
        box.setChecked(keep)
        box.toggled.connect(lambda _c: self._refresh_quarantine())
        self._keep[member.key] = box
        layout.addWidget(thumb)
        layout.addWidget(name)
        layout.addWidget(meta)
        layout.addWidget(box)
        return cell

    def _on_thumb(self, key: str, pixmap: QPixmap) -> None:
        label = self._thumbs.get(key)
        if label is None:
            return
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _drop_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for group in self._groups:
            kept = [member for member in group.members if self._keep.get(member.key) and self._keep[member.key].isChecked()]
            if not kept:
                continue
            for member in group.members:
                box = self._keep.get(member.key)
                if box is not None and not box.isChecked():
                    paths.append(member.path)
        return tuple(paths)

    def _refresh_quarantine(self) -> None:
        count = len(self._drop_paths())
        self.quarantine_button.setText(BUTTON_QUARANTINE.format(n=count))
        self.quarantine_button.setEnabled(count > 0)

    def _quarantine(self) -> None:
        paths = self._drop_paths()
        if not paths:
            return
        if not ask_confirm(
            self,
            CONFIRM_TITLE,
            CONFIRM_TEXT.format(n=len(paths)),
            destructive=True,
        ):
            return
        self.quarantine_button.setEnabled(False)
        self._bridge.quarantine(paths)

    def _on_quarantined(self, _count: object) -> None:
        self.accept()
