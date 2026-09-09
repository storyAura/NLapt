"""Shared scope picker used by 推标 and image-tool launchers."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from nlapt_gui.controller import AppController
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.tools_panel import ScopeSelector

WINDOW_TITLE = "选择应用范围"
HINT_DEFAULT = "选择要处理的训练集范围（文件夹含其下所有子文件夹）。"
LABEL_COUNT = "将处理 {n} 张"
BUTTON_OK = "确定"
BUTTON_CANCEL = "取消"
DIALOG_W = 420


class BatchScopeDialog(CenteredDialog):
    """当前 / 文件夹 / 选中 / 全部 picker that returns ``scope_keys``."""

    def __init__(
        self,
        controller: AppController,
        title: str = WINDOW_TITLE,
        *,
        hint: str = HINT_DEFAULT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle(title)
        self.setMinimumWidth(DIALOG_W)
        self.setModal(True)

        self.hint_label = QLabel(hint, self)
        self.hint_label.setWordWrap(True)
        self.hint_label.setProperty("muted", True)

        self.scope = ScopeSelector(controller, self)
        self.scope.changed.connect(lambda _scope: self._refresh_count())
        self.count_label = QLabel(self)
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.ok_button = QPushButton(BUTTON_OK, self)
        self.ok_button.setProperty("variant", "accent")
        self.ok_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton(BUTTON_CANCEL, self)
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.ok_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.scope)
        layout.addWidget(self.count_label)
        layout.addLayout(buttons)
        self._refresh_count()

    def selected_keys(self) -> tuple[str, ...]:
        """Keys currently covered by the chosen scope."""
        return self._controller.scope_keys(self.scope.scope)

    def _refresh_count(self) -> None:
        count = len(self.selected_keys())
        self.count_label.setText(LABEL_COUNT.format(n=count))
        self.ok_button.setEnabled(count > 0)


def pick_scope_keys(
    controller: AppController,
    title: str,
    *,
    hint: str = HINT_DEFAULT,
    parent: QWidget | None = None,
) -> tuple[str, ...] | None:
    """Show the dialog; return keys on accept, ``None`` on cancel."""
    dialog = BatchScopeDialog(controller, title, hint=hint, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    keys = dialog.selected_keys()
    return keys if keys else None
