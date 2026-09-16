"""Glue between the CHA标注 summary dialog and the image-tools bridge.

Opens :class:`LayeredSummaryDialog` for a finished batch and turns its
``quarantine_requested`` into a confirmed ``ImageToolsBridge.quarantine``
(image + sibling txt → ``.backups/duplicates/``), greying the rows out once
the bridge reports success. Keeps this flow out of :class:`MainWindow`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from nlapt.core.errors import NLaptError

from nlapt_gui.controller import AppController
from nlapt_gui.image_tools_bridge import ImageToolsBridge
from nlapt_gui.layered_prompts import LayeredSummary
from nlapt_gui.widgets.dialogs import ask_confirm
from nlapt_gui.widgets.layered_summary_dialog import LayeredSummaryDialog
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

QUARANTINE_TITLE = "移出未归类图片"
QUARANTINE_TEXT = (
    "将把 {n} 张图片连同同名 txt 移到 .backups/duplicates/，"
    "可用「撤销上次图像操作」恢复。确定移出？"
)


class LayeredSummaryFlow(QObject):
    """Owns the current recap dialog and its 移出 round trip."""

    def __init__(
        self,
        window: QWidget,
        controller: AppController,
        image_tools: ImageToolsBridge,
        loader: ThumbnailLoader,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._controller = controller
        self._image_tools = image_tools
        self._loader = loader
        self.dialog: LayeredSummaryDialog | None = None

    def show(self, summary: LayeredSummary) -> LayeredSummaryDialog:
        """Replace any previous recap with one for ``summary`` and show it."""
        if self.dialog is not None and isValid(self.dialog):
            self.dialog.close()
        dialog = LayeredSummaryDialog(
            summary,
            self._window,
            loader=self._loader,
            resolve_path=self._controller.image_path,
        )
        dialog.key_activated.connect(self._controller.set_current)
        dialog.quarantine_requested.connect(self.quarantine)
        self.dialog = dialog
        dialog.show()
        return dialog

    def quarantine(self, keys: object) -> None:
        """Confirm, then move the ticked 未归类 images (and txt) to .backups/."""
        wanted = tuple(keys) if isinstance(keys, (list, tuple)) else ()
        paths: list[Path] = []
        resolved: list[str] = []
        for key in wanted:
            try:
                paths.append(self._controller.image_path(key))
            except NLaptError:
                continue
            resolved.append(key)
        if not paths:
            return
        if not ask_confirm(
            self._window,
            QUARANTINE_TITLE,
            QUARANTINE_TEXT.format(n=len(paths)),
            destructive=True,
        ):
            return
        dialog = self.dialog
        if not self._image_tools.quarantine(tuple(paths)):
            return
        pending = tuple(resolved)

        def finished(count: object) -> None:
            self._image_tools.quarantine_finished.disconnect(finished)
            if count is None or dialog is None or not isValid(dialog):
                return
            dialog.mark_quarantined(pending)

        self._image_tools.quarantine_finished.connect(finished)
