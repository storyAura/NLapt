"""Width-driven thumbnail geometry for the file panel's folder bodies."""

from __future__ import annotations

import math

from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QWidget

from nlapt_gui.widgets.thumb_cells import ThumbCell

GRID_GAP = 10


class _ThumbGrid(QWidget):
    """Reflowing grid: auto-fill minmax(min_w, 1fr) columns like the design."""

    def __init__(self, cells: list[ThumbCell], min_w: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cells = cells
        self._min_w = max(1, min_w)
        self._last_width = -1
        self._last_cols = 0
        for cell in cells:
            cell.setParent(self)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        width = self.width()
        if not self._cells or width < self._min_w // 2:
            return
        cols = max(1, (width + GRID_GAP) // (self._min_w + GRID_GAP))
        if width == self._last_width and cols == self._last_cols:
            return
        self._last_width = width
        self._last_cols = cols
        cell_w = (width - GRID_GAP * (cols - 1)) / cols
        cell_h = round(cell_w * 4 / 3)  # design aspect-ratio 3/4
        for index, cell in enumerate(self._cells):
            row, col = divmod(index, cols)
            x = round(col * (cell_w + GRID_GAP))
            right = round((col + 1) * (cell_w + GRID_GAP)) - GRID_GAP
            cell.setGeometry(x, row * (cell_h + GRID_GAP), right - x, cell_h)
        rows = math.ceil(len(self._cells) / cols)
        height = rows * cell_h + (rows - 1) * GRID_GAP if rows else 0
        if self.height() != height:
            self.setFixedHeight(height)
