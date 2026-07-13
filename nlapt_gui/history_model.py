"""Per-file labeled caption history backing the right-panel 历史记录 section.

This is UI-level history (labeled snapshots shown to the user), independent
of the core undo stack. Entries are newest-first. A per-file CURSOR marks the
entry whose caption is currently applied: 回退/跳转 moves the cursor without
deleting anything, so the user can move both ways through the timeline. The
window is bounded to :data:`HISTORY_KEEP_NEWER` entries above the cursor and
:data:`HISTORY_KEEP_OLDER` below it (user requirement: 保留上下 5 条内);
whatever falls outside the window is trimmed. A new edit while the cursor sits
in the past first cuts the "future" branch (standard undo-branch semantics).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, Signal

from nlapt.core.errors import ValidationError
from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Window around the cursor (user requirement: keep within 5 above / 5 below).
HISTORY_KEEP_NEWER = 5
HISTORY_KEEP_OLDER = 5
# Maximum total entries (kept for backward compatibility with older imports).
HISTORY_LIMIT = HISTORY_KEEP_NEWER + HISTORY_KEEP_OLDER + 1
# Label of the entry seeded when a dataset is opened.
INITIAL_LABEL = "载入原始标注"
TIME_FORMAT = "%H:%M:%S"


@dataclass(frozen=True)
class HistoryEntry:
    """One labeled caption snapshot."""

    time_label: str  # 'HH:MM:SS'
    label: str  # e.g. '载入原始标注', '编辑分段「long hair」', '查找替换 ×3'
    caption: str


class FileHistory(QObject):
    """Cursor-based labeled caption history per caption key."""

    changed = Signal(str)  # key

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        now: Callable[[], str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._now = now if now is not None else lambda: time.strftime(TIME_FORMAT)
        self._entries: dict[str, list[HistoryEntry]] = {}
        self._cursor: dict[str, int] = {}

    # -- queries ---------------------------------------------------------------
    def entries(self, key: str) -> tuple[HistoryEntry, ...]:
        """Snapshots newest-first (the cursor row is the applied caption)."""
        return tuple(self._entries.get(key, ()))

    def current_index(self, key: str) -> int:
        """Index of the applied snapshot within :meth:`entries` (0 = newest)."""
        return self._cursor.get(key, 0)

    # -- mutations --------------------------------------------------------------
    def seed(self, key: str, caption: str) -> None:
        """Reset ``key``'s history to the single '载入原始标注' entry."""
        self._entries[key] = [HistoryEntry(self._now(), INITIAL_LABEL, caption)]
        self._cursor[key] = 0
        self.changed.emit(key)

    def push(self, key: str, label: str, caption: str) -> None:
        """Prepend a labeled snapshot; no-op when the caption is unchanged.

        When the cursor sits in the past (after a 回退), the newer branch is
        cut first, exactly like a classic undo stack accepting a new edit.
        """
        entries = self._entries.setdefault(key, [])
        cursor = self._cursor.get(key, 0)
        if entries and 0 <= cursor < len(entries) and entries[cursor].caption == caption:
            return
        if cursor > 0:
            del entries[:cursor]
        entries.insert(0, HistoryEntry(self._now(), label, caption))
        self._cursor[key] = 0
        self._trim(key)
        self.changed.emit(key)

    def revert_caption(self, key: str, index: int) -> str:
        """Move the cursor to ``index`` (either direction); return its caption.

        Nothing is deleted - newer entries stay available so the jump can be
        made both ways; only the ±window trim applies afterwards.
        """
        entries = self._entries.get(key)
        if not entries:
            raise ValidationError(f"no history for key {key!r}")
        if not 0 <= index < len(entries):
            raise ValidationError(
                f"history index {index} out of range for {key!r} ({len(entries)} entries)"
            )
        if index == self._cursor.get(key, 0):
            return entries[index].caption
        self._cursor[key] = index
        self._trim(key)
        self.changed.emit(key)
        return self._entries[key][self._cursor[key]].caption

    def step_older(self, key: str) -> str | None:
        """Move the cursor one entry older (Ctrl+Z); ``None`` at the oldest."""
        entries = self._entries.get(key)
        if not entries:
            return None
        cursor = self._cursor.get(key, 0)
        if cursor + 1 >= len(entries):
            return None
        return self.revert_caption(key, cursor + 1)

    def clear_keep_current(self, key: str) -> None:
        """Keep only the applied entry (design: 清空历史,保留当前状态)."""
        entries = self._entries.get(key)
        if not entries:
            return
        cursor = self._cursor.get(key, 0)
        current = entries[cursor] if 0 <= cursor < len(entries) else entries[0]
        self._entries[key] = [current]
        self._cursor[key] = 0
        self.changed.emit(key)

    # -- internals ---------------------------------------------------------------
    def _trim(self, key: str) -> None:
        """Enforce the ±window: ≤KEEP_NEWER above and ≤KEEP_OLDER below cursor."""
        entries = self._entries.get(key)
        if not entries:
            return
        cursor = self._cursor.get(key, 0)
        if cursor > HISTORY_KEEP_NEWER:
            drop = cursor - HISTORY_KEEP_NEWER
            del entries[:drop]
            cursor -= drop
            self._cursor[key] = cursor
        oldest_kept = cursor + HISTORY_KEEP_OLDER
        if len(entries) - 1 > oldest_kept:
            del entries[oldest_kept + 1 :]
