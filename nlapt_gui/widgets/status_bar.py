"""Bottom status strip (design 状态栏).

26px bar. Left side: a live system clock (year + time), the real save-state
indicator (spec 2.3: 已保存 / 保存中 / 保存失败, plus a live 未保存 n count)
and the mono dataset summary. Right side: shortcut hints plus the live
``模式 · {name}`` / ``主题 · {name}`` labels. Driven purely by
:class:`AppController` signals and (optionally) ``ThemeManager.theme_changed``.
"""

from __future__ import annotations

from PySide6.QtCore import QDateTime, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import MONO_STACK, ThemeTokens

_LOGGER = get_logger(__name__)

STATUS_BAR_HEIGHT = 26
# Right-side hint texts, exactly as in the design.
HINTS: tuple[str, ...] = (
    "Ctrl+S 保存",
    "Alt+↑/↓ 切换",
    "Shift+点击 范围选",
    "Alt+点击 取消选",
)
MODE_NAMES: dict[str, str] = {"chips": "胶囊", "sents": "分句", "text": "文本"}
MODE_PREFIX = "模式 · "
THEME_PREFIX = "主题 · "
ENCODING_LABEL = "UTF-8"
COUNT_SUFFIX = " 个标注文件"
NO_DATASET_TEXT = "未打开数据集"
_DOT = " · "

# Live clock (bottom-left): year + time, ticking every second.
CLOCK_FORMAT = "yyyy-MM-dd HH:mm:ss"
CLOCK_INTERVAL_MS = 1000

# Save-state indicator texts (spec 2.3 三态 + live dirty count).
SAVE_STATE_SAVED = "已保存"
SAVE_STATE_SAVING = "保存中…"
SAVE_STATE_FAILED = "保存失败"
SAVE_STATE_DIRTY_FMT = "未保存 {n}"

# Live batch (推标) progress, hidden while idle.
BATCH_PROGRESS_FMT = "{description} {done}/{total}"

_MONO_FAMILY = ", ".join(f'"{name}"' for name in MONO_STACK)


class StatusBar(QFrame):
    """The 26px bottom strip; all colors come from theme tokens."""

    def __init__(
        self,
        controller: AppController,
        theme_manager: ThemeManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._manager = theme_manager
        self._saving = False
        self._save_failed = False
        self.setFixedHeight(STATUS_BAR_HEIGHT)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(12)
        # Bottom-left live clock (system year + time).
        self.clock_label = QLabel(self)
        self.clock_label.setProperty("mono", "true")
        row.addWidget(self.clock_label)
        # Real save-state indicator (spec 2.3).
        self.save_label = QLabel(self)
        row.addWidget(self.save_label)
        # Live 推标 progress (hidden while no batch runs).
        self.batch_label = QLabel(self)
        self.batch_label.setProperty("saveState", "saving")
        self.batch_label.hide()
        row.addWidget(self.batch_label)
        self.left_label = QLabel(self)
        self.left_label.setProperty("mono", "true")
        row.addWidget(self.left_label)
        row.addStretch(1)
        self._hint_labels: list[QLabel] = []
        for hint in HINTS:
            label = QLabel(hint, self)
            row.addWidget(label)
            self._hint_labels.append(label)
        self.mode_label = QLabel(self)
        row.addWidget(self.mode_label)
        self.theme_label = QLabel(self)
        row.addWidget(self.theme_label)

        controller.dataset_opened.connect(lambda _result: self.refresh())
        controller.mode_changed.connect(lambda _mode: self._refresh_mode())
        controller.caption_changed.connect(lambda _key: self._refresh_save_state())
        controller.files_saved.connect(lambda _keys: self._refresh_save_state())
        controller.save_state_changed.connect(self._on_save_state)
        controller.batch_progress.connect(self._on_batch_progress)
        controller.batch_finished.connect(lambda _d, _r: self.batch_label.hide())

        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(CLOCK_INTERVAL_MS)
        self._clock_timer.timeout.connect(self.tick_clock)
        self._clock_timer.start()
        self.tick_clock()

        if theme_manager is not None:
            theme_manager.theme_changed.connect(self.apply_tokens)
            self.apply_tokens(theme_manager.tokens)
        self.refresh()

    # -- clock ------------------------------------------------------------------------
    def tick_clock(self) -> None:
        """Refresh the bottom-left clock (public so tests can tick directly)."""
        self.clock_label.setText(
            QDateTime.currentDateTime().toString(CLOCK_FORMAT)
        )

    # -- batch progress -----------------------------------------------------------------
    def _on_batch_progress(self, description: str, done: int, total: int) -> None:
        self.batch_label.setText(
            BATCH_PROGRESS_FMT.format(description=description, done=done, total=total)
        )
        self.batch_label.show()

    # -- save state ---------------------------------------------------------------------
    def _on_save_state(self, state: str) -> None:
        self._saving = state == "saving"
        self._save_failed = state == "failed"
        self._refresh_save_state()

    def save_state_text(self) -> str:
        """The rendered save-state text (also used by tests)."""
        if self._saving:
            return SAVE_STATE_SAVING
        if self._save_failed:
            return SAVE_STATE_FAILED
        dirty = self._controller.dirty_count()
        if dirty > 0:
            return SAVE_STATE_DIRTY_FMT.format(n=dirty)
        return SAVE_STATE_SAVED

    def _refresh_save_state(self) -> None:
        text = self.save_state_text()
        self.save_label.setText(text)
        state = (
            "saving"
            if self._saving
            else "failed"
            if self._save_failed
            else "dirty"
            if text.startswith("未保存")
            else "saved"
        )
        self.save_label.setProperty("saveState", state)
        style = self.save_label.style()
        style.unpolish(self.save_label)
        style.polish(self.save_label)

    # -- refresh --------------------------------------------------------------------
    def refresh(self) -> None:
        """Recompute the left summary + right dynamic labels."""
        root = self._controller.root
        if root is None:
            self.left_label.setText(NO_DATASET_TEXT)
        else:
            count = len(self._controller.keys())
            self.left_label.setText(
                f"{root}{_DOT}{ENCODING_LABEL}{_DOT}{count}{COUNT_SUFFIX}"
            )
        self._refresh_mode()
        self._refresh_save_state()
        if self._manager is not None:
            self.theme_label.setText(f"{THEME_PREFIX}{self._manager.theme_name}")

    def _refresh_mode(self) -> None:
        mode = self._controller.mode
        self.mode_label.setText(f"{MODE_PREFIX}{MODE_NAMES.get(mode, mode)}")

    # -- theme ----------------------------------------------------------------------
    def apply_tokens(self, tokens: ThemeTokens) -> None:
        """Restyle from tokens (connected to ThemeManager.theme_changed)."""
        self.setStyleSheet(
            f"""
            StatusBar {{
                background: {tokens.panel};
                border-top: 1px solid {tokens.bd};
            }}
            QLabel {{ color: {tokens.text3}; font-size: 11px; background: transparent; }}
            QLabel[mono="true"] {{ font-family: {_MONO_FAMILY}; font-size: 10.5px; }}
            QLabel[saveState="dirty"] {{ color: {tokens.warn}; font-weight: 600; }}
            QLabel[saveState="failed"] {{ color: {tokens.danger}; font-weight: 600; }}
            QLabel[saveState="saving"] {{ color: {tokens.accent}; }}
            QLabel[saveState="saved"] {{ color: {tokens.text3}; }}
            """
        )
        if self._manager is not None:
            self.theme_label.setText(f"{THEME_PREFIX}{self._manager.theme_name}")
