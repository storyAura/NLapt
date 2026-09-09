"""Grouped 工具 popup (当前模型 / 处理 / 标注 / 预处理) launched from the left rail.

The 当前模型 group shows the 文本 / 视觉 targets; clicking one opens a menu of
the model pool (every switched-on model across the API profiles, pi-style
quick switch) and emits ``model_switch_requested(role, ModelRef)``.
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

from nlapt.core.config import ROLE_TEXT, ROLE_VISION, ModelRef

from nlapt_gui.model_targets import (
    LABEL_STALE,
    LABEL_UNSET,
    ModelChoice,
    find_choice,
)
from nlapt_gui.theme.tokens import ACCENT_SOFT_PCT, ThemeTokens, mix

ACTION_FLATTEN_ALPHA = "flatten_alpha"
ACTION_UNDO_IMAGE_OP = "undo_image"
ACTION_INFER_LLM = "infer_llm"
ACTION_INFER_LOCAL = "infer_local"
ACTION_INFER_CHA = "infer_cha"
ACTION_INFER_COMPARE = "infer_compare"
ACTION_FIND_DUPLICATES = "find_duplicates"
ACTION_MODEL_TEXT = "model_text"
ACTION_MODEL_VISION = "model_vision"

LABEL_FLATTEN_ALPHA = "替换透明底…"
LABEL_UNDO_IMAGE_OP = "撤销上次图像操作…"
LABEL_INFER_LLM = "LLM 推标…"
LABEL_INFER_LOCAL = "本地模型推标…"
LABEL_INFER_CHA = "CHA 推标…"
LABEL_INFER_COMPARE = "多对比推标…"
LABEL_FIND_DUPLICATES = "查找雷同图片…"
LABEL_MODEL_TEXT = "文本: {model}"
LABEL_MODEL_VISION = "视觉: {model}"
MENU_POOL_EMPTY = "模型池为空 — 请在设置 ▸ LLM 设置中启用模型"
MENU_CLEAR_TARGET = "不使用"
MENU_VISION_TAG = " · 视觉"

GROUP_MODELS = "当前模型"
GROUP_PROCESS = "处理工具"
GROUP_ANNOTATE = "标注工具"
GROUP_PREPROCESS = "预处理工具"

POPUP_W = 248
MENU_OFFSET_Y = 2
MODEL_ACTIONS: dict[str, str] = {ACTION_MODEL_TEXT: ROLE_TEXT, ACTION_MODEL_VISION: ROLE_VISION}
# Switching the model under a running batch would swap engines mid-flight.
BUSY_ACTIONS = frozenset(
    {
        ACTION_FLATTEN_ALPHA,
        ACTION_UNDO_IMAGE_OP,
        ACTION_FIND_DUPLICATES,
        ACTION_MODEL_TEXT,
        ACTION_MODEL_VISION,
    }
)


class ToolsMenuPopup(QFrame):
    """Frameless popup listing grouped tool actions."""

    action_triggered = Signal(str)
    model_switch_requested = Signal(str, object)  # (role, ModelRef)

    def __init__(
        self,
        tokens: ThemeTokens,
        *,
        busy: bool = False,
        choices: Sequence[ModelChoice] = (),
        text_target: ModelRef = ModelRef(),
        vision_target: ModelRef = ModelRef(),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setFixedWidth(POPUP_W)
        self._buttons: dict[str, QPushButton] = {}
        self._choices = tuple(choices)
        self._targets = {ROLE_TEXT: text_target, ROLE_VISION: vision_target}
        self._model_menu: QMenu | None = None
        models_bg = mix(tokens.text2, tokens.bg, ACCENT_SOFT_PCT)
        process_bg = mix(tokens.accent, tokens.bg, ACCENT_SOFT_PCT)
        annotate_bg = mix(tokens.warn, tokens.bg, ACCENT_SOFT_PCT)
        preprocess_bg = mix(tokens.ok, tokens.bg, ACCENT_SOFT_PCT)
        self.setStyleSheet(
            f"""
            ToolsMenuPopup {{
                background: {tokens.surface};
                border: 1px solid {tokens.bd};
                border-radius: 10px;
            }}
            QLabel[toolGroup="true"] {{
                font-size: 11px;
                font-weight: 600;
                padding: 5px 10px;
                border-radius: 6px;
            }}
            QPushButton[popupAction="true"] {{
                background: transparent;
                color: {tokens.text};
                border: none;
                border-radius: 7px;
                padding: 7px 10px;
                font-size: 12.5px;
                text-align: left;
            }}
            QPushButton[popupAction="true"]:hover {{
                background: {tokens.surface2};
            }}
            QPushButton[popupAction="true"]:disabled {{
                color: {tokens.text3};
            }}
            """
        )
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(4)
        self._add_group(
            column,
            GROUP_MODELS,
            models_bg,
            tokens,
            (
                (ACTION_MODEL_TEXT, self._target_label(ROLE_TEXT)),
                (ACTION_MODEL_VISION, self._target_label(ROLE_VISION)),
            ),
        )
        self._add_group(
            column,
            GROUP_PROCESS,
            process_bg,
            tokens,
            (
                (ACTION_FLATTEN_ALPHA, LABEL_FLATTEN_ALPHA),
                (ACTION_UNDO_IMAGE_OP, LABEL_UNDO_IMAGE_OP),
            ),
        )
        self._add_group(
            column,
            GROUP_ANNOTATE,
            annotate_bg,
            tokens,
            (
                (ACTION_INFER_LLM, LABEL_INFER_LLM),
                (ACTION_INFER_LOCAL, LABEL_INFER_LOCAL),
                (ACTION_INFER_CHA, LABEL_INFER_CHA),
                (ACTION_INFER_COMPARE, LABEL_INFER_COMPARE),
            ),
        )
        self._add_group(
            column,
            GROUP_PREPROCESS,
            preprocess_bg,
            tokens,
            ((ACTION_FIND_DUPLICATES, LABEL_FIND_DUPLICATES),),
        )
        self.set_busy(busy)

    def set_busy(self, busy: bool) -> None:
        """Disable image-mutating actions while a batch is in flight."""
        for action_id in BUSY_ACTIONS:
            button = self._buttons.get(action_id)
            if button is not None:
                button.setEnabled(not busy)

    def _add_group(
        self,
        column: QVBoxLayout,
        title: str,
        background: str,
        tokens: ThemeTokens,
        actions: tuple[tuple[str, str], ...],
    ) -> None:
        header = QLabel(title, self)
        header.setProperty("toolGroup", True)
        header.setStyleSheet(
            f"background: {background}; color: {tokens.text2};"
        )
        column.addWidget(header)
        for action_id, label in actions:
            button = QPushButton(label, self)
            button.setProperty("popupAction", True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, oid=action_id: self._fire(oid))
            column.addWidget(button)
            self._buttons[action_id] = button

    def _fire(self, action_id: str) -> None:
        role = MODEL_ACTIONS.get(action_id)
        if role is not None:
            self._open_model_menu(action_id, role)
            return
        self.action_triggered.emit(action_id)
        self.close()

    # -- model quick switch ----------------------------------------------------------
    def target_display(self, role: str) -> str:
        """Model id shown for ``role``: 未设置 / 已失效 / the id."""
        ref = self._targets[role]
        if not ref.is_set():
            return LABEL_UNSET
        if find_choice(self._choices, ref) is None:
            return LABEL_STALE
        return ref.model

    def _target_label(self, role: str) -> str:
        template = LABEL_MODEL_TEXT if role == ROLE_TEXT else LABEL_MODEL_VISION
        return template.format(model=self.target_display(role))

    def build_model_menu(self, role: str) -> QMenu:
        """Menu of the pool for ``role`` (current entry checked, 不使用 last)."""
        menu = QMenu(self)
        current = self._targets[role]
        if not self._choices:
            empty = menu.addAction(MENU_POOL_EMPTY)
            empty.setEnabled(False)
        for choice in self._choices:
            label = choice.label + (MENU_VISION_TAG if choice.vision_hint else "")
            action = QAction(label, menu)
            action.setCheckable(True)
            action.setChecked(choice.ref == current)
            action.triggered.connect(
                lambda _checked=False, r=role, ref=choice.ref: self._switch(r, ref)
            )
            menu.addAction(action)
        if current.is_set():
            menu.addSeparator()
            clear = menu.addAction(MENU_CLEAR_TARGET)
            clear.triggered.connect(
                lambda _checked=False, r=role: self._switch(r, ModelRef())
            )
        return menu

    def _open_model_menu(self, action_id: str, role: str) -> None:
        button = self._buttons[action_id]
        menu = self.build_model_menu(role)
        self._model_menu = menu
        anchor = button.mapToGlobal(QPoint(0, button.height() + MENU_OFFSET_Y))
        # Non-blocking: the popup stays up behind the menu and closes on pick.
        menu.popup(anchor)

    def _switch(self, role: str, ref: ModelRef) -> None:
        self.model_switch_requested.emit(role, ref)
        self.close()
