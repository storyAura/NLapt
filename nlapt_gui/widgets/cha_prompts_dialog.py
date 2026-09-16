"""Settings dialog: custom CHA character-card and scene prompt templates."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt_gui.layered_prompts import (
    CARD_PLACEHOLDERS,
    CHARACTER_CARD_PROMPT,
    POSE_SCENE_PROMPT,
    SCENE_PLACEHOLDERS,
    effective_card_template,
    effective_scene_template,
    format_card_stats,
    missing_placeholders,
)
from nlapt_gui.widgets.dialogs import (
    CONFIRM_CANCEL_TEXT,
    CONFIRM_OK_TEXT,
    CenteredDialog,
)

WINDOW_TITLE = "CHA标注 · 提示词"
HINT = (
    "两套提示词分别发给人物卡生成与整批画面段。"
    "与内置默认相同则按默认保存，之后内置更新仍会生效。"
)
LABEL_CARD = "人物卡提示词"
LABEL_SCENE = "画面段提示词"
HINT_CARD_PLACEHOLDERS = "占位符：{opening}（角色名 + 可选作品）、{name}"
HINT_SCENE_PLACEHOLDERS = "占位符：{ROSTER}（角色卡列表）、{NAMES}（角色名）"
BUTTON_RESTORE = "恢复默认"
WARN_MISSING_FMT = "缺少占位符：{tokens}"
DIALOG_MIN_W = 960
DIALOG_MIN_H = 640
EDIT_MIN_H = 280


def _normalize(text: str, default: str) -> str:
    """Keep custom text; collapse a match of the built-in default to empty."""
    return "" if text.strip() == default.strip() else text


class CHAPromptsDialog(CenteredDialog):
    """Two-column editor for the CHA card prompt and the scene prompt."""

    def __init__(
        self,
        card_prompt: str = "",
        scene_prompt: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.setModal(True)
        self.setMinimumSize(DIALOG_MIN_W, DIALOG_MIN_H)

        hint = QLabel(HINT, self)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)

        self.card_edit, self.card_stats, self.card_warn, card_column = self._column(
            LABEL_CARD,
            HINT_CARD_PLACEHOLDERS,
            effective_card_template(card_prompt),
            CARD_PLACEHOLDERS,
            self._restore_card,
        )
        self.scene_edit, self.scene_stats, self.scene_warn, scene_column = self._column(
            LABEL_SCENE,
            HINT_SCENE_PLACEHOLDERS,
            effective_scene_template(scene_prompt),
            SCENE_PLACEHOLDERS,
            self._restore_scene,
        )

        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(12)
        columns.addLayout(card_column, 1)
        columns.addLayout(scene_column, 1)

        self.ok_button = QPushButton(CONFIRM_OK_TEXT, self)
        self.ok_button.setProperty("variant", "accent")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton(CONFIRM_CANCEL_TEXT, self)
        self.cancel_button.setProperty("variant", "outline")
        self.cancel_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.ok_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addWidget(hint)
        root.addLayout(columns, 1)
        root.addLayout(buttons)

    def result_prompts(self) -> tuple[str, str]:
        """Custom texts, or empty strings when they still match the built-ins."""
        return (
            _normalize(self.card_edit.toPlainText(), CHARACTER_CARD_PROMPT),
            _normalize(self.scene_edit.toPlainText(), POSE_SCENE_PROMPT),
        )

    def _column(
        self,
        title: str,
        placeholders: str,
        initial: str,
        required: tuple[str, ...],
        restore,
    ) -> tuple[QPlainTextEdit, QLabel, QLabel, QVBoxLayout]:
        heading = QLabel(title, self)
        heading.setProperty("sectionTitle", True)
        hint = QLabel(placeholders, self)
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        editor = QPlainTextEdit(self)
        editor.setMinimumHeight(EDIT_MIN_H)
        editor.setPlainText(initial)
        stats = QLabel(format_card_stats(initial), self)
        stats.setProperty("muted", True)
        warn = QLabel("", self)
        warn.setProperty("muted", True)
        warn.setWordWrap(True)
        restore_button = QPushButton(BUTTON_RESTORE, self)
        restore_button.setProperty("variant", "outline")
        restore_button.clicked.connect(restore)

        def refresh() -> None:
            text = editor.toPlainText()
            stats.setText(format_card_stats(text))
            missing = missing_placeholders(text, required)
            warn.setText(WARN_MISSING_FMT.format(tokens=" ".join(missing)) if missing else "")
            warn.setVisible(bool(missing))

        editor.textChanged.connect(refresh)
        refresh()

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        column.addWidget(heading)
        column.addWidget(hint)
        column.addWidget(editor, 1)
        column.addWidget(stats)
        column.addWidget(warn)
        column.addWidget(restore_button)
        return editor, stats, warn, column

    def _restore_card(self) -> None:
        self.card_edit.setPlainText(CHARACTER_CARD_PROMPT)

    def _restore_scene(self) -> None:
        self.scene_edit.setPlainText(POSE_SCENE_PROMPT)
