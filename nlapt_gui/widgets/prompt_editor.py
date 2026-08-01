"""设置 ▸ 提示词 - custom system/user prompts for image inference (重译).

Layout follows the 自动标注器设置 reference: a template dropdown with
新建自定义 / 保存模板 / 删除 / 导出当前 / 导出全部 actions, the system-prompt
editor (the built-in 默认 template is intentionally empty — the user pastes
their own later) and the 用户提示词 editor below it.

The tab edits an in-memory :class:`VisionPrompts`; the parent dialog calls
:meth:`current_prompts` on 保存 and persists it together with the rest of the
settings. 保存模板 / 新建自定义 / 删除 additionally persist immediately so a
template operation is never lost by a later 取消.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger

from nlapt_gui.prompt_store import (
    DEFAULT_PROMPT_NAME,
    VisionPrompts,
    save_vision_prompts,
)

_LOGGER = get_logger(__name__)

# -- exact UI strings --------------------------------------------------------------
NOTE_SHARED = (
    "提示词默认统一管线:在线 LLM 与本地模型的图片推理都使用这里的"
    "系统 / 用户提示词;取消下方勾选后可为本地推理单独设置一套。"
)
LABEL_TEMPLATE = "推理提示词模板"
LABEL_SYSTEM = "系统提示词(推理图片前发送,默认留空)"
LABEL_USER = "用户提示词(留空使用内置指令)"
LABEL_LOCAL_UNIFIED = "本地推理共用上方提示词(统一管线)"
LABEL_LOCAL_SYSTEM = "本地推理 · 系统提示词"
LABEL_LOCAL_USER = "本地推理 · 用户提示词(留空使用内置指令)"
LOCAL_EDIT_MIN_H = 60
BUTTON_NEW = "新建自定义"
BUTTON_SAVE_TEMPLATE = "保存模板"
BUTTON_DELETE = "删除"
BUTTON_EXPORT_CURRENT = "导出当前"
BUTTON_EXPORT_ALL = "导出全部自定义"
NEW_TEMPLATE_TITLE = "新建自定义模板"
NEW_TEMPLATE_PROMPT = "模板名称:"
EXPORT_CURRENT_CAPTION = "导出当前系统提示词"
EXPORT_ALL_CAPTION = "导出全部自定义模板"
EXPORT_TXT_FILTER = "文本文件 (*.txt)"
EXPORT_JSON_FILTER = "JSON 文件 (*.json)"
HINT_DEFAULT_READONLY = "「默认」模板为内置空模板 — 新建自定义模板后可编辑并保存。"
TOAST_TEMPLATE_SAVED = "已保存模板「{name}」"
TOAST_TEMPLATE_DELETED = "已删除模板「{name}」"
TOAST_EXPORTED = "已导出到 {path}"
TOAST_EXPORT_FAILED = "导出失败: {message}"
TOAST_NAME_EXISTS = "模板「{name}」已存在"
TOAST_NOTHING_TO_EXPORT = "没有自定义模板可导出"

_KIND_OK = "ok"
_KIND_WARN = "warn"
_KIND_ERR = "err"

SYSTEM_EDIT_MIN_H = 150
USER_EDIT_MIN_H = 70


class PromptsTab(QWidget):
    """The 提示词 settings tab editing a VisionPrompts value."""

    toast_requested = Signal(str, str)  # text, kind

    def __init__(
        self, prompts: VisionPrompts, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._prompts = prompts
        self._loading = False

        column = QVBoxLayout(self)
        column.setSpacing(8)

        # One shared prompt set for every LLM (在线 + 本地) — spec issue 3.
        self.shared_note = self._muted_label(NOTE_SHARED)
        self.shared_note.setWordWrap(True)
        column.addWidget(self.shared_note)

        column.addWidget(self._muted_label(LABEL_TEMPLATE))
        top = QHBoxLayout()
        top.setSpacing(6)
        self.template_combo = QComboBox(self)
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        top.addWidget(self.template_combo, 1)
        self.new_button = self._button(BUTTON_NEW, self._on_new)
        top.addWidget(self.new_button)
        self.save_template_button = self._button(BUTTON_SAVE_TEMPLATE, self._on_save_template)
        top.addWidget(self.save_template_button)
        self.delete_button = self._button(BUTTON_DELETE, self._on_delete)
        top.addWidget(self.delete_button)
        column.addLayout(top)

        column.addWidget(self._muted_label(LABEL_SYSTEM))
        self.system_edit = QPlainTextEdit(self)
        self.system_edit.setMinimumHeight(SYSTEM_EDIT_MIN_H)
        column.addWidget(self.system_edit, 1)
        self.hint = QLabel(HINT_DEFAULT_READONLY, self)
        self.hint.setProperty("muted", True)
        self.hint.setWordWrap(True)
        column.addWidget(self.hint)

        column.addWidget(self._muted_label(LABEL_USER))
        self.user_edit = QPlainTextEdit(self)
        self.user_edit.setMinimumHeight(USER_EDIT_MIN_H)
        column.addWidget(self.user_edit)

        # 本地推理: unified by default, optional separate prompt pair.
        self.local_unified_box = QCheckBox(LABEL_LOCAL_UNIFIED, self)
        self.local_unified_box.toggled.connect(self._sync_local_visibility)
        column.addWidget(self.local_unified_box)
        self.local_system_label = self._muted_label(LABEL_LOCAL_SYSTEM)
        column.addWidget(self.local_system_label)
        self.local_system_edit = QPlainTextEdit(self)
        self.local_system_edit.setMinimumHeight(LOCAL_EDIT_MIN_H)
        column.addWidget(self.local_system_edit)
        self.local_user_label = self._muted_label(LABEL_LOCAL_USER)
        column.addWidget(self.local_user_label)
        self.local_user_edit = QPlainTextEdit(self)
        self.local_user_edit.setMinimumHeight(LOCAL_EDIT_MIN_H)
        column.addWidget(self.local_user_edit)

        exports = QHBoxLayout()
        exports.setSpacing(6)
        self.export_current_button = self._button(BUTTON_EXPORT_CURRENT, self._on_export_current)
        exports.addWidget(self.export_current_button)
        self.export_all_button = self._button(BUTTON_EXPORT_ALL, self._on_export_all)
        exports.addWidget(self.export_all_button)
        exports.addStretch(1)
        column.addLayout(exports)

        self._reload_names(select=self._prompts.active)
        self.user_edit.setPlainText(self._prompts.user_prompt)
        self.local_unified_box.setChecked(self._prompts.local_unified)
        self.local_system_edit.setPlainText(self._prompts.local_system)
        self.local_user_edit.setPlainText(self._prompts.local_user_prompt)
        self._sync_local_visibility()

    def _sync_local_visibility(self, *_args: object) -> None:
        """Show the local prompt editors only when 统一管线 is off."""
        separate = not self.local_unified_box.isChecked()
        for widget in (
            self.local_system_label,
            self.local_system_edit,
            self.local_user_label,
            self.local_user_edit,
        ):
            widget.setVisible(separate)

    # -- helpers ------------------------------------------------------------------------
    def _muted_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setProperty("muted", True)
        return label

    def _button(self, text: str, handler) -> QPushButton:
        button = QPushButton(text, self)
        button.setProperty("variant", "outline")
        button.clicked.connect(handler)
        return button

    def _toast(self, text: str, kind: str) -> None:
        self.toast_requested.emit(text, kind)

    # -- state --------------------------------------------------------------------------
    def current_prompts(self) -> VisionPrompts:
        """The VisionPrompts value reflecting the tab's current UI state.

        Unsaved edits of the active CUSTOM template are captured; edits while
        默认 is active are ignored (默认 stays the built-in empty template).
        """
        active = self.template_combo.currentText() or DEFAULT_PROMPT_NAME
        prompts = dict(self._prompts.prompts)
        if active != DEFAULT_PROMPT_NAME and active in prompts:
            prompts[active] = self.system_edit.toPlainText()
        return VisionPrompts(
            active=active if active == DEFAULT_PROMPT_NAME or active in prompts else DEFAULT_PROMPT_NAME,
            prompts=prompts,
            user_prompt=self.user_edit.toPlainText(),
            local_unified=self.local_unified_box.isChecked(),
            local_system=self.local_system_edit.toPlainText(),
            local_user_prompt=self.local_user_edit.toPlainText(),
        )

    def _persist(self, prompts: VisionPrompts) -> bool:
        try:
            save_vision_prompts(prompts)
        except NLaptError as exc:
            self._toast(str(exc), _KIND_ERR)
            return False
        self._prompts = prompts
        return True

    def _reload_names(self, *, select: str) -> None:
        self._loading = True
        self.template_combo.clear()
        for name in self._prompts.names():
            self.template_combo.addItem(name)
        index = self.template_combo.findText(select)
        self.template_combo.setCurrentIndex(index if index >= 0 else 0)
        self._loading = False
        self._sync_editor()

    def _sync_editor(self) -> None:
        name = self.template_combo.currentText()
        is_default = name == DEFAULT_PROMPT_NAME
        text = "" if is_default else self._prompts.prompts.get(name, "")
        self.system_edit.setPlainText(text)
        self.system_edit.setReadOnly(is_default)
        self.hint.setVisible(is_default)
        self.save_template_button.setEnabled(not is_default)
        self.delete_button.setEnabled(not is_default)

    def _on_template_changed(self, _index: int) -> None:
        if not self._loading:
            self._sync_editor()

    # -- actions ------------------------------------------------------------------------
    def _on_new(self) -> None:
        name, ok = QInputDialog.getText(
            self.window(), NEW_TEMPLATE_TITLE, NEW_TEMPLATE_PROMPT, QLineEdit.EchoMode.Normal
        )
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in self._prompts.names():
            self._toast(TOAST_NAME_EXISTS.format(name=name), _KIND_WARN)
            return
        prompts = dict(self._prompts.prompts)
        # Seed the new template with whatever is currently in the editor so
        # 新建 can fork the visible text (默认 forks as empty).
        prompts[name] = self.system_edit.toPlainText()
        if self._persist(self._prompts.with_changes(active=name, prompts=prompts)):
            self._reload_names(select=name)
            self._toast(TOAST_TEMPLATE_SAVED.format(name=name), _KIND_OK)

    def _on_save_template(self) -> None:
        name = self.template_combo.currentText()
        if name == DEFAULT_PROMPT_NAME:
            return
        prompts = dict(self._prompts.prompts)
        prompts[name] = self.system_edit.toPlainText()
        if self._persist(
            self._prompts.with_changes(
                active=name, prompts=prompts, user_prompt=self.user_edit.toPlainText()
            )
        ):
            self._toast(TOAST_TEMPLATE_SAVED.format(name=name), _KIND_OK)

    def _on_delete(self) -> None:
        name = self.template_combo.currentText()
        if name == DEFAULT_PROMPT_NAME:
            return
        prompts = dict(self._prompts.prompts)
        prompts.pop(name, None)
        if self._persist(
            self._prompts.with_changes(active=DEFAULT_PROMPT_NAME, prompts=prompts)
        ):
            self._reload_names(select=DEFAULT_PROMPT_NAME)
            self._toast(TOAST_TEMPLATE_DELETED.format(name=name), _KIND_OK)

    def _on_export_current(self) -> None:
        name = self.template_combo.currentText()
        default_name = f"{name}.txt"
        path_str, _filter = QFileDialog.getSaveFileName(
            self.window(), EXPORT_CURRENT_CAPTION, default_name, EXPORT_TXT_FILTER
        )
        if not path_str:
            return
        self._export_write(Path(path_str), self.system_edit.toPlainText())

    def _on_export_all(self) -> None:
        snapshot = self.current_prompts()
        if not snapshot.prompts:
            self._toast(TOAST_NOTHING_TO_EXPORT, _KIND_WARN)
            return
        path_str, _filter = QFileDialog.getSaveFileName(
            self.window(), EXPORT_ALL_CAPTION, "vision_prompts.json", EXPORT_JSON_FILTER
        )
        if not path_str:
            return
        payload = json.dumps(dict(snapshot.prompts), ensure_ascii=False, indent=2)
        self._export_write(Path(path_str), payload)

    def _export_write(self, path: Path, content: str) -> None:
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            _LOGGER.exception("prompt export failed")
            self._toast(TOAST_EXPORT_FAILED.format(message=exc), _KIND_ERR)
            return
        self._toast(TOAST_EXPORTED.format(path=path), _KIND_OK)
