"""推标 context-menu actions for :class:`FilePanel` (labels + scope wiring)."""

from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMenu, QWidget

from nlapt_gui.controller import FOLDER_ROOT_LABEL, AppController
from nlapt_gui.prompt_store import ENGINE_LLM, ENGINE_LOCAL

MENU_INFER_FOLDER_LLM = "用 LLM 推理此文件夹({n} 张)"
MENU_INFER_FOLDER_LOCAL = "用本地模型推理此文件夹({n} 张)"
MENU_INFER_SELECTED_LLM = "用 LLM 推理已选({n} 张)"
MENU_INFER_SELECTED_LOCAL = "用本地模型推理已选({n} 张)"
MENU_INFER_ALL_LLM = "用 LLM 推理全部({n} 张)"
MENU_INFER_ALL_LOCAL = "用本地模型推理全部({n} 张)"
MENU_INFER_IMAGE_LLM = "用 LLM 推理这张图片"
MENU_INFER_IMAGE_LOCAL = "用本地模型推理这张图片"
MENU_INFER_FOLDER_UNLABELED_LLM = "用 LLM 推理此文件夹未标注({n} 张)"
MENU_INFER_FOLDER_UNLABELED_LOCAL = "用本地模型推理此文件夹未标注({n} 张)"
MENU_INFER_ALL_UNLABELED_LLM = "用 LLM 推理全部未标注({n} 张)"
MENU_INFER_ALL_UNLABELED_LOCAL = "用本地模型推理全部未标注({n} 张)"
MENU_LAYERED_FOLDER = "分层推标此文件夹({n} 张)"
MENU_LAYERED_SELECTED = "分层推标已选({n} 张)"
MENU_LAYERED_ALL = "分层推标全部({n} 张)"
MENU_LAYERED_IMAGE = "分层推标这张图片"
MENU_LAYERED_FOLDER_UNLABELED = "分层推标此文件夹未标注({n} 张)"
MENU_LAYERED_ALL_UNLABELED = "分层推标全部未标注({n} 张)"
MENU_CANCEL_INFER = "取消当前推标"
MENU_COPY_CAPTION = "复制标注"
CONFIRM_INFER_TITLE = "批量推标"
CONFIRM_INFER_TEXT = (
    "将用{word}为 {n} 张图片重新生成标注,覆盖现有内容。\n"
    "执行前会自动备份,完成后可在 历史记录 面板整批回滚。"
)
ENGINE_CONFIRM_WORDS = {ENGINE_LLM: " LLM ", ENGINE_LOCAL: "本地模型"}


class InferMenuHost(Protocol):
    """FilePanel surface used by :func:`build_infer_actions`."""

    _controller: AppController

    def _request_infer(self, keys: tuple[str, ...], engine: str) -> None: ...
    def _request_layered(self, keys: tuple[str, ...]) -> None: ...
    def infer_menu_actions(
        self, folder: str | None, image: str | None = None
    ) -> list[tuple[str, object]]: ...


def build_infer_actions(
    panel: InferMenuHost,
    folder: str | None,
    image: str | None = None,
) -> list[tuple[str, object]]:
    """(label, callable) entries for the 推标 menu.

    Scoped to the right-click target: an image cell (``image``), a folder
    header (``folder``), or the ALL row / 根目录 group (neither).
    """
    controller = panel._controller
    if controller.batch_running():
        return [(MENU_CANCEL_INFER, controller.cancel_batch)]

    def trio(
        llm_label: str,
        local_label: str,
        layered_label: str,
        keys: tuple[str, ...],
    ) -> list[tuple[str, object]]:
        n = len(keys)
        return [
            (llm_label.format(n=n), lambda: panel._request_infer(keys, ENGINE_LLM)),
            (local_label.format(n=n), lambda: panel._request_infer(keys, ENGINE_LOCAL)),
            (layered_label.format(n=n), lambda: panel._request_layered(keys)),
        ]

    actions: list[tuple[str, object]] = []
    if image is not None:
        selected = controller.selected_keys()
        if image in selected and len(selected) > 1:
            actions += trio(
                MENU_INFER_SELECTED_LLM,
                MENU_INFER_SELECTED_LOCAL,
                MENU_LAYERED_SELECTED,
                selected,
            )
            actions += trio(
                MENU_INFER_ALL_LLM, MENU_INFER_ALL_LOCAL, MENU_LAYERED_ALL, controller.keys()
            )
        else:
            actions += trio(
                MENU_INFER_IMAGE_LLM, MENU_INFER_IMAGE_LOCAL, MENU_LAYERED_IMAGE, (image,)
            )
        return actions

    if folder is not None and folder != FOLDER_ROOT_LABEL:
        keys = controller.folder_keys(folder)
        if keys:
            actions += trio(
                MENU_INFER_FOLDER_LLM,
                MENU_INFER_FOLDER_LOCAL,
                MENU_LAYERED_FOLDER,
                keys,
            )
            unlabeled = controller.unlabeled_keys(keys)
            if unlabeled:
                actions += trio(
                    MENU_INFER_FOLDER_UNLABELED_LLM,
                    MENU_INFER_FOLDER_UNLABELED_LOCAL,
                    MENU_LAYERED_FOLDER_UNLABELED,
                    unlabeled,
                )
        every = controller.keys()
        if every:
            actions += trio(
                MENU_INFER_ALL_LLM, MENU_INFER_ALL_LOCAL, MENU_LAYERED_ALL, every
            )
        return actions

    every = controller.keys()
    if every:
        actions += trio(
            MENU_INFER_ALL_LLM, MENU_INFER_ALL_LOCAL, MENU_LAYERED_ALL, every
        )
        unlabeled = controller.unlabeled_keys(every)
        if unlabeled:
            actions += trio(
                MENU_INFER_ALL_UNLABELED_LLM,
                MENU_INFER_ALL_UNLABELED_LOCAL,
                MENU_LAYERED_ALL_UNLABELED,
                unlabeled,
            )
    return actions


def popup_infer_menu(
    panel: InferMenuHost,
    folder: str | None,
    widget: QWidget,
    pos: QPoint,
    image: str | None = None,
) -> None:
    """Popup the 推标 menu for an image cell / folder header / the ALL row."""
    actions = list(panel.infer_menu_actions(folder, image))
    if image is not None:
        actions.insert(
            0, (MENU_COPY_CAPTION, lambda k=image: panel._controller.copy_caption(k))
        )
    if not actions:
        return
    menu = QMenu(panel)  # type: ignore[arg-type]
    menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    previous_scope: object = None
    for index, (label, handler) in enumerate(actions):
        scope = (
            label.split("(", 1)[0]
            .replace("LLM", "")
            .replace("本地模型", "")
            .replace(" ", "")
        )
        if index and scope != previous_scope:
            menu.addSeparator()
        previous_scope = scope
        menu.addAction(label).triggered.connect(
            lambda _checked=False, run=handler: run()
        )
    menu.popup(widget.mapToGlobal(pos))
