"""CHA wizard coordinator: run CL Tagger on a slot's reference image."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QPoint
from PySide6.QtWidgets import QMenu
from shiboken6 import isValid

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger

from nlapt_gui.controller import TOAST_ERR, TOAST_WARN, AppController
from nlapt_gui.tagger_bridge import (
    REL_CHILD,
    REL_DETECTED,
    REL_PARENT,
    REL_SIBLING,
    CharacterCandidate,
    IdentifyResult,
    TaggerBridge,
    is_tagger_ready,
)
from nlapt_gui.widgets.layered_infer_slots import (
    BUTTON_IDENTIFY,
    BUTTON_IDENTIFYING,
    SlotWidgets,
)

_LOGGER = get_logger(__name__)

TOAST_NEED_REF = "请先为此角色卡选择参考图"
TOAST_NO_CHARACTER = "未识别出已知角色"
TOAST_IDENTIFY_FAIL = "识别失败: {message}"
REL_LABELS: dict[str, str] = {
    REL_DETECTED: "",
    REL_PARENT: "父标签",
    REL_CHILD: "子标签",
    REL_SIBLING: "同级",
}
MENU_DETECTED = "{name} · {pct}% · {series}"
MENU_RELATED = "{name} · {relation} · {series}"
MENU_DETECTED_BARE = "{name} · {pct}%"
MENU_RELATED_BARE = "{name} · {relation}"


class IdentifyCoordinator(QObject):
    """Owns one TaggerBridge and fills SlotRoleForm from a candidate menu."""

    def __init__(
        self,
        controller: AppController,
        tagger_bridge: TaggerBridge,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._bridge = tagger_bridge
        self._slots: dict[int, SlotWidgets] = {}
        self._pending: dict[str, SlotWidgets] = {}
        self._bridge.identify_ready.connect(self._on_ready)
        self._bridge.identify_failed.connect(self._on_failed)

    def bind_slot(self, slot: SlotWidgets) -> None:
        """Wire the slot's 识别角色 button and set its enabled state."""
        self._slots[slot.uid] = slot
        slot.role.set_identify_available(is_tagger_ready())
        slot.role.identify_requested.connect(lambda uid=slot.uid: self.start(uid))

    def start(self, uid: int) -> None:
        """Identify the slot's current reference image."""
        slot = self._slot_by_uid(uid)
        if slot is None:
            return
        if not slot.ref_key:
            self._controller.toast_requested.emit(TOAST_NEED_REF, TOAST_WARN)
            return
        try:
            path = self._controller.image_path(slot.ref_key)
        except NLaptError as exc:
            self._controller.toast_requested.emit(str(exc), TOAST_ERR)
            return
        request_id = f"identify-{uid}"
        self._pending[request_id] = slot
        slot.role.identify_button.setText(BUTTON_IDENTIFYING)
        slot.role.identify_button.setEnabled(False)
        if not self._bridge.request_identify(request_id, Path(path)):
            self._finish_button(slot)
            self._pending.pop(request_id, None)

    def _on_ready(self, request_id: str, payload: object) -> None:
        if not isValid(self):
            return
        slot = self._pending.pop(request_id, None)
        if slot is None:
            return
        self._finish_button(slot)
        result = payload if isinstance(payload, IdentifyResult) else IdentifyResult()
        if not result.candidates:
            self._controller.toast_requested.emit(TOAST_NO_CHARACTER, TOAST_WARN)
            return
        self._popup_menu(slot, result.candidates)

    def _on_failed(self, request_id: str, message: str) -> None:
        if not isValid(self):
            return
        slot = self._pending.pop(request_id, None)
        if slot is not None:
            self._finish_button(slot)
        self._controller.toast_requested.emit(
            TOAST_IDENTIFY_FAIL.format(message=message), TOAST_ERR
        )

    def _popup_menu(
        self, slot: SlotWidgets, candidates: tuple[CharacterCandidate, ...]
    ) -> None:
        menu = QMenu(slot.role)
        for candidate in candidates:
            action = menu.addAction(_candidate_label(candidate))
            action.setData(candidate)
        button = slot.role.identify_button
        chosen = menu.exec(button.mapToGlobal(QPoint(0, button.height())))
        if chosen is None:
            return
        data = chosen.data()
        if isinstance(data, CharacterCandidate):
            self.apply_candidate(slot, data)

    @staticmethod
    def apply_candidate(slot: SlotWidgets, candidate: CharacterCandidate) -> None:
        """Fill 角色名 / 作品名 from a menu pick (works名 empty → leave series)."""
        slot.role.name_edit.setText(candidate.display_name)
        if candidate.series_display:
            slot.role.series_edit.setText(candidate.series_display)

    def _finish_button(self, slot: SlotWidgets) -> None:
        slot.role.identify_button.setText(BUTTON_IDENTIFY)
        slot.role.set_identify_available(is_tagger_ready())

    def _slot_by_uid(self, uid: int) -> SlotWidgets | None:
        return self._slots.get(uid)


def _candidate_label(candidate: CharacterCandidate) -> str:
    series = candidate.series_display
    if candidate.relation == REL_DETECTED:
        pct = round(candidate.prob * 100)
        if series:
            return MENU_DETECTED.format(name=candidate.display_name, pct=pct, series=series)
        return MENU_DETECTED_BARE.format(name=candidate.display_name, pct=pct)
    relation = REL_LABELS.get(candidate.relation, candidate.relation)
    if series:
        return MENU_RELATED.format(
            name=candidate.display_name, relation=relation, series=series
        )
    return MENU_RELATED_BARE.format(name=candidate.display_name, relation=relation)
