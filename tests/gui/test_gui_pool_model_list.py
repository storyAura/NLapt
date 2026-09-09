"""Tests for the provider-grouped checkable pool list widget."""

from __future__ import annotations

from PySide6.QtCore import Qt

from nlapt.core.config import ModelRef

from nlapt_gui.model_targets import ModelChoice
from nlapt_gui.widgets.pool_model_list import (
    EMPTY_HINT,
    GROUP_HEADER_FMT,
    VISION_TAG,
    PoolModelList,
)


def _choice(profile: str, model: str, vision: bool = False) -> ModelChoice:
    return ModelChoice(ModelRef(profile, model), f"{profile} · {model}", vision)


GROUPS = (
    ("alpha", (_choice("alpha", "a1"), _choice("alpha", "gpt-4o", True))),
    ("beta", (_choice("beta", "llava", True),)),
)


def _texts(widget: PoolModelList) -> list[str]:
    return [widget.item(i).text() for i in range(widget.count())]


def test_groups_render_headers_and_checks(qtbot) -> None:
    widget = PoolModelList()
    qtbot.addWidget(widget)
    widget.set_groups(GROUPS, (ModelRef("beta", "llava"),))
    assert _texts(widget) == [
        GROUP_HEADER_FMT.format(name="alpha"),
        "alpha · a1",
        f"alpha · gpt-4o{VISION_TAG}",
        GROUP_HEADER_FMT.format(name="beta"),
        f"beta · llava{VISION_TAG}",
    ]
    assert widget.item(0).flags() == Qt.ItemFlag.NoItemFlags
    assert widget.checked_refs() == (ModelRef("beta", "llava"),)


def test_toggle_emits_changed_and_set_checked(qtbot) -> None:
    widget = PoolModelList()
    qtbot.addWidget(widget)
    widget.set_groups(GROUPS, ())
    with qtbot.waitSignal(widget.changed, timeout=1000):
        widget.item(1).setCheckState(Qt.CheckState.Checked)
    assert widget.checked_refs() == (ModelRef("alpha", "a1"),)
    widget.set_checked((ModelRef("alpha", "gpt-4o"), ModelRef("ghost", "x")))
    assert widget.checked_refs() == (ModelRef("alpha", "gpt-4o"),)


def test_empty_pool_shows_hint(qtbot) -> None:
    widget = PoolModelList()
    qtbot.addWidget(widget)
    widget.set_groups((), ())
    assert _texts(widget) == [EMPTY_HINT]
    assert widget.checked_refs() == ()
