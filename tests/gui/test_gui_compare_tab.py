"""Tests for 设置 ▸ 多对比推标 (CompareTab)."""

from __future__ import annotations

from nlapt.core.config import ModelRef

from nlapt_gui.compare_config import MIN_COMPARE_MODELS, CompareSettings
from nlapt_gui.model_targets import ModelChoice
from nlapt_gui.widgets.compare_tab import (
    LABEL_COUNT_FMT,
    LABEL_COUNT_TOO_FEW,
    CompareTab,
)


def _choice(profile: str, model: str) -> ModelChoice:
    return ModelChoice(ModelRef(profile, model), f"{profile} · {model}", False)


GROUPS = (
    ("alpha", (_choice("alpha", "a1"), _choice("alpha", "a2"))),
    ("beta", (_choice("beta", "b1"),)),
)


def test_prefill_current_settings_and_count_label(qtbot) -> None:
    tab = CompareTab(pool_provider=lambda: GROUPS)
    qtbot.addWidget(tab)
    assert tab.count_label.text() == LABEL_COUNT_TOO_FEW.format(n=0, min=MIN_COMPARE_MODELS)
    tab.prefill(CompareSettings(models=(ModelRef("alpha", "a2"), ModelRef("beta", "b1"), ModelRef("gone", "x"))))
    assert tab.current_settings().models == (ModelRef("alpha", "a2"), ModelRef("beta", "b1"))
    assert tab.count_label.text() == LABEL_COUNT_FMT.format(n=2)


def test_refresh_pool_keeps_checks(qtbot) -> None:
    holder = {"groups": GROUPS}
    tab = CompareTab(pool_provider=lambda: holder["groups"])
    qtbot.addWidget(tab)
    tab.prefill(CompareSettings(models=(ModelRef("alpha", "a1"), ModelRef("beta", "b1"))))
    holder["groups"] = (GROUPS[1],)
    tab.refresh_pool()
    assert tab.current_settings().models == (ModelRef("beta", "b1"),)
    holder["groups"] = GROUPS
    tab.refresh_pool()
    # The alpha pick comes back once its provider is enabled again.
    assert tab.current_settings().models == (ModelRef("alpha", "a1"), ModelRef("beta", "b1"))
