"""Tests for the always-visible Hy-MT2 tier list."""

from __future__ import annotations

from nlapt.local.mt_catalog import TIER_FAST
from nlapt_gui.local_bridge import LocalBridge
from nlapt_gui.widgets.mt_tier_picker import (
    HINT_TIER,
    LABEL_TITLE,
    STATUS_MISSING,
    MtTierPicker,
)


def test_lists_three_tiers_with_download_mark(qtbot, tmp_path) -> None:
    LocalBridge().update_settings(models_dir=str(tmp_path))
    picker = MtTierPicker()
    qtbot.addWidget(picker)
    assert picker.title.text() == LABEL_TITLE
    assert picker.hint.text() == HINT_TIER
    assert picker.list.count() == 3
    assert picker.findData("fast") == 0
    assert picker.findData("balanced") == 1
    assert picker.findData("quality") == 2
    assert STATUS_MISSING in picker.list.item(0).text()


def test_set_current_index_updates_data(qtbot, tmp_path) -> None:
    LocalBridge().update_settings(models_dir=str(tmp_path))
    picker = MtTierPicker()
    qtbot.addWidget(picker)
    picker.setCurrentIndex(picker.findData(TIER_FAST))
    assert picker.currentData() == TIER_FAST
