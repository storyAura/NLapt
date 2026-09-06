"""Tests for the Hy-MT2 download management dialog."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.local.mt_catalog import ALL_MT_MODELS, TIER_FAST, find_mt_model, mt_model_path

from nlapt_gui.local_bridge import LocalBridge
from nlapt_gui.mt_bridge import reset_mt_singletons_for_tests
from nlapt_gui.widgets.mt_models_dialog import (
    BUTTON_DOWNLOAD,
    STATUS_MISSING,
    STATUS_READY,
    MTModelsDialog,
)


@pytest.fixture(autouse=True)
def _isolate_mt(tmp_path: Path) -> None:
    reset_mt_singletons_for_tests()
    LocalBridge().update_settings(models_dir=str(tmp_path))


class TestMTModelsDialog:
    def test_lists_three_tiers(self, qtbot, tmp_path: Path) -> None:
        dialog = MTModelsDialog(models_dir=tmp_path)
        qtbot.addWidget(dialog)
        dialog.show()
        assert len(dialog._status_labels) == len(ALL_MT_MODELS)
        for model in ALL_MT_MODELS:
            assert dialog._status_labels[model.tier].text() == STATUS_MISSING
            assert dialog._download_buttons[model.tier].isEnabled()
            assert not dialog._delete_buttons[model.tier].isEnabled()

    def test_ready_tier_enables_delete(
        self, qtbot, tmp_path: Path, monkeypatch
    ) -> None:
        dest = mt_model_path(tmp_path, find_mt_model(TIER_FAST))
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"gguf")
        monkeypatch.setattr(
            "nlapt_gui.widgets.mt_models_dialog.is_tier_downloaded",
            lambda tier, *, models_dir=None: tier == TIER_FAST,
        )
        dialog = MTModelsDialog(models_dir=tmp_path)
        qtbot.addWidget(dialog)
        dialog.show()
        assert dialog._status_labels[TIER_FAST].text() == STATUS_READY
        assert not dialog._download_buttons[TIER_FAST].isEnabled()
        assert dialog._delete_buttons[TIER_FAST].isEnabled()

    def test_download_calls_start(self, qtbot, tmp_path: Path, monkeypatch) -> None:
        seen: list[str] = []

        def fake_start(tier: str, pool=None, *, models_dir=None) -> bool:
            seen.append(tier)
            return True

        monkeypatch.setattr(
            "nlapt_gui.widgets.mt_models_dialog.start_mt_download", fake_start
        )
        dialog = MTModelsDialog(models_dir=tmp_path)
        qtbot.addWidget(dialog)
        dialog.show()
        dialog._download_buttons[TIER_FAST].click()
        assert seen == [TIER_FAST]
        assert dialog._download_buttons[TIER_FAST].text() == BUTTON_DOWNLOAD
