"""Tests for nlapt_gui.compare_config (多对比推标 model list persistence)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlapt.core.config import ModelRef
from nlapt.core.errors import StorageError

from nlapt_gui.compare_config import (
    CompareSettings,
    compare_settings_path,
    load_compare_settings,
    save_compare_settings,
)


def test_settings_normalise_dedupe_and_drop_unset() -> None:
    settings = CompareSettings(
        models=(ModelRef("a", "m"), {"profile": "b", "model": "n"}, ModelRef("a", "m"), "bare", ModelRef())
    )
    # Bare strings have no provider and cannot take part -> dropped with the unset ref.
    assert settings.models == (ModelRef("a", "m"), ModelRef("b", "n"))


def test_round_trip_and_json_shape() -> None:
    settings = CompareSettings(models=(ModelRef("alpha", "gpt-4o"), ModelRef("beta", "llava")))
    save_compare_settings(settings)
    raw = json.loads(compare_settings_path().read_text(encoding="utf-8"))
    assert raw == {
        "models": [
            {"model": "gpt-4o", "profile": "alpha"},
            {"model": "llava", "profile": "beta"},
        ]
    }
    assert load_compare_settings() == settings


def test_missing_corrupt_and_odd_shapes_fall_back(tmp_path: Path) -> None:
    assert load_compare_settings() == CompareSettings()
    broken = tmp_path / "compare_infer.json"
    broken.write_text("{nope", encoding="utf-8")
    assert load_compare_settings(broken) == CompareSettings()
    broken.write_text("[]", encoding="utf-8")
    assert load_compare_settings(broken) == CompareSettings()
    broken.write_text(json.dumps({"models": "x"}), encoding="utf-8")
    assert load_compare_settings(broken) == CompareSettings()


def test_save_rejects_wrong_type() -> None:
    with pytest.raises(StorageError):
        save_compare_settings("nope")  # type: ignore[arg-type]
