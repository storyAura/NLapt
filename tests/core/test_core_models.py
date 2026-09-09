"""Tests for nlapt.core.models and the nlapt package root exports."""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

import nlapt
from nlapt.core.models import IMAGE_EXTENSIONS, DatasetScanResult, ImageFile


def _image(key: str = "sub/img001.jpg") -> ImageFile:
    return ImageFile(
        key=key,
        image_path=Path("/data") / key,
        txt_path=Path("/data/sub/img001.txt"),
        txt_exists=True,
        mtime=123.0,
    )


def test_image_extensions_exact_set() -> None:
    assert IMAGE_EXTENSIONS == frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    assert isinstance(IMAGE_EXTENSIONS, frozenset)


def test_image_file_is_frozen() -> None:
    image = _image()
    with pytest.raises(dataclasses.FrozenInstanceError):
        image.key = "other"  # type: ignore[misc]


def test_image_file_equality_by_value() -> None:
    assert _image() == _image()
    assert _image() != _image(key="other.png")


def test_dataset_scan_result_holds_tuples() -> None:
    result = DatasetScanResult(
        root=Path("/data"), images=(_image(),), orphan_txts=(Path("/data/orphan.txt"),)
    )
    assert isinstance(result.images, tuple)
    assert isinstance(result.orphan_txts, tuple)
    assert result.images[0].txt_exists is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.root = Path("/other")  # type: ignore[misc]


# ---- package root (lazy NLaptApp export, PEP 562) ---------------------------


def test_package_version() -> None:
    assert nlapt.__version__ == "0.2.1"


def test_package_root_lists_lazy_export_without_importing_it() -> None:
    # importing nlapt must not import nlapt.app eagerly
    assert "NLaptApp" in dir(nlapt)


def test_package_root_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        nlapt.does_not_exist  # noqa: B018


def test_package_root_resolves_nlaptapp_lazily() -> None:
    fake = types.ModuleType("nlapt.app")

    class NLaptApp:  # stand-in until the integrator lands nlapt/app.py
        pass

    fake.NLaptApp = NLaptApp
    sys.modules["nlapt.app"] = fake
    try:
        assert nlapt.NLaptApp is NLaptApp
    finally:
        del sys.modules["nlapt.app"]
        nlapt.__dict__.pop("NLaptApp", None)  # drop the cached lazy attribute
