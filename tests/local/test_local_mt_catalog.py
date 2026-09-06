"""Tests for the Hy-MT2 local-translation catalog snapshot."""

from __future__ import annotations

import re

import pytest

from nlapt.core.errors import ValidationError
from nlapt.llm.translate import Direction
from nlapt.local.mt_catalog import (
    ALL_MT_MODELS,
    KNOWN_TIERS,
    MT_CATALOG_SNAPSHOT_DATE,
    TIER_BALANCED,
    TIER_FAST,
    TIER_QUALITY,
    find_mt_model,
    hymt_prompt,
    is_mt_downloaded,
    mt_download_url,
    mt_model_path,
    mt_tier_dir,
)


class TestMTCatalogIntegrity:
    def test_snapshot_date_is_iso(self) -> None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", MT_CATALOG_SNAPSHOT_DATE)

    def test_three_tiers_in_order(self) -> None:
        assert tuple(model.tier for model in ALL_MT_MODELS) == KNOWN_TIERS
        assert KNOWN_TIERS == (TIER_FAST, TIER_BALANCED, TIER_QUALITY)

    def test_every_file_has_size_and_sha256(self) -> None:
        hex64 = re.compile(r"[0-9a-f]{64}")
        for model in ALL_MT_MODELS:
            assert model.size_bytes > 0
            assert hex64.fullmatch(model.sha256)
            assert model.filename.endswith(".gguf")
            assert model.repo_id.startswith("tencent/")
            assert model.params_label

    def test_pinned_sizes_match_hf_snapshot(self) -> None:
        assert find_mt_model(TIER_FAST).size_bytes == 600_534_880
        assert find_mt_model(TIER_BALANCED).size_bytes == 1_133_080_448
        assert find_mt_model(TIER_QUALITY).size_bytes == 4_624_648_896

    def test_unknown_tier_raises(self) -> None:
        with pytest.raises(ValidationError):
            find_mt_model("nope")

    def test_paths_are_namespaced(self, tmp_path) -> None:
        model = find_mt_model(TIER_BALANCED)
        dest = mt_model_path(tmp_path, model)
        assert dest.parent == mt_tier_dir(tmp_path, TIER_BALANCED)
        assert dest.name == model.filename
        assert dest.as_posix().endswith(f"/mt/{TIER_BALANCED}/{model.filename}")

    def test_download_url_is_https_resolve(self) -> None:
        url = mt_download_url(find_mt_model(TIER_FAST))
        assert url.startswith("https://huggingface.co/tencent/")
        assert url.endswith(".gguf")

    def test_is_downloaded_requires_exact_size(self, tmp_path) -> None:
        from nlapt.local.mt_catalog import MTModel

        tiny = MTModel(
            tier=TIER_FAST,
            name="tiny",
            repo_id="tencent/tiny",
            filename="tiny.gguf",
            size_bytes=16,
            sha256="ab" * 32,
            params_label="1",
        )
        dest = mt_model_path(tmp_path, tiny)
        assert not is_mt_downloaded(tmp_path, tiny)
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"x" * 8)
        assert not is_mt_downloaded(tmp_path, tiny)
        dest.write_bytes(b"x" * 16)
        assert is_mt_downloaded(tmp_path, tiny)


class TestHyMTPrompt:
    def test_chinese_target_uses_official_template(self) -> None:
        prompt = hymt_prompt("a red dress", "zh")
        assert prompt.startswith("将以下文本翻译为中文")
        assert "不要额外解释" in prompt
        assert prompt.endswith("a red dress")
        assert "\n\na red dress" in prompt

    def test_english_and_japanese_names(self) -> None:
        assert "英语" in hymt_prompt("外套", "en")
        assert "日语" in hymt_prompt("coat", "ja")

    def test_direction_map(self) -> None:
        from nlapt.local.mt_catalog import HYMT_DIRECTION_LANG

        assert HYMT_DIRECTION_LANG[Direction.EN_TO_ZH] == "zh"
        assert HYMT_DIRECTION_LANG[Direction.ZH_TO_EN] == "en"

    def test_unknown_lang_raises(self) -> None:
        with pytest.raises(ValidationError):
            hymt_prompt("hi", "fr")
