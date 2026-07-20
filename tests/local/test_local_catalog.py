"""Tests for nlapt.local.catalog: structure integrity + lookups + paths."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nlapt.core.errors import ValidationError
from nlapt.local.catalog import (
    ALL_FAMILIES,
    ALL_SERIES,
    CATALOG_SNAPSHOT_DATE,
    all_families,
    all_series,
    download_url,
    families_for,
    family_dir,
    find_family,
    find_quant,
    mmproj_path,
    quant_path,
    recommended_quant,
    repo_page_url,
)


class TestCatalogIntegrity:
    def test_snapshot_date_is_iso(self) -> None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", CATALOG_SNAPSHOT_DATE)

    def test_series_ids_unique_and_nonempty(self) -> None:
        ids = [series.series_id for series in ALL_SERIES]
        assert len(ids) == len(set(ids))
        assert all(series.name and series.description for series in ALL_SERIES)

    def test_family_ids_unique(self) -> None:
        ids = [family.family_id for family in ALL_FAMILIES]
        assert len(ids) == len(set(ids))

    def test_every_family_belongs_to_a_known_series(self) -> None:
        known = {series.series_id for series in ALL_SERIES}
        for family in ALL_FAMILIES:
            assert family.series_id in known

    def test_every_series_has_families(self) -> None:
        for series in ALL_SERIES:
            assert families_for(series.series_id), series.series_id

    def test_quants_have_positive_sizes_and_unique_labels(self) -> None:
        for family in ALL_FAMILIES:
            assert family.quants
            labels = [quant.label for quant in family.quants]
            assert len(labels) == len(set(labels))
            for quant in family.quants:
                assert quant.size_bytes > 0
                assert quant.filename.endswith(".gguf")

    def test_exactly_one_recommended_quant_per_family(self) -> None:
        for family in ALL_FAMILIES:
            recommended = [quant for quant in family.quants if quant.recommended]
            assert len(recommended) == 1, family.family_id
            assert recommended_quant(family) == recommended[0]

    def test_vision_families_carry_mmproj(self) -> None:
        for family in ALL_FAMILIES:
            if family.vision:
                assert family.mmproj_filename.endswith(".gguf"), family.family_id
                assert family.mmproj_bytes > 0
            else:
                assert family.mmproj_filename == ""
                assert family.mmproj_bytes == 0

    def test_every_file_has_a_sha256_digest(self) -> None:
        hex64 = re.compile(r"[0-9a-f]{64}")
        for family in ALL_FAMILIES:
            for quant in family.quants:
                assert hex64.fullmatch(quant.sha256), (family.family_id, quant.label)
            if family.vision:
                assert hex64.fullmatch(family.mmproj_sha256), family.family_id
            else:
                assert family.mmproj_sha256 == ""

    def test_downloads_and_metadata_present(self) -> None:
        for family in ALL_FAMILIES:
            assert family.downloads > 0
            assert family.params_label
            assert family.license
            assert family.kv_bytes_per_token > 0
            assert "/" in family.repo_id

    def test_requested_models_are_present(self) -> None:
        ids = {family.family_id for family in ALL_FAMILIES}
        # Gemma 4 series, heretic variants, ToriiGate-0.5 and JoyCaption were
        # the explicitly requested catalog contents.
        assert {"gemma4-12b", "gemma4-26b-a4b", "gemma4-31b"} <= ids
        assert any(family.series_id == "gemma4-heretic" for family in ALL_FAMILIES)
        assert "toriigate-0.5" in ids
        assert "joycaption-beta-one" in ids


class TestLookups:
    def test_all_series_and_families_accessors(self) -> None:
        assert all_series() == ALL_SERIES
        assert all_families() == ALL_FAMILIES

    def test_find_family(self) -> None:
        family = find_family("toriigate-0.5")
        assert family.name == "ToriiGate 0.5"

    def test_find_family_unknown_raises(self) -> None:
        with pytest.raises(ValidationError):
            find_family("nope")

    def test_find_quant(self) -> None:
        family = find_family("joycaption-beta-one")
        quant = find_quant(family, "Q8_0")
        assert quant.size_bytes == 8_540_772_544

    def test_find_quant_unknown_raises(self) -> None:
        family = find_family("joycaption-beta-one")
        with pytest.raises(ValidationError):
            find_quant(family, "Q0_FAKE")


class TestUrlsAndPaths:
    def test_download_url_format(self) -> None:
        url = download_url("a/b", "model.gguf")
        assert url == "https://huggingface.co/a/b/resolve/main/model.gguf"

    def test_download_url_quotes_path_segments(self) -> None:
        url = download_url("a/b", "sub dir/file name.gguf")
        assert url.endswith("/resolve/main/sub%20dir/file%20name.gguf")

    def test_download_url_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            download_url("", "x.gguf")
        with pytest.raises(ValidationError):
            download_url("a/b", "")

    def test_repo_page_url(self) -> None:
        assert repo_page_url("a/b") == "https://huggingface.co/a/b"

    @pytest.mark.parametrize(
        "bad_repo", ["a", "a/b/c", "../x/y", "a b/c", "https://evil"]
    )
    def test_bad_repo_ids_rejected(self, bad_repo: str) -> None:
        with pytest.raises(ValidationError):
            download_url(bad_repo, "x.gguf")
        with pytest.raises(ValidationError):
            repo_page_url(bad_repo)

    def test_family_dir_rejects_unsafe_ids(self, tmp_path: Path) -> None:
        from dataclasses import replace

        base = find_family("gemma4-12b")
        for bad_id in ("../evil", "a/b", "a\\b", "..", ".hidden", "C:evil"):
            broken = replace(base, family_id=bad_id)
            with pytest.raises(ValidationError):
                family_dir(tmp_path, broken)

    def test_local_paths_are_namespaced_per_family(self, tmp_path: Path) -> None:
        # Several unsloth repos share the literal filename "mmproj-F16.gguf";
        # per-family directories keep them from clobbering each other.
        e2b = find_family("gemma4-e2b")
        e4b = find_family("gemma4-e4b")
        path_a = mmproj_path(tmp_path, e2b)
        path_b = mmproj_path(tmp_path, e4b)
        assert path_a is not None and path_b is not None
        assert path_a != path_b
        assert path_a.parent == tmp_path / "gemma4-e2b"

    def test_quant_path_uses_basename(self, tmp_path: Path) -> None:
        family = find_family("gemma4-12b")
        quant = recommended_quant(family)
        path = quant_path(tmp_path, family, quant)
        assert path == tmp_path / "gemma4-12b" / "gemma-4-12b-it-Q4_K_M.gguf"

    def test_mmproj_path_none_for_text_only(self, tmp_path: Path) -> None:
        family = find_family("gemma4-26b-a4b-heretic")
        assert mmproj_path(tmp_path, family) is None
