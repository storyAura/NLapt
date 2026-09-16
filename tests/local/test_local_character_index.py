"""Tests for nlapt.local.character_index: CSV lookup, family, display names."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.errors import LocalInferenceError
from nlapt.local.character_index import (
    CharacterIndex,
    display_name,
    display_series,
    normalize_tag,
)

CSV = """character_tag,other_names,copyright,parent_tag,post_count
kageyama_shien,Shien,hololive,,1000
kageyama_shien_(1st_costume),,hololive,kageyama_shien,100
kageyama_shien_(2nd_costume),,hololive,kageyama_shien,200
hatsune_miku,Miku,vocaloid_(series),,5000
,
not_a_character_missing_tag,,,,
"""


def _index(tmp_path: Path) -> CharacterIndex:
    path = tmp_path / "danbooru_character_tags.csv"
    path.write_text(CSV, encoding="utf-8")
    return CharacterIndex.load(path)


class TestNormalizeAndDisplay:
    def test_normalize_spaces_and_case(self) -> None:
        assert normalize_tag("Hatsune Miku") == "hatsune_miku"
        assert normalize_tag("  kageyama shien (1st costume) ") == (
            "kageyama_shien_(1st_costume)"
        )

    def test_display_name_strips_qualifier(self) -> None:
        assert display_name("kageyama_shien_(1st_costume)") == "Kageyama Shien"
        assert display_name("hatsune miku") == "Hatsune Miku"

    def test_display_series_strips_series_suffix(self) -> None:
        assert display_series("vocaloid_(series)") == "Vocaloid"
        assert display_series("hololive") == "Hololive"
        assert display_series("") == ""


class TestIndex:
    def test_lookup_after_normalize(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        entry = index.lookup("Kageyama Shien (1st costume)")
        assert entry is not None
        assert entry.tag == "kageyama_shien_(1st_costume)"
        assert entry.parent_tag == "kageyama_shien"
        assert entry.copyright == "hololive"
        assert entry.post_count == 100

    def test_family_expands_parent_children_and_self(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        tags = [entry.tag for entry in index.family("kageyama shien (1st costume)")]
        assert tags == [
            "kageyama_shien",
            "kageyama_shien_(2nd_costume)",
            "kageyama_shien_(1st_costume)",
        ]
        assert index.family("unknown_character") == ()

    def test_children_of_root(self, tmp_path: Path) -> None:
        index = _index(tmp_path)
        children = {entry.tag for entry in index.children_of("kageyama_shien")}
        assert children == {
            "kageyama_shien_(1st_costume)",
            "kageyama_shien_(2nd_costume)",
        }

    def test_broken_csv_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.csv"
        with pytest.raises(LocalInferenceError, match="对照表损坏"):
            CharacterIndex.load(path)
