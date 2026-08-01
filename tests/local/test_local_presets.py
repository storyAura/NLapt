"""Tests for nlapt.local.presets (official caption-model prompt presets)."""

from __future__ import annotations

from nlapt.local.catalog import find_family
from nlapt.local.presets import (
    JOYCAPTION_PRESETS,
    JOYCAPTION_SYSTEM,
    PRESET_CUSTOM,
    PRESETS_BY_FAMILY,
    TORIIGATE_PRESETS,
    TORIIGATE_SYSTEM,
    presets_for,
    resolve_preset,
)


class TestCatalogCoverage:
    def test_every_preset_family_exists_in_catalog(self) -> None:
        for family_id in PRESETS_BY_FAMILY:
            find_family(family_id)  # raises ValidationError for unknown ids

    def test_caption_specialists_have_presets(self) -> None:
        assert presets_for("joycaption-beta-one") == JOYCAPTION_PRESETS
        assert presets_for("toriigate-0.5") == TORIIGATE_PRESETS

    def test_generic_families_have_none(self) -> None:
        assert presets_for("gemma4-12b") == ()
        assert presets_for("gemma4-31b-heretic") == ()
        assert presets_for("florence2-promptgen-v2") == ()  # 指令模式 instead


class TestPresetIntegrity:
    def test_presets_are_complete_and_ids_unique(self) -> None:
        for family_id, presets in PRESETS_BY_FAMILY.items():
            assert presets, family_id
            ids = [preset.preset_id for preset in presets]
            assert len(ids) == len(set(ids)), family_id
            assert PRESET_CUSTOM not in ids, family_id
            for preset in presets:
                assert preset.preset_id, family_id
                assert preset.label, family_id
                assert preset.user_prompt.strip(), preset.preset_id

    def test_joycaption_official_texts(self) -> None:
        joy = {preset.preset_id: preset for preset in JOYCAPTION_PRESETS}
        assert (
            joy["Descriptive"].user_prompt
            == "Write a detailed description for this image."
        )
        assert "comma-separated Danbooru tags" in joy["Danbooru tag list"].user_prompt
        assert joy["Descriptive"].system == JOYCAPTION_SYSTEM
        assert JOYCAPTION_SYSTEM.startswith("You are a helpful assistant")
        # The default (first) preset is the official Descriptive mode.
        assert JOYCAPTION_PRESETS[0].preset_id == "Descriptive"

    def test_toriigate_official_assembly(self) -> None:
        torii = {preset.preset_id: preset for preset in TORIIGATE_PRESETS}
        long = torii["long"].user_prompt
        # Assembled exactly like the official make_user_query helper.
        assert long.startswith("# Captioning format:\n")
        assert "Use 2 to 5 paragraphs" in long
        assert "# Characters on picture:" in long
        assert "Try to recognize the characters" in torii["json"].user_prompt
        assert torii["short"].system == TORIIGATE_SYSTEM
        assert TORIIGATE_SYSTEM.startswith("You are image captioning expert")


class TestResolvePreset:
    def test_empty_id_falls_back_to_family_default(self) -> None:
        assert resolve_preset("joycaption-beta-one", "") is JOYCAPTION_PRESETS[0]

    def test_known_id_matches(self) -> None:
        preset = resolve_preset("toriigate-0.5", "short")
        assert preset is not None
        assert preset.preset_id == "short"

    def test_unknown_id_falls_back_to_family_default(self) -> None:
        assert resolve_preset("toriigate-0.5", "<nope>") is TORIIGATE_PRESETS[0]

    def test_custom_opts_out(self) -> None:
        assert resolve_preset("joycaption-beta-one", PRESET_CUSTOM) is None

    def test_family_without_presets_resolves_to_none(self) -> None:
        assert resolve_preset("gemma4-12b", "") is None
        assert resolve_preset("gemma4-12b", "Descriptive") is None
