"""Tests for nlapt_gui.layered_prompts (built-in 分层推标 skills)."""

from __future__ import annotations

import pytest

from nlapt.core.errors import LLMOutputError, ValidationError

from nlapt_gui.layered_prompts import (
    CARD_APPEARANCE_SKILL,
    CARD_BAN_WORDS,
    CARD_CLOTHING_SKILL,
    CARD_PLACEHOLDERS,
    CARD_VARIANT_HINTS,
    CHARACTER_CARD_PROMPT,
    MAX_CARDS,
    OUTFIT_PREFIX,
    OUTFIT_REFERENCE_MISSING,
    POSE_SCENE_PROMPT,
    SCENE_HEDGE_WORDS,
    SCENE_PLACEHOLDERS,
    CardParts,
    CardRoster,
    CharacterCard,
    LayeredVerdict,
    assemble_caption,
    assemble_layered,
    assemble_roster_caption,
    build_card_prompt,
    build_scene_prompt,
    count_words,
    effective_card_template,
    effective_scene_template,
    estimate_tokens,
    format_card_stats,
    missing_placeholders,
    parse_layered_reply,
    parse_scene_reply,
    split_card,
)

APPEARANCE = (
    "ema from monosaba, has long blonde hair worn in low twintails. "
    "Her eyes are purple with round pupils. "
    "A small black crescent tattoo sits on her left upper thigh."
)
OUTFIT = "She wears a white cropped shirt, blue denim shorts, and white sneakers."
CARD = f"{APPEARANCE} {OUTFIT}"
CARD_B_APPEARANCE = "rin, has short black hair. Her eyes are red."
CARD_B_OUTFIT = "She wears a black school uniform and loafers."
CARD_B = f"{CARD_B_APPEARANCE} {CARD_B_OUTFIT}"
EMA = CharacterCard.from_text("ema", "monosaba", CARD)
RIN = CharacterCard.from_text("rin", "", CARD_B)
ROSTER = CardRoster((EMA, RIN))
SOLO = CardRoster((EMA,))


class TestCardPrompt:
    def test_opening_includes_series(self) -> None:
        text = build_card_prompt("ema", "monosaba")
        assert "ema from monosaba" in text
        assert "ema from monosaba," not in text
        assert "ema from monosaba has" in text
        assert "{name}" not in text
        assert "{opening}" not in text

    def test_opening_without_series(self) -> None:
        text = build_card_prompt("ema")
        assert "ema from " not in text
        assert "ema has" in text
        assert "ema, brown hair, long, wavy" in text

    def test_variants_differ(self) -> None:
        a = build_card_prompt("ema", variant=0)
        b = build_card_prompt("ema", variant=1)
        c = build_card_prompt("ema", variant=2)
        assert a != b != c
        assert "alternative 1" in a
        assert "alternative 3" in c

    def test_skill_bans_pose_and_quality(self) -> None:
        lower = CHARACTER_CARD_PROMPT.lower()
        for word in CARD_BAN_WORDS:
            assert word in lower

    def test_appearance_skill_before_clothing_skill(self) -> None:
        appear_at = CHARACTER_CARD_PROMPT.find(CARD_APPEARANCE_SKILL.strip())
        cloth_at = CHARACTER_CARD_PROMPT.find(CARD_CLOTHING_SKILL.strip())
        assert appear_at != -1
        assert cloth_at != -1
        assert appear_at < cloth_at
        assert "pupil" in CARD_APPEARANCE_SKILL.lower()
        assert "hair" in CARD_APPEARANCE_SKILL.lower()

    def test_appearance_skill_hair_color_before_style(self) -> None:
        lower = CARD_APPEARANCE_SKILL.lower()
        color_at = lower.find("hair color")
        assert color_at != -1
        assert color_at < lower.find("length")
        assert color_at < lower.find("twintails")
        assert color_at < lower.find("bangs")
        assert "hair-worn ornaments" not in lower

    def test_appearance_skill_requires_tattoos(self) -> None:
        lower = CARD_APPEARANCE_SKILL.lower()
        assert "tattoo" in lower
        assert "scar" in lower
        assert "mandatory" in lower
        assert "position" in lower
        prompt_lower = CHARACTER_CARD_PROMPT.lower()
        assert "hair color" in prompt_lower
        assert "tattoo" in prompt_lower
        assert "one short" not in prompt_lower

    def test_clothing_skill_lists_details(self) -> None:
        lower = CARD_CLOTHING_SKILL.lower()
        for word in ("pattern", "print", "strap", "belt", "harness",
                     "choker", "bikini", "jewelry", "lace", "buckle"):
            assert word in lower
        assert "no visible layer" in lower

    def test_clothing_skill_requires_wears_opener(self) -> None:
        assert '"{name} wears"' in CARD_CLOTHING_SKILL
        assert '"ema wears"' in build_card_prompt("ema")

    def test_appearance_skill_moves_headwear_to_clothing(self) -> None:
        appear = CARD_APPEARANCE_SKILL.lower()
        cloth = CARD_CLOTHING_SKILL.lower()
        assert "nothing worn on the head" in appear
        assert "those are clothing" in appear
        assert "hair-worn ornaments" not in appear
        assert cloth.find("headwear and hair ornaments") < cloth.find("neckwear")

    def test_card_prompt_bans_comma_fragments(self) -> None:
        lower = CHARACTER_CARD_PROMPT.lower()
        assert "comma-separated" in lower
        assert "tag lists" in lower
        joined = " ".join(CARD_VARIANT_HINTS).lower()
        assert "no tag-like comma lists" in joined

    def test_variants_keep_appearance_first(self) -> None:
        joined = " ".join(CARD_VARIANT_HINTS).lower()
        assert "never start with garments" in joined
        assert "do not lead with footwear" in joined
        assert "hair color first" in joined
        assert "tattoo" in joined
        assert "straps" in joined
        for hint in CARD_VARIANT_HINTS:
            assert "lead with headwear" not in hint.lower()
            assert not hint.lower().startswith("lead with footwear")
            assert "lead with outer layers" not in hint.lower()


class TestCustomTemplates:
    def test_effective_falls_back_to_builtin(self) -> None:
        assert effective_card_template("") is CHARACTER_CARD_PROMPT
        assert effective_card_template("   ") is CHARACTER_CARD_PROMPT
        assert effective_scene_template("") is POSE_SCENE_PROMPT
        assert effective_card_template("custom card") == "custom card"
        assert effective_scene_template("custom scene") == "custom scene"

    def test_missing_placeholders(self) -> None:
        assert missing_placeholders("hello {name}", CARD_PLACEHOLDERS) == ("{opening}",)
        assert missing_placeholders(CHARACTER_CARD_PROMPT, CARD_PLACEHOLDERS) == ()
        assert missing_placeholders(POSE_SCENE_PROMPT, SCENE_PLACEHOLDERS) == ()

    def test_build_card_prompt_uses_custom_template(self) -> None:
        text = build_card_prompt("ema", "monosaba", template="Lead:{opening} Name:{name}")
        assert "Lead:ema from monosaba Name:ema" in text
        assert "alternative 1" in text

    def test_build_scene_prompt_uses_custom_template(self) -> None:
        text = build_scene_prompt(ROSTER, template="R={ROSTER}\nN={NAMES}")
        assert "Card 1 — name: ema" in text
        assert "N=ema, rin" in text
        assert "You are an image analysis system" not in text


class TestScenePrompt:
    def test_lists_every_card_and_locks_names(self) -> None:
        text = build_scene_prompt(ROSTER)
        assert "{ROSTER}" not in text and "{NAMES}" not in text
        assert "Card 1 — name: ema" in text
        assert "Card 2 — name: rin" in text
        assert APPEARANCE in text and OUTFIT in text
        assert CARD_B_OUTFIT in text
        assert "(ema, rin)" in text

    def test_duplicate_names_listed_once(self) -> None:
        roster = CardRoster((EMA, CharacterCard.from_text("ema", "", CARD_B)))
        assert "(ema)" in build_scene_prompt(roster)

    def test_skill_lists_hedge_bans_and_clothing_ban(self) -> None:
        lower = POSE_SCENE_PROMPT.lower()
        for word in SCENE_HEDGE_WORDS:
            assert word in lower
        assert "clothing" in lower
        assert "tattoos" in lower
        assert "200–300" in POSE_SCENE_PROMPT or "200-300" in POSE_SCENE_PROMPT

    def test_skill_explains_card_identification(self) -> None:
        assert "CARD: NONE" in POSE_SCENE_PROMPT
        assert "CARD: 1, 3" in POSE_SCENE_PROMPT
        assert "OUTFIT <n>: SAME" in POSE_SCENE_PROMPT
        lower = POSE_SCENE_PROMPT.lower()
        assert "appearance first" in lower
        assert "several costumes" in lower
        assert "closest outfit" in lower

    def test_same_requires_every_item_worn_as_described(self) -> None:
        """Held / removed / open items must rewrite the outfit, never SAME."""
        lower = POSE_SCENE_PROMPT.lower()
        assert "only when every garment and accessory" in lower
        for phrase in ("held in the hand", "taken off", "hanging from the arm", "unbuttoned"):
            assert phrase in lower
        assert "holds the black hat in her right hand" in lower
        assert "leave out items that are absent" in lower
        # Cropped / occluded parts are still not a difference.
        assert '"not visible" is not "not worn"' in lower
        assert "out of frame" in lower

    def test_missing_outfit_reference_tells_model_to_answer_same(self) -> None:
        bare = CharacterCard.from_text("ema", "", "ema, black hair and red eyes.")
        text = build_scene_prompt(CardRoster((bare,)))
        assert OUTFIT_REFERENCE_MISSING in text
        assert OUTFIT_PREFIX[:-1] in text
        assert OUTFIT_REFERENCE_MISSING not in build_scene_prompt(SOLO)


class TestRoster:
    def test_from_text_splits_outfit(self) -> None:
        assert EMA.parts == CardParts(APPEARANCE, OUTFIT)
        assert EMA.series == "monosaba"

    def test_size_limits(self) -> None:
        assert MAX_CARDS == 8
        with pytest.raises(ValidationError):
            CardRoster(())
        with pytest.raises(ValidationError):
            CardRoster(tuple([EMA] * (MAX_CARDS + 1)))
        assert len(CardRoster(tuple([EMA] * MAX_CARDS))) == MAX_CARDS


class TestParseLayeredReply:
    def test_single_match_with_same(self) -> None:
        verdict = parse_layered_reply("CARD: 2\nOUTFIT 2: SAME\n\nscene", 2)
        assert verdict == LayeredVerdict((1,), (None,), "scene")

    def test_multi_match_with_rewrite_and_wrapped_line(self) -> None:
        reply = (
            "CARD: 1, 3\n"
            "OUTFIT 1: SAME\n"
            "OUTFIT 3: She wears a blue swimsuit\nand sandals.\n\n"
            "scene one\n\nscene two"
        )
        verdict = parse_layered_reply(reply, 3)
        assert verdict.matched == (0, 2)
        assert verdict.outfits == (None, "She wears a blue swimsuit and sandals.")
        assert verdict.scene == "scene one\n\nscene two"

    def test_none_skips(self) -> None:
        assert parse_layered_reply("CARD: NONE", 3) == LayeredVerdict((), (), "")
        assert parse_layered_reply("card: none.\n\nstray text", 3).matched == ()
        assert parse_layered_reply("   ", 3).matched == ()

    def test_out_of_range_numbers_dropped_and_sorted(self) -> None:
        verdict = parse_layered_reply("CARD: 3, 1, 9\n\nscene", 3)
        assert verdict.matched == (0, 2)
        assert verdict.outfits == (None, None)

    def test_full_roster_reaches_the_last_card(self) -> None:
        reply = f"CARD: 1, {MAX_CARDS}\nOUTFIT {MAX_CARDS}: She wears a cloak.\n\nscene"
        verdict = parse_layered_reply(reply, MAX_CARDS)
        assert verdict.matched == (0, MAX_CARDS - 1)
        assert verdict.outfits == (None, "She wears a cloak.")
        # One past the roster is still dropped.
        assert parse_layered_reply(f"CARD: {MAX_CARDS + 1}\n\nscene", MAX_CARDS).matched == ()

    def test_unnumbered_outfit_applies_to_single_match(self) -> None:
        verdict = parse_layered_reply("CARD: 2\nOUTFIT: She wears a red dress.\n\nscene", 2)
        assert verdict.outfits == ("She wears a red dress.",)
        multi = parse_layered_reply("CARD: 1, 2\nOUTFIT: She wears a red dress.\n\nscene", 2)
        assert multi.outfits == (None, None)

    def test_legacy_single_card_shapes(self) -> None:
        legacy = parse_layered_reply("OUTFIT: She wears a swimsuit.\n\nscene", 1)
        assert legacy == LayeredVerdict((0,), ("She wears a swimsuit.",), "scene")
        assert parse_layered_reply("just a scene", 1) == LayeredVerdict((0,), (None,), "just a scene")

    def test_multi_roster_requires_card_line(self) -> None:
        with pytest.raises(LLMOutputError):
            parse_layered_reply("just a scene", 2)
        with pytest.raises(LLMOutputError):
            parse_layered_reply("OUTFIT 1: SAME\n\nscene", 2)


class TestAssembleRoster:
    def test_none_returns_none(self) -> None:
        assert assemble_roster_caption(ROSTER, LayeredVerdict((), (), "")) is None

    def test_multi_cards_in_roster_order_with_rewrite(self) -> None:
        verdict = LayeredVerdict((0, 1), (None, "She wears a red dress."), "scene")
        result = assemble_roster_caption(ROSTER, verdict)
        assert result == f"{CARD}\n\n{CARD_B_APPEARANCE} She wears a red dress.\n\nscene"
        assert "uniform" not in result

    def test_card_without_outfit_block_never_rewritten(self) -> None:
        bare = CharacterCard.from_text("ema", "", "ema, black hair and red eyes.")
        verdict = LayeredVerdict((0,), ("She wears a dress.",), "scene")
        assert assemble_roster_caption(CardRoster((bare,)), verdict) == f"{bare.text}\n\nscene"

    def test_missing_scene_fails(self) -> None:
        with pytest.raises(LLMOutputError):
            assemble_roster_caption(SOLO, LayeredVerdict((0,), (None,), "  "))


class TestSplitCard:
    def test_splits_at_she_wears(self) -> None:
        parts = split_card(CARD, "ema")
        assert parts == CardParts(APPEARANCE, OUTFIT)

    def test_splits_at_name_wears(self) -> None:
        card = "ema, black hair. Red eyes. ema wears a red dress."
        parts = split_card(card, "ema")
        assert parts.appearance == "ema, black hair. Red eyes."
        assert parts.outfit == "ema wears a red dress."

    def test_garment_word_fallback(self) -> None:
        card = "ema, black hair and red eyes. A black jacket covers a white shirt."
        parts = split_card(card, "ema")
        assert parts.appearance == "ema, black hair and red eyes."
        assert parts.outfit.startswith("A black jacket")

    def test_no_clothing_keeps_whole_card(self) -> None:
        card = "ema, black hair and red eyes."
        assert split_card(card, "ema") == CardParts(card, "")
        assert split_card("", "ema") == CardParts("", "")

    def test_first_sentence_never_becomes_outfit(self) -> None:
        # The opening sentence is the identity line even when it names a garment.
        card = "ema in a dress, black hair. She wears a red dress."
        parts = split_card(card, "ema")
        assert parts.appearance == "ema in a dress, black hair."


class TestParseSceneReply:
    def test_same_keeps_card(self) -> None:
        assert parse_scene_reply("OUTFIT: SAME\n\nscene text") == (None, "scene text")
        assert parse_scene_reply("outfit: Same.\n\nscene") == (None, "scene")
        assert parse_scene_reply("OUTFIT: SAME OUTFIT\n\nscene") == (None, "scene")

    def test_rewritten_outfit_joins_lines(self) -> None:
        reply = "OUTFIT: She wears a blue swimsuit\nand sandals.\n\nscene one\n\nscene two"
        outfit, scene = parse_scene_reply(reply)
        assert outfit == "She wears a blue swimsuit and sandals."
        assert scene == "scene one\n\nscene two"

    def test_without_header_is_plain_scene(self) -> None:
        assert parse_scene_reply("just a scene") == (None, "just a scene")
        assert parse_scene_reply("   ") == (None, "")


class TestAssemble:
    def test_joins_with_blank_line(self) -> None:
        assert assemble_caption("  card  ", "  scene  ") == "card\n\nscene"

    def test_layered_same_keeps_card(self) -> None:
        parts = split_card(CARD, "ema")
        assert assemble_layered(CARD, parts, None, "scene") == f"{CARD}\n\nscene"

    def test_layered_rewrite_replaces_outfit_only(self) -> None:
        parts = split_card(CARD, "ema")
        result = assemble_layered(CARD, parts, "She wears a blue swimsuit.", "scene")
        assert result == f"{APPEARANCE} She wears a blue swimsuit.\n\nscene"
        assert "denim" not in result

    def test_layered_without_outfit_block_never_rewrites(self) -> None:
        parts = CardParts(CARD, "")
        result = assemble_layered(CARD, parts, "She wears a blue swimsuit.", "scene")
        assert result == f"{CARD}\n\nscene"


class TestStats:
    def test_count_words_splits_on_whitespace(self) -> None:
        assert count_words("") == 0
        assert count_words("   ") == 0
        assert count_words("one two\nthree") == 3
        assert count_words("  a   b  ") == 2

    def test_estimate_tokens_empty_and_heuristic(self) -> None:
        assert estimate_tokens("") == 0
        assert estimate_tokens("    ") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcdefgh") == 2

    def test_format_card_stats_mentions_tokens_and_words(self) -> None:
        text = format_card_stats("one two three four")
        assert "tokens" in text
        assert "词" in text
        assert "4" in text


class TestLayeredStore:
    def test_round_trip(self, tmp_path) -> None:
        from nlapt_gui.layered_store import (
            LayeredMemory,
            SlotMemory,
            load_layered_memory,
            save_layered_memory,
        )

        path = tmp_path / "layered_infer.json"
        memory = LayeredMemory(
            slots=(
                SlotMemory(name="ema", series="monosaba", card_text="card"),
                SlotMemory(name="rin", series="", card_text="card b"),
            )
        )
        save_layered_memory(memory, path)
        loaded = load_layered_memory(path)
        assert loaded == memory
        assert loaded.name == "ema" and loaded.series == "monosaba"

    def test_legacy_single_card_file_becomes_first_slot(self, tmp_path) -> None:
        from nlapt_gui.layered_store import LayeredMemory, SlotMemory, load_layered_memory

        path = tmp_path / "layered_infer.json"
        path.write_text(
            '{"name": "ema", "series": "monosaba", "card_text": "card"}', encoding="utf-8"
        )
        assert load_layered_memory(path) == LayeredMemory(
            slots=(SlotMemory(name="ema", series="monosaba", card_text="card"),)
        )

    def test_slots_capped_and_too_many_rejected(self, tmp_path) -> None:
        from nlapt.core.errors import StorageError

        from nlapt_gui.layered_store import (
            LayeredMemory,
            SlotMemory,
            load_layered_memory,
            save_layered_memory,
        )

        path = tmp_path / "layered_infer.json"
        many = [{"name": f"n{i}", "series": "", "card_text": ""} for i in range(MAX_CARDS + 2)]
        path.write_text(f'{{"slots": {many}}}'.replace("'", '"'), encoding="utf-8")
        assert len(load_layered_memory(path).slots) == MAX_CARDS
        with pytest.raises(StorageError):
            save_layered_memory(
                LayeredMemory(slots=tuple(SlotMemory(name="x") for _ in range(MAX_CARDS + 1))),
                path,
            )

    def test_missing_and_corrupt_are_empty(self, tmp_path) -> None:
        from nlapt_gui.layered_store import LayeredMemory, load_layered_memory

        missing = tmp_path / "nope.json"
        assert load_layered_memory(missing) == LayeredMemory()
        bad = tmp_path / "bad.json"
        bad.write_text("{not-json", encoding="utf-8")
        assert load_layered_memory(bad) == LayeredMemory()
        assert load_layered_memory(bad).name == ""
