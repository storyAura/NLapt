"""Tests for nlapt_gui.layered_prompts (built-in 分层推标 skills)."""

from __future__ import annotations

from nlapt_gui.layered_prompts import (
    CARD_APPEARANCE_SKILL,
    CARD_BAN_WORDS,
    CARD_CLOTHING_SKILL,
    CARD_VARIANT_HINTS,
    CHARACTER_CARD_PROMPT,
    OUTFIT_PREFIX,
    OUTFIT_REFERENCE_MISSING,
    POSE_SCENE_PROMPT,
    SCENE_HEDGE_WORDS,
    CardParts,
    assemble_caption,
    assemble_layered,
    build_card_prompt,
    build_scene_prompt,
    count_words,
    estimate_tokens,
    format_card_stats,
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


class TestCardPrompt:
    def test_opening_includes_series(self) -> None:
        text = build_card_prompt("ema", "monosaba")
        assert "ema from monosaba, " in text
        assert "{name}" not in text
        assert "{opening}" not in text

    def test_opening_without_series(self) -> None:
        text = build_card_prompt("ema")
        assert "ema from " not in text
        assert "ema, " in text

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
        assert color_at < lower.find("ribbons")

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


class TestScenePrompt:
    def test_locks_name(self) -> None:
        text = build_scene_prompt("ema")
        assert "{NAME}" not in text
        assert "{OUTFIT}" not in text
        assert text.count("ema") >= 3

    def test_skill_lists_hedge_bans_and_clothing_ban(self) -> None:
        lower = POSE_SCENE_PROMPT.lower()
        for word in SCENE_HEDGE_WORDS:
            assert word in lower
        assert "clothing" in lower
        assert "tattoos" in lower
        assert "200–300" in POSE_SCENE_PROMPT or "200-300" in POSE_SCENE_PROMPT

    def test_outfit_check_injects_official_outfit(self) -> None:
        text = build_scene_prompt("ema", official_outfit=OUTFIT)
        assert OUTFIT in text
        assert OUTFIT_PREFIX in text
        assert "OUTFIT: SAME" in text
        assert OUTFIT_REFERENCE_MISSING not in text

    def test_missing_outfit_reference_tells_model_to_answer_same(self) -> None:
        text = build_scene_prompt("ema")
        assert OUTFIT_REFERENCE_MISSING in text


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
        from nlapt_gui.layered_store import LayeredMemory, load_layered_memory, save_layered_memory

        path = tmp_path / "layered_infer.json"
        memory = LayeredMemory(name="ema", series="monosaba", card_text="card")
        save_layered_memory(memory, path)
        loaded = load_layered_memory(path)
        assert loaded == memory

    def test_missing_and_corrupt_are_empty(self, tmp_path) -> None:
        from nlapt_gui.layered_store import LayeredMemory, load_layered_memory

        missing = tmp_path / "nope.json"
        assert load_layered_memory(missing) == LayeredMemory()
        bad = tmp_path / "bad.json"
        bad.write_text("{not-json", encoding="utf-8")
        assert load_layered_memory(bad) == LayeredMemory()
