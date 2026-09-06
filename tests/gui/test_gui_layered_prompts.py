"""Tests for nlapt_gui.layered_prompts (built-in 分层推标 skills)."""

from __future__ import annotations

from nlapt_gui.layered_prompts import (
    CARD_APPEARANCE_SKILL,
    CARD_BAN_WORDS,
    CARD_CLOTHING_SKILL,
    CARD_VARIANT_HINTS,
    CHARACTER_CARD_PROMPT,
    POSE_SCENE_PROMPT,
    SCENE_HEDGE_WORDS,
    assemble_caption,
    build_card_prompt,
    build_scene_prompt,
    count_words,
    estimate_tokens,
    format_card_stats,
)


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

    def test_variants_keep_appearance_first(self) -> None:
        joined = " ".join(CARD_VARIANT_HINTS).lower()
        assert "never start with garments" in joined
        assert "do not lead with footwear" in joined
        for hint in CARD_VARIANT_HINTS:
            assert "lead with headwear" not in hint.lower()
            assert not hint.lower().startswith("lead with footwear")
            assert "lead with outer layers" not in hint.lower()


class TestScenePrompt:
    def test_locks_name(self) -> None:
        text = build_scene_prompt("ema")
        assert "{NAME}" not in text
        assert text.count("ema") >= 3

    def test_skill_lists_hedge_bans_and_clothing_ban(self) -> None:
        lower = POSE_SCENE_PROMPT.lower()
        for word in SCENE_HEDGE_WORDS:
            assert word in lower
        assert "clothing" in lower
        assert "200–300" in POSE_SCENE_PROMPT or "200-300" in POSE_SCENE_PROMPT


class TestAssemble:
    def test_joins_with_blank_line(self) -> None:
        assert assemble_caption("  card  ", "  scene  ") == "card\n\nscene"


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
