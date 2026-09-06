"""Built-in skills for layered 推标: character card + pose/scene paragraph.

The final caption is two English paragraphs joined by a blank line:

1. A fixed identity/outfit card (no pose, no scene).
2. A per-image pose/scene paragraph that never restates clothing or looks.

Pure functions — no Qt. Chinese UI strings live at the module top.
"""

from __future__ import annotations

# -- UI / history labels -------------------------------------------------------
LAYERED_BATCH_DESCRIPTION_FMT = "CHA标注 · {n} 张"
LAYERED_BATCH_HISTORY = "CHA标注"
TIP_FLORENCE_NO_LAYERED = "当前本地模型是 Florence-2,不支持自定义提示词,请改用 LLM"
# Heuristic: English averages ~4 characters per token (no tokenizer in-app).
CHARS_PER_TOKEN = 4
CARD_STATS_FMT = "约 {tokens} tokens · {words} 词"

# -- Character-card skills (appearance first, then clothing) -------------------
# Two named skills are filled into one card so the model cannot start with
# shoes / jacket. Appearance is always written out before any garment.
CARD_APPEARANCE_SKILL = """\
Appearance skill (write this block first, before any clothing):
- Hair: color, length, cut, bangs, highlights, and how it is worn \
(loose, tied, twintails, ribbons, clips). No mood words.
- Eyes: iris color; pupil color and pupil shape when visible \
(round, slit, cross, ring, symbol). Do not mention gaze direction.
- Other fixed face/body facts only if clearly visible: glasses, a mark, \
horns, or species traits that are part of the design (e.g. animal ears \
worn as appearance). No expression, emotion, or gaze.
"""

CARD_CLOTHING_SKILL = """\
Clothing skill (write this block second, after appearance is complete):
- Full outfit from head to shoes: headwear, outerwear, inner layers, \
bottoms, hosiery, footwear, bags, jewelry, harnesses, straps, and other \
worn accessories.
- Name colors and construction when visible (pleated, ruffled, \
cold-shoulder, crossed straps, cropped, oversized).
"""

CHARACTER_CARD_PROMPT = """\
You are an image analysis system. Output English only.

Write one short objective paragraph that identifies the named subject \
and lists only their visible fixed appearance and full outfit.

Lead sentence must start with this exact opening: "{opening}"
Then continue in the same paragraph.

Subject naming: refer to the primary subject only as {name} (or a pronoun \
after the first mention). Do not invent other proper names.

Required order (do not invert, do not interleave):
1. Appearance skill — hair, eyes / pupils, then other fixed face facts.
2. Clothing skill — the full outfit, only after appearance is finished.

""" + CARD_APPEARANCE_SKILL + "\n" + CARD_CLOTHING_SKILL + """
Hard bans (never include):
- Pose, stance, weight shift, gestures, hand position, or motion.
- Facial expression, gaze direction, emotion, or demeanor.
- Framing, crop, composition, camera angle, lens, depth of field, focus.
- Background, setting, furniture, props not worn on the body, weather.
- Lighting, shadows, color grading, atmosphere.
- Art-style or medium labels (anime, oil painting, illustration, …).
- Quality or scoring words (masterpiece, best quality, detailed, …).
- Hedging (seems, appears, might, probably, likely, suggests).
- Speculation, narrative, praise, or criticism.

Output format: one complete English paragraph. No title, bullets, or notes.
The paragraph MUST finish all appearance facts before the first garment \
word (jacket, dress, skirt, shirt, socks, shoes, …).
"""

# -- Pose/scene skill (SKILL.md, no clothing / appearance) ---------------------
POSE_SCENE_PROMPT = """\
You are an image analysis system. Process visual data and output a fully \
objective English text description. Output English only.

Produce one clear paragraph of 200–300 words describing the provided image.

Rules:

1. Absolute objectivity
Describe only what is visually present. Language must be objective, precise, \
and declarative. Do not speculate, infer, or assume. Do not use words such as \
"seems", "appears", "might", "probably", "likely", or "suggests". State \
observed facts only.

2. Content policy
Describe all visible elements, actions, and interactions calmly and \
objectively. Do not censor, omit, or morally judge sensitive, explicit, \
violent, or crude content. The description is an objective visual report.

3. Subject naming
Refer to the primary subject only as {NAME}. Do not invent other proper \
names. Do not identify real public figures or fictional characters by their \
known names; if a recognizable figure is present and is not {NAME}, describe \
them only as an unnamed person or secondary figure without identity labels.

4. Required content (include all of the following that are visible)
- {NAME}: demeanor / facial expression / gaze direction; body posture; \
position in the frame and relative to other people, creatures, or objects; \
actions and interactions.
- Other people, creatures, or figures: demeanor, posture, position, and \
actions only — same bans as for {NAME}.
- Objects and environment: indoor/outdoor setting; objects; spatial \
relationships between objects and between objects and figures.
- Photography: camera angle (e.g. low angle, eye level), composition, depth \
of field, focus, lighting, lens effects (e.g. fisheye distortion, lens flare) \
when visible.
- On-image text and watermarks: quote all readable text; state where each \
string sits in the frame; for watermarks state location, color, style, and \
relative size.
- Censorship: if any region is pixelated, black-barred, mosaicked, or \
otherwise censored, state that and say which part of the image is covered. \
If nothing is censored, say nothing about censorship.

5. Hard bans (never include)
- Clothing, outfits, accessories worn on the body, hairstyle, hair \
color/length, body type, age cues, skin tone, facial structure, makeup, \
species/breed appearance traits, or any other appearance or costume \
descriptors for {NAME} or anyone else.
- Subjective interpretation: meaning, symbolism, intent, narrative, praise, \
or criticism.
- Art-style labels or medium labels (e.g. impressionism, surrealism, cartoon, \
pixel art, pencil drawing, oil painting, line art). Do not classify render \
style.

6. Output format
Output only one complete English paragraph of 200–300 words. No headings, \
summaries, introductions, or trailing notes before or after the paragraph.
"""

# Extra user-line so three concurrent card calls do not collapse to one wording.
CARD_VARIANT_HINTS: tuple[str, ...] = (
    "This is alternative 1 of 3: after the opening, give more hair detail "
    "(part, ties, highlights), then clothing. Never start with garments.",
    "This is alternative 2 of 3: after the opening, give more eye/pupil "
    "detail, then clothing. Rephrase; do not copy another alternative.",
    "This is alternative 3 of 3: keep appearance-first order, then be more "
    "specific about clothing construction. Do not lead with footwear.",
)

# Words the pose/scene skill must forbid (self-check / tests).
SCENE_HEDGE_WORDS: tuple[str, ...] = (
    "seems",
    "appears",
    "might",
    "probably",
    "likely",
    "suggests",
)
CARD_BAN_WORDS: tuple[str, ...] = (
    "pose",
    "gaze",
    "composition",
    "background",
    "lighting",
    "masterpiece",
)


def _fill(template: str, **values: str) -> str:
    """Replace ``{key}`` placeholders without interpreting other braces."""
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def build_card_prompt(name: str, series: str = "", *, variant: int = 0) -> str:
    """User prompt for one character-card alternative (0..2)."""
    cleaned = name.strip()
    work = series.strip()
    opening = f"{cleaned} from {work}, " if work else f"{cleaned}, "
    filled = _fill(CHARACTER_CARD_PROMPT, name=cleaned, opening=opening)
    hint = CARD_VARIANT_HINTS[max(0, min(variant, len(CARD_VARIANT_HINTS) - 1))]
    return f"{filled.rstrip()}\n\n{hint}"


def build_scene_prompt(name: str) -> str:
    """User prompt for the per-image pose/scene paragraph."""
    return _fill(POSE_SCENE_PROMPT, NAME=name.strip())


def assemble_caption(card: str, scene: str) -> str:
    """Join the locked card and the per-image scene with a blank line."""
    return f"{card.strip()}\n\n{scene.strip()}"


def count_words(text: str) -> int:
    """English word count (whitespace-separated tokens)."""
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Rough token count: ``len // 4`` for non-empty text, else 0.

    This is a preview heuristic, not a model tokenizer — NLapt stays
    stdlib-only and does not ship tiktoken / sentencepiece.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // CHARS_PER_TOKEN)


def format_card_stats(text: str) -> str:
    """Chinese status line for a card / prompt body."""
    return CARD_STATS_FMT.format(
        tokens=estimate_tokens(text), words=count_words(text)
    )
