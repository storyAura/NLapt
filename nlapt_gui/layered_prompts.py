"""Built-in skills for layered 推标: character card + pose/scene paragraph.

The final caption is two English paragraphs joined by a blank line:

1. A fixed identity/outfit card (no pose, no scene).
2. A per-image pose/scene paragraph that never restates clothing or looks.

Pure functions — no Qt. Chinese UI strings live at the module top.
"""

from __future__ import annotations

# -- UI / history labels -------------------------------------------------------
LAYERED_BATCH_DESCRIPTION_FMT = "分层推标 · {n} 张"
LAYERED_BATCH_HISTORY = "分层推标"
TIP_FLORENCE_NO_LAYERED = "当前本地模型是 Florence-2,不支持自定义提示词,请改用 LLM"
# Heuristic: English averages ~4 characters per token (no tokenizer in-app).
CHARS_PER_TOKEN = 4
CARD_STATS_FMT = "约 {tokens} tokens · {words} 词"

# -- Character-card skill (appearance + clothing only) -------------------------
CHARACTER_CARD_PROMPT = """\
You are an image analysis system. Output English only.

Write one short objective paragraph that identifies the named subject and \
lists only their visible fixed appearance and full outfit.

Lead sentence must start with this exact opening: "{opening}"
Then continue in the same paragraph.

Subject naming: refer to the primary subject only as {name} (or a pronoun \
after the first mention). Do not invent other proper names.

Include only what is visible:
- Hair: color, length, style (cut, bangs, ties). No mood words.
- Face-structure facts only if clearly visible (e.g. glasses, a mark) — \
not expression, gaze, or emotion.
- Full clothing from head to shoes, including headwear, outerwear, inner \
layers, bottoms, hosiery, footwear, bags, jewelry, harnesses, and other \
worn accessories. Name colors and construction (pleated, ruffled, \
cold-shoulder, crossed straps) when they are visible.
- Species or body-covering traits that are costume-like (e.g. animal ears \
that are part of the design) only if worn/attached as appearance, not pose.

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
    "This is alternative 1 of 3: lead with headwear and outer layers.",
    "This is alternative 2 of 3: lead with inner garments and accessories; "
    "rephrase; do not copy another alternative.",
    "This is alternative 3 of 3: lead with footwear and lower garments; "
    "use a different sentence order.",
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
