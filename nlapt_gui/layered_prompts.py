"""Built-in skills for layered 推标: character card + pose/scene paragraph.

The final caption is two English paragraphs joined by a blank line:

1. A fixed identity/outfit card (no pose, no scene).
2. A per-image pose/scene paragraph that never restates clothing or looks.

The per-image call also receives the card's official outfit and answers
with an ``OUTFIT:`` header first: ``SAME`` keeps the card verbatim, anything
else replaces the card's clothing block while the appearance block (hair,
eyes, skin marks) stays fixed. Still one LLM call per image.

Pure functions — no Qt. Chinese UI strings live at the module top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# -- UI / history labels -------------------------------------------------------
LAYERED_BATCH_DESCRIPTION_FMT = "CHA标注 · {n} 张"
LAYERED_BATCH_HISTORY = "CHA标注"
TIP_FLORENCE_NO_LAYERED = "当前本地模型是 Florence-2,不支持自定义提示词,请改用 LLM"
LABEL_CARD_APPEARANCE = "固定外貌（每张图保留）"
LABEL_CARD_OUTFIT = "官方服装（按图核对，不同则改写）"
WARN_SPLIT_FAILED = "未识别到服装段：服装将不会按图改写"
# Heuristic: English averages ~4 characters per token (no tokenizer in-app).
CHARS_PER_TOKEN = 4
CARD_STATS_FMT = "约 {tokens} tokens · {words} 词"

# -- Outfit-check protocol (scene reply header) --------------------------------
OUTFIT_PREFIX = "OUTFIT:"
OUTFIT_SAME = "SAME"
OUTFIT_REFERENCE_MISSING = (
    "(no official outfit on record — answer SAME and describe nothing about clothing)"
)
# Sentence openers the card must use for its clothing block (split anchor).
OUTFIT_OPENER_VERBS = ("wears", "is wearing", "is dressed in")
# Fallback anchor when no opener sentence exists: first sentence naming a garment.
GARMENT_WORDS: tuple[str, ...] = (
    "jacket", "dress", "skirt", "shirt", "blouse", "coat", "pants", "trousers",
    "shorts", "socks", "stockings", "thighhighs", "shoes", "boots", "sneakers",
    "heels", "gloves", "choker", "necklace", "bikini", "swimsuit", "uniform",
    "hoodie", "sweater", "cardigan", "belt", "cape", "cloak", "leotard",
    "bodysuit", "top", "vest", "tie", "scarf",
)

# -- Character-card skills (appearance first, then clothing) -------------------
# Two named skills are filled into one card so the model cannot start with
# shoes / jacket. Appearance is always written out before any garment, and
# inside appearance the hair COLOR is always the first fact after the opening.
CARD_APPEARANCE_SKILL = """\
Appearance skill (write this block first, before any clothing):
- Hair, in this fixed sub-order and nothing in between: (a) hair color \
first — including two-tone, gradient, inner color, streaks or highlights; \
(b) length; (c) cut and how it is worn (loose, ponytail, twintails, buns, \
braids, side tail); (d) bangs and parting; (e) hair-worn ornaments \
(ribbons, clips, flowers, pins). No mood words.
- Eyes: iris color; pupil color and pupil shape when visible \
(round, slit, cross, ring, symbol). Do not mention gaze direction.
- Skin marks are mandatory when visible: every tattoo, scar, birthmark, \
mole, painted mark, or body paint must be stated with its position on the \
body, its color, and its motif or shape. Omitting a visible tattoo is an \
error. Do not describe skin tone otherwise.
- Other fixed face/body facts only if clearly visible: glasses, horns, \
halo, or species traits that are part of the design (e.g. animal ears, \
tail, wings). No expression, emotion, or gaze.
"""

CARD_CLOTHING_SKILL = """\
Clothing skill (write this block second, after appearance is complete):
- The clothing block MUST begin a new sentence whose first words are \
"{name} wears" (or "She wears" / "He wears"). Nothing about clothing may \
appear before that sentence; nothing about hair, eyes, or skin may appear \
after it.
- Full outfit from head to shoes; no visible layer or small worn item may \
be skipped. Cover in order: headwear, neckwear (choker, collar, necklace, \
tie, scarf), outerwear, inner layers and tops, bottoms, arm wear (sleeves, \
gloves, arm bands, bracelets), waist wear (belts, sashes, harnesses, \
crossed straps, garter straps), hosiery and leg straps, footwear, bags, \
then remaining jewelry and worn accessories.
- For every garment name its color and any visible pattern, print, logo, \
trim, embroidery, or emblem.
- Name construction when visible: pleated, ruffled, lace, sheer, cutout, \
cold-shoulder, off-shoulder, cropped, oversized, layered, zipper, buckle, \
lacing, buttons.
- Exposed underlayers (bikini, swimsuit, bra strap, underwear, undershirt) \
that are visible must be named as their own item.
"""

CHARACTER_CARD_PROMPT = """\
You are an image analysis system. Output English only.

Write one complete objective paragraph that identifies the named subject \
and lists only their visible fixed appearance and full outfit. Completeness \
matters more than brevity: every visible tattoo and every visible garment \
detail must be present.

Lead sentence must start with this exact opening: "{opening}"
Then continue in the same paragraph. The first fact after the opening is \
the hair color.

Subject naming: refer to the primary subject only as {name} (or a pronoun \
after the first mention). Do not invent other proper names.

Required order (do not invert, do not interleave):
1. Appearance skill — hair color, then hair length / style, then eyes / \
pupils, then tattoos and other fixed skin or body facts.
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
word (jacket, dress, skirt, shirt, socks, shoes, …). A paragraph that omits \
a visible tattoo, an exposed underlayer, a strap, or a garment pattern is \
not acceptable.
"""

# -- Pose/scene skill (SKILL.md, no clothing / appearance) ---------------------
POSE_SCENE_PROMPT = """\
You are an image analysis system. Process visual data and output a fully \
objective English text description. Output English only.

Produce two parts: an OUTFIT line, then one clear paragraph of 200–300 \
words describing the provided image.

Rules:

0. Outfit check (first output line)
The official outfit of {NAME} on record is:
"{OUTFIT}"
Compare the clothing actually worn by {NAME} in this image with that record.
- Treat it as the SAME outfit when the visible main garments (top, bottom, \
dress, outerwear) match in garment type and color. Parts hidden by the crop, \
by occlusion, or simply out of frame (missing shoes, hidden legs) never make \
the outfit different. Then write exactly: OUTFIT: SAME
- Treat it as a DIFFERENT outfit when a main garment differs in type or \
color, or the character is in another costume (swimsuit, uniform, casual \
clothes, sleepwear, formal wear, armor). Then write "OUTFIT: " followed by \
one to three sentences describing only the clothing that is visible, in \
this order: headwear, neckwear, outerwear, tops, bottoms, arm wear, waist \
wear, hosiery and leg straps, footwear, bags, jewelry and accessories. \
Give each garment its color and any visible pattern, print, trim, or \
construction (pleated, lace, sheer, cropped, zipper, buckle, lacing). Name \
exposed underlayers as their own item. Do not invent garments that are not \
visible. Start the first sentence with "{NAME} wears", "She wears", or \
"He wears". Do not repeat hair, eyes, skin marks, or any appearance fact. \
No pose, no expression, no setting in this line.

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

5. Hard bans in the scene paragraph (never include)
- Clothing, outfits, accessories worn on the body, hairstyle, hair \
color/length, tattoos, scars, birthmarks, body paint, body type, age cues, \
skin tone, facial structure, makeup, species/breed appearance traits, or \
any other appearance or costume descriptors for {NAME} or anyone else. \
Clothing belongs only in the OUTFIT line of rule 0.
- Subjective interpretation: meaning, symbolism, intent, narrative, praise, \
or criticism.
- Art-style labels or medium labels (e.g. impressionism, surrealism, cartoon, \
pixel art, pencil drawing, oil painting, line art). Do not classify render \
style.

6. Output format
First the OUTFIT line from rule 0 ("OUTFIT: SAME" or "OUTFIT: <clothing>"). \
Then exactly one blank line. Then one complete English paragraph of 200–300 \
words. No headings, summaries, introductions, or trailing notes anywhere \
else.
"""

# Extra user-line so three concurrent card calls do not collapse to one wording.
CARD_VARIANT_HINTS: tuple[str, ...] = (
    "This is alternative 1 of 3: after the opening, state hair color first, "
    "then give more hair detail (part, ties, highlights), then clothing. "
    "Never start with garments.",
    "This is alternative 2 of 3: after the opening and hair color, give more "
    "eye/pupil detail and name every visible tattoo or skin mark with its "
    "position, then clothing. Rephrase; do not copy another alternative.",
    "This is alternative 3 of 3: keep appearance-first order, then be more "
    "specific about clothing construction, straps, jewelry, and patterns. "
    "Do not lead with footwear.",
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


def build_scene_prompt(name: str, official_outfit: str = "") -> str:
    """User prompt for the per-image OUTFIT line + pose/scene paragraph.

    ``official_outfit`` is the card's clothing block (see :func:`split_card`);
    when empty the model is told there is no record and to answer SAME.
    """
    reference = official_outfit.strip() or OUTFIT_REFERENCE_MISSING
    return _fill(POSE_SCENE_PROMPT, NAME=name.strip(), OUTFIT=reference)


def assemble_caption(card: str, scene: str) -> str:
    """Join the locked card and the per-image scene with a blank line."""
    return f"{card.strip()}\n\n{scene.strip()}"


# -- card split + outfit-check reply handling ----------------------------------
@dataclass(frozen=True)
class CardParts:
    """A locked card split into its fixed appearance block and clothing block.

    ``outfit`` is empty when no clothing block could be located; callers then
    keep the whole card and never substitute clothing.
    """

    appearance: str
    outfit: str


def _sentence_starts(text: str) -> list[int]:
    """Offsets where a sentence begins (text start or after `. ` / `! ` / `? `)."""
    starts = [0]
    for match in re.finditer(r"[.!?]\s+(?=\S)", text):
        starts.append(match.end())
    return starts


def split_card(card: str, name: str = "") -> CardParts:
    """Split ``card`` at the first sentence that opens the clothing block.

    Primary anchor: a sentence starting ``<name>|She|He|They wears / is
    wearing / is dressed in``. Fallback: the first sentence that names a
    garment from :data:`GARMENT_WORDS`. No anchor → ``CardParts(card, "")``.
    """
    text = card.strip()
    if not text:
        return CardParts("", "")
    subjects = ["she", "he", "they"]
    cleaned = name.strip()
    if cleaned:
        subjects.append(re.escape(cleaned))
    verbs = "|".join(re.escape(verb) for verb in OUTFIT_OPENER_VERBS)
    opener = re.compile(rf"^(?:{'|'.join(subjects)})\s+(?:{verbs})\b", re.IGNORECASE)
    garment = re.compile(
        r"\b(?:" + "|".join(re.escape(word) for word in GARMENT_WORDS) + r")\b",
        re.IGNORECASE,
    )
    starts = _sentence_starts(text)
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        if opener.match(text[start:end]):
            return CardParts(text[:start].rstrip(), text[start:].strip())
    for index, start in enumerate(starts[1:], start=1):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        if garment.search(text[start:end]):
            return CardParts(text[:start].rstrip(), text[start:].strip())
    return CardParts(text, "")


def parse_scene_reply(text: str) -> tuple[str | None, str]:
    """Split a per-image reply into ``(outfit_or_None, scene)``.

    The first blank-line-separated block must start with ``OUTFIT:`` to count
    as the outfit header; ``SAME`` (any case, optional punctuation / trailing
    word) yields ``None``. Replies without the header are treated as a plain
    scene paragraph (legacy behavior).
    """
    stripped = text.strip()
    if not stripped:
        return (None, "")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", stripped) if block.strip()]
    head = blocks[0]
    if not head.upper().startswith(OUTFIT_PREFIX):
        return (None, stripped)
    body = head[len(OUTFIT_PREFIX):].strip()
    scene = "\n\n".join(blocks[1:]).strip()
    verdict = re.sub(r"[^A-Za-z ]", "", body).strip().upper()
    if not body or verdict == OUTFIT_SAME or verdict.startswith(OUTFIT_SAME + " "):
        return (None, scene)
    return (" ".join(body.split()), scene)


def assemble_layered(card: str, parts: CardParts, outfit: str | None, scene: str) -> str:
    """Card (or appearance + rewritten outfit) + blank line + scene."""
    if outfit is None or not outfit.strip() or not parts.outfit.strip():
        return assemble_caption(card, scene)
    return assemble_caption(f"{parts.appearance.rstrip()} {outfit.strip()}", scene)


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
