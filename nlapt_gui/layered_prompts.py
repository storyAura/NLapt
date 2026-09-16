"""Built-in skills for layered 推标: character cards + pose/scene paragraph.

The final caption is the matched character card(s) followed by one
per-image pose/scene paragraph, every block separated by a blank line:

1. Fixed identity/outfit cards (no pose, no scene) — one per matched card.
2. A per-image pose/scene paragraph that never restates clothing or looks.

A run carries a :class:`CardRoster` of 1..:data:`MAX_CARDS` cards (several
characters, or one character in several costumes). The per-image call
lists every card and answers with a header block first: ``CARD: 1, 3`` (or
``CARD: NONE`` → the image is skipped), then one ``OUTFIT n:`` line per
matched card — ``SAME`` keeps that card verbatim, anything else replaces its
clothing block while the appearance block (hair, eyes, skin marks) stays
fixed. Still one LLM call per image.

Pure functions — no Qt. Chinese UI strings live at the module top.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nlapt.core.errors import LLMOutputError, ValidationError

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

# -- Roster limits ------------------------------------------------------------
MAX_CARDS = 8
ERR_ROSTER_SIZE = f"角色卡数量须在 1 到 {MAX_CARDS} 之间"
ERR_REPLY_NO_CARD_LINE = "模型回复缺少 CARD: 判定行，无法确定属于哪张角色卡"
ERR_REPLY_NO_SCENE = "模型回复缺少画面段"

# -- Card-identification + outfit-check protocol (scene reply header) ----------
CARD_PREFIX = "CARD:"
CARD_NONE = "NONE"
OUTFIT_PREFIX = "OUTFIT:"
OUTFIT_SAME = "SAME"
OUTFIT_REFERENCE_MISSING = (
    "(no official outfit on record — answer SAME and describe nothing about clothing)"
)
ROSTER_ENTRY_FMT = (
    'Card {n} — name: {name}; appearance: "{appearance}"; official outfit: "{outfit}"'
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
# inside appearance the hair COLOR is always the first fact after the subject.
# Hats / ribbons / clips are clothing — they must not leak into appearance.
CARD_APPEARANCE_SKILL = """\
Appearance skill (write this block first, before any clothing):
- Hair, in this fixed sub-order and nothing in between: (a) hair color \
first — including two-tone, gradient, inner color, streaks or highlights; \
(b) length; (c) cut and how it is worn (loose, ponytail, twintails, buns, \
braids, side tail); (d) bangs and parting. No mood words.
- Nothing worn on the head (hats, caps, hoods, helmets, headbands, \
ribbons, clips, pins, goggles, headphones) belongs here — those are \
clothing.
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
after it. Headwear and hair ornaments may appear only after this opener.
- Full outfit from head to shoes; no visible layer or small worn item may \
be skipped. Cover in order: headwear and hair ornaments (hat, cap, hood, \
ribbon, clip, pin, flower, headband, goggles, headphones), neckwear \
(choker, collar, necklace, tie, scarf), outerwear, inner layers and tops, \
bottoms, arm wear (sleeves, gloves, arm bands, bracelets), waist wear \
(belts, sashes, harnesses, crossed straps, garter straps), hosiery and \
leg straps, footwear, bags, then remaining jewelry and worn accessories.
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

Lead sentence must be one complete grammatical sentence that starts with \
this exact subject: "{opening}"
Write: "{opening} has <color> hair …" using verbs. The first fact after \
the subject is the hair color.

Never write comma-separated tag fragments such as "{name}, brown hair, \
long, wavy"; write flowing prose with verbs.

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
- Comma-separated tag lists instead of grammatical sentences.

Output format: one complete English paragraph. No title, bullets, notes, \
or tag lists.
The paragraph MUST finish all appearance facts before the first garment \
word (jacket, dress, skirt, shirt, socks, shoes, …). A paragraph that omits \
a visible tattoo, an exposed underlayer, a strap, or a garment pattern is \
not acceptable.
"""

# -- Pose/scene skill (SKILL.md, no clothing / appearance) ---------------------
# ``{ROSTER}`` is one ROSTER_ENTRY_FMT line per card; ``{NAMES}`` lists the
# distinct card names (the only proper names the scene may use).
POSE_SCENE_PROMPT = """\
You are an image analysis system. Process visual data and output a fully \
objective English text description. Output English only.

Produce two parts: a header block (one CARD line, then one OUTFIT line per \
matched card), then one clear paragraph of 200–300 words describing the \
provided image.

Rules:

0. Card identification and outfit check (header block)
Character cards on record:
{ROSTER}

Decide which card(s) the figure(s) in this image belong to:
- Match by fixed appearance first — hair color and hair style, eye color, \
skin marks (tattoos, scars, birthmarks), and species traits such as animal \
ears, horns, a tail, or wings — comparing each figure against every card's \
appearance.
- When several cards share the same appearance (one character in several \
costumes), pick the card whose official outfit matches the visible main \
garments (top, bottom, dress, outerwear) in garment type and color.
- When a figure's appearance matches a card but none of the official \
outfits match, still pick the card with the closest outfit and report the \
clothing actually worn in that card's OUTFIT line.
- A figure whose appearance matches no card belongs to no card. If no \
figure in the image matches any card, the whole answer is exactly one \
line: CARD: NONE

First header line: "CARD: " followed by the matched card numbers, \
comma-separated in ascending order (e.g. "CARD: 2" or "CARD: 1, 3"), or \
"CARD: NONE".
Then one line per matched card, in the same order, starting with \
"OUTFIT <n>: " where <n> is that card number.
- Write "OUTFIT <n>: SAME" only when every garment and accessory listed in \
that card's official outfit is worn on the body exactly as the card \
describes it. Parts hidden by the crop, by occlusion, or simply out of \
frame (missing shoes, hidden legs) never make the outfit different: "not \
visible" is not "not worn".
- It is NOT SAME when any listed item is visibly in a different state: held \
in the hand, taken off, hanging from the arm or shoulder, tied around the \
waist, lying beside the figure, open or unbuttoned where the card describes \
it closed, sleeves pushed up, hat / coat / gloves / shoes removed, or an \
accessory that would be visible at this framing is missing. Also NOT SAME \
when a main garment differs in type or color, or the character is in \
another costume (swimsuit, uniform, casual clothes, sleepwear, formal wear, \
armor).
- When NOT SAME, write "OUTFIT <n>: " followed by one to three sentences \
describing the clothing as actually seen on that figure, in this order: \
headwear, neckwear, outerwear, tops, bottoms, arm wear, waist wear, hosiery \
and leg straps, footwear, bags, jewelry and accessories. Start from the \
official outfit: keep every item that is still worn as described, state the \
actual state and position of each item that was removed or displaced (for \
example "holds the black hat in her right hand", "the purple coat is off and \
draped over her left arm", "the jacket hangs open"), and leave out items \
that are absent. Give each garment its color and any visible pattern, \
print, trim, or construction (pleated, lace, sheer, cropped, zipper, \
buckle, lacing). Name exposed underlayers as their own item. Do not invent \
garments that are not visible. Start the first sentence with "<card name> \
wears", "She wears", or "He wears". Do not repeat hair, eyes, skin marks, \
or any appearance fact. No pose, no expression, no setting in these lines.

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
Refer to each matched figure only by the name of the card it matched \
({NAMES}). Do not invent other proper names. Do not identify real public \
figures or fictional characters by their known names; a figure that matches \
no card is described only as an unnamed person or secondary figure without \
identity labels.

4. Required content (include all of the following that are visible)
- Each matched figure ({NAMES}): demeanor / facial expression / gaze \
direction; body posture; position in the frame and relative to other \
people, creatures, or objects; actions and interactions, including \
interactions between matched figures.
- Other people, creatures, or figures: demeanor, posture, position, and \
actions only — same bans as for the matched figures.
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
any other appearance or costume descriptors for the matched figures or \
anyone else. Clothing belongs only in the OUTFIT lines of rule 0.
- Subjective interpretation: meaning, symbolism, intent, narrative, praise, \
or criticism.
- Art-style labels or medium labels (e.g. impressionism, surrealism, cartoon, \
pixel art, pencil drawing, oil painting, line art). Do not classify render \
style.

6. Output format
First the header block from rule 0: the CARD line, then one "OUTFIT <n>: \
SAME" or "OUTFIT <n>: <clothing>" line per matched card. Then exactly one \
blank line. Then one complete English paragraph of 200–300 words. No \
headings, summaries, introductions, or trailing notes anywhere else. When \
the CARD line is NONE, output only that line and nothing else.
"""

# Extra user-line so three concurrent card calls do not collapse to one wording.
CARD_VARIANT_HINTS: tuple[str, ...] = (
    "This is alternative 1 of 3: after the opening, state hair color first "
    "in a complete sentence, then give more hair detail (part, ties, "
    "highlights), then clothing. Never start with garments. Write flowing "
    "prose; no tag-like comma lists.",
    "This is alternative 2 of 3: after the opening and hair color, give more "
    "eye/pupil detail and name every visible tattoo or skin mark with its "
    "position, then clothing. Rephrase; do not copy another alternative. "
    "Write flowing prose; no tag-like comma lists.",
    "This is alternative 3 of 3: keep appearance-first order, then be more "
    "specific about clothing construction, straps, jewelry, and patterns. "
    "Do not lead with footwear. Write flowing prose; no tag-like comma lists.",
)

# Placeholders a custom template should keep so fill-in still works.
CARD_PLACEHOLDERS: tuple[str, ...] = ("{opening}", "{name}")
SCENE_PLACEHOLDERS: tuple[str, ...] = ("{ROSTER}", "{NAMES}")

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


def effective_card_template(custom: str) -> str:
    """Return ``custom`` when it has content, else the built-in card skill."""
    stripped = custom.strip()
    return stripped if stripped else CHARACTER_CARD_PROMPT


def effective_scene_template(custom: str) -> str:
    """Return ``custom`` when it has content, else the built-in scene skill."""
    stripped = custom.strip()
    return stripped if stripped else POSE_SCENE_PROMPT


def missing_placeholders(template: str, required: tuple[str, ...]) -> tuple[str, ...]:
    """Placeholders from ``required`` that do not appear in ``template``."""
    return tuple(token for token in required if token not in template)


def build_card_prompt(
    name: str,
    series: str = "",
    *,
    variant: int = 0,
    template: str | None = None,
) -> str:
    """User prompt for one character-card alternative (0..2).

    ``template`` empty/None uses :data:`CHARACTER_CARD_PROMPT`. The variant
    hint is always appended after the filled template.
    """
    cleaned = name.strip()
    work = series.strip()
    opening = f"{cleaned} from {work}" if work else cleaned
    body = CHARACTER_CARD_PROMPT if not (template or "").strip() else template
    filled = _fill(body, name=cleaned, opening=opening)
    hint = CARD_VARIANT_HINTS[max(0, min(variant, len(CARD_VARIANT_HINTS) - 1))]
    return f"{filled.rstrip()}\n\n{hint}"


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


@dataclass(frozen=True)
class CharacterCard:
    """One locked character card of a roster (name + series + card text)."""

    name: str
    series: str
    text: str
    parts: CardParts

    @classmethod
    def from_text(cls, name: str, series: str, text: str) -> CharacterCard:
        """Build a card from its final English text, splitting the outfit."""
        cleaned = text.strip()
        return cls(name.strip(), series.strip(), cleaned, split_card(cleaned, name))


@dataclass(frozen=True)
class CardRoster:
    """The 1..:data:`MAX_CARDS` cards one CHA标注 run identifies against."""

    cards: tuple[CharacterCard, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.cards) <= MAX_CARDS:
            raise ValidationError(ERR_ROSTER_SIZE)

    def __len__(self) -> int:
        return len(self.cards)

    def names(self) -> tuple[str, ...]:
        """Distinct card names in roster order (several cards may share one)."""
        seen: list[str] = []
        for card in self.cards:
            if card.name and card.name not in seen:
                seen.append(card.name)
        return tuple(seen)


@dataclass(frozen=True)
class LayeredVerdict:
    """Parsed per-image reply.

    ``matched`` holds 0-based roster indices in ascending order (empty means
    ``CARD: NONE``); ``outfits[i]`` belongs to ``matched[i]`` and is ``None``
    for ``SAME``. ``scene`` is the paragraph after the header block.
    """

    matched: tuple[int, ...]
    outfits: tuple[str | None, ...]
    scene: str


@dataclass(frozen=True)
class LayeredSummary:
    """Outcome of one CHA标注 batch, shown in the closing card dialog.

    ``matched_counts[i]`` is how many images included ``roster.cards[i]``;
    ``none_keys`` were skipped (``CARD: NONE``); ``failed_keys`` raised or
    returned unusable replies; ``cancelled`` mirrors the batch status.
    """

    roster: CardRoster
    matched_counts: tuple[int, ...]
    none_keys: tuple[str, ...]
    failed_keys: tuple[str, ...]
    cancelled: bool = False


def build_scene_prompt(roster: CardRoster, template: str | None = None) -> str:
    """User prompt for the per-image CARD/OUTFIT header + pose/scene paragraph.

    Every card is listed with its name, appearance block and official outfit
    (see :func:`split_card`); a card without a clothing block is shown with
    :data:`OUTFIT_REFERENCE_MISSING` so the model answers SAME for it.
    ``template`` empty/None uses :data:`POSE_SCENE_PROMPT`.
    """
    lines = [
        ROSTER_ENTRY_FMT.format(
            n=index + 1,
            name=card.name,
            appearance=" ".join(card.parts.appearance.split()),
            outfit=" ".join(card.parts.outfit.split()) or OUTFIT_REFERENCE_MISSING,
        )
        for index, card in enumerate(roster.cards)
    ]
    body = POSE_SCENE_PROMPT if not (template or "").strip() else template
    return _fill(body, ROSTER="\n".join(lines), NAMES=", ".join(roster.names()))


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
    return (_outfit_verdict(body), scene)


def assemble_layered(card: str, parts: CardParts, outfit: str | None, scene: str) -> str:
    """Card (or appearance + rewritten outfit) + blank line + scene."""
    if outfit is None or not outfit.strip() or not parts.outfit.strip():
        return assemble_caption(card, scene)
    return assemble_caption(f"{parts.appearance.rstrip()} {outfit.strip()}", scene)


def _card_body(card: CharacterCard, outfit: str | None) -> str:
    """Card text verbatim, or its appearance block + a rewritten outfit."""
    if outfit is None or not outfit.strip() or not card.parts.outfit.strip():
        return card.text.strip()
    return f"{card.parts.appearance.rstrip()} {outfit.strip()}"


def _outfit_verdict(body: str) -> str | None:
    """``None`` for SAME (any case, punctuation / trailing word tolerated)."""
    verdict = re.sub(r"[^A-Za-z ]", "", body).strip().upper()
    if not body.strip() or verdict == OUTFIT_SAME or verdict.startswith(OUTFIT_SAME + " "):
        return None
    return " ".join(body.split())


_CARD_LINE = re.compile(rf"^\s*{CARD_PREFIX[:-1]}\s*:\s*(.*)$", re.IGNORECASE)
_OUTFIT_LINE = re.compile(
    rf"^\s*{OUTFIT_PREFIX[:-1]}\s*(\d+)?\s*:\s*(.*)$", re.IGNORECASE
)


def _parse_header(head: str, roster_size: int) -> tuple[tuple[int, ...] | None, list[tuple[int | None, str]]]:
    """Read the CARD line and OUTFIT entries of the header block.

    Returns ``(matched_or_None, [(card_number_or_None, body), ...])`` where
    ``matched`` is ``None`` when no CARD line exists. Continuation lines are
    appended to the previous OUTFIT entry (a rewritten outfit may wrap).
    """
    matched: tuple[int, ...] | None = None
    outfits: list[tuple[int | None, str]] = []
    for line in head.splitlines():
        card_match = _CARD_LINE.match(line)
        if card_match and matched is None:
            body = card_match.group(1)
            if re.sub(r"[^A-Za-z]", "", body).upper() == CARD_NONE:
                matched = ()
            else:
                numbers = sorted(
                    {int(n) - 1 for n in re.findall(r"\d+", body) if 1 <= int(n) <= roster_size}
                )
                matched = tuple(numbers)
            continue
        outfit_match = _OUTFIT_LINE.match(line)
        if outfit_match:
            number = outfit_match.group(1)
            outfits.append((int(number) if number else None, outfit_match.group(2)))
        elif outfits and line.strip():
            index, body = outfits[-1]
            outfits[-1] = (index, f"{body} {line.strip()}")
    return matched, outfits


def parse_layered_reply(text: str, roster_size: int) -> LayeredVerdict:
    """Split a per-image reply into its :class:`LayeredVerdict`.

    The first blank-line-separated block is the header when it starts with
    ``CARD:`` or ``OUTFIT:``; everything after it is the scene. Tolerances:
    ``CARD: NONE`` → no match; numbers outside ``1..roster_size`` are
    dropped; an unnumbered ``OUTFIT:`` applies to the single matched card;
    a reply without any header is the legacy single-card shape and is only
    accepted when ``roster_size == 1`` (else :class:`LLMOutputError`).
    """
    stripped = text.strip()
    if not stripped:
        return LayeredVerdict((), (), "")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", stripped) if block.strip()]
    head_upper = blocks[0].upper().lstrip()
    if not (head_upper.startswith(CARD_PREFIX[:-1]) or head_upper.startswith(OUTFIT_PREFIX[:-1])):
        if roster_size != 1:
            raise LLMOutputError(ERR_REPLY_NO_CARD_LINE)
        return LayeredVerdict((0,), (None,), stripped)
    matched, entries = _parse_header(blocks[0], roster_size)
    scene = "\n\n".join(blocks[1:]).strip()
    if matched is None:
        if roster_size != 1:
            raise LLMOutputError(ERR_REPLY_NO_CARD_LINE)
        matched = (0,)
    if not matched:
        return LayeredVerdict((), (), scene)
    by_card: dict[int, str | None] = {index: None for index in matched}
    for number, body in entries:
        if number is None:
            if len(matched) == 1:
                by_card[matched[0]] = _outfit_verdict(body)
            continue
        index = number - 1
        if index in by_card:
            by_card[index] = _outfit_verdict(body)
    return LayeredVerdict(matched, tuple(by_card[index] for index in matched), scene)


def assemble_roster_caption(roster: CardRoster, verdict: LayeredVerdict) -> str | None:
    """Matched card(s) + blank line + scene; ``None`` when nothing matched.

    Each matched card contributes its text verbatim (SAME / no clothing
    block) or ``appearance + rewritten outfit``. A matched reply without a
    scene paragraph raises :class:`LLMOutputError` so the batch item fails
    instead of writing cards alone.
    """
    if not verdict.matched:
        return None
    if not verdict.scene.strip():
        raise LLMOutputError(ERR_REPLY_NO_SCENE)
    bodies = [
        _card_body(roster.cards[index], outfit)
        for index, outfit in zip(verdict.matched, verdict.outfits)
        if 0 <= index < len(roster.cards)
    ]
    return "\n\n".join([*bodies, verdict.scene.strip()])


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
