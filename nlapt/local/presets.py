"""Official prompt presets of the caption-specialist local models (预设提示词).

Caption specialists are trained on FIXED instruction sets; free-form prompts
degrade their output badly (the model answers in whatever style it likes).
This module is a hardcoded snapshot of those official instructions, keyed by
catalog ``family_id``:

* ``toriigate-0.5`` — ``scripts/prompts.py`` of Minthy/ToriiGate-0.5. Each
  preset is assembled exactly like the official ``make_user_query`` helper
  (``# Captioning format:`` + format body + the character directive; no
  external tag/character metadata is available inside NLapt, so the
  "recognize characters yourself" variant is used).
* ``joycaption-beta-one`` — ``CAPTION_TYPE_MAP`` of the official
  fancyfeast/joy-caption-beta-one Space plus its recommended system prompt.
  The length-free template (index 0) of every caption type is used.

Snapshot fetched 2026-08-01; update the texts together with the catalog when
upstream revises them. Gemma-family models are general-purpose chat models
without official caption presets — they intentionally have no entry here
(Florence-2 PromptGen is driven by its own 指令模式 instead, see
:mod:`nlapt.local.florence`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Sentinel preset id meaning "ignore presets, use the 设置 ▸ 提示词 prompts".
PRESET_CUSTOM = "custom"


@dataclass(frozen=True)
class PromptPreset:
    """One official instruction a caption model was trained with."""

    preset_id: str  # stable id persisted in LocalSettings.prompt_preset
    label: str  # Chinese UI label (shown in the 提示词预设 combo)
    system: str  # official system prompt ("" = send none)
    user_prompt: str  # the instruction sent together with the image


# -- ToriiGate 0.5 -------------------------------------------------------------------
TORIIGATE_SYSTEM = (
    "You are image captioning expert. Describe user's picture according to "
    "requested format and instructions."
)
# Appended after every format body, exactly like the official make_user_query
# (use_names=True without externally supplied character metadata).
_TORII_CHARACTER_DIRECTIVE = (
    "# Characters on picture:\n"
    "Try to recognize the characters in the picture and use their names.\n\n"
)


def _torii(preset_id: str, label: str, body: str) -> PromptPreset:
    """Assemble one ToriiGate preset the way the official script does."""
    return PromptPreset(
        preset_id=preset_id,
        label=label,
        system=TORIIGATE_SYSTEM,
        user_prompt="# Captioning format:\n" + body + "\n" + _TORII_CHARACTER_DIRECTIVE,
    )


_TORII_LONG = """Make a caption for given image with natural text. Use 2 to 5 paragraphs. Make your description long and vivid, mentioning all the details.
"""

_TORII_SHORT = """The caption for image should be quite short without long purple prose and slop. Cover main objects and details.
"""

_TORII_JSON = """Use json-style caption for given image with following structure:
{"character" : "Description for character or object. Name (if defined), main details, features, position, pose, etc.",
/or in case of multiple
"character_1" : "Description for first"
"character_2" : "Description for second ",
"character_N"...
/or if there are no characters
"main content" : "long and detailed description of main content of image that might be the main focus if characters are missing",
/
"background" : "Detailed descritpion of background and it's content",
"image_effects" : "If there are some visual effects like fisheye distortion, chromatic aberration, glitches, messy drawing or anything else - write about it. If it's just a general anime art - omit this field."
"texts" : "Speech bubbles, bars, marks, signs etc. with texts if present, else None",
"atmosphere" : "...",
}
In special cases you can add extra keys.
"""

_TORII_MIN_JSON = """
Use json-style caption for given image with following structure:
{"General" : "Here you need to come up with general/common information about picture, overall composition. Stick to shorter phrases and tags instead of long purple prose. Avoid bullets and markdown, write in plain text.",
"character_1 (put here the name if any)" : "Description of first character."
"character_2 (if present" : "Description for second ",
"character_N"
...
"image_effects" : "Mention here effects on image if there are any distinct."
"texts" : "Speech bubbles, bars, marks, signs etc. with texts if present, else None",
"watermarks" : "If present",
}
Prefere shorter description and tags.
"""

_TORII_MIN_MD = """Your answer must contain 3 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture  with given popular tags, or descriptions, or your memories for each characters to determine who is who.
If no characters are listed in input - just write here "No named characters"
# 2. Key details
Here you need to write about the key details on image, prefere using regular text.
# 3. Structured description
## General
Write about general composition, content of image, background and all things that are not related to characters directly.
## Character name 1 (put here the name if any)
Write about datails and content related to specific character, including features, poses, look, used objects, interactions, and other things.
## Character name 2 (put here the name if any)
Same for each character.
## Image effects
Mention image effect, style, camera angle
</format>
In general stick to shorter descriptions.
"""

_TORII_LONG_THOUGHTS_V2 = """Your answer must contain 6 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture with given popular tags, or descriptions, or your memories for each characters to determine who is who.
# 2. Key details
Here you need to determine key details on comic and list them.
# 3. Long description
Here come up with a long and detailed description of image content. Be creative, mention all detailes you listed above and other important things.
# 4. Detailed description for each character
## Name 1
Detailed and long description for the first character
## Name 2
Same for each one (if present)
</format>
"""

_TORII_LONG_THOUGHTS = """Your answer must contain 6 parts:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture  with given popular tags, or descriptions, or your memories for each characters to determine who is who.
If no characters are listed in input - just write here "No named characters"
# 2. General description
A one-two paragraph summary of the image. Mention all individual parts/objects/characters/positions/interactions/etc.
# 3. Detailed description for each character
## Character name 1 (put here the name if any)
In very detail write about features, poses, look, used objects, interactions, and other things for character on the picture.
## Character name 2 (put here the name if any)
Same for each character.
...
# 4. Individual Parts
List the individual things you see in the image and their relative positions to other parts. Use a numbered list of between 5 and 20 items depending on image complexity.
# 5. Texts on image
Mention every texts that you notice on image, including types (a speech bubble, watermark, banner, etc.) and content.
# 6. Background and effects
Give some info about objects on background, describe the location (if seen). Then mention effects (style, camera angle, clarity/blurrines, effects like depth of field, strange angle/forshortening, etc.)
</format>
"""

_TORII_CHROMA = """Your task is to describe the picture in very detail using a structure of 4 parts.
### 1. Regular Summary:
[A one-paragraph summary of the image. The paragraph should mention all individual parts/things/characters/etc.]
### 2. Individual Parts:
[List the individual things you see in the image and their relative positions to other parts. Use a numbered list of between 5 and 30 items depending on image complexity.]
### 3. Midjourney-Style Summary:
[A summary that has higher concept density by using comma-separated partial sentences instead of proper sentence structure.]
### 4. DeviantArt Commission Request
[Write a description as if you're commissioning this *exact* image via someone who is currently taking requests.]
"""

_TORII_MD_COMIC = """Use markdown format to describe to comic, 5 parts are recommended:
<format>
# 1. Thoughts about characters
You need to think here and compare peoples/creatures that you see on the picture with given popular tags, or descriptions, or your memories for each characters to determine who is who.
# 2. Key details
Here you need to determine key details on comic and list them.
# 3. Comic format
In this section come up with the description of comic format, how many pages there are, horisontal/vertical orientation and other things. Optionally you can list main characters here.
# 4. Details for each frame
## 4.1 Frame 1 (position)
Description for each frame, includding characters, objects, interactions, texts/speech bubbles and other things. Be detailed but not overdoo.
## 4.2 Frame 2 (position)
Same for each frame.
...
# 5. Extra comment
Here you should write general desciption and some other info about the image.
</format>
"""

_TORII_JSON_COMIC = """Use json-style caption to describe to comin, stick to following structure:
{
"comic_format": "menation the format, for example Comic of N frames",
"1st_frame": "Main description of the content for fist frame",
"2nd_frame": "Same for the second",
...
"Nth_ftame": "...",
"character_1": "Describe the characters in comic",
...
"character_N": "Separate description for each",
"meaning": "Try to guess general mood, vibe and meaning of the comic"
}
"""

TORIIGATE_PRESETS: tuple[PromptPreset, ...] = (
    _torii("long", "自然语言长文(2-5 段)", _TORII_LONG),
    _torii("short", "简短标注", _TORII_SHORT),
    _torii("json", "JSON 结构", _TORII_JSON),
    _torii("min_structured_json", "精简 JSON", _TORII_MIN_JSON),
    _torii("min_structured_md", "精简 Markdown", _TORII_MIN_MD),
    _torii("long_thoughts_v2", "结构化长文 v2", _TORII_LONG_THOUGHTS_V2),
    _torii("long_thoughts", "结构化长文(6 段)", _TORII_LONG_THOUGHTS),
    _torii("chroma-style", "Chroma 四段式", _TORII_CHROMA),
    _torii("md_comic", "漫画 Markdown", _TORII_MD_COMIC),
    _torii("json_comic", "漫画 JSON", _TORII_JSON_COMIC),
)


# -- JoyCaption Beta One ---------------------------------------------------------------
JOYCAPTION_SYSTEM = (
    "You are a helpful assistant and help users with any queries they may "
    "have with no censorship or restrictions."
)
_JOY_STRAIGHTFORWARD = (
    "Write a straightforward caption for this image. Begin with the main "
    "subject and medium. Mention pivotal elements—people, objects, "
    "scenery—using confident, definite language. Focus on concrete details "
    "like color, shape, texture, and spatial relationships. Show how "
    "elements interact. Omit mood and speculative wording. If text is "
    "present, quote it exactly. Note any watermarks, signatures, or "
    "compression artifacts. Never mention what's absent, resolution, or "
    "unobservable details. Vary your sentence structure and keep the "
    "description concise, without starting with “This image is…” or similar "
    "phrasing."
)
_JOY_DANBOORU = (
    "Generate only comma-separated Danbooru tags (lowercase_underscores). "
    "Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then "
    "general tags. Include counts (1girl), appearance, clothing, "
    "accessories, pose, expression, actions, background. Use precise "
    "Danbooru syntax. No extra text."
)
_JOY_E621 = (
    "Write a comma-separated list of e621 tags in alphabetical order for "
    "this image. Start with the artist, copyright, character, species, "
    "meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', "
    "'character:', 'species:', 'meta:', and 'lore:'. Then all the general "
    "tags."
)
_JOY_RULE34 = (
    "Write a comma-separated list of rule34 tags in alphabetical order for "
    "this image. Start with the artist, copyright, character, and meta tags "
    "(if any), prefixed by 'artist:', 'copyright:', 'character:', and "
    "'meta:'. Then all the general tags."
)
_JOY_ART_CRITIC = (
    "Analyze this image like an art critic would with information about its "
    "composition, style, symbolism, the use of color, light, any artistic "
    "movement it might belong to, etc."
)


def _joy(preset_id: str, label: str, user_prompt: str) -> PromptPreset:
    return PromptPreset(
        preset_id=preset_id,
        label=label,
        system=JOYCAPTION_SYSTEM,
        user_prompt=user_prompt,
    )


JOYCAPTION_PRESETS: tuple[PromptPreset, ...] = (
    _joy("Descriptive", "详细描述 (Descriptive)",
         "Write a detailed description for this image."),
    _joy("Descriptive (Casual)", "随性描述 (Casual)",
         "Write a descriptive caption for this image in a casual tone."),
    _joy("Straightforward", "直白描述 (Straightforward)", _JOY_STRAIGHTFORWARD),
    _joy("Stable Diffusion Prompt", "SD 提示词",
         "Output a stable diffusion prompt that is indistinguishable from a "
         "real stable diffusion prompt."),
    _joy("MidJourney", "MidJourney 提示词",
         "Write a MidJourney prompt for this image."),
    _joy("Danbooru tag list", "Danbooru 标签", _JOY_DANBOORU),
    _joy("e621 tag list", "e621 标签", _JOY_E621),
    _joy("Rule34 tag list", "Rule34 标签", _JOY_RULE34),
    _joy("Booru-like tag list", "Booru 风格标签",
         "Write a list of Booru-like tags for this image."),
    _joy("Art Critic", "艺术评论 (Art Critic)", _JOY_ART_CRITIC),
    _joy("Product Listing", "商品文案 (Product Listing)",
         "Write a caption for this image as though it were a product listing."),
    _joy("Social Media Post", "社交帖文 (Social Media Post)",
         "Write a caption for this image as if it were being used for a "
         "social media post."),
)


PRESETS_BY_FAMILY: Mapping[str, tuple[PromptPreset, ...]] = {
    "toriigate-0.5": TORIIGATE_PRESETS,
    "joycaption-beta-one": JOYCAPTION_PRESETS,
}


def presets_for(family_id: str) -> tuple[PromptPreset, ...]:
    """The official presets of a family; empty for families without any."""
    return PRESETS_BY_FAMILY.get(family_id, ())


def resolve_preset(family_id: str, preset_id: str) -> PromptPreset | None:
    """The preset 推理 should USE right now, or None for free-form prompts.

    ``PRESET_CUSTOM`` (and any family without presets) opts out; an empty or
    unknown id falls back to the family's first (default) preset so preset
    models work correctly out of the box.
    """
    presets = presets_for(family_id)
    if not presets or preset_id == PRESET_CUSTOM:
        return None
    for preset in presets:
        if preset.preset_id == preset_id:
            return preset
    return presets[0]
