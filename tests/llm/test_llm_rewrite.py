"""Tests for nlapt.llm.rewrite: prompts, model routing, dry runs."""

from __future__ import annotations

import random

import pytest

from nlapt.core.config import LLMProfile
from nlapt.core.errors import LLMConfigError, LLMOutputError, ValidationError
from nlapt.llm.cleaning import OUTPUT_CONSTRAINT
from nlapt.llm.mock import MockLLMClient
from nlapt.llm.rewrite import RewriteResult, RewriteService, RewriteSpec, RewriteType

IMAGE_BYTES = b"\xff\xd8\xfffake-jpeg"
SAMPLE_SEED = 42


def make_profile(**overrides: object) -> LLMProfile:
    defaults: dict = {
        "name": "p",
        "api_type": "openai",
        "base_url": "https://api.test/v1",
        "text_model": "text-model",
        "vision_model": "vision-model",
        "system_prompt": "sys",
    }
    defaults.update(overrides)
    return LLMProfile(**defaults)


def make_service(
    *,
    text_client: MockLLMClient | None = None,
    vision_client: MockLLMClient | None = None,
    profile: LLMProfile | None = None,
    trigger: str = "",
) -> RewriteService:
    return RewriteService(
        text_client=text_client or MockLLMClient(["result text"]),
        vision_client=vision_client,
        profile=profile or make_profile(),
        trigger=trigger,
    )


class TestBuildPrompt:
    def test_polish_prompt_contains_caption_and_constraint(self) -> None:
        service = make_service()
        prompt = service.build_prompt(
            RewriteSpec(type=RewriteType.POLISH), caption="a red fox", filename="a.png"
        )
        assert "a red fox" in prompt
        assert prompt.endswith(OUTPUT_CONSTRAINT)

    def test_condense_substitutes_target_tokens(self) -> None:
        service = make_service()
        prompt = service.build_prompt(
            RewriteSpec(type=RewriteType.CONDENSE, target_tokens=42),
            caption="c",
            filename="f",
        )
        assert "42" in prompt
        assert "{target_tokens}" not in prompt

    def test_condense_invalid_target_tokens_raises(self) -> None:
        service = make_service()
        with pytest.raises(ValidationError):
            service.build_prompt(
                RewriteSpec(type=RewriteType.CONDENSE, target_tokens=0),
                caption="c",
                filename="f",
            )

    def test_custom_without_instruction_raises(self) -> None:
        service = make_service()
        with pytest.raises(ValidationError):
            service.build_prompt(
                RewriteSpec(type=RewriteType.CUSTOM), caption="c", filename="f"
            )

    def test_custom_instruction_used_and_caption_appended(self) -> None:
        service = make_service()
        prompt = service.build_prompt(
            RewriteSpec(type=RewriteType.CUSTOM, custom_instruction="Make it poetic."),
            caption="a red fox",
            filename="f",
        )
        assert "Make it poetic." in prompt
        assert "a red fox" in prompt
        assert prompt.endswith(OUTPUT_CONSTRAINT)

    def test_custom_instruction_with_caption_placeholder(self) -> None:
        service = make_service()
        prompt = service.build_prompt(
            RewriteSpec(
                type=RewriteType.CUSTOM,
                custom_instruction="Shorten this: {caption}",
            ),
            caption="a red fox",
            filename="f",
        )
        assert "Shorten this: a red fox" in prompt
        # Caption must not be appended a second time.
        assert prompt.count("a red fox") == 1

    def test_template_override_wins(self) -> None:
        service = make_service()
        prompt = service.build_prompt(
            RewriteSpec(
                type=RewriteType.POLISH, template_override="OVERRIDE {caption}"
            ),
            caption="a red fox",
            filename="f",
        )
        assert prompt.startswith("OVERRIDE a red fox")

    def test_trigger_variable_rendered(self) -> None:
        service = make_service(trigger="minahamu")
        prompt = service.build_prompt(
            RewriteSpec(
                type=RewriteType.POLISH,
                template_override="trigger={trigger} cap={caption}",
            ),
            caption="c",
            filename="f",
        )
        assert "trigger=minahamu" in prompt

    def test_filename_variable_rendered(self) -> None:
        service = make_service()
        prompt = service.build_prompt(
            RewriteSpec(
                type=RewriteType.POLISH, template_override="file={filename} {caption}"
            ),
            caption="c",
            filename="img_007.png",
        )
        assert "file=img_007.png" in prompt


class TestRunOne:
    def test_text_mode_uses_text_client_and_text_model(self) -> None:
        text_client = MockLLMClient(["polished"])
        vision_client = MockLLMClient(["should-not-be-used"])
        service = make_service(text_client=text_client, vision_client=vision_client)
        result = service.run_one(
            RewriteSpec(type=RewriteType.POLISH),
            key="k1",
            caption="a red fox",
            filename="a.png",
        )
        assert result == RewriteResult(key="k1", original="a red fox", result="polished")
        assert text_client.calls == 1
        assert vision_client.calls == 0
        request = text_client.requests[0]
        assert request.model == "text-model"
        assert request.messages[0].images == ()

    def test_vision_mode_uses_vision_client_and_model(self) -> None:
        text_client = MockLLMClient(["should-not-be-used"])
        vision_client = MockLLMClient(["seen"])
        service = make_service(text_client=text_client, vision_client=vision_client)
        result = service.run_one(
            RewriteSpec(type=RewriteType.POLISH, use_vision=True),
            key="k1",
            caption="",
            filename="a.png",
            image=IMAGE_BYTES,
        )
        assert result.result == "seen"
        assert vision_client.calls == 1
        assert text_client.calls == 0
        request = vision_client.requests[0]
        assert request.model == "vision-model"
        assert request.messages[0].images == (IMAGE_BYTES,)

    def test_vision_without_vision_client_raises_config_error(self) -> None:
        service = make_service(vision_client=None)
        with pytest.raises(LLMConfigError):
            service.run_one(
                RewriteSpec(type=RewriteType.POLISH, use_vision=True),
                key="k",
                caption="c",
                filename="f",
                image=IMAGE_BYTES,
            )

    def test_vision_without_vision_model_raises_config_error(self) -> None:
        service = make_service(
            vision_client=MockLLMClient(["x"]),
            profile=make_profile(vision_model=""),
        )
        with pytest.raises(LLMConfigError):
            service.run_one(
                RewriteSpec(type=RewriteType.POLISH, use_vision=True),
                key="k",
                caption="c",
                filename="f",
                image=IMAGE_BYTES,
            )

    def test_vision_without_image_raises_validation_error(self) -> None:
        service = make_service(vision_client=MockLLMClient(["x"]))
        with pytest.raises(ValidationError):
            service.run_one(
                RewriteSpec(type=RewriteType.POLISH, use_vision=True),
                key="k",
                caption="c",
                filename="f",
            )

    def test_text_mode_without_text_model_raises(self) -> None:
        service = make_service(profile=make_profile(text_model=""))
        with pytest.raises(LLMConfigError):
            service.run_one(
                RewriteSpec(type=RewriteType.POLISH),
                key="k",
                caption="c",
                filename="f",
            )

    def test_output_is_cleaned(self) -> None:
        text_client = MockLLMClient(['"Caption: cleaned text"'])
        service = make_service(text_client=text_client)
        result = service.run_one(
            RewriteSpec(type=RewriteType.POLISH), key="k", caption="c", filename="f"
        )
        assert result.result == "cleaned text"

    def test_empty_output_raises_output_error(self) -> None:
        text_client = MockLLMClient(["``` ```"])
        service = make_service(text_client=text_client)
        with pytest.raises(LLMOutputError):
            service.run_one(
                RewriteSpec(type=RewriteType.POLISH), key="k", caption="c", filename="f"
            )


def make_items(count: int) -> list[tuple[str, str, str]]:
    return [(f"k{i}", f"caption {i}", f"file{i}.png") for i in range(count)]


class TestDryRun:
    def test_deterministic_sampling_with_seeded_rng(self) -> None:
        items = make_items(8)
        expected_sample = random.Random(SAMPLE_SEED).sample(items, 3)

        client = MockLLMClient(lambda request: "out")
        service = make_service(text_client=client)
        results = service.dry_run(
            RewriteSpec(type=RewriteType.POLISH),
            items,
            sample_size=3,
            rng=random.Random(SAMPLE_SEED),
        )
        assert [r.key for r in results] == [item[0] for item in expected_sample]
        assert client.calls == 3

    def test_sample_size_larger_than_items_uses_all_in_order(self) -> None:
        items = make_items(3)
        client = MockLLMClient(lambda request: "out")
        service = make_service(text_client=client)
        results = service.dry_run(
            RewriteSpec(type=RewriteType.POLISH), items, sample_size=10
        )
        assert [r.key for r in results] == ["k0", "k1", "k2"]
        assert [r.original for r in results] == ["caption 0", "caption 1", "caption 2"]

    def test_default_sample_size_is_five(self) -> None:
        items = make_items(20)
        client = MockLLMClient(lambda request: "out")
        service = make_service(text_client=client)
        results = service.dry_run(
            RewriteSpec(type=RewriteType.POLISH), items, rng=random.Random(1)
        )
        assert len(results) == 5

    def test_invalid_sample_size_raises(self) -> None:
        service = make_service()
        with pytest.raises(ValidationError):
            service.dry_run(RewriteSpec(type=RewriteType.POLISH), [], sample_size=0)

    def test_vision_dry_run_requires_image_provider(self) -> None:
        service = make_service(vision_client=MockLLMClient(["x"]))
        with pytest.raises(ValidationError):
            service.dry_run(
                RewriteSpec(type=RewriteType.POLISH, use_vision=True), make_items(2)
            )

    def test_vision_dry_run_uses_image_provider(self) -> None:
        provided: list[str] = []

        def provider(key: str) -> bytes:
            provided.append(key)
            return IMAGE_BYTES

        vision_client = MockLLMClient(lambda request: "seen")
        service = make_service(vision_client=vision_client)
        results = service.dry_run(
            RewriteSpec(type=RewriteType.POLISH, use_vision=True),
            make_items(2),
            sample_size=5,
            image_provider=provider,
        )
        assert len(results) == 2
        assert provided == ["k0", "k1"]
        for request in vision_client.requests:
            assert request.model == "vision-model"
            assert request.messages[0].images == (IMAGE_BYTES,)


class TestServiceConstruction:
    def test_invalid_text_client_raises(self) -> None:
        with pytest.raises(ValidationError):
            RewriteService(
                text_client="nope",  # type: ignore[arg-type]
                vision_client=None,
                profile=make_profile(),
            )

    def test_invalid_vision_client_raises(self) -> None:
        with pytest.raises(ValidationError):
            RewriteService(
                text_client=MockLLMClient(["x"]),
                vision_client="nope",  # type: ignore[arg-type]
                profile=make_profile(),
            )

    def test_invalid_profile_raises(self) -> None:
        with pytest.raises(ValidationError):
            RewriteService(
                text_client=MockLLMClient(["x"]),
                vision_client=None,
                profile="nope",  # type: ignore[arg-type]
            )
