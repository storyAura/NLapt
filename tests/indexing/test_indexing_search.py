"""Tests for nlapt.indexing.search_index (spec 4.2)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from nlapt.core.errors import ValidationError
from nlapt.indexing import SearchIndex


@pytest.fixture()
def index() -> SearchIndex:
    idx = SearchIndex()
    idx.build(
        {
            "a/cat.jpg": "a cat sitting on a mat",
            "b/dog.png": "a happy DOG running",
            "c/1girl.webp": "portrait of a boy",
            "d/bird.jpg": "",
        }
    )
    return idx


class TestQueryBasics:
    def test_blank_query_returns_all_keys_in_order(self, index: SearchIndex) -> None:
        assert index.query("") == ("a/cat.jpg", "b/dog.png", "c/1girl.webp", "d/bird.jpg")
        assert index.query("   ") == index.query("")

    def test_case_insensitive_caption_substring(self, index: SearchIndex) -> None:
        assert index.query("DOG") == ("b/dog.png",)
        assert index.query("Cat") == ("a/cat.jpg",)

    def test_substring_not_whole_word(self, index: SearchIndex) -> None:
        assert index.query("sitt") == ("a/cat.jpg",)

    def test_positive_term_matches_filename(self, index: SearchIndex) -> None:
        # "bird" appears only in the filename, caption is empty.
        assert index.query("bird") == ("d/bird.jpg",)

    def test_positive_term_does_not_match_directory(self, index: SearchIndex) -> None:
        # Directory segments are not part of the searched filename.
        assert index.query("a/") == ()

    def test_multiple_terms_are_anded(self, index: SearchIndex) -> None:
        assert index.query("cat mat") == ("a/cat.jpg",)
        assert index.query("cat running") == ()

    def test_no_match_returns_empty(self, index: SearchIndex) -> None:
        assert index.query("zebra") == ()


class TestNegation:
    def test_negation_checks_caption_only(self, index: SearchIndex) -> None:
        # "1girl" is in the filename of c/ but not in its caption, so the
        # negated query must still return that key.
        result = index.query("-1girl")
        assert "c/1girl.webp" in result

    def test_negation_excludes_caption_matches(self, index: SearchIndex) -> None:
        result = index.query("-cat")
        assert "a/cat.jpg" not in result
        assert "b/dog.png" in result

    def test_negation_is_case_insensitive(self, index: SearchIndex) -> None:
        assert "b/dog.png" not in index.query("-DOG")

    def test_negation_combined_with_positive(self, index: SearchIndex) -> None:
        assert index.query("a -cat") == ("b/dog.png", "c/1girl.webp")

    def test_lone_dash_is_ignored(self, index: SearchIndex) -> None:
        assert index.query("-") == index.query("")


class TestMutation:
    def test_update_reflected_immediately(self, index: SearchIndex) -> None:
        index.update("a/cat.jpg", "now a zebra")
        assert index.query("zebra") == ("a/cat.jpg",)
        assert index.query("cat") == ("a/cat.jpg",)  # filename still matches

    def test_update_preserves_index_order(self, index: SearchIndex) -> None:
        index.update("a/cat.jpg", "changed")
        assert index.query("")[0] == "a/cat.jpg"

    def test_update_unknown_key_inserts_at_end(self, index: SearchIndex) -> None:
        index.update("e/new.jpg", "fresh caption")
        assert index.query("")[-1] == "e/new.jpg"
        assert index.query("fresh") == ("e/new.jpg",)

    def test_remove_reflected_immediately(self, index: SearchIndex) -> None:
        index.remove("b/dog.png")
        assert "b/dog.png" not in index.query("")
        assert index.query("dog") == ()

    def test_remove_unknown_key_is_noop(self, index: SearchIndex) -> None:
        index.remove("missing/key.jpg")
        assert len(index.query("")) == 4

    def test_build_replaces_previous_entries(self, index: SearchIndex) -> None:
        index.build({"x/only.jpg": "solo"})
        assert index.query("") == ("x/only.jpg",)


class TestValidation:
    def test_build_rejects_non_mapping(self) -> None:
        with pytest.raises(ValidationError):
            SearchIndex().build([("k", "v")])  # type: ignore[arg-type]

    def test_build_rejects_non_string_caption(self) -> None:
        with pytest.raises(ValidationError):
            SearchIndex().build({"k.jpg": 42})  # type: ignore[dict-item]

    def test_update_rejects_empty_key(self, index: SearchIndex) -> None:
        with pytest.raises(ValidationError):
            index.update("", "text")

    def test_query_rejects_non_string(self, index: SearchIndex) -> None:
        with pytest.raises(ValidationError):
            index.query(None)  # type: ignore[arg-type]


class TestThreadSafety:
    def test_concurrent_updates_and_queries(self) -> None:
        idx = SearchIndex()
        idx.build({f"f{i}.jpg": f"caption {i}" for i in range(20)})
        iterations = 200

        def writer(worker: int) -> None:
            for i in range(iterations):
                idx.update(f"f{i % 20}.jpg", f"caption {worker}-{i}")

        def reader(_: int) -> None:
            for _ in range(iterations):
                idx.query("caption")
                idx.query("-caption")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(writer, w) for w in range(4)]
            futures += [pool.submit(reader, r) for r in range(4)]
            for future in futures:
                future.result()  # re-raises any worker exception

        assert len(idx.query("")) == 20
        assert idx.query("caption") == idx.query("")
