"""Tests for app.utils.entity_resolver."""
from __future__ import annotations

from app.utils.entity_resolver import fuzzy_match, levenshtein_distance
from app.utils import fuzzy_match as fuzzy_match_via_package


# ── Levenshtein distance: direct unit tests ────────────────


class TestLevenshteinDistance:
    """The DP implementation must match textbook reference values."""

    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_empty_against_non_empty(self):
        assert levenshtein_distance("", "abc") == 3
        assert levenshtein_distance("abc", "") == 3

    def test_single_substitution(self):
        assert levenshtein_distance("cat", "bat") == 1

    def test_single_insertion(self):
        assert levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("cats", "cat") == 1

    def test_classic_example(self):
        # Standard textbook example: kitten -> sitting = 3.
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3

    def test_is_symmetric(self):
        assert levenshtein_distance("flaw", "lawn") == levenshtein_distance(
            "lawn", "flaw"
        )

    def test_unicode(self):
        # 'café' (4 chars including é) vs 'cafe' - one substitution.
        assert levenshtein_distance("café", "cafe") == 1


# ── Exact and case-insensitive matches ──────────────────────


class TestExactMatch:
    def test_basic_exact_match(self):
        candidates = [
            {"id": "abc-123", "name": "Website Redesign"},
            {"id": "def-456", "name": "Mobile App"},
        ]
        assert fuzzy_match("Website Redesign", candidates) == "abc-123"

    def test_lowercase_query_matches_titlecase_candidate(self):
        candidates = [
            {"id": "abc-123", "name": "Website Redesign"},
            {"id": "def-456", "name": "Mobile App"},
        ]
        assert fuzzy_match("website redesign", candidates) == "abc-123"

    def test_uppercase_query_matches(self):
        candidates = [{"id": "id1", "name": "My Project"}]
        assert fuzzy_match("MY PROJECT", candidates) == "id1"


# ── Partial matches (substring-style) ─────────────────────


class TestPartialMatch:
    def test_partial_with_lower_threshold(self):
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("website", candidates, threshold=0.5) == "id1"

    def test_partial_substring_finds_parent(self):
        # 'redesign' alone should still find 'Website Redesign' at a
        # permissive threshold.
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("redesign", candidates, threshold=0.5) == "id1"


# ── Fuzzy match with typos ─────────────────────────────────


class TestFuzzyMatchWithTypos:
    def test_single_typo_resolves(self):
        # 'websit redesign' is missing an 'e'. Threshold of 0.7 should
        # still pick 'Website Redesign' (16 chars, 1 edit ~= 0.94).
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("websit redesign", candidates) == "id1"

    def test_transposed_chars_resolve(self):
        # 'Webiste Redesign' - swapped letters in 'Webiste'.
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("webiste redesign", candidates) == "id1"

    def test_extra_letter_still_resolves(self):
        # 'Websitee Redesign' - one extra letter.
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("websitee redesign", candidates) == "id1"


# ── Threshold filtering ────────────────────────────────────


class TestThresholdFiltering:
    def test_below_threshold_returns_none(self):
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("completely different", candidates, threshold=0.8) is None

    def test_threshold_zero_accepts_everything(self):
        # With threshold=0, any non-empty candidate qualifies. Pick a
        # query that is itself identical to one candidate so the result
        # is unambiguous regardless of tie-breaker behaviour.
        candidates = [{"id": "x", "name": "Other"}]
        assert fuzzy_match("Other", candidates, threshold=0.0) == "x"

    def test_strict_threshold_rejects_weak_match(self):
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        # 'wibsite' (typo) vs 'website' = 1/7 ~= 0.857, so threshold 0.9 should fail.
        assert fuzzy_match("wibsite", candidates, threshold=0.9) is None


# ── Multiple candidates ───────────────────────────────────


class TestMultipleCandidates:
    def test_picks_best_match(self):
        candidates = [
            {"id": "p1", "name": "Cooking Recipes"},
            {"id": "p2", "name": "Website Redesign"},
            {"id": "p3", "name": "Workout Plan"},
        ]
        assert fuzzy_match("website", candidates, threshold=0.4) == "p2"

    def test_first_match_wins_on_tie(self):
        # Two candidates at the same similarity - the first in the
        # list should win (deterministic tie-breaker).
        candidates = [
            {"id": "first", "name": "Alpha"},
            {"id": "second", "name": "Beta"},
        ]
        # 'gamma' is equidistant (5 chars, 5 edits each) from both,
        # so both score 0.0 and neither meets the default threshold.
        # Instead, test with a string that ties at high similarity.
        # Use a custom threshold that accepts both.
        candidates = [
            {"id": "first", "name": "Alpha"},
            {"id": "second", "name": "Alpha"},
        ]
        assert fuzzy_match("alpha", candidates) == "first"

    def test_returns_none_when_no_match(self):
        candidates = [
            {"id": "p1", "name": "Cooking Recipes"},
            {"id": "p2", "name": "Website Redesign"},
        ]
        assert fuzzy_match("xyzabc", candidates, threshold=0.7) is None


# ── Empty inputs ───────────────────────────────────────────


class TestEmptyInputs:
    def test_empty_query_returns_none(self):
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("", candidates) is None

    def test_whitespace_only_query_returns_none(self):
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("   ", candidates) is None

    def test_empty_candidates_returns_none(self):
        assert fuzzy_match("anything", []) is None

    def test_candidate_with_empty_name_skipped(self):
        # The single non-empty candidate still wins.
        candidates = [
            {"id": "empty", "name": ""},
            {"id": "real", "name": "Real Project"},
        ]
        assert fuzzy_match("Real Project", candidates) == "real"


# ── Edge cases: very short / long / unicode ───────────────


class TestEdgeCases:
    def test_single_character_query(self):
        candidates = [{"id": "id1", "name": "a"}]
        assert fuzzy_match("a", candidates) == "id1"

    def test_very_long_strings(self):
        long_name = "Project " + ("very " * 50) + "long"
        candidates = [{"id": "long-id", "name": long_name}]
        assert fuzzy_match(long_name, candidates) == "long-id"

    def test_long_query_with_small_typo(self):
        long_name = "Project " + ("very " * 50) + "long"
        # Drop one letter from the middle.
        typo = long_name[:50] + long_name[51:]
        candidates = [{"id": "long-id", "name": long_name}]
        assert fuzzy_match(typo, candidates, threshold=0.9) == "long-id"

    def test_unicode_candidate(self):
        candidates = [{"id": "u1", "name": "Proyek Desain"}]
        assert fuzzy_match("proyek desain", candidates) == "u1"

    def test_emoji_in_candidate(self):
        # Emoji should not crash; the score is whatever Levenshtein gives.
        candidates = [{"id": "e1", "name": "Fun Project 🎉"}]
        result = fuzzy_match("fun project 🎉", candidates)
        assert result == "e1"

    def test_extra_whitespace_normalised(self):
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match("  website   redesign  ", candidates) == "id1"


# ── Robustness: malformed input ───────────────────────────


class TestMalformedCandidates:
    def test_candidate_missing_id_is_skipped(self):
        candidates = [
            {"name": "No ID"},  # missing 'id' -> skipped
            {"id": "real", "name": "Real"},
        ]
        assert fuzzy_match("real", candidates) == "real"

    def test_candidate_missing_name_is_skipped(self):
        candidates = [
            {"id": "no-name"},  # missing 'name' -> skipped
            {"id": "real", "name": "Real"},
        ]
        assert fuzzy_match("real", candidates) == "real"

    def test_non_dict_candidate_is_skipped(self):
        candidates = [
            "not a dict",  # type: ignore[list-item]
            {"id": "real", "name": "Real"},
        ]
        assert fuzzy_match("real", candidates) == "real"

    def test_non_string_id_or_name_skipped(self):
        candidates = [
            {"id": 123, "name": "Numeric"},  # type: ignore[list-item]
            {"id": "real", "name": "Real"},
        ]
        assert fuzzy_match("real", candidates) == "real"

    def test_non_string_query_returns_none(self):
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        assert fuzzy_match(123, candidates) is None  # type: ignore[arg-type]


# ── Package-level export ───────────────────────────────────


class TestPackageExport:
    """Confirm the public re-export from ``app.utils`` works."""

    def test_fuzzy_match_reexported(self):
        candidates = [{"id": "id1", "name": "Website Redesign"}]
        # The same function must be reachable both ways.
        assert fuzzy_match_via_package("website", candidates, threshold=0.5) == "id1"
        assert fuzzy_match_via_package is fuzzy_match
