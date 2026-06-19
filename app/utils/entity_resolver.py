"""Fuzzy entity name resolver for the Karsa AI Assistant.

When the LLM extracts an entity name from a user prompt (e.g. the
project name in *"tambahkan task ke project Website Redesign"*), we need
to map that free-form string back to a real database record. Exact
equality is rarely good enough - users capitalise inconsistently, the
LLM may paraphrase, and typos sneak in. This module provides a
Levenshtein-distance based fuzzy matcher that returns the ID of the
best-matching candidate above a configurable similarity threshold.

The implementation is intentionally dependency-free: we re-implement
Levenshtein distance directly with the standard dynamic-programming
table so the AI service stays free of optional ``rapidfuzz`` /
``python-Levenshtein`` installs.
"""
from __future__ import annotations

from typing import Any

__all__ = ["fuzzy_match", "levenshtein_distance"]


# ── Helpers ─────────────────────────────────────────────────


def _normalise(raw: str) -> str:
    """Lower-case, collapse internal whitespace, strip ends.

    Mirrors :func:`app.utils.date_parser._normalise` so behaviour is
    consistent across utility modules: identical strings with different
    capitalisation or surrounding whitespace resolve to the same key.
    """
    return " ".join(raw.lower().split())


def _similarity(query: str, candidate: str) -> float:
    """Convert Levenshtein distance into a similarity score in ``[0.0, 1.0]``.

    Formula: ``1 - distance / max(len(query), len(candidate))``. Identical
    strings score ``1.0``; completely disjoint strings score ``0.0``.

    An empty query against an empty candidate is treated as a perfect
    match (``1.0``) so callers don't have to special-case empty inputs.
    A mismatch where one side is empty always scores ``0.0``.

    A pure Levenshtein ratio penalises substring matches harshly
    (e.g. ``"website"`` vs ``"website redesign"`` scores only ``0.44``).
    For entity resolution that's the wrong intuition - if the user's
    query is contained inside a candidate name (or vice versa), the
    candidate is almost certainly the right one. We therefore take the
    *maximum* of the Levenshtein ratio and a containment bonus that
    weights substring matches at ``0.8 + 0.2 * (shorter / longer)``.
    That floor of ``0.8`` is high enough to clear the default
    ``threshold=0.7`` while still letting *very* short queries (e.g.
    a 2-letter prefix of a long title) fall under that threshold if
    the caller is strict.
    """
    if not query and not candidate:
        return 1.0
    if not query or not candidate:
        return 0.0

    longest = max(len(query), len(candidate))
    distance = levenshtein_distance(query, candidate)
    levenshtein_ratio = 1.0 - (distance / longest)

    shorter, longer = (
        (query, candidate) if len(query) <= len(candidate) else (candidate, query)
    )
    if shorter in longer:
        containment = 0.8 + 0.2 * (len(shorter) / len(longer))
    else:
        containment = 0.0

    return max(levenshtein_ratio, containment)


# ── Core algorithm ──────────────────────────────────────────


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Uses the classic O(len(s1) * len(s2)) DP table with two rolling
    rows so memory stays at O(min(len(s1), len(s2))). Accepts any
    string-like input; non-strings are coerced via ``str()`` to keep
    the matcher robust against accidental numeric IDs being passed
    as ``name`` values.
    """
    a = str(s1)
    b = str(s2)

    # Ensure the inner loop walks the shorter string so memory is
    # bounded by the shorter operand.
    if len(a) < len(b):
        a, b = b, a

    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    current_row = [0] * (len(b) + 1)

    for i, ch_a in enumerate(a, start=1):
        current_row[0] = i
        for j, ch_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (0 if ch_a == ch_b else 1)
            current_row[j] = min(insert_cost, delete_cost, replace_cost)
        previous_row, current_row = current_row, previous_row

    return previous_row[len(b)]


# ── Public API ──────────────────────────────────────────────


def fuzzy_match(
    query: str,
    candidates: list[dict[str, Any]],
    threshold: float = 0.7,
) -> str | None:
    """Resolve ``query`` to the best-matching candidate's ``id``.

    Both the query and candidate names are normalised (lower-cased,
    whitespace-collapsed) before scoring. Each candidate receives a
    similarity score in ``[0.0, 1.0]``; the candidate with the highest
    score that meets ``threshold`` wins. Ties are broken by candidate
    position so the first-listed match is preferred (useful when the
    caller has already pre-sorted candidates by relevance).

    Args:
        query: The free-form name to resolve (e.g. ``"website redesign"``).
        candidates: List of dicts each exposing ``"id"`` and ``"name"``.
            Candidates missing either key are skipped silently.
        threshold: Minimum similarity score in ``[0.0, 1.0]`` required
            for a match to be accepted. Defaults to ``0.7``.

    Returns:
        The ``id`` of the best-matching candidate, or ``None`` if no
        candidate clears the threshold, the query is empty, or the
        candidate list is empty.
    """
    if not isinstance(query, str) or not query.strip():
        return None
    if not candidates:
        return None

    normalised_query = _normalise(query)

    best_id: str | None = None
    best_score = threshold  # so candidates below threshold are skipped

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = candidate.get("name")
        cid = candidate.get("id")
        if not isinstance(name, str) or not isinstance(cid, str):
            # Skip malformed candidates; the caller is responsible for
            # well-formed input, but we don't want a single bad row
            # to crash the whole resolution.
            continue
        if not name.strip():
            continue

        score = _similarity(normalised_query, _normalise(name))
        # Strict ``>`` (not ``>=``) keeps the first-scoring candidate as
        # the tie-breaker: when two candidates score identically, the
        # earlier one in the list wins, giving callers a deterministic
        # ordering they can control via the input order.
        if score >= threshold and score > best_score:
            best_score = score
            best_id = cid

    return best_id
