"""Relative date phrase parser for the Karsa AI Assistant.

Converts user-supplied relative date phrases (Indonesian and English)
into ISO 8601 date strings (``YYYY-MM-DD``). Used by the backend to
normalise the ``deadline`` / ``date`` arguments produced by the LLM
into a form the entity executors can store verbatim.

Supported phrases
-----------------

Indonesian:
    - ``besok``              -> tomorrow
    - ``lusa``               -> day after tomorrow
    - ``minggu depan``       -> 7 days from today
    - ``bulan depan``        -> 30 days from today
    - ``akhir bulan``        -> last day of the current month
    - ``tanggal N``          -> N-th day of current or next month

English:
    - ``tomorrow``           -> tomorrow
    - ``day after tomorrow`` -> day after tomorrow
    - ``next week``          -> 7 days from today
    - ``next month``         -> 30 days from today

Passthrough:
    - ``YYYY-MM-DD`` strings are returned unchanged (after validation).

Everything else raises :class:`ValueError` so the caller can surface a
user-friendly error rather than silently storing garbage.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

__all__ = ["parse_relative_date"]


# ── Phrase tables ───────────────────────────────────────────
#
# Keys are the *normalised* form of the phrase: lower-case, single-spaced,
# stripped. Adding a new alias is a one-line change.
_PHRASES: dict[str, str] = {
    # Indonesian
    "besok": "besok",
    "lusa": "lusa",
    "minggu depan": "minggu_depan",
    "bulan depan": "bulan_depan",
    "akhir bulan": "akhir_bulan",
    # English
    "tomorrow": "besok",
    "day after tomorrow": "lusa",
    "next week": "minggu_depan",
    "next month": "bulan_depan",
}

# ``tanggal N`` - Indonesian "date N" (e.g. "tanggal 15").
_TANGGAL_RE = re.compile(r"^tanggal\s+(\d{1,2})$")

# ISO 8601 date validation: YYYY-MM-DD with real calendar values.
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalise(raw: str) -> str:
    """Lower-case, collapse internal whitespace, strip ends."""
    return " ".join(raw.lower().split())


def _resolve_phrase(today: date, kind: str) -> date:
    """Map a phrase identifier to the actual date."""
    if kind == "besok":
        return today + timedelta(days=1)
    if kind == "lusa":
        return today + timedelta(days=2)
    if kind == "minggu_depan":
        return today + timedelta(days=7)
    if kind == "bulan_depan":
        return today + timedelta(days=30)
    if kind == "akhir_bulan":
        # Move to the first of next month, then back one day.
        if today.month == 12:
            first_next_month = date(today.year + 1, 1, 1)
        else:
            first_next_month = date(today.year, today.month + 1, 1)
        return first_next_month - timedelta(days=1)
    # Defensive: should be unreachable because the table is closed.
    raise ValueError(f"Unknown phrase kind: {kind!r}")


def parse_relative_date(date_str: str) -> str:
    """Parse a relative date phrase to ISO 8601 (``YYYY-MM-DD``).

    Args:
        date_str: Relative date phrase (Indonesian or English) or an
            already-formatted ISO 8601 date.

    Returns:
        ISO 8601 date string (``YYYY-MM-DD``).

    Raises:
        ValueError: If ``date_str`` cannot be parsed or represents an
            invalid date.
    """
    if not isinstance(date_str, str):
        raise ValueError(f"date_str must be a string, got {type(date_str).__name__}")

    normalised = _normalise(date_str)

    if not normalised:
        raise ValueError("date_str must not be empty")

    # 1. Exact phrase match (handles besok, lusa, minggu depan, etc.)
    if normalised in _PHRASES:
        result = _resolve_phrase(date.today(), _PHRASES[normalised])
        return result.isoformat()

    # 2. Indonesian "tanggal N" - pick this month if still ahead,
    #    otherwise roll over to next month.
    tanggal_match = _TANGGAL_RE.match(normalised)
    if tanggal_match:
        day = int(tanggal_match.group(1))
        if not 1 <= day <= 31:
            raise ValueError(f"Invalid day in 'tanggal N': {day}")

        today = date.today()
        try:
            candidate = today.replace(day=day)
        except ValueError as exc:
            # e.g. tanggal 31 in a 30-day month - try next month.
            if today.month == 12:
                candidate = date(today.year + 1, 1, day)
            else:
                candidate = date(today.year, today.month + 1, day)
            try:
                # Final validation - if day is still impossible, raise.
                candidate  # noqa: B018 - access for validation only
            except ValueError:
                raise ValueError(
                    f"Invalid date: 'tanggal {day}' cannot be resolved"
                ) from exc

        # If the candidate is today or already past, bump to next month
        # so 'tanggal 15' on the 20th (or even on the 15th itself) means
        # *next* month's 15th - the phrase always implies a future date.
        if candidate <= today:
            if today.month == 12:
                candidate = date(today.year + 1, 1, day)
            else:
                candidate = date(today.year, today.month + 1, day)
        return candidate.isoformat()

    # 3. ISO 8601 passthrough - validate, return unchanged.
    if _ISO_RE.match(normalised):
        try:
            parsed = date.fromisoformat(normalised)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO 8601 date: {date_str!r}") from exc
        return parsed.isoformat()

    raise ValueError(f"Cannot parse date phrase: {date_str!r}")