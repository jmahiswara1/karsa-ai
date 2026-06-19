"""Tests for app.utils.date_parser.parse_relative_date."""
from __future__ import annotations

from datetime import date

import pytest
from unittest.mock import patch

from app.utils.date_parser import parse_relative_date


# ── Helpers ─────────────────────────────────────────────────


def _freeze_today(mock_date, today: date) -> None:
    """Patch ``date.today()`` inside the parser module to return ``today``."""
    mock_date.today.return_value = today
    # Keep the constructor intact so calls like ``date(year, month, day)``
    # still build real ``date`` instances when the parser needs them.
    mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)


# ── Indonesian phrases ──────────────────────────────────────


class TestIndonesianPhrases:
    @patch("app.utils.date_parser.date")
    def test_besok(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("besok") == "2026-06-20"

    @patch("app.utils.date_parser.date")
    def test_lusa(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("lusa") == "2026-06-21"

    @patch("app.utils.date_parser.date")
    def test_minggu_depan(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("minggu depan") == "2026-06-26"

    @patch("app.utils.date_parser.date")
    def test_bulan_depan(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("bulan depan") == "2026-07-19"

    @patch("app.utils.date_parser.date")
    def test_akhir_bulan(self, mock_date):
        # June has 30 days, so 'akhir bulan' on the 19th is the 30th.
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("akhir bulan") == "2026-06-30"

    @patch("app.utils.date_parser.date")
    def test_tanggal_future_this_month(self, mock_date):
        # Today is the 10th, so 'tanggal 15' stays in this month.
        _freeze_today(mock_date, date(2026, 6, 10))
        assert parse_relative_date("tanggal 15") == "2026-06-15"

    @patch("app.utils.date_parser.date")
    def test_tanggal_past_rolls_to_next_month(self, mock_date):
        # Today is the 20th, so 'tanggal 15' rolls to next month's 15th.
        _freeze_today(mock_date, date(2026, 6, 20))
        assert parse_relative_date("tanggal 15") == "2026-07-15"

    @patch("app.utils.date_parser.date")
    def test_tanggal_same_day_rolls_to_next_month(self, mock_date):
        # Today is the 15th itself - 'tanggal 15' must be next month's 15th
        # because the user clearly doesn't mean "today".
        _freeze_today(mock_date, date(2026, 6, 15))
        assert parse_relative_date("tanggal 15") == "2026-07-15"


# ── English phrases ─────────────────────────────────────────


class TestEnglishPhrases:
    @patch("app.utils.date_parser.date")
    def test_tomorrow(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("tomorrow") == "2026-06-20"

    @patch("app.utils.date_parser.date")
    def test_day_after_tomorrow(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("day after tomorrow") == "2026-06-21"

    @patch("app.utils.date_parser.date")
    def test_next_week(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("next week") == "2026-06-26"

    @patch("app.utils.date_parser.date")
    def test_next_month(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("next month") == "2026-07-19"


# ── ISO 8601 passthrough ────────────────────────────────────


class TestIsoPassthrough:
    def test_iso_date_unchanged(self):
        assert parse_relative_date("2026-06-15") == "2026-06-15"

    def test_iso_date_leap_day(self):
        # 2024 is a leap year; 2024-02-29 is valid.
        assert parse_relative_date("2024-02-29") == "2024-02-29"

    def test_iso_invalid_calendar_value_raises(self):
        # 2026-02-30 does not exist.
        with pytest.raises(ValueError):
            parse_relative_date("2026-02-30")

    def test_iso_wrong_shape_raises(self):
        # Missing zero-padding is NOT a passthrough.
        with pytest.raises(ValueError):
            parse_relative_date("2026-6-15")


# ── Edge cases: month and year boundaries ───────────────────


class EdgeCases:
    @patch("app.utils.date_parser.date")
    def test_besok_crosses_month_boundary(self, mock_date):
        # End of June -> July 1.
        _freeze_today(mock_date, date(2026, 6, 30))
        assert parse_relative_date("besok") == "2026-07-01"

    @patch("app.utils.date_parser.date")
    def test_besok_crosses_year_boundary(self, mock_date):
        # Dec 31 -> next year's Jan 1.
        _freeze_today(mock_date, date(2026, 12, 31))
        assert parse_relative_date("besok") == "2027-01-01"

    @patch("app.utils.date_parser.date")
    def test_akhir_bulan_december(self, mock_date):
        # December -> 31st of December.
        _freeze_today(mock_date, date(2026, 12, 5))
        assert parse_relative_date("akhir bulan") == "2026-12-31"

    @patch("app.utils.date_parser.date")
    def test_akhir_bulan_february_non_leap(self, mock_date):
        # 2026 is not a leap year; Feb ends on the 28th.
        _freeze_today(mock_date, date(2026, 2, 10))
        assert parse_relative_date("akhir bulan") == "2026-02-28"

    @patch("app.utils.date_parser.date")
    def test_akhir_bulan_february_leap(self, mock_date):
        # 2024 is a leap year; Feb ends on the 29th.
        _freeze_today(mock_date, date(2024, 2, 10))
        assert parse_relative_date("akhir bulan") == "2024-02-29"

    @patch("app.utils.date_parser.date")
    def test_tanggal_rolls_year_boundary(self, mock_date):
        # 'tanggal 10' on Dec 20 -> next year Jan 10.
        _freeze_today(mock_date, date(2026, 12, 20))
        assert parse_relative_date("tanggal 10") == "2027-01-10"

    @patch("app.utils.date_parser.date")
    def test_tanggal_february_29_in_non_leap_year(self, mock_date):
        # 2027 is not a leap year, so 'tanggal 29' rolls to March 1.
        # We still expect a valid ISO date as the answer.
        _freeze_today(mock_date, date(2027, 1, 5))
        result = parse_relative_date("tanggal 29")
        # February 2027 has 28 days, so 'tanggal 29' spills into March.
        assert result == "2027-03-01"


# ── Case insensitivity and whitespace ───────────────────────


class TestNormalisation:
    @patch("app.utils.date_parser.date")
    def test_uppercase_besok(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("BESOK") == "2026-06-20"

    @patch("app.utils.date_parser.date")
    def test_mixed_case_tomorrow(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("Tomorrow") == "2026-06-20"

    @patch("app.utils.date_parser.date")
    def test_extra_whitespace(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 19))
        assert parse_relative_date("  besok  ") == "2026-06-20"
        assert parse_relative_date("minggu   depan") == "2026-06-26"

    @patch("app.utils.date_parser.date")
    def test_tanggal_with_extra_space(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 10))
        assert parse_relative_date("  tanggal   15 ") == "2026-06-15"


# ── Invalid input ───────────────────────────────────────────


class TestInvalidInput:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_relative_date("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            parse_relative_date("   ")

    def test_unknown_phrase_raises(self):
        with pytest.raises(ValueError):
            parse_relative_date("next century")

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            parse_relative_date("not a date at all")

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            parse_relative_date(12345)  # type: ignore[arg-type]

    @patch("app.utils.date_parser.date")
    def test_tanggal_out_of_range_raises(self, mock_date):
        _freeze_today(mock_date, date(2026, 6, 10))
        with pytest.raises(ValueError):
            parse_relative_date("tanggal 0")
        with pytest.raises(ValueError):
            parse_relative_date("tanggal 32")