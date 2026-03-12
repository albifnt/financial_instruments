"""Day count convention helpers."""

from __future__ import annotations

from datetime import date, datetime


def _to_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def year_fraction(start: date | datetime, end: date | datetime, convention: str = "ACT/365") -> float:
    """Return year fraction between two dates under a day count convention."""
    start_d = _to_date(start)
    end_d = _to_date(end)
    if end_d < start_d:
        raise ValueError("End date must be on or after start date.")

    days = (end_d - start_d).days
    conv = convention.upper().strip()

    if conv == "ACT/365":
        return days / 365.0
    if conv == "ACT/360":
        return days / 360.0
    if conv == "30/360":
        d1 = min(start_d.day, 30)
        d2 = min(end_d.day, 30)
        total = (end_d.year - start_d.year) * 360 + (end_d.month - start_d.month) * 30 + (d2 - d1)
        return total / 360.0

    raise ValueError(f"Unsupported day count convention: {convention}")
