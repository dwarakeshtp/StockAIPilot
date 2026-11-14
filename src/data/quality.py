"""Data-quality assessment for a single item's daily sales history."""

from __future__ import annotations

from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

Severity = Literal["INFO", "WARNING", "CRITICAL"]

STOCKOUT_STREAK_DAYS: int = 7
STOCKOUT_MIN_MEAN_UNITS: float = 3.0


class DataQualityIssue(BaseModel):
    """A single data-quality finding."""

    code: str
    message: str
    severity: Severity = "WARNING"


class DataQualityReport(BaseModel):
    """Summary of data quality for one item, including a 0-1 score."""

    item_id: str
    n_rows: int
    start_date: date
    end_date: date
    missing_dates: int = 0
    missing_units: int = 0
    negative_units: int = 0
    missing_prices: int = 0
    duplicate_dates: int = 0
    zero_share: float = 0.0
    longest_zero_streak: int = 0
    suspected_stockout_days: int = 0
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: list[DataQualityIssue] = Field(default_factory=list)


def zero_streak_mask(units: pd.Series, min_len: int = STOCKOUT_STREAK_DAYS) -> pd.Series:
    """Boolean mask of days inside runs of at least ``min_len`` consecutive zeros."""
    is_zero = units.fillna(0).eq(0).to_numpy()
    mask = np.zeros(len(is_zero), dtype=bool)
    start = None
    for i, z in enumerate(np.append(is_zero, False)):
        if z and start is None:
            start = i
        elif not z and start is not None:
            if i - start >= min_len:
                mask[start:i] = True
            start = None
    return pd.Series(mask, index=units.index)


def _longest_zero_streak(units: pd.Series) -> int:
    longest = current = 0
    for v in units.fillna(0).to_numpy():
        current = current + 1 if v == 0 else 0
        longest = max(longest, current)
    return longest


def assess_quality(df: pd.DataFrame, item_id: str | None = None) -> DataQualityReport:
    """Assess the raw (pre-cleaning) sales history of a single item.

    Args:
        df: Rows of one item with at least ``date``, ``units`` and ``sell_price``.
        item_id: Item identifier (taken from ``df`` when omitted).

    Returns:
        A :class:`DataQualityReport` with counts, issues and a quality score.
    """
    item = item_id or str(df["item_id"].iloc[0])
    dates = pd.to_datetime(df["date"])
    full_range = pd.date_range(dates.min(), dates.max(), freq="D")
    n = len(full_range)
    duplicates = int(dates.duplicated().sum())
    missing_dates = int(n - dates.nunique())
    missing_units = int(df["units"].isna().sum())
    negative_units = int((df["units"] < 0).sum())
    missing_prices = int(df["sell_price"].isna().sum()) if "sell_price" in df else n
    units = df.sort_values("date")["units"].reset_index(drop=True)
    zero_share = float(units.fillna(0).eq(0).mean())
    longest = _longest_zero_streak(units)
    mean_units = float(units[units > 0].mean()) if (units > 0).any() else 0.0
    stockout_days = (
        int(zero_streak_mask(units).sum()) if mean_units >= STOCKOUT_MIN_MEAN_UNITS else 0
    )

    issues: list[DataQualityIssue] = []
    score = 1.0
    if missing_dates:
        issues.append(DataQualityIssue(code="missing_dates",
                                       message=f"{missing_dates} calendar days missing"))
        score -= min(0.3, 2 * missing_dates / n)
    if missing_units:
        issues.append(DataQualityIssue(code="missing_units",
                                       message=f"{missing_units} days without a units value"))
        score -= min(0.2, missing_units / n)
    if negative_units:
        issues.append(DataQualityIssue(code="negative_units", severity="CRITICAL",
                                       message=f"{negative_units} negative unit values"))
        score -= 0.05
    if missing_prices:
        issues.append(DataQualityIssue(code="missing_prices", severity="INFO",
                                       message=f"{missing_prices} days without a price"))
        score -= min(0.1, 0.5 * missing_prices / n)
    if duplicates:
        issues.append(DataQualityIssue(code="duplicate_dates",
                                       message=f"{duplicates} duplicated dates"))
        score -= 0.05
    if stockout_days:
        issues.append(
            DataQualityIssue(
                code="suspected_stockouts",
                message=(
                    f"{stockout_days} days in zero-sales streaks of {STOCKOUT_STREAK_DAYS}+ days "
                    f"(longest {longest}); likely stockouts, treated as censored demand"
                ),
            )
        )
        score -= min(0.3, 0.1 + 0.5 * stockout_days / n)
    if zero_share > 0.5:
        issues.append(DataQualityIssue(code="intermittent_demand", severity="INFO",
                                       message=f"{zero_share:.0%} zero-sales days"))

    return DataQualityReport(
        item_id=item,
        n_rows=len(df),
        start_date=full_range.min().date(),
        end_date=full_range.max().date(),
        missing_dates=missing_dates,
        missing_units=missing_units,
        negative_units=negative_units,
        missing_prices=missing_prices,
        duplicate_dates=duplicates,
        zero_share=round(zero_share, 4),
        longest_zero_streak=longest,
        suspected_stockout_days=stockout_days,
        score=round(max(0.0, min(1.0, score)), 3),
        issues=issues,
    )
