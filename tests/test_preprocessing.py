"""Tests for data quality, cleaning and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_item_sales
from src.data.features import (
    build_features,
    build_future_frame,
    chronological_split,
    prophet_holidays,
)
from src.data.quality import assess_quality, zero_streak_mask


def _dirty_frame() -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", periods=60, freq="D")
    units = np.full(60, 10.0)
    units[20:30] = 0  # 10-day zero streak -> suspected stockout
    units[5] = -3  # negative value
    df = pd.DataFrame(
        {
            "date": dates,
            "item_id": "X",
            "dept_id": "D",
            "cat_id": "FOODS",
            "store_id": "S",
            "units": units,
            "sell_price": 2.0,
            "event_name": "",
            "event_type": "",
            "snap": 0,
        }
    )
    df.loc[10, "units"] = np.nan
    df.loc[12, "sell_price"] = np.nan
    df = df.drop(index=[40, 41])  # two missing dates
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate date


def test_zero_streak_mask_only_long_runs() -> None:
    s = pd.Series([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2])
    mask = zero_streak_mask(s, min_len=7)
    assert mask.sum() == 7
    assert not mask.iloc[1]


def test_assess_quality_detects_issues() -> None:
    report = assess_quality(_dirty_frame())
    codes = {i.code for i in report.issues}
    assert {"missing_dates", "missing_units", "negative_units", "duplicate_dates",
            "suspected_stockouts", "missing_prices"} <= codes
    assert report.missing_dates == 2
    assert report.score < 1.0


def test_assess_quality_clean_series_scores_high(item_sales: pd.DataFrame) -> None:
    report = assess_quality(item_sales)
    assert report.score >= 0.9


def test_clean_item_sales_produces_complete_series() -> None:
    clean, report = clean_item_sales(_dirty_frame())
    assert len(clean) == 60
    assert clean["date"].is_monotonic_increasing and clean["date"].is_unique
    assert (clean["units"] >= 0).all()
    assert clean["sell_price"].notna().all()
    assert clean["is_imputed"].sum() == 3  # one NaN + two missing dates
    assert clean["is_suspected_stockout"].sum() == 10
    assert clean.loc[clean["is_suspected_stockout"], "y"].isna().all()
    assert report.item_id == "X"


def test_clean_rejects_multiple_items(item_sales: pd.DataFrame) -> None:
    other = item_sales.assign(item_id="OTHER")
    with pytest.raises(ValueError):
        clean_item_sales(pd.concat([item_sales, other]))


def test_features_are_leakage_safe(item_sales: pd.DataFrame) -> None:
    clean, _ = clean_item_sales(item_sales)
    feats = build_features(clean)
    expected = clean["units"].iloc[:7].mean()
    assert feats.loc[7, "rolling_mean_7"] == pytest.approx(expected)
    assert np.isnan(feats.loc[0, "rolling_mean_7"])
    for col in ("dow", "is_weekend", "is_event", "price_ratio", "lag_7"):
        assert col in feats


def test_future_frame_uses_calendar(item_sales: pd.DataFrame, synthetic_dataset) -> None:
    clean, _ = clean_item_sales(item_sales)
    feats = build_features(clean)
    future = build_future_frame(feats, synthetic_dataset.future_calendar, horizon=7)
    assert len(future) == 7
    assert future["date"].iloc[0] == feats["date"].max() + pd.Timedelta(days=1)
    assert future["sell_price"].nunique() == 1
    assert {"snap", "price_ratio", "is_event"} <= set(future.columns)


def test_chronological_split() -> None:
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10), "v": range(10)})
    train, test = chronological_split(df, 3)
    assert len(train) == 7 and len(test) == 3
    assert train["date"].max() < test["date"].min()
    with pytest.raises(ValueError):
        chronological_split(df, 10)


def test_prophet_holidays_frame(item_sales: pd.DataFrame) -> None:
    clean, _ = clean_item_sales(item_sales)
    hol = prophet_holidays(build_features(clean))
    assert list(hol.columns) == ["holiday", "ds"]
    assert (hol["holiday"] != "").all()
