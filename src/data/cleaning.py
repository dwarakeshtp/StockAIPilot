"""Cleaning of a single item's daily sales history.

Steps (all deterministic and logged in the returned frame via flag columns):

1. Drop duplicate dates (keep the last record).
2. Re-index to a complete daily calendar.
3. Impute missing units with 0 and flag them (``is_imputed``).
4. Clip negative units to 0.
5. Forward/back-fill prices, events and SNAP flags.
6. Flag long zero streaks on selling items as suspected stockouts
   (``is_suspected_stockout``); these days become ``NaN`` in the modelling
   target ``y`` so models learn demand rather than censored sales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.quality import (
    STOCKOUT_MIN_MEAN_UNITS,
    DataQualityReport,
    assess_quality,
    zero_streak_mask,
)
from src.logging.logger import get_logger

logger = get_logger(__name__)

_CATEGORICAL_COLUMNS: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "store_id")


def clean_item_sales(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Clean one item's sales and return the cleaned frame plus a quality report.

    Args:
        df: Raw rows of a single item following the ingestion schema.

    Returns:
        ``(clean_df, report)`` where ``clean_df`` has one row per calendar day and
        the extra columns ``is_imputed``, ``is_suspected_stockout`` and ``y``.

    Raises:
        ValueError: if ``df`` is empty or contains more than one item.
    """
    if df.empty:
        raise ValueError("Cannot clean an empty frame")
    if df["item_id"].nunique() != 1:
        raise ValueError("clean_item_sales expects exactly one item")

    report = assess_quality(df)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset="date", keep="last")
    full_index = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    out = out.set_index("date").reindex(full_index)
    out.index.name = "date"

    out["is_imputed"] = out["units"].isna()
    out["units"] = out["units"].fillna(0).clip(lower=0).astype(float)
    for col in _CATEGORICAL_COLUMNS:
        if col in out:
            out[col] = out[col].ffill().bfill()
    out["sell_price"] = out["sell_price"].ffill().bfill()
    out["event_name"] = out["event_name"].fillna("")
    out["event_type"] = out["event_type"].fillna("")
    out["snap"] = out["snap"].fillna(0).astype(int)

    positive = out.loc[out["units"] > 0, "units"]
    mean_positive = float(positive.mean()) if not positive.empty else 0.0
    if mean_positive >= STOCKOUT_MIN_MEAN_UNITS:
        out["is_suspected_stockout"] = zero_streak_mask(out["units"]).to_numpy()
    else:
        out["is_suspected_stockout"] = False
    out["y"] = np.where(out["is_suspected_stockout"], np.nan, out["units"])

    out = out.reset_index()
    logger.debug(
        "item_cleaned",
        item_id=report.item_id,
        rows=len(out),
        imputed=int(out["is_imputed"].sum()),
        stockout_days=int(out["is_suspected_stockout"].sum()),
        quality_score=report.score,
    )
    return out, report
