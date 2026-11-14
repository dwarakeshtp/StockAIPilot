"""Synthetic, M5-shaped daily sales generator.

Used for unit tests and as an offline fallback when the processed M5 subset is
unavailable. The generator injects weekly and yearly seasonality, SNAP uplift,
event uplift, price promotions, random spikes and (for one item) intermittency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EVENTS: dict[str, tuple[str, str]] = {
    "02-07": ("SuperBowl", "Sporting"),
    "02-14": ("ValentinesDay", "Cultural"),
    "05-08": ("Mother's day", "Cultural"),
    "07-04": ("IndependenceDay", "National"),
    "10-31": ("Halloween", "Cultural"),
    "11-24": ("Thanksgiving", "National"),
    "12-25": ("Christmas", "National"),
    "05-30": ("MemorialDay", "National"),
    "06-19": ("Father's day", "Cultural"),
}


def _calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    keys = dates.strftime("%m-%d")
    event_name = [(_EVENTS.get(k, ("", ""))[0]) for k in keys]
    event_type = [(_EVENTS.get(k, ("", ""))[1]) for k in keys]
    snap = (dates.day <= 10).astype(int)
    return pd.DataFrame(
        {"date": dates, "event_name": event_name, "event_type": event_type, "snap": snap}
    )


def generate_synthetic_sales(
    n_items: int = 6,
    history_days: int = 730,
    end_date: pd.Timestamp | str = "2016-05-22",
    seed: int = 42,
    horizon_days: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a long-format sales frame and the future calendar.

    Args:
        n_items: Number of synthetic SKUs.
        history_days: Number of daily observations per SKU.
        end_date: Last observed date.
        seed: Random seed for reproducibility.
        horizon_days: Number of future calendar rows to emit.

    Returns:
        Tuple ``(sales, future_calendar)`` following the ingestion schema.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(end=end, periods=history_days, freq="D")
    cal = _calendar(dates)
    future = _calendar(pd.date_range(end + pd.Timedelta(days=1), periods=horizon_days, freq="D"))

    t = np.arange(history_days)
    weekly_profile = np.array([1.0, 0.9, 0.85, 0.85, 0.95, 1.25, 1.3])
    frames = []
    for i in range(n_items):
        base = float(rng.uniform(8, 60))
        trend = 1 + rng.uniform(-0.15, 0.25) * t / history_days
        weekly = weekly_profile[dates.dayofweek.to_numpy()] ** rng.uniform(0.5, 1.5)
        yearly = 1 + 0.1 * np.sin(2 * np.pi * t / 365.25 + rng.uniform(0, 2 * np.pi))
        snap_lift = 1 + 0.12 * cal["snap"].to_numpy()
        event_lift = np.where(cal["event_name"].to_numpy() != "", 1.3, 1.0)

        base_price = float(rng.uniform(1.0, 6.0))
        price = np.full(history_days, base_price)
        for start in rng.choice(history_days - 14, size=6, replace=False):
            price[start : start + 7] = base_price * 0.8
        price_lift = (price / base_price) ** -1.5

        lam = base * trend * weekly * yearly * snap_lift * event_lift * price_lift
        units = rng.poisson(lam).astype(float)
        if i == n_items - 1 and n_items > 2:
            units = np.where(rng.random(history_days) < 0.45, 0.0, units * 0.4).round()
        if i == 0:
            units[-5:] = units[-5:] * 2.2
        spikes = rng.choice(history_days - 30, size=3, replace=False)
        units[spikes] = units[spikes] * 3

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "item_id": f"FOODS_SYN_{i + 1:03d}",
                    "dept_id": "FOODS_SYN",
                    "cat_id": "FOODS",
                    "store_id": "SYN_1",
                    "units": units.round(),
                    "sell_price": price.round(2),
                    "event_name": cal["event_name"].to_numpy(),
                    "event_type": cal["event_type"].to_numpy(),
                    "snap": cal["snap"].to_numpy(),
                }
            )
        )
    sales = pd.concat(frames, ignore_index=True)
    return sales, future
