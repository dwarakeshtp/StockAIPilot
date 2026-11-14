"""Stockout risk estimation and stock projection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from scipy.stats import norm

from src.inventory.policy import lead_time_demand_std


def stockout_probability(
    inventory_position: float,
    mean_demand: float,
    demand_std: float,
    lead_time: float,
    lead_time_std: float = 0.0,
) -> float:
    """Probability that lead-time demand exceeds the current inventory position.

    Interpreted as the chance of running out before a replenishment ordered
    today could arrive, assuming normally distributed lead-time demand.
    """
    mu = mean_demand * lead_time
    sigma = lead_time_demand_std(demand_std, mean_demand, lead_time, lead_time_std)
    if sigma <= 1e-9:
        return 1.0 if inventory_position < mu else 0.0
    return float(1 - norm.cdf((inventory_position - mu) / sigma))


def days_of_cover(stock: float, mean_demand: float, cap: float = 999.0) -> float:
    """Days the given stock lasts at the mean demand rate (capped)."""
    if mean_demand <= 1e-9:
        return cap
    return float(min(max(stock, 0.0) / mean_demand, cap))


@dataclass(frozen=True)
class StockProjection:
    """Day-by-day projected stock and the first projected stockout date."""

    dates: list[date]
    demand: list[float]
    arrivals: list[float]
    stock: list[float]
    stockout_date: date | None
    stockout_day_index: int | None


def project_stock(
    on_hand: float,
    daily_demand: Sequence[float],
    dates: Sequence[date],
    on_order: float = 0.0,
    on_order_eta_days: int = 0,
) -> StockProjection:
    """Project end-of-day stock assuming no new orders are placed.

    Args:
        on_hand: Current on-hand units.
        daily_demand: Expected demand per future day.
        dates: Calendar date of each future day.
        on_order: Units already ordered.
        on_order_eta_days: Day index (1-based) when the open order arrives.

    Returns:
        A :class:`StockProjection`; stock may go negative (unmet demand).
    """
    if len(daily_demand) != len(dates):
        raise ValueError("daily_demand and dates must have the same length")
    stock = float(on_hand)
    stocks, arrivals = [], []
    stockout_date, stockout_idx = None, None
    for i, (d, demand) in enumerate(zip(dates, daily_demand, strict=True), start=1):
        arrival = float(on_order) if on_order and i == max(on_order_eta_days, 1) else 0.0
        stock = stock + arrival - float(demand)
        arrivals.append(arrival)
        stocks.append(round(stock, 3))
        if stock < 0 and stockout_date is None:
            stockout_date, stockout_idx = d, i
    return StockProjection(
        dates=list(dates),
        demand=[float(x) for x in daily_demand],
        arrivals=arrivals,
        stock=stocks,
        stockout_date=stockout_date,
        stockout_day_index=stockout_idx,
    )
