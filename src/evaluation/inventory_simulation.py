"""Day-by-day inventory replay comparing the agentic policy with a naive rule.

Both policies face the same actual demand, the same starting stock and the same
deterministic lead time. Unmet demand is lost (no backorders).

* **Agentic policy** - every ``review_days`` the item's selected forecasting model
  is refit on data up to that day; safety stock, reorder point and order-up-to
  quantity are recomputed with the same rules as the Inventory Planning Agent.
  Each day, if the inventory position is at or below the reorder point, an order
  is placed (rounded to pack size / MOQ).
* **Naive policy** - spreadsheet-style rule fixed at the start of the window:
  reorder ``naive_cover_days`` of average sales when the inventory position drops
  below ``naive_trigger_days`` of average sales (no forecast, no safety stock).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.inventory import (
    reorder_point,
    reorder_quantity,
    round_order_quantity,
    safety_stock,
    z_value,
)

OrderPolicy = Callable[[int, float], int]


@dataclass
class SimulationResult:
    """KPIs of one simulated policy."""

    policy: str
    days: int
    total_demand: float
    served: float
    lost_units: float
    stockout_days: int
    fill_rate: float
    avg_inventory: float
    n_orders: int

    def to_dict(self) -> dict:
        """Plain-dict representation."""
        return asdict(self)


def simulate_policy(
    demand: np.ndarray,
    lead_time: int,
    initial_on_hand: float,
    decide_order: OrderPolicy,
    policy_name: str,
) -> SimulationResult:
    """Replay ``demand`` day by day under an ordering policy.

    Args:
        demand: Actual daily demand over the window.
        lead_time: Days between ordering and receipt (>= 1).
        initial_on_hand: Stock at the start of day 0.
        decide_order: ``f(day_index, inventory_position) -> order_qty``.
        policy_name: Label for the result.
    """
    lead_time = max(int(lead_time), 1)
    on_hand = float(initial_on_hand)
    pipeline: dict[int, float] = {}
    served_total = lost_total = 0.0
    stockout_days = n_orders = 0
    inventory_levels = []
    for day, d in enumerate(demand):
        on_hand += pipeline.pop(day, 0.0)
        served = min(on_hand, float(d))
        lost = float(d) - served
        on_hand -= served
        served_total += served
        lost_total += lost
        stockout_days += int(lost > 0)
        position = on_hand + sum(pipeline.values())
        qty = decide_order(day, position)
        if qty > 0:
            pipeline[day + lead_time] = pipeline.get(day + lead_time, 0.0) + qty
            n_orders += 1
        inventory_levels.append(on_hand)
    total = float(np.sum(demand))
    return SimulationResult(
        policy=policy_name,
        days=len(demand),
        total_demand=total,
        served=round(served_total, 1),
        lost_units=round(lost_total, 1),
        stockout_days=stockout_days,
        fill_rate=round(served_total / total, 4) if total else 1.0,
        avg_inventory=round(float(np.mean(inventory_levels)), 1),
        n_orders=n_orders,
    )


def naive_policy(avg_daily: float, trigger_days: float, cover_days: float, pack_size: int,
                 moq: int) -> OrderPolicy:
    """Fixed reorder threshold and fixed order quantity."""
    threshold = avg_daily * trigger_days
    qty = round_order_quantity(avg_daily * cover_days, pack_size, moq)

    def decide(_day: int, position: float) -> int:
        return qty if position <= threshold else 0

    return decide


def agentic_policy(
    forecast_means: list[float],
    review_days: int,
    demand_std: float,
    lead_time: float,
    lead_time_std: float,
    service_level: float,
    review_period: int,
    pack_size: int,
    moq: int,
) -> OrderPolicy:
    """Forecast-driven SS/ROP/order-up-to policy refreshed every ``review_days``."""
    z = z_value(service_level)

    def decide(day: int, position: float) -> int:
        mu = forecast_means[min(day // review_days, len(forecast_means) - 1)]
        ss = safety_stock(z, demand_std, mu, lead_time, lead_time_std)
        if position > reorder_point(mu, lead_time, ss):
            return 0
        return reorder_quantity(mu, lead_time, review_period, ss, position, pack_size, moq)

    return decide


def rolling_forecast_means(
    features: pd.DataFrame,
    start_index: int,
    days: int,
    review_days: int,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
) -> list[float]:
    """Mean forecast for each review block, refitting on data available at block start."""
    means = []
    for block_start in range(start_index, start_index + days, review_days):
        train = features.iloc[:block_start]
        future = features.iloc[block_start : block_start + review_days]
        yhat = fit_predict(train, future)
        means.append(float(np.nanmean(yhat)) if len(yhat) else float(train["units"].tail(28)
                                                                     .mean()))
    return means


def compare_policies(
    features: pd.DataFrame,
    params: dict,
    demand_std: float,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    days: int = 56,
    review_days: int = 7,
    service_level: float = 0.95,
    review_period: int = 7,
    naive_trigger_days: float = 7,
    naive_cover_days: float = 14,
) -> dict[str, SimulationResult]:
    """Run both policies on the last ``days`` of ``features`` for one item."""
    start = len(features) - days
    demand = features["units"].iloc[start:].to_numpy(dtype=float)
    lead_time = max(int(round(params["lead_time_days"])), 1)
    avg = float(features["units"].iloc[start - 28 : start].mean())
    initial = avg * (lead_time + review_period)
    pack, moq = int(params["pack_size"]), int(params["moq"])

    means = rolling_forecast_means(features, start, days, review_days, fit_predict)
    agentic = simulate_policy(
        demand, lead_time, initial,
        agentic_policy(means, review_days, demand_std, lead_time, float(params["lead_time_std"]),
                       service_level, review_period, pack, moq),
        "agentic",
    )
    naive = simulate_policy(
        demand, lead_time, initial,
        naive_policy(avg, naive_trigger_days, naive_cover_days, pack, moq), "naive",
    )
    return {"agentic": agentic, "naive": naive}


def aggregate(results: list[dict[str, SimulationResult]]) -> dict[str, dict[str, float]]:
    """Sum / average KPIs across items for each policy."""
    totals: dict[str, dict[str, float]] = {}
    for policy in ("agentic", "naive"):
        rows = [r[policy] for r in results]
        demand = sum(r.total_demand for r in rows)
        served = sum(r.served for r in rows)
        totals[policy] = {
            "stockout_days": float(sum(r.stockout_days for r in rows)),
            "lost_units": round(sum(r.lost_units for r in rows), 1),
            "fill_rate": round(served / demand, 4) if demand else 1.0,
            "avg_inventory": round(sum(r.avg_inventory for r in rows), 1),
            "n_orders": float(sum(r.n_orders for r in rows)),
        }
    return totals
