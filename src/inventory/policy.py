"""Rule-based inventory policy: safety stock, reorder point and order quantity.

Formulas (continuous review with periodic ordering opportunity):

* z            = Phi^-1(service_level)
* sigma_LT     = sqrt(L * sigma_d^2 + mu_d^2 * sigma_L^2)
* safety stock = z * sigma_LT
* reorder point (ROP) = mu_d * L + SS
* order-up-to level   = mu_d * (L + R) + SS
* reorder qty  = max(0, order-up-to - inventory position), rounded up to the
  pack size and to at least the MOQ when positive
* EOQ (reference only) = sqrt(2 * D_annual * S / (c * h))
"""

from __future__ import annotations

import math

from scipy.stats import norm


def z_value(service_level: float) -> float:
    """Standard-normal quantile for a cycle service level in (0.5, 1).

    Raises:
        ValueError: if ``service_level`` is outside (0.5, 1).
    """
    if not 0.5 < service_level < 1.0:
        raise ValueError("service_level must be in (0.5, 1.0)")
    return float(norm.ppf(service_level))


def lead_time_demand_std(
    demand_std: float, mean_demand: float, lead_time: float, lead_time_std: float = 0.0
) -> float:
    """Standard deviation of demand over a stochastic lead time."""
    if lead_time <= 0:
        raise ValueError("lead_time must be positive")
    variance = lead_time * demand_std**2 + (mean_demand**2) * (lead_time_std**2)
    return math.sqrt(max(variance, 0.0))


def safety_stock(
    z: float, demand_std: float, mean_demand: float, lead_time: float, lead_time_std: float = 0.0
) -> float:
    """Safety stock covering demand and lead-time variability."""
    return max(z, 0.0) * lead_time_demand_std(demand_std, mean_demand, lead_time, lead_time_std)


def reorder_point(mean_demand: float, lead_time: float, safety: float) -> float:
    """Inventory position at which a replenishment order should be placed."""
    return mean_demand * lead_time + safety


def order_up_to_level(mean_demand: float, lead_time: float, review_period: float,
                      safety: float) -> float:
    """Target inventory position after ordering (covers lead time + review period)."""
    return mean_demand * (lead_time + review_period) + safety


def round_order_quantity(raw_qty: float, pack_size: int = 1, moq: int = 1) -> int:
    """Round a raw quantity up to full packs and to at least the MOQ (0 stays 0)."""
    if raw_qty <= 0:
        return 0
    pack = max(int(pack_size), 1)
    qty = math.ceil(raw_qty / pack) * pack
    min_qty = math.ceil(max(int(moq), 1) / pack) * pack
    return int(max(qty, min_qty))


def reorder_quantity(
    mean_demand: float,
    lead_time: float,
    review_period: float,
    safety: float,
    inventory_position: float,
    pack_size: int = 1,
    moq: int = 1,
) -> int:
    """Order quantity bringing the inventory position up to the order-up-to level."""
    target = order_up_to_level(mean_demand, lead_time, review_period, safety)
    return round_order_quantity(target - inventory_position, pack_size, moq)


def economic_order_quantity(
    annual_demand: float, ordering_cost: float, unit_cost: float, holding_rate: float
) -> float:
    """Classic EOQ; returns 0 when demand or holding cost is non-positive."""
    holding = unit_cost * holding_rate
    if annual_demand <= 0 or holding <= 0 or ordering_cost <= 0:
        return 0.0
    return math.sqrt(2 * annual_demand * ordering_cost / holding)
