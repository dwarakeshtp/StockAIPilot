"""Inventory Planning Agent: safety stock, reorder point, reorder quantity, stockout risk."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.schemas import (
    ForecastOutput,
    InventoryAction,
    InventoryOutput,
    InventoryParams,
    ProjectedStockPoint,
    RiskOutput,
)
from src.inventory import (
    days_of_cover,
    economic_order_quantity,
    project_stock,
    reorder_point,
    reorder_quantity,
    safety_stock,
    stockout_probability,
    z_value,
)

SUPPORTED_OVERRIDES: frozenset[str] = frozenset(
    {"lead_time_days", "service_level", "demand_multiplier", "on_hand", "lead_time_std"}
)


class InventoryAgent(BaseAgent):
    """Turns the forecast and risk assessment into an inventory recommendation."""

    name = "inventory_agent"
    role = "Inventory Planning Agent - replenishment planner"

    def run(
        self,
        item_id: str,
        forecast: ForecastOutput,
        risk: RiskOutput,
        params: dict[str, Any] | InventoryParams,
        overrides: dict[str, float] | None = None,
    ) -> InventoryOutput:
        """Compute the replenishment recommendation for ``item_id``.

        Args:
            item_id: Item identifier.
            forecast: Forecast Agent output (mean demand and backtest RMSE).
            risk: Risk Agent output (uncertainty multiplier).
            params: Inventory parameters (simulated in this POC).
            overrides: Optional what-if overrides (see :data:`SUPPORTED_OVERRIDES`).

        Returns:
            A validated :class:`InventoryOutput`.

        Raises:
            ValueError: on unsupported override keys.
        """
        overrides = {k: float(v) for k, v in (overrides or {}).items() if v is not None}
        unknown = set(overrides) - SUPPORTED_OVERRIDES
        if unknown:
            raise ValueError(f"Unsupported overrides: {sorted(unknown)}")

        base = params if isinstance(params, InventoryParams) else InventoryParams(**params)
        updates = {k: overrides[k] for k in ("lead_time_days", "on_hand", "lead_time_std")
                   if k in overrides}
        p = base.model_copy(update=updates)
        cfg = self.settings.inventory
        service_level = overrides.get("service_level", cfg.service_level)
        multiplier = overrides.get("demand_multiplier", 1.0)

        daily = [pt.yhat * multiplier for pt in forecast.forecast]
        mu = sum(daily) / len(daily)
        sigma = forecast.selected_metrics.rmse * multiplier
        sigma_adj = sigma * risk.uncertainty_multiplier
        z = z_value(service_level)
        lt = p.lead_time_days
        ss = safety_stock(z, sigma_adj, mu, lt, p.lead_time_std)
        rop = reorder_point(mu, lt, ss)
        ip = p.on_hand + p.on_order
        p_stockout = stockout_probability(ip, mu, sigma_adj, lt, p.lead_time_std)

        horizon_days = max(len(daily), math.ceil(lt))
        demand_path = daily + [mu] * (horizon_days - len(daily))
        start = forecast.forecast[0].date
        dates = [start + timedelta(days=i) for i in range(horizon_days)]
        projection = project_stock(p.on_hand, demand_path, dates, p.on_order,
                                   p.on_order_eta_days)

        needs_order = ip <= rop
        qty = (
            reorder_quantity(mu, lt, cfg.review_period_days, ss, ip, p.pack_size, p.moq)
            if needs_order else 0
        )
        stockout_before_arrival = (
            projection.stockout_day_index is not None and projection.stockout_day_index <= lt
        )
        if needs_order and stockout_before_arrival:
            action = InventoryAction.EXPEDITE
        elif needs_order:
            action = InventoryAction.REORDER
        else:
            action = InventoryAction.HOLD

        eoq = economic_order_quantity(mu * 365, cfg.ordering_cost, p.unit_cost,
                                      p.holding_cost_rate)
        cover = days_of_cover(ip, mu)
        reasoning = [
            f"Mean daily demand {mu:.1f} (7-day forecast, x{multiplier:g} what-if multiplier).",
            f"Demand std {sigma:.2f} (backtest RMSE) x{risk.uncertainty_multiplier:.1f} risk "
            f"multiplier = {sigma_adj:.2f}.",
            f"Service level {service_level:.0%} -> z = {z:.2f}; lead time {lt:g}d "
            f"(std {p.lead_time_std:g}d).",
            f"Safety stock {ss:.1f}; reorder point {rop:.1f}; inventory position {ip:.0f} "
            f"({p.on_hand:.0f} on hand + {p.on_order:.0f} on order).",
            f"Probability of stockout before replenishment: {p_stockout:.0%}; "
            f"days of cover {cover:.1f}.",
        ]
        if projection.stockout_date:
            reasoning.append(f"Projected stockout on {projection.stockout_date} "
                             f"(day {projection.stockout_day_index}) without a new order.")
        reasoning.append(
            f"Action {action.value}: "
            + (f"order {qty} units (pack {p.pack_size}, MOQ {p.moq})." if qty
               else "inventory position above reorder point.")
        )

        output = InventoryOutput(
            item_id=item_id,
            params=p,
            service_level=service_level,
            z_value=round(z, 4),
            review_period_days=cfg.review_period_days,
            mean_daily_demand=round(mu, 3),
            demand_std=round(sigma, 3),
            adjusted_demand_std=round(sigma_adj, 3),
            lead_time_demand=round(mu * lt, 2),
            safety_stock=round(ss, 2),
            reorder_point=round(rop, 2),
            inventory_position=round(ip, 2),
            reorder_qty=qty,
            eoq=round(eoq, 1),
            order_value=round(qty * p.unit_cost, 2),
            days_of_cover=round(cover, 2),
            stockout_probability=round(min(max(p_stockout, 0.0), 1.0), 4),
            projected_stockout_date=projection.stockout_date,
            projected_stock=[
                ProjectedStockPoint(date=d, demand=round(dm, 2), arrivals=a, stock=s)
                for d, dm, a, s in zip(projection.dates, projection.demand, projection.arrivals,
                                       projection.stock, strict=True)
            ],
            action=action,
            overrides_applied=overrides,
            reasoning=reasoning,
        )
        output.narrative = self.narrate(
            "Explain the replenishment recommendation and the main driver behind it.",
            {
                "item_id": item_id,
                "action": action.value,
                "reorder_qty": qty,
                "safety_stock": round(ss, 1),
                "reorder_point": round(rop, 1),
                "inventory_position": round(ip, 1),
                "days_of_cover": round(cover, 1),
                "stockout_probability": round(p_stockout, 2),
                "lead_time_days": lt,
                "service_level": service_level,
                "mean_daily_demand": round(mu, 1),
                "projected_stockout_date": str(projection.stockout_date or "none"),
            },
        )
        self.logger.info("inventory_planned", item_id=item_id, action=action.value, qty=qty,
                         stockout_probability=round(p_stockout, 3))
        return output
