"""Tests for inventory policy functions and the Inventory Planning Agent."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from factories import make_forecast, make_params, make_risk

from src.agents.inventory_agent import InventoryAgent
from src.agents.schemas import InventoryAction, RiskLevel
from src.inventory import (
    days_of_cover,
    economic_order_quantity,
    lead_time_demand_std,
    project_stock,
    reorder_point,
    reorder_quantity,
    round_order_quantity,
    safety_stock,
    stockout_probability,
    z_value,
)


def test_z_value() -> None:
    assert z_value(0.95) == pytest.approx(1.6449, abs=1e-4)
    with pytest.raises(ValueError):
        z_value(1.0)


def test_safety_stock_hand_calculation() -> None:
    # sigma_LT = sqrt(4 * 3^2 + 10^2 * 1^2) = sqrt(136)
    assert lead_time_demand_std(3, 10, 4, 1) == pytest.approx(math.sqrt(136))
    assert safety_stock(1.65, 3, 10, 4, 1) == pytest.approx(1.65 * math.sqrt(136))
    assert safety_stock(1.65, 3, 10, 4, 0) == pytest.approx(1.65 * 6)


def test_reorder_point() -> None:
    assert reorder_point(10, 4, 9.9) == pytest.approx(49.9)


@pytest.mark.parametrize(
    ("raw", "pack", "moq", "expected"),
    [(0, 6, 12, 0), (-5, 6, 12, 0), (1, 6, 12, 12), (13, 6, 12, 18), (25, 1, 1, 25),
     (7, 12, 5, 12)],
)
def test_round_order_quantity(raw: float, pack: int, moq: int, expected: int) -> None:
    assert round_order_quantity(raw, pack, moq) == expected


def test_reorder_quantity_order_up_to() -> None:
    # target = 10 * (4 + 7) + 10 = 120; IP 50 -> raw 70 -> packs of 6 -> 72
    assert reorder_quantity(10, 4, 7, 10, 50, pack_size=6, moq=12) == 72
    assert reorder_quantity(10, 4, 7, 10, 500) == 0


def test_eoq() -> None:
    assert economic_order_quantity(3650, 25, 2.0, 0.25) == pytest.approx(
        math.sqrt(2 * 3650 * 25 / 0.5))
    assert economic_order_quantity(0, 25, 2, 0.25) == 0


def test_stockout_probability_monotonic() -> None:
    low = stockout_probability(100, 10, 3, 4)
    mid = stockout_probability(40, 10, 3, 4)
    high = stockout_probability(10, 10, 3, 4)
    assert low < mid < high
    assert mid == pytest.approx(0.5)
    assert stockout_probability(30, 10, 0, 4) == 1.0
    assert stockout_probability(50, 10, 0, 4) == 0.0


def test_days_of_cover() -> None:
    assert days_of_cover(50, 10) == 5
    assert days_of_cover(50, 0) == 999.0


def test_project_stock_with_arrival() -> None:
    dates = [date(2016, 5, 23) + timedelta(days=i) for i in range(5)]
    proj = project_stock(25, [10] * 5, dates, on_order=20, on_order_eta_days=3)
    assert proj.stock == [15, 5, 15, 5, -5]
    assert proj.arrivals[2] == 20
    assert proj.stockout_date == dates[4] and proj.stockout_day_index == 5
    with pytest.raises(ValueError):
        project_stock(1, [1, 2], dates)


def test_agent_hold_when_well_stocked(settings) -> None:
    out = InventoryAgent(settings).run("X", make_forecast(10), make_risk(),
                                       make_params(on_hand=500))
    assert out.action == InventoryAction.HOLD
    assert out.reorder_qty == 0
    assert out.stockout_probability < 0.01


def test_agent_reorder_and_expedite(settings) -> None:
    agent = InventoryAgent(settings)
    # ROP = 10*4 + 1.645*2*2 = 46.6 > IP 45; stockout projected on day 5 > lead time 4
    reorder = agent.run("X", make_forecast(10), make_risk(), make_params(on_hand=45))
    assert reorder.action == InventoryAction.REORDER
    assert reorder.reorder_qty > 0 and reorder.reorder_qty % 6 == 0
    assert reorder.order_value == pytest.approx(reorder.reorder_qty * 2.0)

    expedite = agent.run("X", make_forecast(10), make_risk(), make_params(on_hand=20))
    assert expedite.action == InventoryAction.EXPEDITE
    assert expedite.projected_stockout_date is not None


def test_agent_hand_calculated_values(settings) -> None:
    out = InventoryAgent(settings).run("X", make_forecast(10, rmse=3), make_risk(),
                                       make_params(on_hand=50, lead_time_days=4))
    z = z_value(0.95)
    assert out.safety_stock == pytest.approx(z * 3 * 2, abs=0.01)
    assert out.reorder_point == pytest.approx(40 + z * 6, abs=0.01)
    assert out.inventory_position == 50


def test_agent_risk_multiplier_increases_safety_stock(settings) -> None:
    agent = InventoryAgent(settings)
    low = agent.run("X", make_forecast(), make_risk(), make_params())
    high = agent.run("X", make_forecast(), make_risk(RiskLevel.HIGH, 30, 1.4), make_params())
    assert high.safety_stock == pytest.approx(low.safety_stock * 1.4, rel=1e-3)


def test_agent_what_if_overrides(settings) -> None:
    agent = InventoryAgent(settings)
    base = agent.run("X", make_forecast(), make_risk(), make_params(on_hand=500))
    shocked = agent.run("X", make_forecast(), make_risk(), make_params(on_hand=500),
                        overrides={"demand_multiplier": 3.0, "lead_time_days": 12,
                                   "service_level": 0.99})
    assert shocked.reorder_point > base.reorder_point
    assert shocked.params.lead_time_days == 12
    assert shocked.overrides_applied["demand_multiplier"] == 3.0
    with pytest.raises(ValueError):
        agent.run("X", make_forecast(), make_risk(), make_params(), overrides={"bogus": 1})
