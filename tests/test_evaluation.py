"""Tests for the Phase 9 evaluation modules."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_item_sales
from src.data.features import build_features
from src.data.repository import DataRepository
from src.evaluation.holdout import SELECTED, holdout_accuracy
from src.evaluation.inventory_simulation import (
    aggregate,
    compare_policies,
    naive_policy,
    simulate_policy,
)
from src.evaluation.report import render_report
from src.evaluation.scenario_eval import Scenario, detected_flags, evaluate_scenarios
from src.orchestration.runner import PipelineRunner


def test_simulate_policy_no_orders_depletes_stock() -> None:
    res = simulate_policy(np.full(10, 5.0), lead_time=2, initial_on_hand=20,
                          decide_order=lambda d, p: 0, policy_name="none")
    assert res.served == 20 and res.lost_units == 30
    assert res.stockout_days == 6
    assert res.fill_rate == pytest.approx(0.4)


def test_simulate_policy_order_arrives_after_lead_time() -> None:
    orders = {0: 50}
    res = simulate_policy(np.full(6, 10.0), lead_time=3, initial_on_hand=20,
                          decide_order=lambda d, p: orders.get(d, 0), policy_name="one")
    # Days 0-1 are served from stock, day 2 is short, the order lands on day 3.
    assert res.n_orders == 1
    assert res.stockout_days == 1
    assert res.lost_units == 10


def test_naive_policy_fixed_threshold() -> None:
    decide = naive_policy(avg_daily=10, trigger_days=7, cover_days=14, pack_size=12, moq=1)
    assert decide(0, 80) == 0
    assert decide(0, 70) == 144  # 140 rounded up to a pack of 12


def test_compare_policies_runs_on_item_history(item_sales) -> None:
    feats = build_features(clean_item_sales(item_sales)[0])
    params = {"lead_time_days": 3, "lead_time_std": 0.5, "pack_size": 6, "moq": 6}

    def fit_predict(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        return np.full(len(future), train["units"].tail(28).mean())

    out = compare_policies(feats, params, demand_std=feats["units"].std(),
                           fit_predict=fit_predict, days=28)
    assert set(out) == {"agentic", "naive"}
    for res in out.values():
        assert res.days == 28
        assert 0 <= res.fill_rate <= 1
    totals = aggregate([out])
    assert totals["agentic"]["stockout_days"] == out["agentic"].stockout_days


@pytest.fixture
def pipeline_result(settings, repository):
    return PipelineRunner(settings, repository=repository).run_item(repository.item_ids[1])


def test_holdout_accuracy_uses_future_actuals(pipeline_result, repository) -> None:
    assert holdout_accuracy([pipeline_result], repository) == {}
    dates = pd.to_datetime([p.date for p in pipeline_result.forecast.forecast])
    actuals = pd.DataFrame({"date": dates, "item_id": pipeline_result.item_id, "units": 10.0})
    with_actuals = DataRepository(repository.settings, dataset=dataclasses.replace(
        repository.dataset, future_actuals=actuals))
    out = holdout_accuracy([pipeline_result], with_actuals)
    assert SELECTED in out["by_model"]
    assert out["by_model"][SELECTED]["n"] == 7
    assert out["by_item"][0]["item_id"] == pipeline_result.item_id


def test_detected_flags_include_critic_rules(pipeline_result) -> None:
    flags = detected_flags(pipeline_result)
    for issue in pipeline_result.critic.issues:
        if issue.severity in ("WARNING", "CRITICAL"):
            assert issue.rule in flags


def test_evaluate_scenarios_zero_stock(settings, repository) -> None:
    item = repository.item_ids[0]
    scenarios = [
        Scenario("zero_stock", item, "no stock", {"high_stockout_probability"},
                 {"APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 params_fn=lambda p, m: {**p, "on_hand": 0, "on_order": 0},
                 acceptable_flags={"risk:spike", "excessive_order"}),
        Scenario("missing_item", "NOT_AN_ITEM", "skipped", set(), {"APPROVED"}),
    ]
    out = evaluate_scenarios(scenarios, repository.dataset, repository.inventory_params, settings)
    assert out["n_scenarios"] == 1
    assert out["recall"] == 1.0
    assert out["scenarios"][0]["action"] == "EXPEDITE"


def test_render_report_sections() -> None:
    summary = {
        "generated_at": "2026-01-01T00:00:00", "data_source": "test", "n_items": 1,
        "holdout_accuracy": {"by_model": {"arima": {"mae": 1.0, "rmse": 1.5, "mape": 0.2,
                                                    "wape": 0.1, "n": 7}}, "by_item": []},
        "inventory_simulation": {
            "days": 56,
            "totals": {p: {"stockout_days": 1.0, "lost_units": 2.0, "fill_rate": 0.9,
                           "avg_inventory": 10.0, "n_orders": 3.0} for p in ("agentic", "naive")},
            "by_item": [{"item_id": "A", "stockout_days_agentic": 1}],
        },
        "scenario_evaluation": {"n_scenarios": 1, "recall": 1.0, "precision": 1.0,
                                "verdict_accuracy": 1.0, "grounding_pass_rate": 1.0,
                                "scenarios": [{"scenario": "s", "item_id": "A",
                                               "verdict": "APPROVED"}]},
    }
    md = render_report(summary)
    for heading in ("# Evaluation Report", "## 1.", "## 2.", "## 3.", "## Limitations"):
        assert heading in md
