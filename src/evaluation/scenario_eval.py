"""Scenario-based evaluation of the Critic (agent collaboration quality).

Each scenario perturbs one item's sales history and/or inventory parameters and
declares which flags the system should raise and which verdicts are acceptable.
Flags are Critic rule ids (WARNING/CRITICAL) plus Risk-Agent signals
(``risk:spike``, ``risk:drop``) and ``revision`` when the Critic loop fired.
Precision is computed over the monitored flag universe (the union of all
expected flags). ``acceptable_flags`` are legitimate side effects of a scenario
(e.g. a 21-day lead time also produces a large order); they are neither required
nor counted as false positives in ``precision`` but are in ``strict_precision``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.agents.schemas import PipelineResult
from src.config.settings import Settings
from src.data.ingestion import SalesDataset
from src.data.repository import DataRepository
from src.orchestration.runner import PipelineRunner

SalesFn = Callable[[pd.DataFrame], pd.DataFrame]
ParamsFn = Callable[[dict, float], dict]


@dataclass
class Scenario:
    """One evaluation scenario."""

    name: str
    item_id: str
    description: str
    expected_flags: set[str]
    expected_verdicts: set[str]
    sales_fn: SalesFn | None = None
    params_fn: ParamsFn | None = None
    acceptable_flags: set[str] = field(default_factory=set)


def _scale_tail(days: int, factor: float) -> SalesFn:
    def fn(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.loc[out.index[-days:], "units"] = (out["units"].iloc[-days:] * factor).round()
        return out
    return fn


def _gappy(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    out = df.copy()
    out.loc[out.index[300:340], "units"] = 0  # 40-day zero streak
    drop = rng.choice(np.arange(30, len(out) - 30), size=40, replace=False)
    return out.drop(index=out.index[drop])


def _noisy(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    out = df.copy()
    tail = out.index[-120:]
    out.loc[tail, "units"] = (out.loc[tail, "units"] * rng.lognormal(0, 1.0, len(tail))).round()
    return out


def _params(**updates) -> ParamsFn:
    def fn(params: dict, mean_daily: float) -> dict:
        resolved = {k: (v(mean_daily) if callable(v) else v) for k, v in updates.items()}
        return {**params, **resolved}
    return fn


def default_scenarios() -> list[Scenario]:
    """The scenario suite used in the evaluation report."""
    healthy = _params(on_hand=lambda m: round(m * 18), on_order=0)
    return [
        Scenario("clean_baseline", "FOODS_3_252", "Stable item, healthy stock",
                 set(), {"APPROVED"}, params_fn=healthy),
        Scenario("demand_spike", "FOODS_3_586", "Last 7 days x2.5 (viral / promo)",
                 {"risk:spike"}, {"APPROVED", "APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 sales_fn=_scale_tail(7, 2.5), params_fn=healthy),
        Scenario("demand_drop", "FOODS_3_064", "Last 7 days x0.3 (competitor / recall)",
                 {"risk:drop"}, {"APPROVED", "APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 sales_fn=_scale_tail(7, 0.3), params_fn=healthy),
        Scenario("data_gaps", "FOODS_3_714", "40 missing days + 40-day zero streak",
                 {"data_quality"}, {"APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 sales_fn=_gappy, params_fn=healthy),
        Scenario("noisy_history", "FOODS_3_252", "Heavy multiplicative noise in last 120 days",
                 {"forecast_accuracy"}, {"APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 sales_fn=_noisy, params_fn=healthy),
        Scenario("intermittent_item", "FOODS_3_519", "Low-volume intermittent SKU",
                 {"forecast_accuracy", "model_disagreement", "revision"},
                 {"APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 acceptable_flags={"high_stockout_probability", "risk:spike"}),
        Scenario("long_lead_time", "FOODS_3_282", "Supplier lead time 21 days, low stock",
                 {"high_stockout_probability"}, {"APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 params_fn=_params(lead_time_days=21, on_hand=lambda m: round(m * 5),
                                   on_order=0),
                 acceptable_flags={"excessive_order"}),
        Scenario("zero_stock", "FOODS_3_587", "Out of stock today",
                 {"high_stockout_probability"}, {"APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 params_fn=_params(on_hand=0, on_order=0),
                 acceptable_flags={"risk:spike"}),
        Scenario("huge_moq", "FOODS_3_455", "Supplier MOQ of 2,400 units",
                 {"excessive_order"}, {"APPROVED_WITH_WARNINGS", "NEEDS_HUMAN_REVIEW"},
                 params_fn=_params(moq=2400, pack_size=12, on_hand=lambda m: round(m * 2),
                                   on_order=0),
                 acceptable_flags={"high_stockout_probability"}),
        Scenario("late_open_order", "FOODS_3_120", "Open order arrives after stock runs out",
                 {"stockout_before_open_order"}, {"NEEDS_HUMAN_REVIEW"},
                 params_fn=_params(on_hand=lambda m: round(m * 2), on_order=lambda m: round(m * 20),
                                   on_order_eta_days=9, lead_time_days=3)),
        Scenario("overstocked", "FOODS_3_090", "60 days of cover already on hand",
                 set(), {"APPROVED", "APPROVED_WITH_WARNINGS"},
                 params_fn=_params(on_hand=lambda m: round(m * 60), on_order=0)),
    ]


def detected_flags(result: PipelineResult) -> set[str]:
    """Flags raised by the system for a pipeline result."""
    flags = {i.rule for i in result.critic.issues if i.severity in ("WARNING", "CRITICAL")}
    if result.risk.spike_detected:
        flags.add("risk:spike")
    if result.risk.drop_detected:
        flags.add("risk:drop")
    if result.revision_count > 0:
        flags.add("revision")
    return flags


def run_scenario(scenario: Scenario, base: SalesDataset, base_params: pd.DataFrame,
                 settings: Settings, runner_kwargs: dict | None = None) -> PipelineResult:
    """Apply a scenario's perturbations and run the full pipeline for its item."""
    sales = base.sales
    if scenario.sales_fn is not None:
        item_rows = sales[sales["item_id"] == scenario.item_id].reset_index(drop=True)
        others = sales[sales["item_id"] != scenario.item_id]
        sales = pd.concat([others, scenario.sales_fn(item_rows)], ignore_index=True)
    dataset = SalesDataset(sales=sales, future_calendar=base.future_calendar,
                           source=f"{base.source}+scenario", future_actuals=None)
    repo = DataRepository(settings, dataset=dataset)
    params = base_params.copy()
    if scenario.params_fn is not None:
        idx = params.index[params["item_id"] == scenario.item_id][0]
        mean_daily = float(base.sales[base.sales["item_id"] == scenario.item_id]["units"]
                           .tail(28).mean())
        updated = scenario.params_fn(params.loc[idx].to_dict(), mean_daily)
        for key, value in updated.items():
            params.loc[idx, key] = value
    repo.__dict__["inventory_params"] = params
    return PipelineRunner(settings, repository=repo, **(runner_kwargs or {})).run_item(
        scenario.item_id
    )


def evaluate_scenarios(
    scenarios: list[Scenario], base: SalesDataset, base_params: pd.DataFrame, settings: Settings,
    runner_kwargs: dict | None = None,
) -> dict:
    """Run every scenario and compute detection precision/recall and verdict accuracy."""
    universe = set().union(*(s.expected_flags for s in scenarios))
    rows, tp, fp, fn, fp_strict, risk_hits = [], 0, 0, 0, 0, 0
    verdict_hits = grounded = 0
    available = set(base.sales["item_id"].unique())
    for s in scenarios:
        if s.item_id not in available:
            continue
        result = run_scenario(s, base, base_params, settings, runner_kwargs)
        found = detected_flags(result) & universe
        expected = s.expected_flags
        hit, miss, extra = found & expected, expected - found, found - expected
        tp, fn, fp_strict = tp + len(hit), fn + len(miss), fp_strict + len(extra)
        fp += len(extra - s.acceptable_flags)
        risk_hits += sum(1 for f in hit if f.startswith("risk:"))
        verdict_ok = result.critic.verdict.value in s.expected_verdicts
        verdict_hits += int(verdict_ok)
        grounded += int(result.critic.grounding_passed)
        rows.append({
            "scenario": s.name,
            "item_id": s.item_id,
            "description": s.description,
            "expected_flags": ", ".join(sorted(s.expected_flags)) or "-",
            "detected_flags": ", ".join(sorted(found)) or "-",
            "missed": ", ".join(sorted(miss)) or "-",
            "unexpected": ", ".join(sorted(extra - s.acceptable_flags)) or "-",
            "action": result.inventory.action.value,
            "qty": result.inventory.reorder_qty,
            "verdict": result.critic.verdict.value,
            "verdict_ok": verdict_ok,
        })
    n = len(rows)
    return {
        "n_scenarios": n,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 1.0,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
        "strict_precision": round(tp / (tp + fp_strict), 4) if tp + fp_strict else 1.0,
        "verdict_accuracy": round(verdict_hits / n, 4) if n else 0.0,
        "grounding_pass_rate": round(grounded / n, 4) if n else 0.0,
        "recall_without_critic": round(risk_hits / (tp + fn), 4) if tp + fn else 0.0,
        "scenarios": rows,
    }
