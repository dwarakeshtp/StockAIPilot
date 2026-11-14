"""Run the full evaluation suite and write ``outputs/evaluation/``.

1. Hold-out accuracy of the 7-day forecasts vs real M5 sales.
2. Inventory policy simulation (agentic vs naive) over the last 56 days.
3. Critic scenario evaluation (injected anomalies / data issues / supply issues).

Usage:
    python scripts/run_evaluation.py            # requires outputs/runs/latest.json
    python scripts/run_evaluation.py --skip-scenarios
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mpl"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config.settings import load_settings  # noqa: E402
from src.data.cleaning import clean_item_sales  # noqa: E402
from src.data.features import PROPHET_REGRESSORS, build_features, prophet_holidays  # noqa: E402
from src.data.repository import DataRepository  # noqa: E402
from src.evaluation.holdout import holdout_accuracy  # noqa: E402
from src.evaluation.inventory_simulation import aggregate, compare_policies  # noqa: E402
from src.evaluation.report import render_report  # noqa: E402
from src.evaluation.scenario_eval import default_scenarios, evaluate_scenarios  # noqa: E402
from src.logging.logger import configure_logging  # noqa: E402
from src.models import ARIMAForecaster, ProphetForecaster  # noqa: E402
from src.orchestration.runner import PipelineRunner, load_latest_results  # noqa: E402
from src.utils.ids import utc_now  # noqa: E402
from src.utils.io import read_json, write_json  # noqa: E402


def _fit_predict_factory(model: str, arima_order, holidays, settings):
    cfg = settings.forecast

    def arima(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        m = ARIMAForecaster(order=tuple(arima_order or (1, 0, 1)),
                            seasonal_order=cfg.arima_seasonal_order,
                            fit_window=cfg.arima_fit_window)
        return m.fit(train).predict(future[["date"]])["yhat"].to_numpy()

    def prophet(train: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        m = ProphetForecaster(holidays=holidays, uncertainty_samples=20)
        return m.fit(train).predict(future[["date", *PROPHET_REGRESSORS]])["yhat"].to_numpy()

    if model == "prophet":
        return prophet
    if model == "ensemble":
        return lambda tr, fu: (arima(tr, fu) + prophet(tr, fu)) / 2
    return arima


def run_inventory_simulation(results, repo: DataRepository, settings, days: int = 56) -> dict:
    """Simulate both policies for every item."""
    per_item, raw = [], []
    for r in results:
        clean, _ = clean_item_sales(repo.item_sales(r.item_id))
        feats = build_features(clean)
        holidays = prophet_holidays(feats)
        fit_predict = _fit_predict_factory(r.forecast.model_used, r.forecast.arima_order,
                                           holidays, settings)
        params = repo.item_inventory_params(r.item_id)
        sigma = r.forecast.selected_metrics.rmse * r.risk.uncertainty_multiplier
        out = compare_policies(feats, params, sigma, fit_predict, days=days,
                               service_level=settings.inventory.service_level,
                               review_period=settings.inventory.review_period_days)
        raw.append(out)
        a, n = out["agentic"], out["naive"]
        per_item.append({
            "item_id": r.item_id, "model": r.forecast.model_used,
            "stockout_days_agentic": a.stockout_days, "stockout_days_naive": n.stockout_days,
            "fill_rate_agentic": a.fill_rate, "fill_rate_naive": n.fill_rate,
            "avg_inv_agentic": a.avg_inventory, "avg_inv_naive": n.avg_inventory,
            "orders_agentic": a.n_orders, "orders_naive": n.n_orders,
        })
        print(f"  {r.item_id}: stockout days agentic={a.stockout_days} naive={n.stockout_days}, "
              f"fill {a.fill_rate:.1%} vs {n.fill_rate:.1%}", flush=True)
    return {"days": days, "totals": aggregate(raw), "by_item": per_item}


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-scenarios", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.outputs.logs_dir, level="WARNING")
    settings.ensure_output_dirs()
    repo = DataRepository(settings)
    results = load_latest_results(settings)
    if not results:
        print("No saved results - running the pipeline first ...")
        runner = PipelineRunner(settings, repository=repo)
        results = runner.run_items()
        runner.save_results(results)

    t0 = time.perf_counter()
    summary_path = settings.outputs.evaluation_dir / "evaluation_summary.json"
    previous = read_json(summary_path) if summary_path.exists() else {}
    summary: dict = {
        "generated_at": utc_now(), "data_source": repo.dataset.source, "n_items": len(results),
        **{k: previous[k] for k in ("inventory_simulation", "scenario_evaluation")
           if k in previous},
    }

    print("1/3 Hold-out accuracy ...")
    summary["holdout_accuracy"] = holdout_accuracy(results, repo)

    if not args.skip_simulation:
        print("2/3 Inventory policy simulation (refits models weekly) ...")
        summary["inventory_simulation"] = run_inventory_simulation(results, repo, settings)

    if not args.skip_scenarios:
        print("3/3 Critic scenario evaluation ...")
        scen_settings = settings.model_copy(update={
            "outputs": settings.outputs.model_copy(
                update={"root": settings.outputs.evaluation_dir / "scenario_runs"})})
        summary["scenario_evaluation"] = evaluate_scenarios(
            default_scenarios(), repo.dataset, repo.inventory_params, scen_settings)

    out_dir = settings.outputs.evaluation_dir
    path = write_json(summary_path, summary)
    if summary.get("holdout_accuracy"):
        pd.DataFrame(summary["holdout_accuracy"]["by_model"]).T.to_csv(
            out_dir / "holdout_by_model.csv")
    if summary.get("inventory_simulation"):
        pd.DataFrame(summary["inventory_simulation"]["by_item"]).to_csv(
            out_dir / "inventory_simulation.csv", index=False)
    if summary.get("scenario_evaluation"):
        pd.DataFrame(summary["scenario_evaluation"]["scenarios"]).to_csv(
            out_dir / "scenario_evaluation.csv", index=False)
    report = ROOT / "docs" / "evaluation_report.md"
    report.write_text(render_report(summary), encoding="utf-8")
    print(f"\nEvaluation written to {path} and {report} in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
