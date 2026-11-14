"""Out-of-sample accuracy against the real sales after the forecast origin."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.agents.schemas import PipelineResult
from src.data.repository import Repository
from src.evaluation.metrics import compute_metrics

SELECTED = "agent_selected"


def holdout_accuracy(results: Sequence[PipelineResult], repository: Repository) -> dict:
    """Compare each model's 7-day forecast with the observed sales.

    Returns:
        ``{"by_model": {model: metrics}, "by_item": [rows]}`` or an empty dict when
        no future actuals are available.
    """
    frames = []
    for r in results:
        actuals = repository.future_actuals(r.item_id)
        if actuals is None:
            continue
        actual_map = dict(zip(actuals["date"].dt.date, actuals["units"], strict=True))
        series = dict(r.forecast.model_forecasts)
        series[SELECTED] = r.forecast.forecast
        for model, points in series.items():
            for p in points:
                if p.date in actual_map:
                    frames.append({"item_id": r.item_id, "model": model, "date": p.date,
                                   "actual": float(actual_map[p.date]), "yhat": p.yhat})
    if not frames:
        return {}
    df = pd.DataFrame(frames)
    by_model = {
        model: compute_metrics(g["actual"], g["yhat"]).model_dump()
        for model, g in df.groupby("model")
    }
    by_item = []
    for item, g in df[df["model"] == SELECTED].groupby("item_id"):
        m = compute_metrics(g["actual"], g["yhat"])
        by_item.append({"item_id": item, "actual_7d": float(g["actual"].sum()),
                        "forecast_7d": round(float(g["yhat"].sum()), 1), **m.model_dump()})
    return {"by_model": by_model, "by_item": by_item}
