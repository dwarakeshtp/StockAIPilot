"""Forecast Agent: backtests ARIMA and Prophet, selects a model, forecasts 7 days.

Responsibilities
----------------
1. Backtest Seasonal-Naive (reference), ARIMA and Prophet with rolling origins.
2. Select ARIMA or Prophet by the configured metric (the naive model is only a
   yardstick for forecast skill; the ensemble is used on Critic request).
3. Refit on the full history and produce the horizon forecast with intervals.
4. Report MAE / RMSE / MAPE / WAPE, model agreement and reasoning steps.

Model runs are cached per item so a Critic-requested revision only re-selects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.agents.llm_client import LLMClient
from src.agents.schemas import (
    BacktestPoint,
    ForecastOutput,
    ForecastPoint,
    RevisionHint,
    RevisionRequest,
)
from src.config.settings import Settings
from src.data.features import PROPHET_REGRESSORS
from src.evaluation.backtest import BacktestResult, ensemble_backtest, rolling_origin_backtest
from src.models import (
    ARIMAForecaster,
    ProphetForecaster,
    SeasonalNaiveForecaster,
    average_forecasts,
)

PRIMARY_MODELS: tuple[str, ...] = ("arima", "prophet")
ENSEMBLE_NAME = "ensemble"
NAIVE_NAME = "seasonal_naive"


@dataclass
class _ModelRun:
    backtests: dict[str, BacktestResult] = field(default_factory=dict)
    forecasts: dict[str, pd.DataFrame] = field(default_factory=dict)
    arima_order: tuple[int, int, int] | None = None
    invalid: set[str] = field(default_factory=set)


def is_plausible(forecast: pd.DataFrame, history_units: pd.Series, factor: float = 10.0) -> bool:
    """A forecast is plausible if finite and below ``factor`` x the historical maximum."""
    cap = factor * max(float(history_units.max()), 1.0)
    values = forecast[["yhat", "lower", "upper"]].to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and forecast["yhat"].max() <= cap)


class ForecastAgent(BaseAgent):
    """Produces the 7-day demand forecast for one item."""

    name = "forecast_agent"
    role = "Forecast Agent - demand forecasting specialist"

    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        super().__init__(settings, llm)
        self._cache: dict[str, _ModelRun] = {}

    def _factories(self, holidays: pd.DataFrame, arima_order: tuple[int, int, int]) -> dict:
        cfg = self.settings.forecast
        return {
            NAIVE_NAME: lambda: SeasonalNaiveForecaster(cfg.seasonal_period, cfg.interval_width),
            "arima": lambda: ARIMAForecaster(
                order=arima_order,
                seasonal_order=cfg.arima_seasonal_order,
                seasonal_period=cfg.seasonal_period,
                fit_window=cfg.arima_fit_window,
                interval_width=cfg.interval_width,
            ),
            "prophet": lambda: ProphetForecaster(
                holidays=holidays,
                regressors=PROPHET_REGRESSORS,
                interval_width=cfg.interval_width,
                uncertainty_samples=cfg.prophet_uncertainty_samples,
            ),
        }

    def _run_models(
        self, item_id: str, features: pd.DataFrame, future: pd.DataFrame, holidays: pd.DataFrame
    ) -> _ModelRun:
        key = f"{item_id}:{features['date'].max()}:{len(features)}:{features['y'].sum():.1f}"
        if key in self._cache:
            return self._cache[key]
        cfg = self.settings.forecast
        selector = ARIMAForecaster(
            seasonal_order=cfg.arima_seasonal_order,
            seasonal_period=cfg.seasonal_period,
            order_grid=cfg.arima_order_grid,
            fit_window=cfg.arima_fit_window,
        )
        arima_order = selector.select_order(features)
        factories = self._factories(holidays, arima_order)
        run = _ModelRun(arima_order=arima_order)
        regressors = list(PROPHET_REGRESSORS)
        for name, factory in factories.items():
            run.backtests[name] = rolling_origin_backtest(
                factory, features, cfg.horizon, cfg.backtest_folds, regressors, model_name=name
            )
            run.forecasts[name] = factory().fit(features).predict(future[["date", *regressors]])
            if not is_plausible(run.forecasts[name], features["units"]):
                run.invalid.add(name)
                self.logger.warning("implausible_forecast_excluded", item_id=item_id, model=name)
        members = [m for m in PRIMARY_MODELS if m not in run.invalid] or [NAIVE_NAME]
        run.backtests[ENSEMBLE_NAME] = ensemble_backtest(
            [run.backtests[m] for m in members], name=ENSEMBLE_NAME
        )
        run.forecasts[ENSEMBLE_NAME] = average_forecasts([run.forecasts[m] for m in members])
        self._cache[key] = run
        return run

    def _metric(self, result: BacktestResult) -> float:
        value = getattr(result.metrics, self.settings.forecast.selection_metric)
        return float("inf") if value is None or not np.isfinite(value) else float(value)

    def _select(self, run: _ModelRun, revision: RevisionRequest | None) -> tuple[str, list[str]]:
        valid = [m for m in PRIMARY_MODELS if m not in run.invalid]
        ranked = sorted(valid, key=lambda m: self._metric(run.backtests[m])) or [NAIVE_NAME]
        metric = self.settings.forecast.selection_metric.upper()
        notes = [
            f"Backtest {metric}: "
            + ", ".join(f"{m}={self._metric(run.backtests[m]):.2f}"
                        for m in (NAIVE_NAME, *PRIMARY_MODELS, ENSEMBLE_NAME))
        ]
        if run.invalid:
            notes.append(f"Excluded implausible forecasts from: {sorted(run.invalid)}.")
        if revision is None:
            notes.append(f"Selected {ranked[0]} (lowest backtest {metric} among ARIMA/Prophet).")
            return ranked[0], notes
        if revision.hint == RevisionHint.USE_ENSEMBLE:
            notes.append(f"Critic requested ensemble: {revision.reason}")
            return ENSEMBLE_NAME, notes
        alternatives = [m for m in (*ranked, ENSEMBLE_NAME) if m != revision.previous_model]
        notes.append(f"Critic requested alternative model ({revision.reason}); "
                     f"switching to {alternatives[0]}.")
        return alternatives[0], notes

    @staticmethod
    def _points(frame: pd.DataFrame) -> list[ForecastPoint]:
        return [
            ForecastPoint(date=r.date.date(), yhat=round(r.yhat, 3), lower=round(r.lower, 3),
                          upper=round(r.upper, 3))
            for r in frame.itertuples()
        ]

    def run(
        self,
        item_id: str,
        features: pd.DataFrame,
        future: pd.DataFrame,
        holidays: pd.DataFrame,
        revision: RevisionRequest | None = None,
    ) -> ForecastOutput:
        """Forecast demand for ``item_id``.

        Args:
            item_id: Item identifier.
            features: Feature frame (history) from :func:`build_features`.
            future: Regressor frame for the horizon from :func:`build_future_frame`.
            holidays: Prophet holidays frame.
            revision: Optional Critic request to change the selected model.

        Returns:
            A validated :class:`ForecastOutput`.
        """
        run = self._run_models(item_id, features, future, holidays)
        model_used, reasoning = self._select(run, revision)
        reasoning.insert(0, f"ARIMA order selected by AIC: {run.arima_order} x "
                            f"{self.settings.forecast.arima_seasonal_order}"
                            f"[{self.settings.forecast.seasonal_period}].")
        selected = run.backtests[model_used]
        naive_mae = run.backtests[NAIVE_NAME].metrics.mae
        skill = None if not naive_mae else round(1 - selected.metrics.mae / naive_mae, 4)

        total_arima = float(run.forecasts["arima"]["yhat"].sum())
        total_prophet = float(run.forecasts["prophet"]["yhat"].sum())
        denom = max((total_arima + total_prophet) / 2, 1e-6)
        agreement = 1 - abs(total_arima - total_prophet) / denom
        agreement = 0.0 if run.invalid or not np.isfinite(agreement) else max(0.0, min(1.0,
                                                                                  agreement))
        reasoning.append(
            f"7-day totals: ARIMA={total_arima:.1f}, Prophet={total_prophet:.1f} "
            f"(agreement {agreement:.0%})."
        )
        if skill is not None:
            reasoning.append(f"Skill vs seasonal-naive (MAE): {skill:+.0%}.")

        backtest = [
            BacktestPoint(
                date=r.date.date(), fold=int(r.fold),
                actual=None if pd.isna(r.actual) else float(r.actual),
                yhat=round(r.yhat, 3), lower=round(r.lower, 3), upper=round(r.upper, 3),
            )
            for r in selected.predictions.itertuples()
        ]
        output = ForecastOutput(
            item_id=item_id,
            model_used=model_used,
            horizon=self.settings.forecast.horizon,
            forecast=self._points(run.forecasts[model_used]),
            model_forecasts={
                k: self._points(v) for k, v in run.forecasts.items() if k not in run.invalid
            },
            metrics_by_model={k: v.metrics for k, v in run.backtests.items()},
            selected_metrics=selected.metrics,
            skill_vs_naive=skill,
            model_agreement_pct=round(agreement, 4),
            arima_order=run.arima_order,
            backtest=backtest,
            revision_applied=revision.hint if revision else None,
            reasoning=reasoning,
        )
        output.narrative = self.narrate(
            "Summarise the 7-day demand forecast and how reliable the chosen model was.",
            {
                "item_id": item_id,
                "model_used": model_used,
                "forecast_total_units_7d": round(output.total_forecast, 1),
                "mean_daily_demand": round(output.mean_daily_demand, 1),
                "backtest_mae": selected.metrics.mae,
                "backtest_mape": selected.metrics.mape,
                "backtest_wape": selected.metrics.wape,
                "skill_vs_naive": skill,
                "arima_prophet_agreement": round(agreement, 2),
                "horizon_days": self.settings.forecast.horizon,
            },
        )
        self.logger.info("forecast_completed", item_id=item_id, model=model_used,
                         mae=selected.metrics.mae, agreement=round(agreement, 3))
        return output
