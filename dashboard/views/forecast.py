"""Forecast Agent page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import backtest_figure, forecast_figure
from dashboard.components.formatting import pct
from dashboard.components.state import get_future_actuals, get_history, require_result

st.title("1. Forecast Agent")
st.caption("Backtests Seasonal-Naive (reference), ARIMA and Prophet on four rolling 7-day "
           "folds, selects the best model and produces a 7-day forecast with an 80% interval.")

result = require_result()
if result is None:
    st.stop()
fc = result.forecast
history = get_history(result.item_id)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Selected model", fc.model_used,
          delta="revised by Critic" if fc.revision_applied else None, delta_color="off")
c2.metric("7-day forecast", f"{fc.total_forecast:,.0f} units")
c3.metric("Backtest MAE", f"{fc.selected_metrics.mae:.2f}")
c4.metric("Backtest MAPE / WAPE",
          f"{pct(fc.selected_metrics.mape)} / {pct(fc.selected_metrics.wape)}")
c5.metric("Skill vs naive", pct(fc.skill_vs_naive), help="1 - MAE(model) / MAE(seasonal naive)")

with st.container(border=True):
    show = st.segmented_control("Overlay models", ["arima", "prophet", "ensemble",
                                                   "seasonal_naive"],
                                selection_mode="multi", default=["arima", "prophet"])
    show_actuals = st.toggle("Show actual hold-out sales (evaluation only)", value=False,
                             help="M5 provides the 28 days after the forecast origin.")
    actuals = get_future_actuals(result.item_id) if show_actuals else None
    st.plotly_chart(forecast_figure(history, result, show_models=tuple(show or []),
                                    actuals=actuals))

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(backtest_figure(result))
with right:
    st.markdown("**Backtest metrics by model**")
    metrics = pd.DataFrame(
        [{"model": m, **v.model_dump()} for m, v in fc.metrics_by_model.items()]
    ).set_index("model")
    st.dataframe(metrics.style.format({"mae": "{:.2f}", "rmse": "{:.2f}", "mape": "{:.1%}",
                                       "wape": "{:.1%}"}, na_rep="-")
                 .highlight_min(subset=["mae", "rmse"], color="#CDEBD3"))
    st.caption(f"ARIMA order {fc.arima_order}; ARIMA vs Prophet agreement "
               f"{pct(fc.model_agreement_pct)}.")
    st.markdown("**7-day forecast**")
    st.dataframe(pd.DataFrame([p.model_dump() for p in fc.forecast]).round(1), hide_index=True)

with st.expander("Agent reasoning", expanded=True):
    for step in fc.reasoning:
        st.markdown(f"- {step}")
    if fc.narrative:
        st.info(fc.narrative, icon=":material/smart_toy:")
