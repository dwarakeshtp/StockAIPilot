"""Risk & Uncertainty Agent page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import anomaly_figure, components_figure, confidence_gauge
from dashboard.components.formatting import risk_badge
from dashboard.components.state import get_history, require_result

st.title("2. Risk & Uncertainty Agent")
st.caption("Scores forecast confidence from backtest accuracy, interval width, model agreement "
           "and data quality; detects anomalies and demand spikes; assigns a risk level that "
           "scales safety stock.")

result = require_result()
if result is None:
    st.stop()
risk = result.risk

top = st.columns([2, 2, 3])
with top[0]:
    st.plotly_chart(confidence_gauge(risk.confidence_score))
with top[1]:
    st.plotly_chart(components_figure(risk.confidence_components))
with top[2]:
    st.markdown("**Assessment**")
    risk_badge(risk.risk_level)
    st.metric("Safety-stock multiplier", f"x{risk.uncertainty_multiplier:.1f}")
    flags = {
        "Recent demand spike": risk.spike_detected,
        "Recent demand drop": risk.drop_detected,
        "Forecast spike vs recent demand": risk.forecast_spike,
    }
    for label, on in flags.items():
        st.markdown(f"{':red[:material/check_circle:]' if on else ':gray[:material/remove:]'} "
                    f"{label}")
    st.caption(f"Recent 7 days = {risk.recent_to_baseline_ratio:.2f}x prior 28-day baseline; "
               f"relative interval width {risk.relative_interval_width:.2f}.")

st.plotly_chart(anomaly_figure(get_history(result.item_id), result))

left, right = st.columns(2)
with left:
    st.markdown(f"**Detected anomalies** ({len(risk.anomalies)} in last 90 days, "
                f"{risk.recent_anomaly_count} in last 14)")
    if risk.anomalies:
        st.dataframe(pd.DataFrame([a.model_dump() for a in risk.anomalies]), hide_index=True)
    else:
        st.caption("None detected.")
with right:
    dq = result.data_quality
    st.markdown(f"**Data quality** - score {dq.score:.2f}")
    if dq.issues:
        for issue in dq.issues:
            st.markdown(f"- `{issue.severity}` {issue.message}")
    else:
        st.caption("No data-quality issues.")

with st.expander("Agent reasoning", expanded=True):
    for step in risk.reasoning:
        st.markdown(f"- {step}")
    if risk.narrative:
        st.info(risk.narrative, icon=":material/smart_toy:")
