"""Plotly figure builders (pure functions - easy to unit test)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.components.formatting import PLOT_COLORS
from src.agents.schemas import AgentTraceEntry, PipelineResult

_LAYOUT = {"margin": {"l": 10, "r": 10, "t": 40, "b": 10}, "hovermode": "x unified",
           "legend": {"orientation": "h", "y": -0.15}}


def _stockout_spans(history: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if "is_suspected_stockout" not in history:
        return []
    spans, start = [], None
    for d, flag in zip(history["date"], history["is_suspected_stockout"], strict=True):
        if flag and start is None:
            start = d
        elif not flag and start is not None:
            spans.append((start, d))
            start = None
    if start is not None:
        spans.append((start, history["date"].iloc[-1]))
    return spans


def forecast_figure(
    history: pd.DataFrame,
    result: PipelineResult,
    history_days: int = 120,
    show_models: tuple[str, ...] = ("arima", "prophet"),
    actuals: pd.DataFrame | None = None,
) -> go.Figure:
    """History, selected forecast with interval band and other model forecasts."""
    hist = history.tail(history_days)
    fig = go.Figure()
    for start, end in _stockout_spans(hist):
        fig.add_vrect(x0=start, x1=end, fillcolor=PLOT_COLORS["stockout"], line_width=0,
                      annotation_text="suspected stockout", annotation_position="top left",
                      annotation_font_size=9)
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["units"], name="Actual sales",
                             line={"color": PLOT_COLORS["actual"], "width": 1.5}))
    events = hist[hist["event_name"] != ""]
    if not events.empty:
        fig.add_trace(go.Scatter(x=events["date"], y=events["units"], mode="markers",
                                 name="Event day", text=events["event_name"],
                                 marker={"symbol": "star", "size": 9, "color": "#2CA02C"}))
    fc = pd.DataFrame([p.model_dump() for p in result.forecast.forecast])
    fig.add_trace(go.Scatter(
        x=list(fc["date"]) + list(fc["date"])[::-1],
        y=list(fc["upper"]) + list(fc["lower"])[::-1],
        fill="toself", fillcolor=PLOT_COLORS["band"], line={"width": 0},
        name="80% interval", hoverinfo="skip"))
    selected = result.forecast.model_used
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], name=f"Forecast ({selected})",
                             mode="lines+markers",
                             line={"color": PLOT_COLORS.get(selected, "#0071CE"), "width": 3}))
    for model in show_models:
        if model == selected or model not in result.forecast.model_forecasts:
            continue
        mf = pd.DataFrame([p.model_dump() for p in result.forecast.model_forecasts[model]])
        fig.add_trace(go.Scatter(x=mf["date"], y=mf["yhat"], name=model,
                                 line={"color": PLOT_COLORS.get(model), "dash": "dot"}))
    if actuals is not None and not actuals.empty:
        window = actuals[actuals["date"] <= pd.Timestamp(fc["date"].max())]
        fig.add_trace(go.Scatter(x=window["date"], y=window["units"], mode="markers",
                                 name="Actual (hold-out)",
                                 marker={"color": "black", "symbol": "x", "size": 8}))
    fig.update_layout(title="Daily demand: history and 7-day forecast", yaxis_title="Units",
                      **_LAYOUT)
    return fig


def backtest_figure(result: PipelineResult) -> go.Figure:
    """Backtest actual vs predicted for the selected model across folds."""
    bt = pd.DataFrame([p.model_dump() for p in result.forecast.backtest])
    fig = go.Figure()
    if bt.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=list(bt["date"]) + list(bt["date"])[::-1],
        y=list(bt["upper"]) + list(bt["lower"])[::-1], fill="toself",
        fillcolor=PLOT_COLORS["band"], line={"width": 0}, name="80% interval",
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["actual"], name="Actual", mode="lines+markers",
                             line={"color": PLOT_COLORS["actual"]}))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["yhat"], name="Predicted",
                             line={"color": PLOT_COLORS.get(result.forecast.model_used,
                                                            "#0071CE")}))
    fig.update_layout(title=f"Rolling-origin backtest ({result.forecast.model_used})",
                      **_LAYOUT)
    return fig


def anomaly_figure(history: pd.DataFrame, result: PipelineResult, days: int = 120) -> go.Figure:
    """Recent history with detected anomalies highlighted."""
    hist = history.tail(days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["units"], name="Actual sales",
                             line={"color": PLOT_COLORS["actual"]}))
    rolling = hist["units"].rolling(28, min_periods=7).mean()
    fig.add_trace(go.Scatter(x=hist["date"], y=rolling, name="28-day mean",
                             line={"color": "#0071CE", "dash": "dash"}))
    if result.risk.anomalies:
        an = pd.DataFrame([a.model_dump() for a in result.risk.anomalies])
        fig.add_trace(go.Scatter(
            x=an["date"], y=an["units"], mode="markers", name="Anomaly",
            marker={"color": PLOT_COLORS["anomaly"], "size": 11, "symbol": "circle-open",
                    "line": {"width": 2}},
            text=[f"{k} ({m}, z={s})" for k, m, s in zip(an["kind"], an["method"], an["score"],
                                                          strict=True)]))
    fig.update_layout(title="Demand anomalies (last 120 days)", **_LAYOUT)
    return fig


def confidence_gauge(score: float) -> go.Figure:
    """Gauge of the 0-100 confidence score with risk bands."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, 45], "color": "#F8D0CC"},
                         {"range": [45, 70], "color": "#FCE6C3"},
                         {"range": [70, 100], "color": "#CDEBD3"}]}))
    fig.update_layout(height=230, margin={"l": 20, "r": 20, "t": 30, "b": 10},
                      title="Forecast confidence")
    return fig


def components_figure(components: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of confidence components."""
    fig = go.Figure(go.Bar(x=list(components.values()), y=list(components.keys()),
                           orientation="h", marker_color="#0071CE",
                           text=[f"{v:.2f}" for v in components.values()],
                           textposition="auto"))
    fig.update_layout(title="Confidence components (0-1)", xaxis={"range": [0, 1]}, height=230,
                      margin={"l": 10, "r": 10, "t": 40, "b": 10})
    return fig


def inventory_projection_figure(result: PipelineResult) -> go.Figure:
    """Projected stock vs reorder point and safety stock."""
    inv = result.inventory
    proj = pd.DataFrame([p.model_dump() for p in inv.projected_stock])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=proj["date"], y=proj["demand"], name="Forecast demand",
                         marker_color="rgba(0,113,206,0.35)"))
    fig.add_trace(go.Scatter(x=proj["date"], y=proj["stock"], name="Projected on-hand",
                             mode="lines+markers", line={"color": "#333", "width": 3}))
    if proj["arrivals"].sum() > 0:
        arr = proj[proj["arrivals"] > 0]
        fig.add_trace(go.Scatter(x=arr["date"], y=arr["stock"], mode="markers",
                                 name="Open order arrives",
                                 marker={"symbol": "triangle-up", "size": 14, "color": "green"}))
    fig.add_hline(y=inv.reorder_point, line_dash="dash", line_color="#E07B00",
                  annotation_text=f"Reorder point {inv.reorder_point:.0f}")
    fig.add_hline(y=inv.safety_stock, line_dash="dot", line_color="#7B3FBF",
                  annotation_text=f"Safety stock {inv.safety_stock:.0f}")
    fig.add_hline(y=0, line_color="#D62728", line_width=1)
    fig.update_layout(title="Projected stock without a new order", yaxis_title="Units",
                      **_LAYOUT)
    return fig


def trace_timeline_figure(trace: list[AgentTraceEntry]) -> go.Figure:
    """Gantt-style timeline of graph nodes."""
    if not trace:
        return go.Figure()
    t0 = trace[0].started_at
    colors = {"data_pipeline": "#9E9E9E", "forecast_agent": "#0071CE", "risk_agent": "#E07B00",
              "inventory_agent": "#2CA02C", "critic_agent": "#7B3FBF"}
    fig = go.Figure()
    for entry in trace:
        start = (entry.started_at - t0).total_seconds() * 1000
        fig.add_trace(go.Bar(
            y=[f"{entry.step}. {entry.node}"], x=[max(entry.duration_ms, 1)], base=[start],
            orientation="h", marker_color=colors.get(entry.agent, "#555"), showlegend=False,
            hovertemplate=f"{entry.agent}<br>{entry.duration_ms:.0f} ms<extra></extra>"))
    fig.update_layout(title="Execution timeline (ms)", yaxis={"autorange": "reversed"},
                      height=80 + 32 * len(trace), margin={"l": 10, "r": 10, "t": 40, "b": 10})
    return fig


def metrics_comparison_figure(frame: pd.DataFrame, metric: str) -> go.Figure:
    """Grouped bars of a metric per item and model."""
    fig = go.Figure()
    for model in frame["model"].unique():
        sub = frame[frame["model"] == model]
        fig.add_trace(go.Bar(x=sub["item_id"], y=sub[metric], name=model,
                             marker_color=PLOT_COLORS.get(model)))
    fig.update_layout(barmode="group", title=f"Backtest {metric.upper()} by item and model",
                      **_LAYOUT)
    return fig
