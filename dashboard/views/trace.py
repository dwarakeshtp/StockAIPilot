"""Agent trace page: step-by-step view of the LangGraph run."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from dashboard.components.charts import trace_timeline_figure
from dashboard.components.state import require_result, what_if_results

st.title("Agent Trace")
st.caption("Every LangGraph node records its inputs, outputs, reasoning, duration and the LLM "
           "provider used. Traces are stored as JSON in `outputs/traces/`.")

result = require_result()
if result is None:
    st.stop()

options = {"Pipeline run": result}
if result.item_id in what_if_results():
    options["Latest what-if run"] = what_if_results()[result.item_id]
label = st.radio("Run", list(options), horizontal=True) if len(options) > 1 else "Pipeline run"
run = options[label]

c = st.columns(4)
c[0].metric("Run id", run.run_id.split("-", 1)[1][:15])
c[1].metric("Steps", len(run.trace))
c[2].metric("Total time", f"{sum(t.duration_ms for t in run.trace) / 1000:.1f}s")
c[3].metric("Forecast revisions", run.revision_count)

st.plotly_chart(trace_timeline_figure(run.trace))

for entry in run.trace:
    title = (f"Step {entry.step} - {entry.node} ({entry.agent}) - {entry.duration_ms:.0f} ms"
             + (f" - LLM: {entry.llm_provider}" if entry.llm_provider else ""))
    with st.expander(title, expanded=entry.node == "critic"):
        cols = st.columns(2)
        cols[0].markdown("**Input summary**")
        cols[0].json(entry.input_summary, expanded=True)
        cols[1].markdown("**Output summary**")
        cols[1].json(entry.output_summary, expanded=True)
        if entry.reasoning:
            st.markdown("**Reasoning**")
            for step in entry.reasoning:
                st.markdown(f"- {step}")

if run.trace_path and Path(run.trace_path).exists():
    st.download_button("Download trace JSON", Path(run.trace_path).read_text(encoding="utf-8"),
                       file_name=Path(run.trace_path).name, mime="application/json")
else:
    st.download_button("Download trace JSON",
                       json.dumps([t.model_dump(mode="json") for t in run.trace], indent=2),
                       file_name=f"{run.run_id}.json", mime="application/json")
