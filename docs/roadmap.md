# Implementation Roadmap

Every phase ended with a re-evaluation on the real M5 subset before moving on.

| Phase | Scope | Key modules | Exit check (re-evaluation) | Status |
|---|---|---|---|---|
| 1. Setup and ingestion | Repo layout, settings (YAML + `.env`), structlog, M5 download and subset, simulated inventory | `src/config`, `src/logging`, `src/data/ingestion.py`, `simulate_inventory.py`, `scripts/build_subset.py` | 8,760 rows, 12 items, 730 days; future calendar and actuals present | Done |
| 2. Preprocessing and features | Quality report, censored-demand handling, features, future frame, holidays | `quality.py`, `cleaning.py`, `features.py`, `repository.py` | Quality scores 0.76-1.0; suspected stockout streaks flagged on 7 items | Done |
| 3. Forecast Agent | ARIMA, Prophet, seasonal-naive, ensemble, rolling backtest, metrics, selection | `src/models`, `src/evaluation/metrics.py`, `backtest.py`, `forecast_agent.py` | WAPE 0.16-0.30 for most items; an exploding ARIMA (1e38) was caught and fixed with sanity checks | Done |
| 4. Risk Agent | Confidence score, anomalies, level shifts, risk level | `confidence.py`, `anomaly.py`, `risk_agent.py` | 4 LOW, 5 MEDIUM, 3 HIGH; scores track WAPE | Done |
| 5. Inventory Agent | SS / ROP / ROQ, stockout probability, projection, overrides | `src/inventory`, `inventory_agent.py` | EXPEDITE, REORDER and HOLD all occur; quantities respect pack and MOQ | Done |
| 6. Critic Agent | 13 rules, verdicts, revision request, grounded LLM explanation | `validation_rules.py`, `critic_agent.py`, `llm_client.py`, `grounding.py` | Catches the stockout before an open order on FOODS_3_090 and requests an ensemble for FOODS_3_519 | Done |
| 7. LangGraph | State graph, Critic revision loop, what-if graph, traces, runner, decisions | `src/orchestration` | 12 items end to end in about 25 s; traces saved | Done |
| 8. Streamlit | 7 pages, what-if, Approve / Modify / Reject, PO download | `dashboard/` | Headless AppTest renders every page; what-if and decisions verified | Done |
| 9. Evaluation and docs | Hold-out, inventory simulation, Critic scenarios, report, docs, notebooks | `src/evaluation/*`, `scripts/run_evaluation.py`, `docs/` | See [evaluation_report.md](evaluation_report.md) | Done |

## Possible next steps

1. **Global ML model.** LightGBM with quantile objectives across SKUs, compared against
   ARIMA and Prophet.
2. **Multi-store.** Extend the subset to several stores; reuse the agents per store-item.
3. **Weather and demand shocks.** External regressors (NOAA) and promo calendars.
4. **Real inventory feed.** Replace the simulated parameters with an ERP or WMS export (same
   schema).
5. **Learning from decisions.** Use `outputs/decisions.jsonl` (Modify and Reject reasons)
   to tune service levels and Critic thresholds.
6. **Deployment.** Streamlit Community Cloud with `st.secrets` for the LLM keys, and
   scheduled nightly runs.
