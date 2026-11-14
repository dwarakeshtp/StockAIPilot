# Agentic Demand Forecasting & Reorder Recommendation System

Status of each requirement, with where it is implemented or evidenced.

## Project Scope

- [x] POC only
- [x] Single product category - FOODS ([dataset.md](dataset.md))
- [x] Single store or limited stores - CA_1
- [x] Historical sales forecasting - `src/agents/forecast_agent.py`
- [x] Inventory recommendation generation - `src/agents/inventory_agent.py`
- [x] Dashboard visualization - `dashboard/app.py`

---

## Data Requirements

- [x] Walmart Sales Dataset - M5, `scripts/build_subset.py`
- [x] Clean sales history - `src/data/cleaning.py` (censored-demand handling)
- [x] Train/Test split - rolling-origin backtest plus real hold-out (`src/evaluation/backtest.py`, `holdout.py`)
- [x] Data quality validation - `src/data/quality.py`, Critic `data_quality` rule

---

## Agent Requirements

### Forecast Agent

- [x] Load demand data - via `DataRepository` and graph nodes ingest, clean, features
- [x] Generate demand forecast - ARIMA / Prophet / ensemble, 7-day horizon with 80% interval
- [x] Output forecast values - `ForecastOutput`

### Risk Agent

- [x] Calculate forecast confidence - `src/agents/confidence.py`
- [x] Detect anomalies - `src/agents/anomaly.py` (z-score, IQR, level shift)
- [x] Assign risk score - confidence 0-100 and LOW / MEDIUM / HIGH

### Inventory Planner Agent

- [x] Calculate safety stock - `src/inventory/policy.py`
- [x] Recommend reorder quantity - order-up-to policy with pack / MOQ rounding
- [x] Estimate stockout risk - `src/inventory/stockout.py`

### Critic Agent (Capstone Excellence)

- [x] Validate recommendations - 13 rules, `src/agents/validation_rules.py`
- [x] Generate reasoning summary - `CriticOutput.summary` and grounded explanation
- [x] Explain agent decisions - LLM or template explanation, plus per-agent narratives

---

## Machine Learning Requirements

- [x] Baseline forecast model - seasonal-naive
- [x] Forecast evaluation - rolling-origin backtest and hold-out
- [x] MAE calculation - `src/evaluation/metrics.py`
- [x] RMSE calculation
- [x] MAPE calculation (plus WAPE)

---

## Agentic AI Requirements

- [x] Minimum 3 agents - 4 agents
- [x] Agent communication workflow - LangGraph with the Critic revision loop (`src/orchestration/graph.py`)
- [x] Agent reasoning logs - `outputs/traces/*.json`, `outputs/logs/app.jsonl`
- [x] Agent output persistence - `outputs/runs/latest.json`, `outputs/decisions.jsonl`

---

## Dashboard Requirements

- [x] Forecast visualization - Forecast page
- [x] Inventory recommendation view - Overview and Inventory pages (plus what-if)
- [x] Agent trace view - Agent Trace page
- [x] Performance metrics page - Performance Metrics page

---

## Evaluation Requirements

### Quantitative

- [x] Forecast accuracy - [evaluation_report.md](evaluation_report.md) section 1
- [x] Inventory KPI improvement - section 2 (fill rate, inventory, lost sales)
- [x] Stockout reduction simulation - section 2 (8 vs 51 stockout days)

### Qualitative

- [x] Explainability assessment - [final_report.md](final_report.md) section 4.1
- [x] Agent collaboration review - final report section 4.2 and Critic scenario evaluation
- [ ] User feedback/demo review - template in final report section 4.3 (fill in after the demo)

---

## Capstone Deliverables

- [x] Architecture diagram - [architecture.md](architecture.md)
- [x] Dataset documentation - [dataset.md](dataset.md)
- [x] Agent workflow diagram - architecture.md sections 2-4
- [x] Working prototype - `streamlit run dashboard/app.py`
- [x] Demo dashboard - [demo_script.md](demo_script.md)
- [x] Final report - [final_report.md](final_report.md)
- [ ] Presentation deck - outline in [presentation_outline.md](presentation_outline.md); slides to be built

---

## Stretch Goals

- [ ] Weather data integration
- [x] Demand shock detection - spike / drop level-shift detection in the Risk Agent
- [ ] Multi-store forecasting
- [x] LLM-powered inventory explanations - OpenRouter / Cohere with grounding
