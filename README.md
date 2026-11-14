# StockPilot AI - Agentic Demand Forecasting & Inventory Reorder Recommendations

A capstone proof of concept in which four cooperating AI agents, orchestrated with
**LangGraph**, turn Walmart (M5) daily sales history into **7-day demand forecasts** and
**explainable reorder recommendations**. A planner reviews each recommendation (Approve /
Modify / Reject) in a **Streamlit** dashboard.

It is built like a small production system with three tiers:

* **Data layer:** the M5 CSVs are loaded into **SQLite** at initialisation, covering the
  whole CA_1 FOODS category (1,437 items, 2.79M daily rows). Agents read through a
  repository, never from files.
* **API layer:** a **FastAPI** backend exposes data, forecasting, inventory,
  agent-workflow, background-job, decision and evaluation endpoints.
* **Presentation:** a **Streamlit** dashboard that talks to the API over HTTP.

```
Ingestion -> Cleaning -> Feature Engineering -> Forecast Agent -> Risk & Uncertainty Agent
          -> Inventory Planning Agent -> Critic Agent --(revise once)--> Forecast Agent
                                                    \-> API -> Dashboard + human decision
```

| Agent | Responsibility | Core technique |
|---|---|---|
| Forecast Agent | Train ARIMA, Prophet and a seasonal-naive baseline; backtest; choose a model; 7-day forecast with 80% interval | statsmodels SARIMAX, Prophet, rolling-origin backtest (MAE / RMSE / MAPE / WAPE) |
| Risk & Uncertainty Agent | Confidence score 0-100, anomaly / spike / drop detection, risk level | Weighted accuracy + interval precision + model agreement + data quality |
| Inventory Planning Agent | Safety stock, reorder point, order quantity, stockout probability, stock projection | Normal-approximation SS with lead-time variance, order-up-to policy, pack / MOQ rounding |
| Critic Agent | 13 validation rules, verdict, revision request, grounded business explanation | Rule engine + LLM explanation with numeric grounding check and template fallback |

Every agent returns a **pydantic** model. Every step is written to a **JSON trace**, and the
LLM is optional: with no API key the system runs fully deterministic and uses template
explanations.

## Quick start

Requirements: Python 3.11+ (tested on 3.11 and 3.14).

```bash
git clone <this-repo> && cd StockAIPilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/init_db.py --download --source raw   # optional: whole category from raw M5 (~450 MB download)
scripts/dev.sh                                      # API on :8000 and dashboard on :8501
```

Or start the two tiers separately:

```bash
uvicorn src.api.main:app --port 8000     # API docs: http://localhost:8000/docs
streamlit run dashboard/app.py           # talks to STOCKPILOT_API_URL (default http://localhost:8000)
```

On its first start the API creates `data/db/stockpilot.db`. If the raw M5 CSVs are in
`data/raw/`, it loads the whole CA_1 FOODS category (about 10 s). Otherwise it loads the
committed 12-item seed (`data/processed/`), so a fresh clone works without downloads. The
simulated inventory parameters (`data/simulated/`) and the product catalog
(`data/reference/`, illustrative food names for the anonymised M5 item codes) are committed
too.

In the dashboard, click **Run all** to run the agents for the 12 demo items as a background
job (about 30 s), or use **Search all items** to analyse any other SKU. The API reference
with curl examples is in [docs/api.md](docs/api.md).

<details>
<summary>pip blocked on your network? Use conda-forge instead</summary>

```bash
# micromamba / mamba / conda all work
micromamba create -p ./.venv -c conda-forge python=3.11 pandas numpy pyarrow scipy statsmodels \
  prophet pydantic pydantic-settings pyyaml python-dotenv structlog streamlit plotly pytest \
  pytest-cov ruff langgraph langchain-core langchain-openai sqlalchemy fastapi uvicorn httpx
```
</details>

## Enable the LLM (optional)

```bash
cp .env.example .env
# then edit .env:
LLM_PROVIDER=openrouter            # or cohere
OPENROUTER_API_KEY=sk-or-...       # free models end with ":free"
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
COHERE_API_KEY=...                 # optional second provider (fallback)
```

Both providers are called through OpenAI-compatible endpoints with `langchain-openai`.
Fallbacks run in order: primary provider, then the secondary provider (if
`LLM_ENABLE_FALLBACK=true`), then deterministic templates. Any LLM sentence that contains a
number not present in the agent facts is rejected (grounding check) and replaced by the
template.

## Command line

```bash
python scripts/init_db.py                          # create / load the database (idempotent)
python scripts/init_db.py --source raw --rebuild   # reload from the raw CSVs (runs and decisions kept)
python scripts/run_pipeline.py                     # demo items -> pipeline_runs table
python scripts/run_pipeline.py --items FOODS_3_090 FOODS_3_202
python scripts/run_evaluation.py                   # hold-out, inventory simulation, scenarios
pytest                                             # unit, integration, API and dashboard tests
ruff check src tests scripts dashboard
```

The same workflows over HTTP:

```bash
curl -s -X POST localhost:8000/jobs -H 'content-type: application/json' -d '{"all_demo": true}'
curl -s localhost:8000/recommendations
curl -s -X POST localhost:8000/forecasts/FOODS_3_090
```

Rebuild the 12-item parquet seed from raw M5 (optional):

```bash
python scripts/build_subset.py --download          # Kaggle CLI or public GitHub mirror
```

## Results (12 SKUs, CA_1 / FOODS)

Full numbers: [docs/evaluation_report.md](docs/evaluation_report.md), regenerated by
`scripts/run_evaluation.py`.

* **Hold-out accuracy (real next-7-day sales):** the agent-selected model has WAPE 35.8% vs
  42.5% for seasonal-naive, and MAE is 16% lower.
* **Inventory simulation (56 days, actual demand):** 8 stockout days vs 51 for a naive
  fixed-reorder rule. Fill rate is 99.2% vs 90.5%, with 12% less average inventory.
* **Critic scenarios (11 injected cases):** 100% detection recall, 100% verdict accuracy and
  a 100% grounding pass rate. The Risk Agent's signals alone would surface only 18% of the
  injected issues.

## Project layout

```
config/settings.yaml        all tunable parameters (horizon, service level, critic thresholds, database, api)
data/                       raw M5 (gitignored) / db (SQLite, gitignored) / processed seed / simulated / reference
src/
  config/                   pydantic settings + .env overrides
  db/                       SQLite engine, schema, loader (init_db), SQLiteRepository, run/decision/job stores
  api/                      FastAPI app, schemas, routers (system, items, runs, jobs, decisions, evaluation)
  services/                 pipeline, job, decision and evaluation services used by the API
  data/                     ingestion (M5 / parquet / synthetic), quality, cleaning, features, Repository protocol
  models/                   BaseForecaster, ARIMA, Prophet, seasonal-naive, ensemble
  evaluation/               metrics, backtest, hold-out, inventory simulation, scenario eval, report
  inventory/                safety stock / ROP / ROQ / stockout formulas
  agents/                   schemas, LLM client, grounding, prompts, 4 agents, critic rules
  orchestration/            LangGraph state, graph, tracing, runner, human decisions
  logging/                  structlog JSON logging
dashboard/                  Streamlit app (7 pages) + components (api_client, state, charts, formatting)
scripts/                    init_db / dev.sh / download / build subset / run pipeline / run evaluation
tests/                      pytest suite (synthetic data, no network)
docs/                       architecture, agent design, dataset, roadmap, evaluation, demo script
notebooks/                  EDA and forecasting experiments
outputs/                    traces, logs, evaluation (generated)
```

## Documentation

* [Architecture, sequence and ER diagrams](docs/architecture.md)
* [HTTP API reference](docs/api.md)
* [Agent design](docs/agent_design.md)
* [Dataset](docs/dataset.md)
* [Implementation roadmap](docs/roadmap.md)
* [Evaluation report](docs/evaluation_report.md)
* [Final report](docs/final_report.md)
* [Demo script](docs/demo_script.md) and [presentation outline](docs/presentation_outline.md)
* [Requirement checklist](docs/requirement_checklist.md)

## Limitations

* M5 has no inventory data. On-hand stock, open orders, lead times, MOQs and costs are
  **simulated** (seeded) and labelled as such in the UI and in Critic issues.
* M5 records sales, not demand. Long zero streaks on otherwise active items are treated as
  suspected stockouts and excluded from training.
* One store and one category (all 1,437 FOODS SKUs are in the database; 12 named demo
  SKUs) and a 7-day horizon, as scoped for a POC.
* No authentication on the API, and agent runs are serialised (one at a time). Postgres,
  Docker and scheduled runs are natural next steps; only the SQLAlchemy URL would change for
  Postgres.
