# StockPilot AI - HTTP API

The FastAPI backend (`src/api/`) exposes the SQLite data layer, the four agents and the
human-approval workflow. The Streamlit dashboard is a client of this API and never reads
the database itself.

```bash
python scripts/init_db.py              # optional - the API also initialises an empty DB on start
uvicorn src.api.main:app --port 8000   # interactive docs: http://localhost:8000/docs
```

* Base URL: `http://localhost:8000`. The dashboard reads `STOCKPILOT_API_URL`, falling back to
  `api.base_url` in `config/settings.yaml`.
* All bodies are JSON. Dates are ISO `YYYY-MM-DD` strings and missing values are `null`.
* Agent payloads reuse the pydantic models in `src/agents/schemas.py`. `PipelineResult`
  carries `forecast`, `risk`, `inventory`, `critic` and `trace`.
* Agent runs are serialised: one pipeline run at a time. Batches go through background
  jobs; poll the job until it finishes.

## Errors

| Status | When | Body |
|---|---|---|
| 404 | Unknown item, run or job | `{"detail": "Unknown item: FOODS_9_999"}` |
| 422 | Invalid what-if override, invalid inventory field, malformed request | `{"detail": "Unsupported overrides: ['foo']. Supported: [...]"}` |
| 400 | Decision breaks an approval rule | `{"detail": "Please give a reason (at least 3 characters) when the decision is MODIFIED."}` |
| 503 | Database empty or backend still starting | `{"detail": "Database ... is empty. Run python scripts/init_db.py."}` |

## System

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness, plus `database: ready / empty` and the loader source |
| GET | `/system/info` | Data source, store/category, row counts, history window, forecast origin, horizon, LLM chain |
| GET | `/system/graph` | Mermaid source of the compiled LangGraph pipeline |

```bash
curl -s localhost:8000/health
# {"status":"ok","database":"ready","source":"m5_raw"}
curl -s localhost:8000/system/info
# {"data_source":"sqlite:m5_raw","items":1437,"demo_items":["FOODS_3_064",...],
#  "history_start":"2014-05-24","history_end":"2016-05-22","horizon_days":7,
#  "counts":{"daily_sales":2789217,"prices":5570,...},"llm":"openrouter -> template"}
```

## Data

| Method | Path | Description |
|---|---|---|
| GET | `/items?search=&department=&demo_only=&limit=50&offset=0` | Page through the item master; demo items first. `search` matches the item code or product name |
| GET | `/items/{item_id}` | Item metadata, inventory position, sales stats and latest run id |
| GET | `/items/{item_id}/sales` | Raw daily rows the pipeline consumes (units, price, events, SNAP); last 730 days |
| GET | `/items/{item_id}/history` | Cleaned history with `is_imputed` / `is_suspected_stockout` flags and the training target `y` |
| GET | `/items/{item_id}/actuals` | Hold-out sales after the forecast origin (evaluation only) |
| GET | `/calendar/future` | Calendar of the next 28 days (events, SNAP) |

```bash
curl -s "localhost:8000/items?search=milk"
curl -s "localhost:8000/items?search=FOODS_3_0&limit=5&offset=5"
curl -s localhost:8000/items/FOODS_3_090/history | jq '.rows[-1]'
```

## Inventory

| Method | Path | Description |
|---|---|---|
| GET | `/items/{item_id}/inventory` | On hand, on order, lead time, MOQ, pack size, unit cost, `is_simulated`, `updated_at` |
| PUT | `/items/{item_id}/inventory` | Partial update, like a stock feed or a manual count. Sets `is_simulated=false` |

```bash
curl -s -X PUT localhost:8000/items/FOODS_3_090/inventory \
  -H 'content-type: application/json' -d '{"on_hand": 120, "on_order": 0}'
```

Editable fields: `on_hand`, `on_order`, `on_order_eta_days`, `lead_time_days`,
`lead_time_std`, `moq`, `pack_size`, `unit_cost`, `holding_cost_rate`. The next pipeline run
uses the new position.

## Forecasting

| Method | Path | Description |
|---|---|---|
| POST | `/forecasts/{item_id}` | Data pipeline plus the Forecast and Risk agents only (no planning, nothing stored) |

```bash
curl -s -X POST localhost:8000/forecasts/FOODS_3_090 | jq '{model: .forecast.model_used,
  total: .forecast.forecast | map(.yhat) | add, risk: .risk.risk_level}'
```

## Agent workflows

| Method | Path | Description |
|---|---|---|
| POST | `/runs` | Run all four agents for one item synchronously. Body: `{"item_id": "...", "overrides": {}}`. Returns `PipelineResult` |
| GET | `/runs?item_id=&kind=&limit=50` | Run log (summary columns), newest first. `kind` is `pipeline` or `whatif` |
| GET | `/runs/latest?item_ids=..&kind=pipeline` | Latest run per item (full `PipelineResult`s) |
| GET | `/runs/{run_id}` | One stored run |
| GET | `/runs/{run_id}/trace` | Step-by-step agent trace (`AgentTraceEntry` list) |
| POST | `/runs/{run_id}/what-if` | Re-plan inventory and re-run the Critic with overrides, reusing the run's forecast and risk |

What-if overrides: `lead_time_days`, `lead_time_std`, `on_hand`, `service_level`,
`demand_multiplier`.

```bash
curl -s -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"item_id": "FOODS_3_519"}' | jq '.critic.recommendation'
RUN=$(curl -s "localhost:8000/runs/latest?item_ids=FOODS_3_519" | jq -r '.[0].run_id')
curl -s -X POST localhost:8000/runs/$RUN/what-if -H 'content-type: application/json' \
  -d '{"overrides": {"lead_time_days": 10, "service_level": 0.98, "demand_multiplier": 1.3}}' \
  | jq '{action: .inventory.action, qty: .inventory.reorder_qty, verdict: .critic.verdict}'
```

## Background jobs

| Method | Path | Description |
|---|---|---|
| POST | `/jobs` | Queue pipeline runs. Body: `{"item_ids": [...]}` or `{"all_demo": true}`. Returns **202** with the job |
| GET | `/jobs` | Recent jobs |
| GET | `/jobs/{job_id}` | Status (`queued`, `running`, `succeeded`, `partial`, `failed`), `done/total`, `progress`, `run_ids`, `errors` |

```bash
JOB=$(curl -s -X POST localhost:8000/jobs -H 'content-type: application/json' \
  -d '{"all_demo": true}' | jq -r .job_id)
watch -n1 "curl -s localhost:8000/jobs/$JOB | jq '{status, done, total}'"
```

A failing item does not stop the batch. The job ends `partial` and the error is listed.

## Recommendations and decisions

| Method | Path | Description |
|---|---|---|
| GET | `/recommendations?item_ids=..` | Overview rows: latest action, quantity, order value, risk, verdict and decision status per item |
| GET | `/purchase-orders/draft?format=csv` | Purchase-order lines from the latest approved or modified decisions (`format=json` also available) |
| POST | `/decisions` | Record a decision on a stored run. Returns **201** |
| GET | `/decisions?item_id=` | Decision log, oldest first |
| GET | `/decisions/latest` | Most recent decision per item |

Decision rules (enforced server-side, 400 otherwise):

* `APPROVED`: keeps the recommended quantity. The reason is optional; it defaults to
  "Approved as recommended".
* `MODIFIED`: `final_qty` must differ from the recommendation, and a reason of at least 3
  characters is required.
* `REJECTED`: final quantity 0, and a reason is required.

```bash
curl -s -X POST localhost:8000/decisions -H 'content-type: application/json' \
  -d "{\"run_id\": \"$RUN\", \"decision\": \"MODIFIED\", \"final_qty\": 288,
       \"reason\": \"Supplier offers a case discount\", \"reviewer\": \"dana\"}"
curl -s localhost:8000/purchase-orders/draft -o po_draft.csv
```

## Evaluation

| Method | Path | Description |
|---|---|---|
| GET | `/evaluation/summary` | Output of `scripts/run_evaluation.py` (404 until it has run) |
| GET | `/evaluation/tables/{name}` | `holdout_by_model`, `inventory_simulation` or `scenario_evaluation` as records |
| GET | `/evaluation/holdout?item_ids=..` | Live hold-out accuracy of the latest stored runs against `sales_holdout` |
