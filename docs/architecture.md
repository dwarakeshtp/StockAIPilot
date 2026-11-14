# Architecture

## 1. Layered view

Three tiers: a Streamlit client, a FastAPI service with the agent orchestration behind it,
and a SQLite data layer that the M5 CSVs are loaded into at initialisation.

```mermaid
flowchart TB
    subgraph UI["Presentation - dashboard/"]
        APP["Streamlit app<br/>Overview, Forecast, Risk, Inventory + What-if,<br/>Critic + Decision, Agent Trace, Metrics"]
        CL["api_client.py (httpx)"]
    end
    subgraph API["API layer - src/api/"]
        RT["FastAPI routers<br/>system, items, inventory, forecasts, runs,<br/>jobs, recommendations, decisions, evaluation"]
    end
    subgraph SVC["Services - src/services/"]
        PS["PipelineService<br/>(lock: one agent run at a time)"]
        JS["JobService<br/>(single-worker executor)"]
        DS["DecisionService"]
        ES["EvaluationService"]
    end
    subgraph ORCH["Orchestration - src/orchestration/"]
        RUN["PipelineRunner"] --> G["LangGraph StateGraphs<br/>pipeline / forecast / what-if"]
        G --> TR["traced_node -> AgentTraceEntry"]
    end
    subgraph AG["Agents - src/agents/"]
        FA["Forecast Agent"]
        RA["Risk & Uncertainty Agent"]
        IA["Inventory Planning Agent"]
        CA["Critic Agent<br/>13 ValidationRules"]
        LLM["LLMClient<br/>OpenRouter / Cohere / Template"]
    end
    subgraph DOM["Domain - src/models, src/inventory, src/evaluation"]
        M["ARIMA - Prophet - Seasonal-naive - Ensemble"]
        INV["SS / ROP / ROQ / stockout formulas"]
        EV["Hold-out, inventory simulation, scenario eval"]
    end
    subgraph DATA["Data layer - src/db/ + src/data/"]
        REPO["Repository protocol<br/>SQLiteRepository (prod) / InMemoryRepository (tests)"]
        ST["RunStore / SQLDecisionStore / JobStore"]
        LD["Loader: init_db"]
        DB[("SQLite<br/>data/db/stockpilot.db")]
        CLN["Quality + cleaning + features"]
    end
    CSV["M5 CSVs (data/raw)"] --> LD
    SEED["12-item parquet seed"] --> LD

    APP --> CL -->|HTTP JSON| RT
    RT --> PS & JS & DS & ES
    JS --> PS --> RUN
    G --> FA & RA & IA & CA
    FA --> M
    IA --> INV
    CA --> LLM
    G --> REPO --> DB
    G --> CLN
    PS & DS & JS --> ST --> DB
    LD --> DB
    ES --> EV
```

Design principles:

* **Single responsibility.** Each agent owns one decision. Formulas live in pure functions
  (`src/inventory`, `src/agents/confidence.py`), so they can be unit-tested without agents.
* **Open/closed.** The Critic is a list of `ValidationRule` strategy objects, data sources
  implement `SalesDataSource`, forecasters implement `BaseForecaster` and LLM providers
  implement `LLMClient`. Each can be extended without touching its callers.
* **Dependency inversion.** Agents receive `Settings` and an `LLMClient`, and the graph
  receives a `PipelineDependencies` bundle whose repository satisfies the `Repository`
  protocol. Production uses `SQLiteRepository`; tests and the scenario evaluation inject an
  `InMemoryRepository`. Moving from SQLite to Postgres only changes the SQLAlchemy URL.
* **Thin HTTP layer.** Routers only translate HTTP to service calls. Business rules (decision
  validation, override checks, PO lines) live in services and stores, so scripts, tests and
  the API share them.
* **Typed contracts.** Agents communicate only through pydantic models (`ForecastOutput`,
  `RiskOutput`, `InventoryOutput`, `CriticOutput`). LangGraph state is a `TypedDict`.
* **Deterministic numbers, generative words.** Every number comes from code. The LLM only
  phrases explanations, and the grounding check rejects any sentence with an unknown number.

## 2. Agent workflow (LangGraph)

```mermaid
flowchart LR
    S((START)) --> ingest --> clean --> features --> forecast
    forecast --> risk --> inventory --> critic
    critic -- "NEEDS_REVISION<br/>(max 1 loop)" --> forecast
    critic -- "APPROVED / WITH_WARNINGS /<br/>NEEDS_HUMAN_REVIEW" --> E((END))
    E --> H{{"Planner: Approve / Modify / Reject"}}
```

Forecast graph (`POST /forecasts/{item_id}`): `ingest -> clean -> features -> forecast ->
risk`, with no planning or critique and nothing stored.

What-if graph (reuses cached forecast and risk outputs; recomputes only the policy and
validation):

```mermaid
flowchart LR
    S((START)) --> inventory["inventory<br/>(overrides: lead time, service level,<br/>demand multiplier, on hand)"] --> critic --> E((END))
```

`PipelineState.trace` uses an `operator.add` reducer. Every node appends an
`AgentTraceEntry` (node, agent, input and output summary, reasoning, LLM provider, duration,
status). The trace is stored with the run in `pipeline_runs.result_json` and exported to
`outputs/traces/{run_id}.json`.

## 3. Sequence: background job, UI -> API -> graph -> SQLite

```mermaid
sequenceDiagram
    autonumber
    actor P as Planner
    participant UI as Streamlit
    participant A as FastAPI
    participant J as JobService
    participant S as PipelineService
    participant G as LangGraph
    participant R as SQLiteRepository
    participant DB as SQLite
    participant L as LLM (optional)

    P->>UI: "Run all"
    UI->>A: POST /jobs {all_demo: true}
    A->>J: submit(demo items)
    J->>DB: INSERT jobs (queued)
    A-->>UI: 202 {job_id}
    loop every second until finished
        UI->>A: GET /jobs/{job_id}
        A-->>UI: status, done/total -> progress bar
    end
    loop each item (worker thread)
        J->>S: run_item(item_id)
        S->>G: invoke (under the pipeline lock)
        G->>R: item_sales / inventory / future calendar
        R->>DB: SELECT v_daily_sales (last 730 days), inventory_positions, calendar
        G->>G: clean -> features -> Forecast -> Risk -> Inventory -> Critic (revise once)
        G-->>L: narrate / explain (grounded) or template
        S->>DB: INSERT pipeline_runs (result_json + summary columns)
        J->>DB: UPDATE jobs (done += 1, run_ids)
    end
    J->>DB: UPDATE jobs (succeeded / partial / failed)
    UI->>A: GET /runs/latest?item_ids=...
    A-->>UI: PipelineResult[] -> pages
    P->>UI: Approve / Modify (qty, reason) / Reject
    UI->>A: POST /decisions {run_id, decision, final_qty, reason}
    A->>DB: validate rules -> INSERT decisions (400 with message if invalid)
```

Inside the graph, one item runs as follows:

```mermaid
sequenceDiagram
    participant G as LangGraph
    participant F as Forecast Agent
    participant K as Risk Agent
    participant I as Inventory Agent
    participant C as Critic Agent
    G->>F: run(features, future, holidays)
    F->>F: backtest ARIMA / Prophet / naive (4 x 7 days), pick best by MAE
    F-->>G: ForecastOutput
    G->>K: run(features, forecast, data_quality)
    K-->>G: RiskOutput (confidence, anomalies, risk level, multiplier)
    G->>I: run(forecast, risk, params)
    I-->>G: InventoryOutput (SS, ROP, qty, stockout prob, projection, action)
    G->>C: run(forecast, risk, inventory, quality)
    alt NEEDS_REVISION (first time)
        C-->>G: revision hint -> back to the Forecast Agent
    else final verdict
        C-->>G: CriticOutput (verdict, issues, grounded explanation)
    end
```

## 4. Sequence: what-if analysis

```mermaid
sequenceDiagram
    actor P as Planner
    participant UI as Inventory page
    participant A as FastAPI
    participant S as PipelineService
    participant W as What-if graph
    participant DB as SQLite
    P->>UI: lead time = 10, service level = 0.98, demand x1.3
    UI->>A: POST /runs/{run_id}/what-if {overrides}
    A->>S: validate override names (422 if unknown)
    S->>DB: load base run (forecast, risk)
    S->>W: invoke(forecast, risk, overrides)
    W->>W: Inventory Agent (overrides) -> Critic (revision budget spent)
    S->>DB: INSERT pipeline_runs (kind = whatif)
    A-->>UI: PipelineResult -> side-by-side baseline vs what-if
```

## 5. Data layer and persistence

`init_db` (run by `scripts/init_db.py` or on API start-up when the database is empty) loads
the whole CA_1 FOODS category from the raw M5 CSVs: 1,437 items, 2,789,217 daily rows and
1,969 calendar days, in about 10 seconds. Without the raw files it loads the committed
12-item parquet seed instead. Reloading (`--rebuild`) replaces only the data tables; runs,
decisions and jobs are kept.

```mermaid
erDiagram
    items ||--o{ daily_sales : "sold"
    items ||--o{ prices : "priced"
    items ||--o{ sales_holdout : "holdout"
    items ||--|| inventory_positions : "stocked"
    calendar ||--o{ daily_sales : "date"
    items ||--o{ pipeline_runs : "analysed"
    pipeline_runs ||--o{ decisions : "reviewed"
    items {
        text item_id PK
        text dept_id
        text product_name
        text department
        bool is_demo
    }
    calendar {
        text date PK
        text d
        int wm_yr_wk
        text event_name
        int snap
    }
    daily_sales {
        text item_id PK
        text date PK
        int units
    }
    prices {
        text item_id PK
        text start_date PK
        text end_date
        real sell_price
    }
    sales_holdout {
        text item_id PK
        text date PK
        int units
    }
    inventory_positions {
        text item_id PK
        int on_hand
        int on_order
        int lead_time_days
        bool is_simulated
        text updated_at
    }
    pipeline_runs {
        text run_id PK
        text kind
        text item_id
        text action
        text verdict
        text result_json
    }
    decisions {
        text decision_id PK
        text run_id
        text decision
        int final_qty
        text reason
    }
    jobs {
        text job_id PK
        text status
        int done
        int total
        text run_ids
    }
```

* The `v_daily_sales` view joins `daily_sales`, `items`, `calendar` and `prices` back into
  the long `SALES_COLUMNS` frame the pipeline has always used. `SQLiteRepository` serves the
  last `history_days` (730) of that view per item, so for the 12 demo items the agents see
  exactly the frames the parquet subset produced. A regression test enforces this.
* Prices are stored as **price spells**: consecutive days with the same price compress to
  one row (5,570 rows instead of about 2.8M). M5 prices change weekly, but spells also
  represent sources whose prices change mid-week (for example the synthetic test data)
  without loss.
* Inventory positions for the demo items come from `data/simulated/inventory_params.csv`.
  All other items are simulated with the same seeded generator. `PUT
  /items/{id}/inventory` updates a row, like a stock feed would.
* The engine uses WAL journaling, so the API can serve reads while a background job writes.

| Artifact | Producer | Format |
|---|---|---|
| `data/db/stockpilot.db` | `init_db`, RunStore, DecisionStore, JobStore | SQLite (gitignored, rebuilt from the CSVs) |
| `data/processed/*.parquet` | `scripts/build_subset.py` | 12-item seed used when `data/raw` is absent |
| `data/simulated/inventory_params.csv` | seeded simulator | Demo-item inventory seed |
| `outputs/traces/{run_id}.json` | `save_trace` | Agent trace export per run |
| `outputs/logs/app.jsonl` | structlog | Structured JSON logs |
| `outputs/evaluation/*` | `scripts/run_evaluation.py` | Summary JSON and CSVs (served by `/evaluation/*`) |

## 6. Failure handling

| Failure | Behaviour |
|---|---|
| Raw M5 CSVs missing | `init_db` loads the 12-item parquet seed (then synthetic data, if allowed) and `/system/info` reports the source |
| Database empty | Data endpoints return 503 with the `init_db` command; `/health` reports `database: empty` |
| API unreachable | The dashboard shows a banner with the command to start the API |
| One item fails inside a job | The job continues; it ends `partial` with the error listed |
| ARIMA unstable or explosive | Sanity check, then enforced stationarity, then fallback orders; the model is excluded if still implausible |
| Prophet fails | The Forecast Agent continues with the remaining models |
| Critic finds a fixable forecast issue | One bounded revision loop (ensemble or alternative model) |
| Critic finds a critical or escalating issue | `NEEDS_HUMAN_REVIEW`, highlighted in the UI |
| LLM unavailable, rate-limited or ungrounded | Secondary provider, then deterministic template |
| Unknown what-if override | 422 before any agent runs, shown as a UI error |
| Invalid planner decision | 400 with a plain-language message, shown in the decision form |
