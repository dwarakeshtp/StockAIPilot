# Architecture

## 1. Layered view

```mermaid
flowchart TB
    subgraph UI["Presentation - dashboard/"]
        APP["Streamlit app<br/>Overview, Forecast, Risk, Inventory + What-if,<br/>Critic + Decision, Agent Trace, Metrics"]
    end
    subgraph ORCH["Orchestration - src/orchestration/"]
        RUN["PipelineRunner"] --> G["LangGraph StateGraph<br/>(pipeline + what-if graphs)"]
        G --> TR["traced_node -> AgentTraceEntry"]
        DEC["DecisionStore (JSONL)"]
    end
    subgraph AG["Agents - src/agents/"]
        FA["Forecast Agent"]
        RA["Risk & Uncertainty Agent"]
        IA["Inventory Planning Agent"]
        CA["Critic Agent<br/>13 ValidationRules"]
        LLM["LLMClient<br/>OpenRouter / Cohere / Template"]
        GR["Grounding check"]
    end
    subgraph DOM["Domain - src/models, src/inventory, src/evaluation"]
        M["ARIMA - Prophet - Seasonal-naive - Ensemble"]
        BT["Rolling-origin backtest + metrics"]
        INV["SS / ROP / ROQ / stockout formulas"]
        EV["Hold-out, inventory simulation, scenario eval"]
    end
    subgraph DATA["Data - src/data/"]
        ING["Ingestion: Parquet / M5 raw / Synthetic"]
        CL["Quality + Cleaning (censored demand)"]
        FE["Feature engineering + future frame"]
        SIM["Simulated inventory parameters"]
        REPO["DataRepository"]
    end
    subgraph X["Cross-cutting"]
        CFG["Settings (YAML + .env, pydantic)"]
        LOG["structlog JSON logs"]
        OUT["outputs/: runs, traces, logs, evaluation, decisions"]
    end

    APP --> RUN
    APP --> DEC
    G --> FA & RA & IA & CA
    FA --> M --> BT
    IA --> INV
    CA --> LLM
    FA & RA & IA -. narrative .-> LLM
    LLM --> GR
    G --> REPO
    REPO --> ING & SIM
    G --> CL --> FE
    RUN --> OUT
    TR --> OUT
```

Design principles:

* **Single responsibility.** Each agent owns one decision. Formulas live in pure functions
  (`src/inventory`, `src/agents/confidence.py`), so they can be unit-tested without agents.
* **Open/closed.** The Critic is a list of `ValidationRule` strategy objects, data sources
  implement `SalesDataSource`, forecasters implement `BaseForecaster` and LLM providers
  implement `LLMClient`. Each can be extended without touching its callers.
* **Dependency inversion.** Agents receive `Settings` and an `LLMClient`, and the graph
  receives a `PipelineDependencies` bundle. Tests inject synthetic data and fake LLMs.
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

What-if graph (reuses cached forecast and risk outputs; recomputes only the policy and
validation):

```mermaid
flowchart LR
    S((START)) --> inventory["inventory<br/>(overrides: lead time, service level,<br/>demand multiplier, on hand)"] --> critic --> E((END))
```

`PipelineState.trace` uses an `operator.add` reducer. Every node appends an
`AgentTraceEntry` (node, agent, input and output summary, reasoning, LLM provider, duration,
status), which is saved to `outputs/traces/{run_id}.json`.

## 3. Sequence: one item, end to end

```mermaid
sequenceDiagram
    autonumber
    actor P as Planner
    participant UI as Streamlit
    participant R as PipelineRunner
    participant G as LangGraph
    participant D as Data layer
    participant F as Forecast Agent
    participant K as Risk Agent
    participant I as Inventory Agent
    participant C as Critic Agent
    participant L as LLM (optional)

    P->>UI: Select item, click "Run item"
    UI->>R: run_item(item_id)
    R->>G: invoke(initial state)
    G->>D: ingest -> clean (quality report) -> features + future frame
    G->>F: run(features, future, holidays)
    F->>F: backtest ARIMA / Prophet / naive (4 x 7 days), pick best by MAE
    F-->>L: narrate(facts)
    L-->>F: text (grounded) or None -> template
    F-->>G: ForecastOutput
    G->>K: run(features, forecast, data_quality)
    K-->>G: RiskOutput (confidence, anomalies, risk level, multiplier)
    G->>I: run(forecast, risk, params)
    I-->>G: InventoryOutput (SS, ROP, qty, stockout prob, projection, action)
    G->>C: run(forecast, risk, inventory, quality)
    C->>C: 13 rules -> issues -> verdict
    alt NEEDS_REVISION (first time)
        C-->>G: revision hint (USE_ENSEMBLE / USE_ALTERNATIVE_MODEL)
        G->>F: rerun with revision
        Note over G,C: risk -> inventory -> critic again
    else final verdict
        C-->>L: explain(facts, issues)
        L-->>C: explanation (grounded) or template
        C-->>G: CriticOutput
    end
    G-->>R: final state + trace
    R->>R: save trace + latest.json
    R-->>UI: PipelineResult
    UI-->>P: Forecast, risk, recommendation, verdict, explanation
    P->>UI: Approve / Modify (qty, reason) / Reject
    UI->>R: DecisionStore.record() -> outputs/decisions.jsonl
```

## 4. Sequence: what-if analysis

```mermaid
sequenceDiagram
    actor P as Planner
    participant UI as Inventory page
    participant R as PipelineRunner
    participant W as What-if graph
    P->>UI: lead time = 10, service level = 0.98, demand x1.3
    UI->>R: what_if(base_result, overrides)
    R->>W: invoke(forecast, risk from base result)
    W->>W: Inventory Agent (overrides) -> Critic (revision budget spent)
    W-->>R: new InventoryOutput + CriticOutput
    R-->>UI: side-by-side baseline vs what-if
```

## 5. Data flow and persistence

| Artifact | Producer | Format |
|---|---|---|
| `data/processed/sales_subset.parquet` | `scripts/build_subset.py` | Long format: date, item, units, price, events, SNAP |
| `data/processed/calendar_future.parquet` | same | Next 28 days of calendar, prices and events |
| `data/processed/actuals_future.parquet` | same | Real sales after the origin (evaluation only) |
| `data/simulated/inventory_params.csv` | `DataRepository` (seeded) | On hand, on order, lead time, MOQ, pack, cost |
| `outputs/runs/latest.json` | `PipelineRunner.save_results` | List of `PipelineResult` |
| `outputs/traces/{run_id}.json` | `save_trace` | Agent trace per run |
| `outputs/logs/app.jsonl` | structlog | Structured JSON logs |
| `outputs/decisions.jsonl` | `DecisionStore` | Human Approve / Modify / Reject audit log |
| `outputs/evaluation/*` | `scripts/run_evaluation.py` | Summary JSON and CSVs |

## 6. Failure handling

| Failure | Behaviour |
|---|---|
| No processed data | Ingestion falls back to synthetic data (dev) and the UI shows the data source |
| ARIMA unstable or explosive | Sanity check, then enforced stationarity, then fallback orders; the model is excluded if still implausible |
| Prophet fails | The Forecast Agent continues with the remaining models |
| Critic finds a fixable forecast issue | One bounded revision loop (ensemble or alternative model) |
| Critic finds a critical or escalating issue | `NEEDS_HUMAN_REVIEW`, highlighted in the UI |
| LLM unavailable, rate-limited or ungrounded | Secondary provider, then deterministic template |
| Unknown what-if override | `ValueError`, shown as a UI error |
