"""HTTP API tests: every router against a temporary synthetic database."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.data.ingestion import SyntheticSource
from src.db.engine import get_engine
from src.db.loader import load_dataset
from src.services import Services
from tests.conftest import make_test_settings


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """API over a 2-item synthetic database; one job has already run all items."""
    settings = make_test_settings(tmp_path_factory.mktemp("api"))
    engine = get_engine(settings.database.url)
    load_dataset(engine, settings, SyntheticSource(n_items=2, history_days=300, seed=11).load())
    services = Services.build(settings, engine)
    with TestClient(create_app(services=services)) as test_client:
        job = test_client.post("/jobs", json={"all_demo": True}).json()
        services.jobs.wait(job["job_id"], timeout=300)
        yield test_client
    services.close()


@pytest.fixture(scope="module")
def item(client) -> str:
    return client.get("/items", params={"demo_only": True}).json()["items"][0]["item_id"]


@pytest.fixture(scope="module")
def latest_run(client, item) -> dict:
    return client.get("/runs/latest", params={"item_ids": [item]}).json()[0]


def test_health_and_system(client) -> None:
    assert client.get("/health").json() == {"status": "ok", "database": "ready",
                                            "source": "synthetic"}
    info = client.get("/system/info").json()
    assert info["data_source"] == "sqlite:synthetic" and info["items"] == 2
    assert info["counts"]["daily_sales"] == 600 and info["llm"] == "template"
    assert "ingest" in client.get("/system/graph").json()["mermaid"]


def test_items_and_data_endpoints(client, item) -> None:
    page = client.get("/items", params={"limit": 1}).json()
    assert page["total"] == 2 and len(page["items"]) == 1
    assert client.get("/items", params={"search": "nothing-like-this"}).json()["total"] == 0
    detail = client.get(f"/items/{item}").json()
    assert detail["item"]["item_id"] == item and detail["latest_run_id"]
    assert detail["stats"]["history_days"] == 300
    sales = client.get(f"/items/{item}/sales").json()["rows"]
    assert len(sales) == 300 and {"date", "units", "sell_price", "snap"} <= set(sales[0])
    history = client.get(f"/items/{item}/history").json()["rows"]
    assert "is_suspected_stockout" in history[0]
    assert client.get(f"/items/{item}/actuals").json()["rows"] == []
    assert len(client.get("/calendar/future").json()["rows"]) == 28


def test_unknown_item_returns_404(client) -> None:
    for path in ("/items/NOPE", "/items/NOPE/sales", "/items/NOPE/history",
                 "/items/NOPE/actuals", "/items/NOPE/inventory"):
        assert client.get(path).status_code == 404, path
    assert client.post("/forecasts/NOPE").status_code == 404
    assert client.post("/runs", json={"item_id": "NOPE"}).status_code == 404
    assert client.post("/jobs", json={"item_ids": ["NOPE"]}).status_code == 404


def test_inventory_update(client, item) -> None:
    before = client.get(f"/items/{item}/inventory").json()
    assert before["is_simulated"] is True
    after = client.put(f"/items/{item}/inventory", json={"on_hand": 11}).json()
    assert after["on_hand"] == 11 and after["is_simulated"] is False
    assert client.put(f"/items/{item}/inventory", json={"item_id": "x"}).status_code == 422
    assert client.put(f"/items/{item}/inventory", json={"on_hand": -1}).status_code == 422
    assert client.put(f"/items/{item}/inventory", json={}).status_code == 422


def test_forecast_endpoint(client, item) -> None:
    body = client.post(f"/forecasts/{item}").json()
    assert body["item_id"] == item and len(body["forecast"]["forecast"]) > 0
    assert body["risk"]["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert [t["node"] for t in body["trace"]] == ["ingest", "clean", "features", "forecast",
                                                  "risk"]


def test_runs_endpoints(client, item, latest_run) -> None:
    run = client.post("/runs", json={"item_id": item}).json()
    assert run["item_id"] == item and run["critic"]["verdict"]
    assert client.get(f"/runs/{run['run_id']}").json()["run_id"] == run["run_id"]
    trace = client.get(f"/runs/{run['run_id']}/trace").json()
    assert trace[0]["node"] == "ingest"
    assert any(r["run_id"] == run["run_id"] for r in client.get("/runs").json())
    assert client.get("/runs/run-missing").status_code == 404
    bad = client.post("/runs", json={"item_id": item, "overrides": {"bogus": 1}})
    assert bad.status_code == 422 and "Unsupported overrides" in bad.json()["detail"]


def test_what_if(client, latest_run) -> None:
    resp = client.post(f"/runs/{latest_run['run_id']}/what-if",
                       json={"overrides": {"lead_time_days": 9, "demand_multiplier": 1.5}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"].startswith("whatif") and body["overrides"]["lead_time_days"] == 9
    assert body["forecast"] == latest_run["forecast"]
    assert client.post(f"/runs/{latest_run['run_id']}/what-if",
                       json={"overrides": {"nope": 1}}).status_code == 422
    assert client.post("/runs/run-missing/what-if", json={"overrides": {}}).status_code == 404


def test_job_lifecycle(client, item) -> None:
    job = client.post("/jobs", json={"item_ids": [item]})
    assert job.status_code == 202 and job.json()["status"] == "queued"
    job_id = job.json()["job_id"]
    for _ in range(300):
        state = client.get(f"/jobs/{job_id}").json()
        if state["status"] not in ("queued", "running"):
            break
        time.sleep(0.2)
    assert state["status"] == "succeeded" and state["progress"] == 1
    assert len(state["run_ids"]) == 1
    assert client.get("/jobs").json()[0]["job_id"] == job_id
    assert client.get("/jobs/job-missing").status_code == 404


def test_recommendations_decisions_and_po(client) -> None:
    recs = client.get("/recommendations").json()
    assert len(recs) == 2 and all(r["decision"] == "PENDING" for r in recs)
    rec = recs[0]
    invalid = client.post("/decisions", json={"run_id": rec["run_id"], "decision": "MODIFIED",
                                              "final_qty": rec["quantity"] + 6})
    assert invalid.status_code == 400 and "reason" in invalid.json()["detail"]
    ok = client.post("/decisions", json={"run_id": rec["run_id"], "decision": "MODIFIED",
                                         "final_qty": rec["quantity"] + 6,
                                         "reason": "Promotion next week"})
    assert ok.status_code == 201 and ok.json()["final_qty"] == rec["quantity"] + 6
    assert client.post("/decisions", json={"run_id": "run-missing",
                                           "decision": "APPROVED"}).status_code == 404
    assert len(client.get("/decisions").json()) == 1
    assert client.get("/decisions/latest").json()[0]["decision"] == "MODIFIED"
    updated = {r["item_id"]: r for r in client.get("/recommendations").json()}
    assert updated[rec["item_id"]]["decision"] == "MODIFIED"

    csv = client.get("/purchase-orders/draft")
    assert csv.headers["content-type"].startswith("text/csv")
    lines = csv.text.strip().splitlines()
    assert lines[0].startswith("item_id,product") and len(lines) == 2
    assert client.get("/purchase-orders/draft", params={"format": "json"}).json()[0][
        "quantity"] == rec["quantity"] + 6


def test_evaluation_endpoints(client) -> None:
    assert client.get("/evaluation/summary").status_code == 404
    assert client.get("/evaluation/tables/unknown").status_code == 404
    assert client.get("/evaluation/tables/holdout_by_model").status_code == 404
    assert client.get("/evaluation/holdout").json() == {}


def test_uninitialised_database_returns_503(tmp_path) -> None:
    settings = make_test_settings(tmp_path)
    services = Services.build(settings, get_engine(settings.database.url))
    with TestClient(create_app(services=services)) as test_client:
        assert test_client.get("/health").json()["database"] == "empty"
        assert test_client.get("/system/info").status_code == 503
        assert test_client.post("/jobs", json={"all_demo": True}).status_code == 503
    services.close()
