"""Typed HTTP client for the StockPilot API (the dashboard never touches the database).

The base URL comes from ``STOCKPILOT_API_URL`` (or ``api.base_url`` in the settings).
Tests inject FastAPI's ``TestClient`` - an ``httpx.Client`` subclass - via :func:`set_client`.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

from src.agents.schemas import PipelineResult
from src.config.settings import load_settings
from src.orchestration.decisions import Decision, DecisionType

DEFAULT_TIMEOUT = httpx.Timeout(300.0, connect=5.0)


class ApiError(RuntimeError):
    """The API answered with an error status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ApiUnavailableError(ApiError):
    """The API could not be reached."""


def _rows_frame(payload: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(payload.get("rows", []))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


class StockPilotClient:
    """Thin wrapper over the REST endpoints returning domain objects and DataFrames."""

    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    @property
    def base_url(self) -> str:
        """Where requests go."""
        return str(self.http.base_url)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise ApiUnavailableError(0, f"Cannot reach the API at {self.base_url}: {exc}") \
                from None
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            if isinstance(detail, list):
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            raise ApiError(response.status_code, str(detail))
        return response

    def _get(self, path: str, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", path, params=clean).json()

    # System -----------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """``GET /health``."""
        return self._get("/health")

    def system_info(self) -> dict[str, Any]:
        """``GET /system/info``."""
        return self._get("/system/info")

    # Data ---------------------------------------------------------------------
    def items(self, search: str | None = None, demo_only: bool = False, limit: int = 50,
              offset: int = 0) -> dict[str, Any]:
        """``GET /items`` page (``items``, ``total``)."""
        return self._get("/items", search=search or None, demo_only=demo_only, limit=limit,
                         offset=offset)

    def item(self, item_id: str) -> dict[str, Any]:
        """``GET /items/{id}`` (item, inventory, stats, latest_run_id)."""
        return self._get(f"/items/{item_id}")

    def history(self, item_id: str) -> pd.DataFrame:
        """Cleaned daily history."""
        return _rows_frame(self._get(f"/items/{item_id}/history"))

    def actuals(self, item_id: str) -> pd.DataFrame | None:
        """Hold-out sales after the forecast origin (``None`` if not loaded)."""
        df = _rows_frame(self._get(f"/items/{item_id}/actuals"))
        return df if not df.empty else None

    def update_inventory(self, item_id: str, **changes: Any) -> dict[str, Any]:
        """``PUT /items/{id}/inventory``."""
        return self._request("PUT", f"/items/{item_id}/inventory", json=changes).json()

    # Agent workflows --------------------------------------------------------------
    def latest_runs(self, item_ids: list[str] | None = None) -> dict[str, PipelineResult]:
        """Latest pipeline run per item."""
        payload = self._get("/runs/latest", item_ids=item_ids or None)
        return {r["item_id"]: PipelineResult.model_validate(r) for r in payload}

    def run(self, run_id: str) -> PipelineResult:
        """One stored run."""
        return PipelineResult.model_validate(self._get(f"/runs/{run_id}"))

    def run_item(self, item_id: str, overrides: dict[str, float] | None = None) -> PipelineResult:
        """``POST /runs`` (synchronous)."""
        payload = self._request("POST", "/runs",
                                json={"item_id": item_id, "overrides": overrides or {}}).json()
        return PipelineResult.model_validate(payload)

    def what_if(self, run_id: str, overrides: dict[str, float]) -> PipelineResult:
        """``POST /runs/{run_id}/what-if``."""
        payload = self._request("POST", f"/runs/{run_id}/what-if",
                                json={"overrides": overrides}).json()
        return PipelineResult.model_validate(payload)

    def submit_job(self, item_ids: list[str] | None = None) -> dict[str, Any]:
        """``POST /jobs`` (empty = all demo items)."""
        return self._request("POST", "/jobs", json={"item_ids": item_ids or [],
                                                    "all_demo": not item_ids}).json()

    def job(self, job_id: str) -> dict[str, Any]:
        """``GET /jobs/{job_id}``."""
        return self._get(f"/jobs/{job_id}")

    # Recommendations & decisions ----------------------------------------------------
    def recommendations(self) -> list[dict[str, Any]]:
        """``GET /recommendations``."""
        return self._get("/recommendations")

    def decisions(self, item_id: str | None = None) -> list[Decision]:
        """Decision log, oldest first."""
        return [Decision.model_validate(d) for d in self._get("/decisions", item_id=item_id)]

    def latest_decisions(self) -> dict[str, Decision]:
        """Most recent decision per item."""
        return {d["item_id"]: Decision.model_validate(d)
                for d in self._get("/decisions/latest")}

    def record_decision(self, run_id: str, decision: DecisionType, reason: str = "",
                        final_qty: int | None = None, reviewer: str = "planner") -> Decision:
        """``POST /decisions``; raises :class:`ApiError` (400) with the rule message."""
        payload = self._request("POST", "/decisions", json={
            "run_id": run_id, "decision": decision.value, "reason": reason,
            "final_qty": final_qty, "reviewer": reviewer}).json()
        return Decision.model_validate(payload)

    def po_draft_csv(self) -> str:
        """``GET /purchase-orders/draft`` as CSV text."""
        return self._request("GET", "/purchase-orders/draft").text

    # Evaluation -----------------------------------------------------------------
    def evaluation_summary(self) -> dict[str, Any] | None:
        """``GET /evaluation/summary`` (``None`` until the evaluation has run)."""
        try:
            return self._get("/evaluation/summary")
        except ApiError as exc:
            if exc.status_code == 404:
                return None
            raise


_client: StockPilotClient | None = None


def default_base_url() -> str:
    """``STOCKPILOT_API_URL`` or the configured ``api.base_url``."""
    return os.getenv("STOCKPILOT_API_URL") or load_settings().api.base_url


def get_client() -> StockPilotClient:
    """Process-wide client (created on first use)."""
    global _client
    if _client is None:
        _client = StockPilotClient(httpx.Client(base_url=default_base_url(),
                                                timeout=DEFAULT_TIMEOUT))
    return _client


def set_client(http: httpx.Client | None) -> None:
    """Use ``http`` for all requests (``None`` resets to the default)."""
    global _client
    _client = StockPilotClient(http) if http is not None else None
