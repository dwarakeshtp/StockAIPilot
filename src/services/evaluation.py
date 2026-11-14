"""Evaluation service: offline evaluation artefacts plus live hold-out accuracy."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.settings import Settings
from src.db.repository import SQLiteRepository
from src.db.stores import RunStore
from src.evaluation.holdout import holdout_accuracy
from src.utils.io import read_json

TABLES = {
    "holdout_by_model": "holdout_by_model.csv",
    "inventory_simulation": "inventory_simulation.csv",
    "scenario_evaluation": "scenario_evaluation.csv",
}


class EvaluationService:
    """Serves ``scripts/run_evaluation.py`` outputs and recomputes hold-out accuracy."""

    def __init__(self, settings: Settings, repository: SQLiteRepository,
                 run_store: RunStore) -> None:
        self.settings = settings
        self.repository = repository
        self.run_store = run_store

    def summary(self) -> dict[str, Any] | None:
        """``evaluation_summary.json`` (``None`` until the evaluation script has run)."""
        path = self.settings.outputs.evaluation_dir / "evaluation_summary.json"
        return read_json(path) if path.exists() else None

    def table(self, name: str) -> list[dict[str, Any]]:
        """One evaluation CSV as records.

        Raises:
            KeyError: if ``name`` is not a known table.
            FileNotFoundError: if the evaluation has not produced it yet.
        """
        path = self.settings.outputs.evaluation_dir / TABLES[name]
        if not path.exists():
            raise FileNotFoundError(path.name)
        df = pd.read_csv(path)
        return df.astype(object).where(df.notna(), None).to_dict(orient="records")

    def holdout(self, item_ids: list[str] | None = None) -> dict[str, Any]:
        """Hold-out accuracy of the latest stored runs against ``sales_holdout``."""
        results = list(self.run_store.latest_by_item(item_ids=item_ids).values())
        return holdout_accuracy(results, self.repository)
