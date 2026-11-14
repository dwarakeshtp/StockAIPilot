"""Product catalog: human-readable names for the anonymised M5 item codes.

M5 hides real product identities, so the names in ``data/reference/product_catalog.csv``
are illustrative labels chosen to match each item's sales profile (volume, price,
intermittency). Items missing from the catalog fall back to their item code.
Names must not contain digits: the grounding check treats any number in LLM text
as a fact that has to be verified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class Product(BaseModel):
    """Display metadata for one item."""

    item_id: str
    name: str
    department: str = "Unknown"
    unit: str = "unit"
    perishable: bool = False
    profile_note: str = ""

    @property
    def label(self) -> str:
        """Name plus item code, e.g. ``Bananas (FOODS_3_090)``."""
        return self.name if self.name == self.item_id else f"{self.name} ({self.item_id})"


def fallback_product(item_id: str) -> Product:
    """Product entry for an item that is not in the catalog."""
    return Product(item_id=item_id, name=item_id)


def load_catalog(path: Path) -> dict[str, Product]:
    """Load the catalog CSV keyed by item id (empty dict if the file is missing)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"item_id": str}).fillna("")
    return {
        row["item_id"]: Product(
            item_id=row["item_id"],
            name=row["product_name"] or row["item_id"],
            department=row.get("department") or "Unknown",
            unit=row.get("unit") or "unit",
            perishable=str(row.get("perishable", "")).lower() == "true",
            profile_note=row.get("profile_note", ""),
        )
        for _, row in df.iterrows()
    }
