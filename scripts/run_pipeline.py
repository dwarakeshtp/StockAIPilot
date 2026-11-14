"""Run the agentic pipeline for all (or selected) items and save the results.

Usage:
    python scripts/run_pipeline.py                     # all demo items
    python scripts/run_pipeline.py --items FOODS_3_090 FOODS_3_586
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mpl"))

from src.config.settings import load_settings  # noqa: E402
from src.logging.logger import configure_logging  # noqa: E402
from src.orchestration.runner import PipelineRunner  # noqa: E402


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", nargs="*",
                        help="Item ids, any of the ~1,400 in the database (default: demo items)")
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.outputs.logs_dir, level="WARNING")
    runner = PipelineRunner(settings)
    print(f"Data source: {runner.repository.source} | LLM: {runner.llm_description}")

    t0 = time.perf_counter()
    results = runner.run_items(
        args.items, progress=lambda i, n, item: print(f"[{i}/{n}] {item} ...", flush=True)
    )
    runner.save_results(results)

    print(f"\n{'product':<26}{'item':<14}{'model':<10}{'risk':<8}{'conf':>6}  "
          f"{'action':<9}{'qty':>6}  {'verdict':<24}{'rev':>4}")
    for r in results:
        print(f"{(r.product_name or '-')[:25]:<26}{r.item_id:<14}{r.forecast.model_used:<10}"
              f"{r.risk.risk_level.value:<8}"
              f"{r.risk.confidence_score:>6.1f}  {r.inventory.action.value:<9}"
              f"{r.inventory.reorder_qty:>6}  {r.critic.verdict.value:<24}{r.revision_count:>4}")
    print(f"\nSaved {len(results)} results to {settings.database.url} in "
          f"{time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
