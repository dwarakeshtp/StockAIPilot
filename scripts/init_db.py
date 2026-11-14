"""Create and populate the StockPilot SQLite database.

Usage:
    python scripts/init_db.py                      # raw M5 CSVs if present, else the 12-item seed
    python scripts/init_db.py --source raw         # whole CA_1 FOODS category (~1,400 items)
    python scripts/init_db.py --source processed   # committed 12-item seed only
    python scripts/init_db.py --rebuild            # reload data (runs/decisions/jobs are kept)
    python scripts/init_db.py --download --source raw   # fetch the M5 CSVs first
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mpl"))

from src.config.settings import load_settings  # noqa: E402
from src.db.loader import init_db  # noqa: E402
from src.logging.logger import configure_logging  # noqa: E402


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["auto", "raw", "processed"], default="auto")
    parser.add_argument("--rebuild", action="store_true", help="Reload even if populated")
    parser.add_argument("--download", action="store_true", help="Download the raw M5 CSVs")
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings.outputs.logs_dir, level="WARNING")
    if args.download:
        from scripts.download_m5 import download_m5

        download_m5(settings.data.raw_dir)

    print(f"Database: {settings.database.url}")
    report = init_db(settings, source=args.source, rebuild=args.rebuild)
    if report.skipped:
        print(f"Already initialised from '{report.source}' (use --rebuild to reload).")
    else:
        print(f"Loaded from '{report.source}' in {report.seconds:.1f}s")
    print(f"  items:         {report.items:,} ({len(report.demo_items)} demo items)")
    print(f"  daily sales:   {report.sales_rows:,} rows, {report.first_date} .. {report.last_date}")
    print(f"  price spells:  {report.price_spells:,}")
    print(f"  calendar days: {report.calendar_days:,}")
    print(f"  holdout rows:  {report.holdout_rows:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
