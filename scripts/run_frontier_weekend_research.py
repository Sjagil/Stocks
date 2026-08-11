from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stocks.analysis.weekend_frontier import (  # noqa: E402
    run_frontier_weekend_research,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded read-only quantum/nuclear weekend research."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run outside Saturday/Sunday; still read-only.",
    )
    parser.add_argument(
        "--no-bars",
        action="store_true",
        help="Reuse the current local multi-timeframe cache.",
    )
    args = parser.parse_args()
    report = run_frontier_weekend_research(
        PROJECT_ROOT,
        force=args.force,
        refresh_bars=not args.no_bars,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] in {"GO", "SKIPPED_NOT_WEEKEND"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
