from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stocks.analysis.themes import build_frontier_theme_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run research-only quantum and nuclear/uranium theme analysis."
        )
    )
    parser.add_argument(
        "--theme",
        choices=["quantum_computing", "nuclear_uranium"],
    )
    args = parser.parse_args()
    report = build_frontier_theme_analysis(
        PROJECT_ROOT,
        theme=args.theme,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"GO", "GO_WITH_DOCUMENTED_GAPS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
