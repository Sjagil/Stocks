from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stocks.analysis.theme_provisional import (  # noqa: E402
    build_theme_provisional_assessment,
)


def main() -> int:
    report = build_theme_provisional_assessment(PROJECT_ROOT)
    print(json.dumps(report, indent=2))
    return 0 if not str(report["status"]).startswith("BLOCKED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
