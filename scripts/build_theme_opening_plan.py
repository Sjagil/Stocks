from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stocks.analysis.theme_session_plan import (  # noqa: E402
    build_theme_opening_session_plan,
)


def main() -> int:
    report = build_theme_opening_session_plan(PROJECT_ROOT)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"GO_WITH_BLOCKERS", "NO_CURRENT_CANDIDATES"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
