from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stocks.analysis.theme_news import collect_theme_news  # noqa: E402


def main() -> int:
    report = collect_theme_news(PROJECT_ROOT)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"GO", "NO_CURRENT_THEME_NEWS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
