from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from stocks.analysis.theme_contracts import collect_theme_contracts  # noqa: E402


def main() -> int:
    report = collect_theme_contracts(PROJECT_ROOT)
    print(json.dumps(report, indent=2, ensure_ascii=True, default=str))
    return 0 if report["status"] in {"GO", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
