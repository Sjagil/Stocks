from __future__ import annotations

import json
from pathlib import Path

from stocks.p3.publisher import publish_p3_evidence


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    result = publish_p3_evidence(project_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
