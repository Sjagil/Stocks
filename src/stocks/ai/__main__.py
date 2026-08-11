from __future__ import annotations

import json
from pathlib import Path

from stocks.ai.plane import publish_ai_research_plane


def main() -> int:
    result = publish_ai_research_plane(Path.cwd())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
