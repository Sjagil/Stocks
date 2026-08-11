from __future__ import annotations

from pathlib import Path
from typing import Any


def dynamic_command(
    project_root: Path,
    command: str,
    *,
    symbol: str | None = None,
) -> dict[str, Any]:
    from stocks.dynamic.service import dynamic_command as run_dynamic_command

    return run_dynamic_command(project_root, command, symbol=symbol)

__all__ = ["dynamic_command"]
