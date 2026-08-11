"""Portfolio control-plane contracts."""

from __future__ import annotations

from typing import Any

__all__ = [
    "active_portfolio_command",
    "build_active_portfolio_report",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from stocks.portfolio import manager

        return getattr(manager, name)
    raise AttributeError(name)
