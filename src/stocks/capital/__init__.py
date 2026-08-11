from __future__ import annotations

from typing import Any

__all__ = [
    "capital_command",
    "capital_level_limits",
    "portfolio_management_command",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from stocks.capital import service

        return getattr(service, name)
    raise AttributeError(name)
