from __future__ import annotations

from dataclasses import dataclass

from .config import IbkrSettings, load_ibkr_settings


@dataclass(frozen=True)
class AppContext:
    ibkr: IbkrSettings


def load_app_context(env_file: str = ".env.ibkr") -> AppContext:
    return AppContext(ibkr=load_ibkr_settings(env_file))
