from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class EodhdSettings:
    api_key_configured: bool
    requested_enabled: bool = False
    data_phase_enabled: bool = False

    @property
    def enabled(self) -> bool:
        return self.api_key_configured and self.requested_enabled and self.data_phase_enabled

    @property
    def status(self) -> str:
        if not self.requested_enabled:
            return "DISABLED"
        if not self.api_key_configured:
            return "NO_GO_MISSING_API_KEY"
        if not self.data_phase_enabled:
            return "DISABLED_UNTIL_DATA_PHASE"
        return "GO"

    @property
    def authority(self) -> str:
        if self.enabled:
            return "research_data_read_only"
        return "disabled_until_data_phase"

    def safe_dict(self) -> dict[str, bool | str]:
        return {
            "provider": "EODHD",
            "enabled": self.enabled,
            "requested_enabled": self.requested_enabled,
            "api_key_configured": self.api_key_configured,
            "authority": self.authority,
            "status": self.status,
        }


def load_eodhd_settings(env_file: str | Path = ".env", *, data_phase_enabled: bool = False) -> EodhdSettings:
    env_path = Path(env_file)
    values = dotenv_values(env_path) if env_path.exists() else {}
    raw_key = os.environ.get("EODHD_API_KEY") or values.get("EODHD_API_KEY")
    enabled_raw = os.environ.get("EODHD_ENABLED") or values.get("EODHD_ENABLED") or ""
    requested_enabled = enabled_raw.strip().lower() in {"1", "true", "yes", "on"}
    return EodhdSettings(
        api_key_configured=bool(raw_key),
        requested_enabled=requested_enabled,
        data_phase_enabled=data_phase_enabled,
    )
