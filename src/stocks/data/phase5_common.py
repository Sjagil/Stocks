from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from dotenv import dotenv_values


PHASE5_CALCULATION_VERSION = "phase5_total_return_fx_v1"


def zero_financial_calls() -> dict[str, int]:
    return {"place_order": 0, "cancel_order": 0, "global_cancel": 0}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def decimal_from(value: Any, *, default: Decimal | None = None) -> Decimal | None:
    if value is None or str(value).strip() == "":
        return default
    return Decimal(str(value))


def parse_date(value: Any) -> date | None:
    if value is None or str(value).strip() == "":
        return None
    return date.fromisoformat(str(value)[:10])


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.ParquetFile(path).read().to_pylist()


def eodhd_api_token(env_file: str | Path = ".env") -> str | None:
    env_path = Path(env_file)
    values = dotenv_values(env_path) if env_path.exists() else {}
    if not any(values.get(key) for key in ("EODHD_API_KEY", "EOD_API_KEY", "EODHISTORICALDATA_API_KEY")):
        fallback = env_path.parent / ".env"
        if fallback.exists() and fallback != env_path:
            values = {**dotenv_values(fallback), **values}
    return (
        os.environ.get("EODHD_API_KEY")
        or values.get("EODHD_API_KEY")
        or os.environ.get("EOD_API_KEY")
        or values.get("EOD_API_KEY")
        or os.environ.get("EODHISTORICALDATA_API_KEY")
        or values.get("EODHISTORICALDATA_API_KEY")
    )


def eodhd_get_json(path: str, *, token: str, params: dict[str, Any], timeout_seconds: float = 30.0) -> Any:
    url = f"https://eodhd.com/api/{path.lstrip('/')}"
    safe_params = {**params, "api_token": token, "fmt": "json"}
    response = httpx.get(url, params=safe_params, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()
