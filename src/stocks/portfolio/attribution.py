from __future__ import annotations

import json
import sqlite3
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.portfolio.coverage import normalize_asset_class


PUBLIC_PATH = Path("output/portfolio/performance-attribution.json")
PRIVATE_PATH = Path("data/portfolio/private/performance-attribution.json")


def publish_performance_attribution(project_root: Path) -> dict[str, Any]:
    metadata = _metadata(project_root)
    facts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for environment, relative in (
        (
            "PAPER",
            "data/execution/phase9/private/paper_execution.sqlite3",
        ),
        ("LIVE", "data/execution/live/private/live_execution.sqlite3"),
    ):
        path = project_root / relative
        executions, commissions, intents = _canonical_rows(path)
        sources.append(
            {
                "environment": environment,
                "path": relative,
                "execution_count": len(executions),
                "commission_count": len(commissions),
                "intent_count": len(intents),
            }
        )
        commission_by_exec = {
            str(row["exec_identity"]): row for row in commissions
        }
        intent_by_id = {str(row["intent_id"]): row for row in intents}
        for execution in executions:
            payload = execution["payload"]
            intent = intent_by_id.get(str(execution["intent_id"]), {})
            intent_payload = intent.get("payload", {})
            symbol = str(
                payload.get("symbol")
                or intent_payload.get("symbol")
                or f"CON{payload.get('con_id', 'UNKNOWN')}"
            ).upper()
            instrument = metadata.get(symbol, {})
            quantity = _decimal(payload.get("quantity"))
            price = _decimal(payload.get("price"))
            fx = _decimal(payload.get("fx_rate"), default=Decimal("1"))
            capital = abs(quantity * price * fx)
            commission = commission_by_exec.get(
                str(execution["exec_identity"])
            )
            commission_value = _commission_value(commission)
            side = str(payload.get("side") or intent_payload.get("side") or "UNKNOWN")
            facts.append(
                {
                    "environment": environment,
                    "execution_identity": str(execution["exec_identity"]),
                    "intent_identity": str(execution["intent_id"]),
                    "timestamp": payload.get("execution_time")
                    or execution.get("created_at"),
                    "asset": symbol,
                    "asset_class": normalize_asset_class(instrument),
                    "strategy_family": str(
                        intent_payload.get("strategy_id")
                        or intent_payload.get("intent_source")
                        or "UNATTRIBUTED_MANUAL"
                    ),
                    "sector": str(instrument.get("sector") or "UNKNOWN"),
                    "industry": str(instrument.get("industry") or "UNKNOWN"),
                    "factor": str(instrument.get("sleeve") or "UNKNOWN"),
                    "regime": "REGIME_AT_EXECUTION_NOT_RECORDED",
                    "entry_type": side,
                    "exit_type": None,
                    "gross_pnl_eur": None,
                    "fees_eur": (
                        str(commission_value)
                        if commission_value is not None
                        else None
                    ),
                    "slippage_eur": None,
                    "net_pnl_eur": None,
                    "capital_employed_eur": str(capital),
                    "risk_employed_eur": None,
                    "realized": False,
                    "source": "CANONICAL_EXECUTION_AND_COMMISSION_RECORDS",
                }
            )
    missing = Counter()
    for row in facts:
        for field in (
            "gross_pnl_eur", "fees_eur", "slippage_eur", "net_pnl_eur",
            "risk_employed_eur", "exit_type",
        ):
            if row[field] is None:
                missing[field] += 1
    private: dict[str, Any] = {
        "schema": "canonical_derived_performance_attribution_v1",
        "status": (
            "GO_EMPTY"
            if not facts
            else "PARTIAL_PENDING_REALIZED_ROUND_TRIP"
            if missing.get("net_pnl_eur")
            else "GO"
        ),
        "sources": sources,
        "fact_count": len(facts),
        "realized_fact_count": sum(bool(row["realized"]) for row in facts),
        "facts": facts,
        "missing_measure_counts": dict(sorted(missing.items())),
        "dimensions": [
            "asset", "asset_class", "strategy_family", "sector", "industry",
            "factor", "regime", "entry_type", "exit_type",
        ],
        "measures": [
            "gross_pnl_eur", "fees_eur", "slippage_eur", "net_pnl_eur",
            "capital_employed_eur", "risk_employed_eur",
        ],
        "derived_read_model_only": True,
        "rebuildable_from_canonical_records": True,
        "parallel_financial_ledger_created": False,
        "canonical_records_mutated": False,
        "execution_authority": "NONE",
        "orders_generated": 0,
    }
    private["content_hash"] = stable_hash(private)
    public = {
        key: value
        for key, value in private.items()
        if key not in {"facts", "content_hash"}
    }
    public.update(
        {
            "private_fact_reference": PRIVATE_PATH.as_posix(),
            "financial_values_public": False,
            "content_hash": stable_hash(
                {
                    key: value
                    for key, value in private.items()
                    if key not in {"facts", "content_hash"}
                }
            ),
        }
    )
    _write(project_root, private, public)
    return public


def _canonical_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        return [], [], []
    try:
        with sqlite3.connect(path) as connection:
            executions = _payload_rows(
                connection,
                "SELECT exec_identity, intent_id, payload_json, created_at FROM executions",
                ("exec_identity", "intent_id", "payload_json", "created_at"),
            )
            commissions = _payload_rows(
                connection,
                "SELECT commission_identity, exec_identity, payload_json, created_at FROM commissions",
                (
                    "commission_identity", "exec_identity", "payload_json",
                    "created_at",
                ),
            )
            intents = _payload_rows(
                connection,
                "SELECT intent_id, economic_order_key, payload_json, created_at FROM intents",
                (
                    "intent_id", "economic_order_key", "payload_json",
                    "created_at",
                ),
            )
    except sqlite3.Error:
        return [], [], []
    return executions, commissions, intents


def _payload_rows(
    connection: sqlite3.Connection,
    query: str,
    columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for values in connection.execute(query):
        row = dict(zip(columns, values, strict=True))
        try:
            row["payload"] = json.loads(str(row.pop("payload_json")))
        except (TypeError, ValueError):
            row["payload"] = {}
        rows.append(row)
    return rows


def _metadata(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "output/universe/instruments.parquet"
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    return {
        str(row.get("symbol") or "").upper(): row
        for row in frame.to_dict(orient="records")
        if row.get("symbol")
    }


def _commission_value(row: dict[str, Any] | None) -> Decimal | None:
    if row is None:
        return None
    payload = row.get("payload", {})
    for field in ("amount", "commission", "commission_eur"):
        if payload.get(field) is not None:
            return _decimal(payload[field])
    return None


def _decimal(value: Any, *, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default


def _write(
    project_root: Path,
    private: dict[str, Any],
    public: dict[str, Any],
) -> None:
    private_path = project_root / PRIVATE_PATH
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    public_path = project_root / PUBLIC_PATH
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["publish_performance_attribution"]
