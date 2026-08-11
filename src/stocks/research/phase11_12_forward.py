from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from stocks.research.autopilot.contracts import stable_hash


SCHEMA = "phase11_12_prospective_forward_evidence_v1"
COST_BPS_PER_SIDE = 50.0
ACTIVE_STATUSES = ("PENDING_ENTRY", "OPEN", "PENDING_EXIT")


def update_lower_timeframe_forward(
    project_root: Path,
    observation: Mapping[str, Any],
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
) -> dict[str, Any]:
    path = _database_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        _initialize(connection)
        identity = _observation_identity(observation)
        with connection:
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO observation_audit (
                    observation_hash, observed_at, active_signal_count,
                    recorded_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    identity,
                    str(observation.get("observed_at")),
                    int(observation.get("active_signal_count", 0)),
                    _now(),
                ),
            ).rowcount
            if inserted:
                _advance_pending_entries(connection, frames_by_timeframe)
                _advance_pending_exits(connection, frames_by_timeframe)
                _apply_current_desired_state(
                    connection,
                    observation,
                )
        report = _build_report(connection, path)
        report["observation_inserted"] = bool(inserted)
        report["observation_hash"] = identity
    finally:
        connection.close()
    _write_json(_output(project_root) / "forward-performance.json", report)
    return report


def lower_timeframe_forward_status(project_root: Path) -> dict[str, Any]:
    path = _database_path(project_root)
    if not path.exists():
        return {
            "schema": SCHEMA,
            "status": "NOT_STARTED",
            "private_database": str(path),
            **_authority(),
        }
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        _initialize(connection)
        report = _build_report(connection, path)
    finally:
        connection.close()
    _write_json(_output(project_root) / "forward-performance.json", report)
    return report


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS observation_audit (
            observation_hash TEXT PRIMARY KEY,
            observed_at TEXT NOT NULL,
            active_signal_count INTEGER NOT NULL,
            recorded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            formula TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            profile TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            symbol TEXT NOT NULL,
            status TEXT NOT NULL,
            signal_first_seen_at TEXT NOT NULL,
            entry_trigger_at TEXT NOT NULL,
            entry_filled_at TEXT,
            entry_price_eur REAL,
            exit_trigger_at TEXT,
            exit_filled_at TEXT,
            exit_price_eur REAL,
            gross_return REAL,
            net_return REAL,
            cost_bps_per_side REAL NOT NULL,
            last_seen_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_episode
        ON episodes(strategy_id, symbol)
        WHERE status IN ('PENDING_ENTRY', 'OPEN', 'PENDING_EXIT');
        CREATE TABLE IF NOT EXISTS episode_events (
            event_id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_at TEXT NOT NULL,
            bar_at TEXT,
            value_eur REAL,
            payload_hash TEXT NOT NULL,
            FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
        );
        """
    )


def _advance_pending_entries(
    connection: sqlite3.Connection,
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    rows = connection.execute(
        "SELECT * FROM episodes WHERE status='PENDING_ENTRY'"
    ).fetchall()
    for row in rows:
        fill = _next_open(
            frames_by_timeframe,
            timeframe=str(row["timeframe"]),
            symbol=str(row["symbol"]),
            after=str(row["entry_trigger_at"]),
        )
        if fill is None:
            continue
        filled_at, price = fill
        connection.execute(
            """
            UPDATE episodes
            SET status='OPEN', entry_filled_at=?, entry_price_eur=?,
                last_seen_at=?
            WHERE episode_id=? AND status='PENDING_ENTRY'
            """,
            (filled_at, price, _now(), row["episode_id"]),
        )
        _append_event(
            connection,
            str(row["episode_id"]),
            "ENTRY_FILLED_NEXT_BAR_OPEN",
            bar_at=filled_at,
            value_eur=price,
        )


def _advance_pending_exits(
    connection: sqlite3.Connection,
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    rows = connection.execute(
        "SELECT * FROM episodes WHERE status='PENDING_EXIT'"
    ).fetchall()
    for row in rows:
        fill = _next_open(
            frames_by_timeframe,
            timeframe=str(row["timeframe"]),
            symbol=str(row["symbol"]),
            after=str(row["exit_trigger_at"]),
        )
        if fill is None:
            continue
        entry = _finite(row["entry_price_eur"])
        if entry is None or entry <= 0:
            continue
        filled_at, exit_price = fill
        gross_return = (exit_price / entry) - 1.0
        cost = COST_BPS_PER_SIDE / 10_000.0
        net_return = (
            (exit_price * (1.0 - cost)) / (entry * (1.0 + cost))
        ) - 1.0
        connection.execute(
            """
            UPDATE episodes
            SET status='CLOSED', exit_filled_at=?, exit_price_eur=?,
                gross_return=?, net_return=?, last_seen_at=?
            WHERE episode_id=? AND status='PENDING_EXIT'
            """,
            (
                filled_at,
                exit_price,
                gross_return,
                net_return,
                _now(),
                row["episode_id"],
            ),
        )
        _append_event(
            connection,
            str(row["episode_id"]),
            "EXIT_FILLED_NEXT_BAR_OPEN",
            bar_at=filled_at,
            value_eur=exit_price,
        )


def _apply_current_desired_state(
    connection: sqlite3.Connection,
    observation: Mapping[str, Any],
) -> None:
    observed = {
        str(row["strategy_id"]): row
        for row in observation.get("observations", [])
        if row.get("observation_status") == "OBSERVATION_COMPLETE"
        and int(row.get("stale_signal_count", 0)) == 0
    }
    desired = {
        (str(row["strategy_id"]), str(row["symbol"])): row
        for row in observation.get("active_signals", [])
        if row.get("data_freshness") == "FRESH_CLOSED_BAR"
    }
    active_rows = connection.execute(
        """
        SELECT * FROM episodes
        WHERE status IN ('PENDING_ENTRY', 'OPEN', 'PENDING_EXIT')
        """
    ).fetchall()
    active_by_key = {
        (str(row["strategy_id"]), str(row["symbol"])): row
        for row in active_rows
    }

    for key, row in active_by_key.items():
        strategy_id, _symbol = key
        if strategy_id not in observed or key in desired:
            connection.execute(
                "UPDATE episodes SET last_seen_at=? WHERE episode_id=?",
                (_now(), row["episode_id"]),
            )
            continue
        trigger_at = str(observed[strategy_id]["closed_bar_timestamp"])
        if row["status"] == "PENDING_ENTRY":
            connection.execute(
                """
                UPDATE episodes
                SET status='CANCELLED_BEFORE_ENTRY', last_seen_at=?
                WHERE episode_id=? AND status='PENDING_ENTRY'
                """,
                (_now(), row["episode_id"]),
            )
            _append_event(
                connection,
                str(row["episode_id"]),
                "ENTRY_CANCELLED_SIGNAL_REMOVED_BEFORE_NEXT_BAR",
            )
        elif row["status"] == "OPEN":
            connection.execute(
                """
                UPDATE episodes
                SET status='PENDING_EXIT', exit_trigger_at=?, last_seen_at=?
                WHERE episode_id=? AND status='OPEN'
                """,
                (trigger_at, _now(), row["episode_id"]),
            )
            _append_event(
                connection,
                str(row["episode_id"]),
                "EXIT_TRIGGERED_CLOSED_BAR",
                bar_at=trigger_at,
            )

    for key, signal in desired.items():
        if key in active_by_key:
            continue
        trigger_at = str(signal["closed_bar_timestamp"])
        payload = {
            "strategy_id": signal["strategy_id"],
            "symbol": signal["symbol"],
            "trigger_at": trigger_at,
        }
        episode_id = "FWD-" + stable_hash(payload)[:24]
        connection.execute(
            """
            INSERT OR IGNORE INTO episodes (
                episode_id, strategy_id, formula, timeframe, profile,
                asset_class, symbol, status, signal_first_seen_at,
                entry_trigger_at, cost_bps_per_side, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING_ENTRY', ?, ?, ?, ?)
            """,
            (
                episode_id,
                str(signal["strategy_id"]),
                str(signal["formula"]),
                str(signal["timeframe"]),
                str(signal["profile"]),
                str(signal["asset_class"]),
                str(signal["symbol"]),
                str(observation.get("observed_at")),
                trigger_at,
                COST_BPS_PER_SIDE,
                _now(),
            ),
        )
        _append_event(
            connection,
            episode_id,
            "ENTRY_TRIGGERED_CLOSED_BAR",
            bar_at=trigger_at,
        )


def _next_open(
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    timeframe: str,
    symbol: str,
    after: str,
) -> tuple[str, float] | None:
    frame = frames_by_timeframe.get(timeframe, {}).get(symbol)
    if frame is None or frame.empty or "open" not in frame:
        return None
    trigger = _timestamp(after)
    candidates = frame.loc[pd.DatetimeIndex(frame.index) > trigger]
    if candidates.empty:
        return None
    timestamp = pd.Timestamp(candidates.index[0])
    price = _finite(candidates.iloc[0]["open"])
    if price is None or price <= 0:
        return None
    return timestamp.isoformat(), price


def _append_event(
    connection: sqlite3.Connection,
    episode_id: str,
    event_type: str,
    *,
    bar_at: str | None = None,
    value_eur: float | None = None,
) -> None:
    payload = {
        "episode_id": episode_id,
        "event_type": event_type,
        "bar_at": bar_at,
        "value_eur": value_eur,
    }
    payload_hash = stable_hash(payload)
    event_id = "EVT-" + payload_hash[:24]
    connection.execute(
        """
        INSERT OR IGNORE INTO episode_events (
            event_id, episode_id, event_type, event_at, bar_at,
            value_eur, payload_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            episode_id,
            event_type,
            _now(),
            bar_at,
            value_eur,
            payload_hash,
        ),
    )


def _build_report(
    connection: sqlite3.Connection,
    path: Path,
) -> dict[str, Any]:
    episodes = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM episodes ORDER BY signal_first_seen_at, episode_id"
        ).fetchall()
    ]
    closed = [row for row in episodes if row["status"] == "CLOSED"]
    strategy_ids = sorted({str(row["strategy_id"]) for row in episodes})
    strategies = [
        _metrics(
            [
                row
                for row in closed
                if str(row["strategy_id"]) == strategy_id
            ],
            strategy_id=strategy_id,
            all_rows=[
                row
                for row in episodes
                if str(row["strategy_id"]) == strategy_id
            ],
        )
        for strategy_id in strategy_ids
    ]
    aggregate = _metrics(closed, strategy_id="ALL", all_rows=episodes)
    observation_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM observation_audit"
        ).fetchone()[0]
    )
    event_count = int(
        connection.execute("SELECT COUNT(*) FROM episode_events").fetchone()[0]
    )
    return {
        "schema": SCHEMA,
        "status": "GO",
        "generated_at": _now(),
        "cost_model": {
            "cost_bps_per_side": COST_BPS_PER_SIDE,
            "round_trip_cost_bps": COST_BPS_PER_SIDE * 2.0,
            "entry": "NEXT_AVAILABLE_BAR_OPEN",
            "exit": "NEXT_AVAILABLE_BAR_OPEN_AFTER_SIGNAL_REMOVAL",
            "same_bar_execution": False,
        },
        "observation_count": observation_count,
        "event_count": event_count,
        "episode_count": len(episodes),
        "pending_entry_count": sum(
            row["status"] == "PENDING_ENTRY" for row in episodes
        ),
        "open_episode_count": sum(
            row["status"] == "OPEN" for row in episodes
        ),
        "pending_exit_count": sum(
            row["status"] == "PENDING_EXIT" for row in episodes
        ),
        "cancelled_before_entry_count": sum(
            row["status"] == "CANCELLED_BEFORE_ENTRY" for row in episodes
        ),
        "closed_episode_count": len(closed),
        "aggregate": aggregate,
        "strategies": strategies,
        "promotion_policy": {
            "minimum_closed_episodes": 30,
            "minimum_net_profit_factor": 1.05,
            "positive_cumulative_net_return": True,
            "maximum_drawdown_limit": -0.25,
            "automatic_promotion": False,
            "financial_authority_from_this_report": False,
        },
        "private_database": str(path),
        **_authority(),
    }


def _metrics(
    closed: list[dict[str, Any]],
    *,
    strategy_id: str,
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    returns = [
        float(row["net_return"])
        for row in closed
        if _finite(row.get("net_return")) is not None
    ]
    positive = [value for value in returns if value > 0]
    negative = [value for value in returns if value < 0]
    if not returns:
        profit_factor = None
        pf_reason = "NO_CLOSED_EPISODES"
    elif not negative:
        profit_factor = None
        pf_reason = "PERFECT_NO_LOSSES"
    elif not positive:
        profit_factor = 0.0
        pf_reason = "NO_POSITIVE_EPISODES"
    else:
        profit_factor = sum(positive) / abs(sum(negative))
        pf_reason = "EVALUATED"
    cumulative = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in returns:
        cumulative *= 1.0 + value
        peak = max(peak, cumulative)
        maximum_drawdown = min(maximum_drawdown, (cumulative / peak) - 1.0)
    count = len(returns)
    sample_status = (
        "INSUFFICIENT_SAMPLE"
        if count < 10
        else "LOW_CONFIDENCE"
        if count < 30
        else "EVALUABLE"
    )
    financial_gate = bool(
        count >= 30
        and profit_factor is not None
        and profit_factor >= 1.05
        and cumulative > 1.0
        and maximum_drawdown >= -0.25
    )
    example = all_rows[0] if all_rows else {}
    return {
        "strategy_id": strategy_id,
        "formula": example.get("formula"),
        "timeframe": example.get("timeframe"),
        "profile": example.get("profile"),
        "asset_class": example.get("asset_class"),
        "closed_episode_count": count,
        "positive_episode_count": len(positive),
        "negative_episode_count": len(negative),
        "zero_episode_count": sum(value == 0 for value in returns),
        "net_profit_factor": profit_factor,
        "profit_factor_reason": pf_reason,
        "mean_net_return": sum(returns) / count if count else None,
        "cumulative_net_return": cumulative - 1.0 if count else None,
        "win_rate": len(positive) / count if count else None,
        "maximum_drawdown": maximum_drawdown if count else None,
        "sample_status": sample_status,
        "forward_financial_gate_pass": financial_gate,
        "authority_granted": False,
    }


def _observation_identity(observation: Mapping[str, Any]) -> str:
    payload = {
        "bars": sorted(
            (
                str(row.get("strategy_id")),
                str(row.get("closed_bar_timestamp")),
            )
            for row in observation.get("observations", [])
        ),
        "active": sorted(
            (
                str(row.get("strategy_id")),
                str(row.get("symbol")),
                str(row.get("closed_bar_timestamp")),
            )
            for row in observation.get("active_signals", [])
        ),
    }
    return stable_hash(payload)


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _authority() -> dict[str, Any]:
    return {
        "FINANCIAL_FINALIST_GO": False,
        "FORWARD_RESEARCH_SHADOW": "OBSERVATION_ONLY",
        "STRATEGY_AUTHORITY": "NONE",
        "EXECUTION_AUTHORITY": "NONE",
        "PAPER_STRATEGY_AUTHORITY": "NONE",
        "LIVE_STRATEGY_AUTHORITY": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
        "automatic_orders": 0,
    }


def _database_path(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "research"
        / "phase11_12"
        / "private"
        / "shadow_forward.sqlite3"
    )


def _output(project_root: Path) -> Path:
    return project_root / "output" / "research" / "phase11_12"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {**payload, "content_hash": stable_hash(payload)}
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(body, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        last_error: PermissionError | None = None
        for delay in (0.0, 0.01, 0.05, 0.1, 0.25):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temporary, path)
                return
            except PermissionError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    finally:
        temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "lower_timeframe_forward_status",
    "update_lower_timeframe_forward",
]
