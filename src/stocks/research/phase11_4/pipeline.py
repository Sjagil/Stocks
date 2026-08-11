from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stocks.execution.idempotency import stable_hash

from .acquisition import acquire_security_master, load_security_master
from .engine import (
    PitTrade,
    generate_trade_grid_for_frame,
    generate_trades,
    portfolio_simulation,
    split_summary,
    trade_summary,
)
from .store import Phase114Store


PHASE11_4_COMMANDS = (
    "preregister",
    "build-universe",
    "trade-backtest",
    "portfolio-backtest",
    "cost-sensitivity",
    "shariah-cohort",
    "robustness",
    "bootstrap",
    "concentration",
    "status",
    "freeze",
)
DECISIONS = (
    "REJECTED_NO_NET_ALPHA",
    "REJECTED_SURVIVORSHIP_DEPENDENT",
    "REJECTED_COST_SENSITIVE",
    "REJECTED_CONCENTRATION",
    "REJECTED_SHARIAH_SAMPLE",
    "PROMISING_RESEARCH_CANDIDATE",
    "PIT_VALIDATED_RESEARCH_CANDIDATE",
    "FORWARD_SHADOW_ELIGIBLE",
)
SPLITS = {
    "train": ("2000-01-01", "2011-12-31"),
    "validation": ("2012-01-01", "2018-12-31"),
    "test": ("2019-01-01", "2026-07-21"),
}
DATA_SCRAPER = Path(r"C:\Users\alhar\Documents\datascraper")


class Layout:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.output = root / "output" / "research" / "rsi_pit"
        self.database = root / "data" / "research" / "phase11_4" / "private" / "rsi_mean_reversion_pit.sqlite3"
        self.phase113_database = root / "data" / "research" / "phase11_3" / "private" / "causal_research.sqlite3"
        self.phase112_database = root / "data" / "research" / "phase11_2" / "private" / "pit_foundation.sqlite3"

    def prepare(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)

    def artifact(self, name: str) -> Path:
        return self.output / name


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _base(schema: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "generated_at": _now(),
        "strategy": "RSI_MEAN_REVERSION_CAUSAL_V1",
        "candidate_pre_registered": True,
        "post_PIT_parameter_selection": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
    }


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    document["content_hash"] = stable_hash({key: value for key, value in document.items() if key != "content_hash"})
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return document


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def preregister(root: Path) -> dict[str, Any]:
    layout = Layout(root)
    layout.prepare()
    payload = {
        **_base("phase11_4_rsi_pit_preregistration_v1"),
        "status": "PREREGISTERED_GO",
        "frozen_strategy": {
            "rsi_length": 3,
            "entry_threshold": 5,
            "entry_signal_timing": "AFTER_CLOSE_T",
            "entry_execution": "OPEN_T_PLUS_1",
            "exit_signal": "FIRST_CLOSE_GREATER_THAN_PREVIOUS_CLOSE",
            "exit_execution": "OPEN_T_PLUS_1",
            "direction": "LONG_ONLY",
            "cost_bps_per_side": 10,
            "maximum_one_position_per_symbol": True,
        },
        "periods": SPLITS,
        "dividend_policy": "EXCLUDED_SHORT_HOLDING_PERIOD_PRICE_RETURN_STUDY",
        "price_policy": "SPLIT_ADJUSTED_OHLC_USING_ADJUSTED_CLOSE_TO_CLOSE_FACTOR",
        "universe_policy": {
            "minimum_price": 5.0,
            "minimum_20_session_median_dollar_volume": 1_000_000.0,
            "listing_and_delisting_dates_required": True,
            "daily_historical_membership_required": True,
            "current_membership_backprojection": False,
        },
        "portfolio": {"initial_capital_eur": 2000, "max_positions": 4, "max_allocation": 0.25, "leverage": False},
        "concurrency_variants": [1, 2, 4, 8],
        "cost_scenarios_bps_per_side": [10, 15, 25, 50],
        "uniform_extra_slippage_bps_per_execution": 25,
        "delisting_recovery_stress": [-0.5, -0.8, -1.0],
        "robustness_grid": {"rsi_lengths": [2, 3, 5, 10, 14, 20], "entry_thresholds": [5, 10, 20, 30, 40], "trial_count": 30},
        "allowed_decisions": list(DECISIONS),
        "candidate_gates": {
            "test_trade_pf_min": 1.20,
            "cost_25_bps_pf_strictly_above": 1.0,
            "portfolio_test_cagr_strictly_above": 0.0,
            "portfolio_maximum_drawdown_max": 0.25,
            "median_yearly_pf_strictly_above": 1.0,
            "positive_test_year_ratio_min": 0.60,
            "bootstrap_probability_expectancy_positive_min": 0.95,
            "DSR_min": 0.95,
            "PBO_max": 0.20,
            "single_year_net_profit_share_max": 0.35,
            "single_symbol_net_profit_share_max": 0.15,
        },
    }
    existing = _read(layout.artifact("preregistration.json"))
    if existing:
        old = {key: value for key, value in existing.items() if key not in {"generated_at", "content_hash"}}
        new = {key: value for key, value in payload.items() if key not in {"generated_at", "content_hash"}}
        if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True):
            raise RuntimeError("PREREGISTRATION_IMMUTABILITY_VIOLATION")
        return existing
    payload = _write(layout.artifact("preregistration.json"), payload)
    Phase114Store(layout.database).append("preregistration", payload["content_hash"], [("FROZEN", payload)])
    return _read(layout.artifact("preregistration.json"))


def _latest_memberships(layout: Layout) -> dict[str, dict[str, Any]]:
    if not layout.phase112_database.is_file():
        return {}
    with sqlite3.connect(layout.phase112_database) as db:
        rows = db.execute("SELECT payload_json FROM universe_memberships ORDER BY row_id").fetchall()
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = json.loads(row[0])
        result[str(payload.get("symbol", "")).removesuffix(".US")] = payload
    return result


def _price_coverage(layout: Layout) -> list[dict[str, Any]]:
    if not layout.phase113_database.is_file():
        return []
    query = """
      SELECT json_extract(payload_json,'$.symbol') AS symbol,
             MIN(json_extract(payload_json,'$.timestamp')) AS first_date,
             MAX(json_extract(payload_json,'$.timestamp')) AS last_date,
             COUNT(*) AS row_count
      FROM records WHERE dataset='prices' GROUP BY 1 ORDER BY 1
    """
    with sqlite3.connect(layout.phase113_database) as db:
        rows = db.execute(query).fetchall()
    return [{"provider_symbol": row[0], "symbol": str(row[0]).removesuffix(".US"), "first_reliable_price_date": row[1], "last_reliable_price_date": row[2], "price_rows": row[3]} for row in rows]


def _match_security_identity(master: pd.DataFrame, coverage: dict[str, Any]) -> dict[str, Any]:
    if master.empty:
        return {}
    candidates = master.loc[master["ticker"].eq(coverage["symbol"])]
    if candidates.empty:
        return {}
    observed_first = pd.Timestamp(coverage["first_reliable_price_date"])
    observed_last = pd.Timestamp(coverage["last_reliable_price_date"])
    matches: list[tuple[int, dict[str, Any]]] = []
    for row in candidates.to_dict("records"):
        expected_first = pd.Timestamp(max("2000-01-01", str(row["first_price_date"])))
        expected_last = pd.Timestamp(row["last_price_date"])
        start_gap = abs((observed_first - expected_first).days)
        end_gap = abs((observed_last - expected_last).days)
        if start_gap <= 10 and end_gap <= 10:
            matches.append((start_gap + end_gap, row))
    matches.sort(key=lambda item: item[0])
    if not matches or (len(matches) > 1 and matches[0][0] == matches[1][0]):
        return {}
    return matches[0][1]


def _corporate_action_conflicts(root: Path) -> set[str]:
    import duckdb

    layout = Layout(root)
    parquet = root / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet"
    private = root / "data" / "research" / "phase11_4" / "private" / "corporate-action-conflicts.json"
    source_hash = _file_hash(parquet)
    cached = _read(private)
    if cached.get("source_hash") == source_hash:
        conflicts = {str(value) for value in cached.get("security_ids", [])}
    elif parquet.is_file():
        with duckdb.connect() as db:
            rows = db.execute(
                """
                WITH ordered AS (
                  SELECT security_id, date, open,
                         LAG(close) OVER(PARTITION BY security_id ORDER BY date) AS prior_close
                  FROM read_parquet(?)
                )
                SELECT DISTINCT security_id
                FROM ordered
                WHERE prior_close > 0
                  AND (open / prior_close < 0.25 OR open / prior_close > 4.0)
                ORDER BY security_id
                """,
                [str(parquet.resolve())],
            ).fetchall()
        conflicts = {str(row[0]) for row in rows}
        private.write_text(
            json.dumps(
                {
                    "schema": "phase11_4_private_corporate_action_conflicts_v1",
                    "source_hash": source_hash,
                    "security_ids": sorted(conflicts),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        conflicts = set()
    public = {
        **_base("phase11_4_corporate_action_audit_v1"),
        "status": "CONFLICTS_BLOCKED_GO" if conflicts else "NO_CONFLICTS_GO",
        "split_adjusted_overnight_ratio_bounds": [0.25, 4.0],
        "conflicted_security_count": len(conflicts),
        "conflicted_security_ids_public": False,
        "conflict_set_hash": stable_hash(sorted(conflicts)),
        "source_price_hash": source_hash,
    }
    _write(layout.artifact("corporate-action-audit.json"), public)
    return conflicts


def build_universe(root: Path) -> dict[str, Any]:
    preregister(root)
    layout = Layout(root)
    security_master_status = acquire_security_master(root)
    security_master = load_security_master(root)
    memberships = _latest_memberships(layout)
    coverage = _price_coverage(layout)
    acquisition = _read(layout.artifact("pit-data-acquisition.json"))
    compaction = _read(layout.artifact("pit-price-compaction.json"))
    rows: list[dict[str, Any]] = []
    for item in coverage:
        meta = memberships.get(item["symbol"], {})
        identity = _match_security_identity(security_master, item)
        rows.append(
            {
                **item,
                "active_on_source_snapshot": bool(meta.get("active_on_date")),
                "source_snapshot_date": meta.get("universe_date"),
                "security_id": identity.get("security_id"),
                "listed_on_date": identity.get("first_price_date") or meta.get("listed_on_date"),
                "delisted_on_date": identity.get("last_price_date") if identity.get("is_delisted") else meta.get("delisted_on_date"),
                "security_type": meta.get("security_type", "UNKNOWN"),
                "sector": identity.get("sector") or meta.get("sector") or "UNKNOWN",
                "industry": identity.get("industry") or meta.get("industry") or "UNKNOWN",
                "exchange": meta.get("exchange") or "UNKNOWN",
                "listing_start_used": max(item["first_reliable_price_date"], str(identity.get("first_price_date") or "2000-01-01")),
                "listing_start_classification": "SHARADAR_SECURITY_WINDOW" if identity else "INFERRED_FROM_FIRST_PRICE_NOT_PROVEN",
            }
        )
    snapshot_dates = sorted(
        {
            str(row["source_snapshot_date"])
            for row in rows
            if row.get("source_snapshot_date")
        }
    )
    expanded_available = compaction.get("status") == "GO" and not security_master.empty
    corporate_action_conflicts = _corporate_action_conflicts(root) if expanded_available else set()
    listing_window_complete = (
        float(security_master_status.get("listing_window_coverage_ratio") or 0) == 1.0
        if expanded_available
        else bool(rows) and all(row["listed_on_date"] for row in rows)
    )
    population_size = int(security_master_status.get("security_count") or len(rows)) if expanded_available else len(rows)
    raw_valid_population = int(compaction.get("security_count") or 0) if expanded_available else len(rows)
    valid_population = raw_valid_population - len(corporate_action_conflicts)
    full_population_acquired = expanded_available and valid_population == population_size
    historical_membership_complete = listing_window_complete and full_population_acquired
    delisted = (
        int(security_master_status.get("delisted_security_count") or 0)
        if expanded_available
        else sum(not row["active_on_source_snapshot"] for row in rows)
    )
    payload = {
        **_base("phase11_4_universe_audit_v1"),
        "status": "SURVIVORSHIP_RISK_BLOCKED" if not historical_membership_complete else "PIT_UNIVERSE_GO",
        "PIT_universe_size": population_size,
        "active_symbols": population_size - delisted,
        "delisted_symbols": delisted,
        "valid_price_security_count": valid_population,
        "raw_price_security_count": raw_valid_population,
        "corporate_action_conflict_security_count": len(corporate_action_conflicts),
        "population_price_coverage_ratio": valid_population / population_size if population_size else 0.0,
        "legacy_selected_ticker_count": len(rows),
        "historical_snapshot_count": len(snapshot_dates),
        "historical_snapshot_dates": snapshot_dates,
        "symbols_with_listed_on_date": sum(bool(row["listed_on_date"]) for row in rows),
        "symbols_with_delisted_on_date": sum(bool(row["delisted_on_date"]) for row in rows),
        "listing_window_coverage_ratio": float(security_master_status.get("listing_window_coverage_ratio") or 0) if expanded_available else (sum(bool(row["listed_on_date"]) for row in rows) / len(rows) if rows else 0.0),
        "sector_coverage_ratio": float(security_master_status.get("sector_coverage_ratio") or 0) if expanded_available else (sum(row["sector"] != "UNKNOWN" for row in rows) / len(rows) if rows else 0.0),
        "security_master": security_master_status,
        "price_acquisition": acquisition,
        "price_compaction": compaction,
        "full_PIT_population_acquired": full_population_acquired,
        "survivorship_bias_blocked": historical_membership_complete,
        "current_membership_backprojection_used": False,
        "signal_eligibility_valid": historical_membership_complete,
        "delisted_endpoint_included": delisted > 0,
        "source_assessment": "PIT_DATA_PARTIAL",
        "blocking_reasons": [] if historical_membership_complete else [
            "FULL_PIT_POPULATION_PRICE_ACQUISITION_INCOMPLETE",
            *(["CORPORATE_ACTION_CONFLICT_SECURITIES_BLOCKED"] if corporate_action_conflicts else []),
            *([] if listing_window_complete else ["LISTING_DATES_INCOMPLETE"]),
        ],
        "private_universe_hash": str(security_master_status.get("provider_identity_map_hash") or stable_hash(rows)),
        "provenance": {
            "phase11_2_universe_audit": _file_hash(root / "output" / "ibkr" / "phase11_2" / "universe-audit.json"),
            "phase11_3_universe_history": _file_hash(root / "output" / "ibkr" / "phase11_3" / "universe-history.json"),
            "datascraper_delisted_reference": _file_hash(DATA_SCRAPER / "data" / "historical" / "eodhd" / "eodhd_delisted_reference_v7.parquet"),
        },
    }
    payload = _write(layout.artifact("universe-audit.json"), payload)
    run_id = payload["content_hash"]
    store = Phase114Store(layout.database)
    universe_records = security_master.to_dict("records") if expanded_available else rows
    store.append(
        "universe_snapshots",
        run_id,
        [
            (
                str(row.get("security_id") or f"{row.get('symbol')}:{row.get('source_snapshot_date')}"),
                row,
            )
            for row in universe_records
        ],
    )
    store.append("provenance", run_id, [("UNIVERSE", payload["provenance"])])
    return payload


def _load_frames(layout: Layout) -> dict[str, pd.DataFrame]:
    query = """
      SELECT json_extract(payload_json,'$.symbol') AS symbol,
             json_extract(payload_json,'$.timestamp') AS timestamp,
             CAST(json_extract(payload_json,'$.open') AS REAL) AS open,
             CAST(json_extract(payload_json,'$.high') AS REAL) AS high,
             CAST(json_extract(payload_json,'$.low') AS REAL) AS low,
             CAST(json_extract(payload_json,'$.close') AS REAL) AS close,
             CAST(json_extract(payload_json,'$.adjusted_close') AS REAL) AS adjusted_close,
             CAST(json_extract(payload_json,'$.volume') AS REAL) AS volume
      FROM records WHERE dataset='prices' ORDER BY symbol,timestamp
    """
    with sqlite3.connect(layout.phase113_database) as db:
        data = pd.read_sql_query(query, db)
    data["symbol"] = data["symbol"].str.replace(r"\.US$", "", regex=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    factor = data["adjusted_close"] / data["close"].replace(0, np.nan)
    for column in ("open", "high", "low"):
        data[column] = data[column] * factor
    data["close"] = data["adjusted_close"]
    valid = (
        data[["open", "high", "low", "close"]].notna().all(axis=1)
        & (data[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (data["high"] >= data[["open", "close", "low"]].max(axis=1))
        & (data["low"] <= data[["open", "close", "high"]].min(axis=1))
        & data["volume"].fillna(0).ge(0)
    )
    data = data.loc[valid].drop_duplicates(["symbol", "timestamp"], keep="last")
    return {
        symbol: group.set_index("timestamp")[["open", "high", "low", "close", "volume"]].sort_index()
        for symbol, group in data.groupby("symbol", sort=True)
        if len(group) >= 24
    }


def _sectors(layout: Layout) -> dict[str, str]:
    return {symbol: str(row.get("sector") or "UNKNOWN") for symbol, row in _latest_memberships(layout).items()}


def _candidate_trades(root: Path, *, cost_bps: float = 10.0) -> tuple[dict[str, pd.DataFrame], list[PitTrade], list[dict[str, Any]]]:
    layout = Layout(root)
    frames = _load_frames(layout)
    trades, signals = generate_trades(frames, cost_bps_per_side=cost_bps, sectors=_sectors(layout))
    return frames, trades, signals


def _expanded_candidate_trades(
    root: Path,
    *,
    period: int = 3,
    threshold: float = 5.0,
    cost_bps: float = 10.0,
) -> tuple[list[PitTrade], list[dict[str, Any]], int]:
    trades: list[PitTrade] = []
    signals: list[dict[str, Any]] = []
    count = 0
    for security_id, sector, frame in _iter_expanded_frames(root):
        generated, generated_signals = generate_trades(
            {security_id: frame},
            period=period,
            threshold=threshold,
            cost_bps_per_side=cost_bps,
            sectors={security_id: sector},
        )
        trades.extend(generated)
        signals.extend(generated_signals)
        count += 1
    return trades, signals, count


def _iter_expanded_frames(root: Path):
    path = root / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet"
    if not path.is_file():
        return
    conflicts = _corporate_action_conflicts(root)
    parquet = pq.ParquetFile(path)
    carry = pd.DataFrame()
    columns = ["security_id", "date", "open", "high", "low", "close", "volume", "sector"]
    for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
        frame = batch.to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        last_security = str(frame["security_id"].iloc[-1])
        carry = frame.loc[frame["security_id"].eq(last_security)].copy()
        complete = frame.loc[~frame["security_id"].eq(last_security)]
        for security_id, group in complete.groupby("security_id", sort=False):
            if str(security_id) in conflicts:
                continue
            sector = str(group["sector"].iloc[-1] or "UNKNOWN")
            values = group.set_index(pd.to_datetime(group["date"]))[["open", "high", "low", "close", "volume"]]
            yield str(security_id), sector, values
    if not carry.empty:
        security_id = str(carry["security_id"].iloc[0])
        if security_id in conflicts:
            return
        sector = str(carry["sector"].iloc[-1] or "UNKNOWN")
        values = carry.set_index(pd.to_datetime(carry["date"]))[["open", "high", "low", "close", "volume"]]
        yield security_id, sector, values


def trade_backtest(root: Path) -> dict[str, Any]:
    universe = build_universe(root)
    layout = Layout(root)
    expanded_path = root / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet"
    if expanded_path.is_file():
        trades, signals, universe_count = _expanded_candidate_trades(root)
        frames: dict[str, pd.DataFrame] = {}
        data_source = "SHARADAR_WINDOWS_EODHD_SPLIT_ADJUSTED"
    else:
        frames, trades, signals = _candidate_trades(root)
        universe_count = len(frames)
        data_source = "PHASE11_3_LEGACY_SELECTED_TICKERS"
    yearly = _grouped_trade_summaries(trades, lambda trade: trade.entry_date[:4])
    monthly = _grouped_trade_summaries(trades, lambda trade: trade.entry_date[:7])
    delisting_stress = {str(int(abs(stress) * 100)): _delisting_stress(trades, stress) for stress in (-0.5, -0.8, -1.0)}
    payload = {
        **_base("phase11_4_trade_results_v1"),
        "status": "PROVISIONAL_BACKTEST_COMPLETE",
        "valid_for_candidate_gate": bool(universe["survivorship_bias_blocked"]),
        "price_contract": {
            "split_consistent_ohlc": True,
            "corporate_action_conflict_security_count": len(_corporate_action_conflicts(root)) if expanded_path.is_file() else 0,
            "corporate_action_conflicts_produced_signals": 0,
            "dividends": "EXCLUDED_SHORT_HOLDING_PERIOD_PRICE_RETURN_STUDY",
            "future_filled_bars": 0,
            "forward_filled_execution_prices": 0,
            "same_close_executions": 0,
        },
        "universe_symbol_count": universe_count,
        "data_source": data_source,
        "signal_count": len(signals),
        "overall": trade_summary(trades),
        "periods": {name: split_summary(trades, *bounds) for name, bounds in SPLITS.items()},
        "yearly": yearly,
        "monthly_return_distribution": _distribution([row.get("expectancy") for row in monthly.values()]),
        "rolling_3_year": _rolling_years(trades),
        "delisting_stress": delisting_stress,
        "execution_status_counts": dict(Counter(trade.status for trade in trades)),
    }
    payload = _write(layout.artifact("trade-results.json"), payload)
    run_id = payload["content_hash"]
    store = Phase114Store(layout.database)
    store.append("signals", run_id, [(f"{row['symbol']}:{row['signal_date']}", row) for row in signals])
    store.append("trades", run_id, [(f"{row.symbol}:{row.entry_date}", row.as_dict()) for row in trades])
    return payload


def _select_capacity_trades(trades: list[PitTrade], max_positions: int) -> tuple[list[PitTrade], int]:
    """Select entries causally; 25% sizing limits an unlevered book to four slots."""
    candidates = [trade for trade in trades if trade.exit_date and trade.net_return is not None]
    by_entry: dict[str, list[PitTrade]] = defaultdict(list)
    by_exit: dict[str, list[PitTrade]] = defaultdict(list)
    for trade in candidates:
        by_entry[trade.entry_date].append(trade)
    effective_slots = min(max_positions, 4)
    active: dict[str, PitTrade] = {}
    accepted: list[PitTrade] = []
    missed = 0
    dates = sorted({trade.entry_date for trade in candidates} | {str(trade.exit_date) for trade in candidates})
    for day in dates:
        for trade in by_exit.pop(day, []):
            active.pop(trade.symbol, None)
        ranked = sorted(by_entry.get(day, []), key=lambda item: (item.rsi, -item.historical_dollar_volume, item.symbol))
        for trade in ranked:
            if trade.symbol in active or len(active) >= effective_slots:
                missed += 1
                continue
            active[trade.symbol] = trade
            accepted.append(trade)
            by_exit[str(trade.exit_date)].append(trade)
    return accepted, missed


def _expanded_holding_marks(
    root: Path, accepted: list[PitTrade], date_bounds: tuple[str, str] | None
) -> tuple[list[str], dict[tuple[str, str], float]]:
    import duckdb

    if not accepted:
        return [], {}
    parquet = root / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet"
    start = date_bounds[0] if date_bounds else min(trade.entry_date for trade in accepted)
    end = date_bounds[1] if date_bounds else max(str(trade.exit_date) for trade in accepted)
    ranges = pd.DataFrame(
        {
            "security_id": [trade.symbol for trade in accepted],
            "entry_date": pd.to_datetime([trade.entry_date for trade in accepted]),
            "exit_date": pd.to_datetime([trade.exit_date for trade in accepted]),
        }
    )
    with duckdb.connect() as db:
        db.register("holding_ranges", ranges)
        marks = db.execute(
            """
            SELECT b.security_id, CAST(b.date AS VARCHAR) AS date, b.close
            FROM read_parquet(?) b
            JOIN holding_ranges h ON b.security_id=h.security_id
              AND CAST(b.date AS DATE) BETWEEN CAST(h.entry_date AS DATE) AND CAST(h.exit_date AS DATE)
            """,
            [str(parquet.resolve())],
        ).fetchdf()
        dates = [
            str(row[0])
            for row in db.execute(
                "SELECT DISTINCT CAST(date AS VARCHAR) FROM read_parquet(?) "
                "WHERE CAST(date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) ORDER BY 1",
                [str(parquet.resolve()), start, end],
            ).fetchall()
        ]
    price_marks = {
        (str(row.security_id), str(row.date)): float(row.close)
        for row in marks.itertuples(index=False)
        if pd.notna(row.close)
    }
    return dates, price_marks


def _eur_per_usd_by_date(root: Path, dates: list[str]) -> tuple[dict[str, float], int]:
    layout = Layout(root)
    path = root / "data" / "research" / "phase11_4" / "private" / "eurusd.parquet"
    if not path.is_file() or not dates:
        return {}, len(dates)
    frame = pd.read_parquet(path, columns=["date", "usd_per_eur"])
    frame.index = pd.to_datetime(frame["date"])
    requested = pd.DatetimeIndex(pd.to_datetime(dates))
    aligned = frame["usd_per_eur"].reindex(frame.index.union(requested)).sort_index().ffill().reindex(requested)
    rates = {
        day: 1.0 / float(value)
        for day, value in zip(dates, aligned.to_numpy(), strict=True)
        if pd.notna(value) and float(value) > 0
    }
    missing = len(dates) - len(rates)
    payload = {
        **_base("phase11_4_currency_normalization_v1"),
        "status": "GO" if missing == 0 else "FX_HISTORY_INCOMPLETE",
        "source": "ECB_EXR_D_USD_EUR_SP00_A",
        "source_private_path": str(path),
        "source_hash": _file_hash(path),
        "interpretation": "USD_PER_EUR_INVERTED_TO_EUR_PER_USD",
        "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "requested_date_count": len(dates),
        "missing_date_count_after_backward_asof": missing,
    }
    _write(layout.artifact("currency-normalization.json"), payload)
    return rates, missing


def _expanded_portfolio_simulation(
    root: Path,
    trades: list[PitTrade],
    *,
    max_positions: int,
    date_bounds: tuple[str, str] | None = None,
    initial_cash: float = 2_000.0,
    allocation: float = 0.25,
    cost_bps_per_side: float = 10.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, missed = _select_capacity_trades(trades, max_positions)
    dates, marks = _expanded_holding_marks(root, accepted, date_bounds)
    eur_per_usd, fx_missing_days = _eur_per_usd_by_date(root, dates)
    accepted_ids = {id(trade) for trade in accepted}
    by_entry: dict[str, list[PitTrade]] = defaultdict(list)
    by_exit: dict[str, list[PitTrade]] = defaultdict(list)
    for trade in accepted:
        by_entry[trade.entry_date].append(trade)
        by_exit[str(trade.exit_date)].append(trade)
    cash = initial_cash
    positions: dict[str, dict[str, Any]] = {}
    last_prices: dict[str, float] = {}
    equity_rows: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    turnover_notional = 0.0
    costs_paid = 0.0
    missing_marks = 0
    cost = cost_bps_per_side / 10_000.0
    for day in dates:
        fx = eur_per_usd.get(day)
        if fx is None:
            continue
        for trade in by_exit.get(day, []):
            position = positions.pop(trade.symbol, None)
            exit_price = trade.exit_price
            if position is None or exit_price is None:
                continue
            proceeds = position["shares"] * exit_price * fx
            fee = proceeds * cost
            cash += proceeds - fee
            turnover_notional += proceeds
            costs_paid += fee
            fills.append({"date": day, "symbol": trade.symbol, "side": "SELL", "notional": proceeds, "cost": fee})
        marked_before = sum(
            position["shares"] * last_prices.get(symbol, position["trade"].entry_price)
            for symbol, position in positions.items()
        )
        equity_before = cash + marked_before
        for trade in sorted(by_entry.get(day, []), key=lambda item: (item.rsi, -item.historical_dollar_volume, item.symbol)):
            if id(trade) not in accepted_ids:
                continue
            target = min(equity_before * allocation, cash / (1 + cost))
            if target <= 0:
                missed += 1
                continue
            fee = target * cost
            entry_price_eur = trade.entry_price * fx
            shares = target / entry_price_eur
            cash -= target + fee
            turnover_notional += target
            costs_paid += fee
            positions[trade.symbol] = {"trade": trade, "shares": shares}
            last_prices[trade.symbol] = entry_price_eur
            fills.append({"date": day, "symbol": trade.symbol, "side": "BUY", "notional": target, "cost": fee})
        for symbol in positions:
            value = marks.get((symbol, day))
            if value is not None:
                last_prices[symbol] = value * fx
            elif day != positions[symbol]["trade"].entry_date:
                missing_marks += 1
        marked = sum(position["shares"] * last_prices[symbol] for symbol, position in positions.items())
        nav = cash + marked
        equity_rows.append(
            {"date": day, "nav": nav, "cash": cash, "open_positions": len(positions), "exposure": marked / nav if nav > 0 else 0.0}
        )
    if not equity_rows:
        return {"status": "INSUFFICIENT_SAMPLE", "maximum_positions": max_positions}, [], []
    equity = pd.DataFrame(equity_rows).set_index(pd.to_datetime([row["date"] for row in equity_rows]))
    returns = equity["nav"].pct_change().fillna(0.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    peak = equity["nav"].cummax()
    downside = returns[returns < 0].std(ddof=0)
    positive = float(returns[returns > 0].sum())
    negative = abs(float(returns[returns < 0].sum()))
    drawdown = float((equity["nav"] / peak - 1).min())
    cagr = float((equity["nav"].iloc[-1] / initial_cash) ** (1 / years) - 1)
    summary = {
        "status": "GO" if missing_marks == 0 and fx_missing_days == 0 else "GO_WITH_EXPLICIT_DATA_CARRY",
        "data_source": "SHARADAR_WINDOWS_EODHD_SPLIT_ADJUSTED",
        "base_currency": "EUR",
        "currency_normalization": "ECB_USD_TO_EUR_DAILY_BACKWARD_ASOF",
        "fx_missing_days": fx_missing_days,
        "initial_capital_eur": initial_cash,
        "terminal_nav_eur": float(equity["nav"].iloc[-1]),
        "CAGR": cagr,
        "Sharpe": float(returns.mean() / returns.std(ddof=0) * math.sqrt(252)) if returns.std(ddof=0) > 0 else None,
        "Sortino": float(returns.mean() / downside * math.sqrt(252)) if downside and downside > 0 else None,
        "maximum_drawdown": drawdown,
        "Calmar": cagr / abs(drawdown) if drawdown < 0 else None,
        "period_profit_factor": None if negative == 0 else positive / negative,
        "turnover": turnover_notional / initial_cash,
        "transaction_costs": costs_paid,
        "average_exposure": float(equity["exposure"].mean()),
        "maximum_exposure": float(equity["exposure"].max()),
        "cash_drag": float((equity["cash"] / equity["nav"]).mean()),
        "accepted_trades": len(accepted),
        "missed_signals_due_to_capacity": missed,
        "maximum_positions": max_positions,
        "missing_holding_marks_carried": missing_marks,
        "maximum_concurrent_sector_concentration": None,
    }
    return summary, equity_rows, fills


def _benchmark_metrics(rows: pd.DataFrame, return_column: str) -> dict[str, Any]:
    if return_column not in rows.columns:
        return {"status": "INSUFFICIENT_SAMPLE"}
    clean = rows.loc[rows[return_column].notna()].copy()
    if clean.empty:
        return {"status": "INSUFFICIENT_SAMPLE"}
    returns = clean[return_column].astype(float)
    nav = (1 + returns).cumprod()
    years = max((pd.Timestamp(clean["date"].iloc[-1]) - pd.Timestamp(clean["date"].iloc[0])).days / 365.25, 1 / 365.25)
    peak = nav.cummax()
    downside = returns.loc[returns < 0].std(ddof=0)
    drawdown = float((nav / peak - 1).min())
    cagr = float(nav.iloc[-1] ** (1 / years) - 1)
    return {
        "status": "GO",
        "observation_count": len(clean),
        "CAGR": cagr,
        "Sharpe": float(returns.mean() / returns.std(ddof=0) * math.sqrt(252)) if returns.std(ddof=0) > 0 else None,
        "Sortino": float(returns.mean() / downside * math.sqrt(252)) if downside and downside > 0 else None,
        "maximum_drawdown": drawdown,
        "Calmar": cagr / abs(drawdown) if drawdown < 0 else None,
        "terminal_growth_of_one_eur": float(nav.iloc[-1]),
    }


def _expanded_benchmarks(root: Path) -> dict[str, Any]:
    import duckdb

    layout = Layout(root)
    parquet = root / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet"
    conflicts = _corporate_action_conflicts(root)
    cache = _read(layout.artifact("benchmarks.json"))
    source_hash = _file_hash(parquet)
    if cache.get("source_price_hash") == source_hash and cache.get("conflict_set_hash") == stable_hash(sorted(conflicts)):
        return cache
    blocked = pd.DataFrame({"security_id": sorted(conflicts)})
    with duckdb.connect() as db:
        db.register("blocked", blocked)
        daily = db.execute(
            """
            WITH clean AS (
              SELECT b.security_id, CAST(b.date AS DATE) AS date, b.close, b.volume,
                     COALESCE(NULLIF(b.sector,''),'UNKNOWN') AS sector
              FROM read_parquet(?) b ANTI JOIN blocked x USING(security_id)
            ), lagged AS (
              SELECT *, LAG(close) OVER w AS prior_close,
                     MEDIAN(close*volume) OVER(
                       PARTITION BY security_id ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                     ) AS historical_dollar_volume,
                     COUNT(*) OVER(
                       PARTITION BY security_id ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                     ) AS history_count
              FROM clean WINDOW w AS (PARTITION BY security_id ORDER BY date)
            ), eligible AS (
              SELECT date, sector, close/prior_close-1 AS local_return
              FROM lagged
              WHERE prior_close>=5 AND historical_dollar_volume>=1000000 AND history_count>=20
            ), sector_returns AS (
              SELECT date,sector,AVG(local_return) AS sector_return FROM eligible GROUP BY date,sector
            )
            SELECT e.date,AVG(e.local_return) AS equal_weight_local_return,
                   s.sector_neutral_local_return
            FROM eligible e
            JOIN (SELECT date,AVG(sector_return) AS sector_neutral_local_return FROM sector_returns GROUP BY date) s USING(date)
            GROUP BY e.date,s.sector_neutral_local_return ORDER BY e.date
            """,
            [str(parquet.resolve())],
        ).fetchdf()
        master = pd.read_parquet(
            root / "data" / "research" / "phase11_4" / "private" / "security-master.parquet",
            columns=["security_id", "ticker", "is_delisted"],
        )
        spy = master.loc[master["ticker"].eq("SPY") & ~master["is_delisted"].astype(bool), "security_id"]
        market = pd.DataFrame()
        if len(spy) == 1 and str(spy.iloc[0]) not in conflicts:
            market = db.execute(
                """SELECT CAST(date AS DATE) AS date,close/LAG(close) OVER(ORDER BY date)-1 AS local_return
                   FROM read_parquet(?) WHERE security_id=? ORDER BY date""",
                [str(parquet.resolve()), str(spy.iloc[0])],
            ).fetchdf()
    if market.empty:
        from stocks.research.critical_trading import load_yfinance_cache

        yahoo_spy = load_yfinance_cache(root).get("SPY")
        if yahoo_spy is not None and not yahoo_spy.empty:
            market = pd.DataFrame(
                {
                    "date": pd.to_datetime(yahoo_spy.index),
                    "local_return": yahoo_spy["close"].astype(float).pct_change().to_numpy(),
                }
            )
    all_dates = sorted({str(value.date()) for value in daily["date"]} | ({str(value.date()) for value in market["date"]} if not market.empty else set()))
    eur_per_usd, fx_missing = _eur_per_usd_by_date(root, all_dates)
    fx = pd.Series(eur_per_usd, name="eur_per_usd")
    fx.index = pd.to_datetime(fx.index)
    fx_return = fx.pct_change()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.join(fx_return.rename("fx_return"), on="date")
    for column in ("equal_weight", "sector_neutral"):
        daily[f"{column}_eur_return"] = (1 + daily[f"{column}_local_return"]) * (1 + daily["fx_return"]) - 1
    if not market.empty:
        market["date"] = pd.to_datetime(market["date"])
        market = market.join(fx_return.rename("fx_return"), on="date")
        market["eur_return"] = (1 + market["local_return"]) * (1 + market["fx_return"]) - 1
    payload = {
        **_base("phase11_4_pit_benchmarks_v1"),
        "status": "GO" if fx_missing == 0 else "FX_HISTORY_INCOMPLETE",
        "source_price_hash": source_hash,
        "conflict_set_hash": stable_hash(sorted(conflicts)),
        "currency": "EUR",
        "cash": {"status": "GO", "CAGR": 0.0},
        "equal_weight_PIT": _benchmark_metrics(daily, "equal_weight_eur_return"),
        "market_buy_and_hold": {"symbol": "SPY", **_benchmark_metrics(market, "eur_return")},
        "sector_adjusted": {
            "classification_limitation": "CURRENT_STATIC_SECTOR_LABELS",
            **_benchmark_metrics(daily, "sector_neutral_eur_return"),
        },
        "Shariah_equity": {"status": "BLOCKED_SHARIAH_HISTORY_INCOMPLETE"},
    }
    return _write(layout.artifact("benchmarks.json"), payload)


def portfolio_backtest(root: Path) -> dict[str, Any]:
    layout = Layout(root)
    if not layout.artifact("trade-results.json").is_file():
        trade_backtest(root)
    trade_artifact = _read(layout.artifact("trade-results.json"))
    expanded = trade_artifact.get("data_source") == "SHARADAR_WINDOWS_EODHD_SPLIT_ADJUSTED"
    if expanded:
        frames: dict[str, pd.DataFrame] = {}
        trades = _stored_trades(layout)
    else:
        frames, trades, _ = _candidate_trades(root)
    variants: dict[str, Any] = {}
    base_equity: list[dict[str, Any]] = []
    base_fills: list[dict[str, Any]] = []
    expanded_cache: dict[int, tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for maximum in (1, 2, 4, 8):
        effective_slots = min(maximum, 4)
        if expanded:
            if effective_slots not in expanded_cache:
                expanded_cache[effective_slots] = _expanded_portfolio_simulation(root, trades, max_positions=maximum)
            result, equity, fills = expanded_cache[effective_slots]
            result = {**result, "maximum_positions": maximum, "effective_cash_limited_positions": effective_slots}
        else:
            result, equity, fills = portfolio_simulation(trades, frames, max_positions=maximum)
        variants[str(maximum)] = result
        if maximum == 4:
            base_equity, base_fills = equity, fills
    test_trades = [trade for trade in trades if SPLITS["test"][0] <= trade.entry_date <= SPLITS["test"][1]]
    if expanded:
        test_result, _, _ = _expanded_portfolio_simulation(
            root, test_trades, max_positions=4, date_bounds=SPLITS["test"]
        )
    else:
        test_start = pd.Timestamp("2019-01-01")
        test_end = pd.Timestamp("2026-07-21")
        test_frames = {
            symbol: frame.loc[
                (frame.index >= test_start) & (frame.index <= test_end)
            ]
            for symbol, frame in frames.items()
        }
        test_result, _, _ = portfolio_simulation(test_trades, test_frames, max_positions=4)
    yearly_returns = _equity_yearly_returns(base_equity)
    if expanded:
        benchmark_artifact = _expanded_benchmarks(root)
        benchmarks = {
            key: benchmark_artifact[key]
            for key in ("cash", "equal_weight_PIT", "market_buy_and_hold", "sector_adjusted", "Shariah_equity")
        }
    else:
        benchmarks = {
        "cash": {"CAGR": 0.0, "status": "GO"},
        "equal_weight_PIT": {"status": "BLOCKED_HISTORICAL_MEMBERSHIP_MISSING"},
        "market_buy_and_hold": _market_benchmark(frames),
        "sector_adjusted": {"status": "BLOCKED_SECTOR_HISTORY_MISSING"},
        "Shariah_equity": {"status": "BLOCKED_SHARIAH_HISTORY_INCOMPLETE"},
        }
    payload = {
        **_base("phase11_4_portfolio_results_v1"),
        "status": "PROVISIONAL_PORTFOLIO_BACKTEST_COMPLETE",
        "valid_for_candidate_gate": bool(_read(layout.artifact("universe-audit.json")).get("survivorship_bias_blocked")),
        "base_max_positions_4": variants["4"],
        "test_max_positions_4": test_result,
        "concurrency_variants": variants,
        "yearly_returns": yearly_returns,
        "positive_test_year_ratio": _positive_year_ratio(yearly_returns, 2019),
        "benchmarks": benchmarks,
    }
    payload = _write(layout.artifact("portfolio-results.json"), payload)
    store = Phase114Store(layout.database)
    run_id = payload["content_hash"]
    store.append("daily_equity", run_id, [(row["date"], row) for row in base_equity])
    store.append("portfolio_fills", run_id, [(f"{i}:{row['date']}:{row['symbol']}:{row['side']}", row) for i, row in enumerate(base_fills)])
    store.append("portfolio_orders", run_id, [(f"FILL_DERIVED:{i}", {**row, "order_status": "SIMULATED_FILLED"}) for i, row in enumerate(base_fills)])
    return payload


def cost_sensitivity(root: Path) -> dict[str, Any]:
    layout = Layout(root)
    trade_artifact = _read(layout.artifact("trade-results.json"))
    if trade_artifact.get("data_source") == "SHARADAR_WINDOWS_EODHD_SPLIT_ADJUSTED":
        base_trades = _stored_trades(layout)
    else:
        _, base_trades, _ = _candidate_trades(root, cost_bps=10.0)
    results: dict[str, Any] = {}
    for bps in (10, 15, 25, 50):
        results[str(bps)] = _repriced_summary(base_trades, float(bps), *SPLITS["test"])
    results["10_plus_uniform_25"] = _repriced_summary(base_trades, 35.0, *SPLITS["test"])
    liquidity = _liquidity_slippage(base_trades)
    payload = {
        **_base("phase11_4_cost_sensitivity_v1"),
        "status": "GO",
        "per_side_bps": results,
        "liquidity_class_slippage": liquidity,
        "cost_sensitive": (results["25"].get("trade_profit_factor") or 0) <= 1.0,
    }
    payload = _write(layout.artifact("cost-sensitivity.json"), payload)
    Phase114Store(layout.database).append("cost_scenarios", payload["content_hash"], [(name, row) for name, row in results.items()])
    return payload


def shariah_cohort(root: Path) -> dict[str, Any]:
    layout = Layout(root)
    trade_backtest(root)
    screens: list[dict[str, Any]] = []
    with sqlite3.connect(layout.phase113_database) as db:
        for row in db.execute("SELECT payload_json FROM research_events WHERE event_type='SHARIAH_SCREEN'"):
            screens.append(json.loads(row[0]))
    eligible = [row for row in screens if row.get("final_status") == "SHARIAH_ELIGIBLE_PIT"]
    complete = [row for row in eligible if all(row.get("available_components", {}).get(key) for key in ("business_activity", "debt_ratio", "cash_interest_ratio", "receivables_ratio", "non_permissible_income_ratio"))]
    trades = _stored_trades(layout)
    retained = [trade for trade in trades if _eligible_on(complete, trade.symbol, trade.signal_date)] if complete else []
    payload = {
        **_base("phase11_4_shariah_cohort_v1"),
        "status": "SHARIAH_SAMPLE_INSUFFICIENT" if len(retained) < 100 else "SHARIAH_COHORT_GO",
        "cohort_A": {"purpose": "PIT_PRICE_EFFECT_ONLY", "trade_count": trade_summary(trades)["trade_count"], "strategy_authority": "NONE"},
        "cohort_B": {
            "complete_screen_count": len(complete),
            "eligible_symbol_days": 0,
            "excluded_symbol_days": len(screens),
            "trades_retained": len(retained),
            "trades_excluded": len(trades) - len(retained),
            "results": trade_summary(retained),
        },
        "screen_count": len(screens),
        "eligible_complete_screen_count": len(complete),
        "Shariah_coverage_ratio": len(complete) / len(screens) if screens else 0.0,
        "current_status_backprojection_used": False,
    }
    payload = _write(layout.artifact("shariah-cohort.json"), payload)
    Phase114Store(layout.database).append("Shariah_cohorts", payload["content_hash"], [("COHORTS", payload)])
    return payload


def _new_streaming_summary() -> dict[str, float | int]:
    return {
        "trade_count": 0,
        "uncertain_delisting_exits": 0,
        "wins": 0,
        "net_sum": 0.0,
        "net_gains": 0.0,
        "net_losses": 0.0,
        "gross_sum": 0.0,
        "gross_gains": 0.0,
        "gross_losses": 0.0,
        "holding_sum": 0,
        "maximum_holding_period": 0,
    }


def _update_streaming_summary(summary: dict[str, float | int], trade: PitTrade) -> None:
    if trade.net_return is None or trade.gross_return is None:
        summary["uncertain_delisting_exits"] += int(trade.status == "DELISTING_EXECUTION_UNCERTAIN")
        return
    net = float(trade.net_return)
    gross = float(trade.gross_return)
    holding = int(trade.holding_sessions or 0)
    summary["trade_count"] += 1
    summary["wins"] += int(net > 0)
    summary["net_sum"] += net
    summary["net_gains"] += max(net, 0.0)
    summary["net_losses"] += abs(min(net, 0.0))
    summary["gross_sum"] += gross
    summary["gross_gains"] += max(gross, 0.0)
    summary["gross_losses"] += abs(min(gross, 0.0))
    summary["holding_sum"] += holding
    summary["maximum_holding_period"] = max(int(summary["maximum_holding_period"]), holding)


def _finish_streaming_summary(summary: dict[str, float | int]) -> dict[str, Any]:
    count = int(summary["trade_count"])
    losses = float(summary["net_losses"])
    gross_losses = float(summary["gross_losses"])
    return {
        "trade_count": count,
        "uncertain_delisting_exits": int(summary["uncertain_delisting_exits"]),
        "win_rate": float(summary["wins"]) / count if count else None,
        "trade_profit_factor": float(summary["net_gains"]) / losses if losses else None,
        "raw_profit_factor": float(summary["gross_gains"]) / gross_losses if gross_losses else None,
        "expectancy": float(summary["net_sum"]) / count if count else None,
        "average_trade": float(summary["net_sum"]) / count if count else None,
        "median_trade": None,
        "median_trade_status": "NOT_RETAINED_STREAMING_GRID",
        "average_holding_period": float(summary["holding_sum"]) / count if count else None,
        "maximum_holding_period": int(summary["maximum_holding_period"]) if count else None,
    }


def robustness(root: Path) -> dict[str, Any]:
    layout = Layout(root)
    expanded = (root / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet").is_file()
    periods = (2, 3, 5, 10, 14, 20)
    thresholds = (5.0, 10.0, 20.0, 30.0, 40.0)
    keys = [(period, threshold) for period in periods for threshold in thresholds]
    by_configuration: dict[tuple[int, float], list[PitTrade]] = {key: [] for key in keys}
    streaming = {key: _new_streaming_summary() for key in keys}
    daily_by_configuration: dict[tuple[int, float], dict[str, float]] = {
        key: defaultdict(float) for key in keys
    }
    if expanded:
        for security_id, sector, frame in _iter_expanded_frames(root):
            generated = generate_trade_grid_for_frame(
                security_id,
                sector,
                frame,
                periods=periods,
                thresholds=thresholds,
            )
            for key, trades in generated.items():
                for trade in trades:
                    if SPLITS["test"][0] <= trade.entry_date <= SPLITS["test"][1]:
                        _update_streaming_summary(streaming[key], trade)
                        if trade.net_return is not None and trade.exit_date:
                            daily_by_configuration[key][trade.exit_date] += float(trade.net_return)
    else:
        frames = _load_frames(layout)
        sectors = _sectors(layout)
        for period in periods:
            for threshold in thresholds:
                trades, _ = generate_trades(frames, period=period, threshold=threshold, sectors=sectors)
                by_configuration[(period, threshold)] = trades
    grid: list[dict[str, Any]] = []
    test_daily: list[pd.Series] = []
    for period in periods:
        for threshold in thresholds:
            key = (period, threshold)
            if expanded:
                summary = _finish_streaming_summary(streaming[key])
                daily = pd.Series(daily_by_configuration[key], dtype=float).sort_index()
            else:
                trades = by_configuration[key]
                summary = split_summary(trades, *SPLITS["test"])
                daily = _trade_daily_series(trades, *SPLITS["test"])
            row = {"period": period, "threshold": threshold, **summary}
            grid.append(row)
            test_daily.append(daily)
    positive = sum((row.get("trade_profit_factor") or 0) > 1 for row in grid)
    candidate = next(row for row in grid if row["period"] == 3 and row["threshold"] == 5)
    candidate_index = keys.index((3, 5.0))
    dsr = _deflated_sharpe(test_daily[candidate_index], 30)
    pbo = _pbo(test_daily)
    payload = {
        **_base("phase11_4_robustness_v1"),
        "status": "GO",
        "trial_count": 30,
        "grid": grid,
        "candidate": candidate,
        "positive_test_configuration_count": positive,
        "configuration_plateau_width": positive,
        "parameter_plateau": "PRESENT" if positive >= 21 else "NARROW_OR_ABSENT",
        "Deflated_Sharpe_Ratio": dsr,
        "Probability_of_Backtest_Overfitting": pbo,
        "White_reality_check_like": _white_like(test_daily),
        "candidate_pre_registered": True,
        "post_PIT_parameter_selection": False,
    }
    payload = _write(layout.artifact("robustness.json"), payload)
    Phase114Store(layout.database).append("robustness_runs", payload["content_hash"], [(f"{row['period']}:{row['threshold']}", row) for row in grid])
    return payload


def bootstrap(root: Path) -> dict[str, Any]:
    layout = Layout(root)
    trades = _stored_trades(layout)
    test = [trade for trade in trades if SPLITS["test"][0] <= trade.entry_date <= SPLITS["test"][1] and trade.net_return is not None]
    test_returns = [
        value
        for trade in test
        if (value := trade.net_return) is not None
    ]
    values = np.array(test_returns, dtype=float)
    entry_clusters: dict[str, list[float]] = defaultdict(list)
    sector_clusters: dict[str, list[float]] = defaultdict(list)
    for trade in test:
        net_return = trade.net_return
        if net_return is None:
            continue
        entry_clusters[trade.entry_date].append(net_return)
        sector_clusters[trade.sector].append(net_return)
    equity = [row for row in Phase114Store(layout.database).latest("daily_equity") if SPLITS["test"][0] <= row["date"] <= SPLITS["test"][1]]
    daily = pd.Series([row["nav"] for row in equity]).pct_change().fillna(0.0).to_numpy()
    rng = np.random.default_rng(114)
    entry_stats = _cluster_bootstrap(entry_clusters, rng)
    sector_stats = _cluster_bootstrap(sector_clusters, rng)
    stationary = _stationary_bootstrap(daily, rng)
    ordinary = _value_bootstrap(values, rng)
    payload = {
        **_base("phase11_4_bootstrap_v1"),
        "status": "GO" if len(values) >= 100 else "INSUFFICIENT_SAMPLE",
        "seed": 114,
        "replications": 1000,
        "entry_date_block_bootstrap": entry_stats,
        "cluster_robust_entry_day": entry_stats,
        "cluster_robust_sector": sector_stats,
        "stationary_portfolio_daily_returns": stationary,
        "trade_bootstrap": ordinary,
        "expectancy_95_confidence_interval": ordinary["expectancy_ci_95"],
        "PF_95_confidence_interval": ordinary["profit_factor_ci_95"],
        "probability_expectancy_positive": entry_stats.get("probability_expectancy_positive", 0.0),
        "probability_portfolio_return_positive": stationary["probability_cumulative_return_positive"],
    }
    payload = _write(layout.artifact("bootstrap.json"), payload)
    Phase114Store(layout.database).append("bootstrap_runs", payload["content_hash"], [("BOOTSTRAP", payload)])
    return payload


def concentration(root: Path) -> dict[str, Any]:
    layout = Layout(root)
    trades = _stored_trades(layout)
    test = [trade for trade in trades if SPLITS["test"][0] <= trade.entry_date <= SPLITS["test"][1] and trade.net_return is not None]
    test_returns = [
        value
        for trade in test
        if (value := trade.net_return) is not None
    ]
    positive_total = sum(max(value, 0.0) for value in test_returns)
    ranked = sorted(test_returns, reverse=True)
    by_year = _positive_contribution(test, lambda trade: trade.entry_date[:4])
    by_sector = _positive_contribution(test, lambda trade: trade.sector)
    by_symbol = _positive_contribution(test, lambda trade: trade.symbol)
    def share(value: float) -> float:
        return value / positive_total if positive_total > 0 else 0.0
    top_year = _top(by_year)
    top_sector = _top(by_sector)
    top_symbol = _top(by_symbol)
    payload = {
        **_base("phase11_4_concentration_v1"),
        "status": "GO",
        "top_1_trade_share": share(sum(ranked[:1])),
        "top_10_trades_share": share(sum(ranked[:10])),
        "top_1_percent_trades_share": share(sum(ranked[: max(1, math.ceil(len(ranked) * 0.01))])),
        "top_5_percent_trades_share": share(sum(ranked[: max(1, math.ceil(len(ranked) * 0.05))])),
        "best_year": {"key": top_year[0], "share": share(top_year[1])},
        "best_sector": {"key": top_sector[0], "share": share(top_sector[1])},
        "best_symbol": {"key": top_symbol[0], "share": share(top_symbol[1])},
        "warnings": {
            "single_year_over_25_percent": share(top_year[1]) > 0.25,
            "single_sector_over_25_percent": share(top_sector[1]) > 0.25,
            "single_symbol_over_15_percent": share(top_symbol[1]) > 0.15,
        },
        "leave_one_year_out": _leave_one_out(test, lambda trade: trade.entry_date[:4]),
        "leave_one_sector_out": _leave_one_out(test, lambda trade: trade.sector),
        "cap_vol_regime_breadth_liquidity": {
            "status": "BLOCKED_REQUIRED_POINT_IN_TIME_CLASSIFICATIONS_MISSING",
            "available_liquidity_split": _liquidity_groups(test),
        },
    }
    payload = _write(layout.artifact("concentration.json"), payload)
    Phase114Store(layout.database).append("concentration_results", payload["content_hash"], [("CONCENTRATION", payload)])
    return payload


def yfinance_validation(root: Path) -> dict[str, Any]:
    from stocks.research.critical_trading import load_yfinance_cache

    layout = Layout(root)
    frames = load_yfinance_cache(root)
    master_path = root / "data" / "research" / "phase11_4" / "private" / "security-master.parquet"
    if not frames or not master_path.is_file():
        return _write(
            layout.artifact("yfinance-validation.json"),
            {
                **_base("phase11_4_yfinance_validation_v1"),
                "status": "NO_YFINANCE_CACHE",
                "candidate_gate_used": False,
            },
        )
    master = pd.read_parquet(master_path, columns=["security_id", "ticker", "is_delisted", "sector"])
    active = master.loc[~master["is_delisted"].astype(bool)].copy()
    active = active.loc[~active["ticker"].duplicated(keep=False)]
    lookup = active.set_index("ticker").to_dict("index")
    overlap = sorted(symbol for symbol in frames if symbol in lookup and not symbol.startswith("^"))
    yahoo_frames = {symbol: frames[symbol] for symbol in overlap}
    yahoo_trades, _ = generate_trades(
        yahoo_frames,
        sectors={symbol: str(lookup[symbol].get("sector") or "UNKNOWN") for symbol in overlap},
    )
    security_ids = {str(lookup[symbol]["security_id"]) for symbol in overlap}
    eodhd_trades = [trade for trade in _stored_trades(layout) if trade.symbol in security_ids]
    payload = {
        **_base("phase11_4_yfinance_validation_v1"),
        "status": "GO",
        "source": "yfinance_cached_auto_adjusted",
        "independent_provider": True,
        "candidate_gate_used": False,
        "PIT_universe_evidence": False,
        "current_active_symbol_overlap": len(overlap),
        "adjustment_difference": "YFINANCE_AUTO_ADJUST_INCLUDES_DIVIDENDS_VS_EODHD_SPLIT_ONLY",
        "yfinance_periods": {name: split_summary(yahoo_trades, *bounds) for name, bounds in SPLITS.items()},
        "eodhd_overlap_periods": {name: split_summary(eodhd_trades, *bounds) for name, bounds in SPLITS.items()},
        "interpretation": "CROSS_PROVIDER_SANITY_CHECK_ONLY_NOT_SURVIVORSHIP_OR_PIT_PROOF",
    }
    return _write(layout.artifact("yfinance-validation.json"), payload)


def status(root: Path) -> dict[str, Any]:
    layout = Layout(root)
    required: dict[str, Callable[[Path], dict[str, Any]]] = {
        "preregistration.json": preregister,
        "universe-audit.json": build_universe,
        "trade-results.json": trade_backtest,
        "portfolio-results.json": portfolio_backtest,
        "cost-sensitivity.json": cost_sensitivity,
        "shariah-cohort.json": shariah_cohort,
        "robustness.json": robustness,
        "bootstrap.json": bootstrap,
        "concentration.json": concentration,
        "yfinance-validation.json": yfinance_validation,
    }
    for name, operation in required.items():
        if not layout.artifact(name).is_file():
            operation(root)
    universe = _read(layout.artifact("universe-audit.json"))
    trades = _read(layout.artifact("trade-results.json"))
    portfolio = _read(layout.artifact("portfolio-results.json"))
    costs = _read(layout.artifact("cost-sensitivity.json"))
    shariah = _read(layout.artifact("shariah-cohort.json"))
    robust = _read(layout.artifact("robustness.json"))
    boot = _read(layout.artifact("bootstrap.json"))
    concentration_result = _read(layout.artifact("concentration.json"))
    yahoo = _read(layout.artifact("yfinance-validation.json"))
    currency = _read(layout.artifact("currency-normalization.json"))
    corporate_actions = _read(layout.artifact("corporate-action-audit.json"))
    decision = _decision(universe, trades, portfolio, costs, shariah, robust, boot, concentration_result)
    payload = {
        **_base("phase11_4_status_v1"),
        "status": "PHASE11_4_RSI_MEAN_REVERSION_PIT_VALIDATION_EVIDENCE_COMPLETE",
        "candidate_decision": decision,
        "financial_assessment_without_universe_gate": (
            "REJECTED_NO_NET_ALPHA"
            if (trades.get("periods", {}).get("test", {}).get("trade_profit_factor") or 0) < 1.20
            else "GATE_EVALUATION_REQUIRED"
        ),
        "forward_shadow_readiness": False,
        "FINANCIAL_FINALIST_GO": False,
        "PIT_universe_size": universe.get("PIT_universe_size"),
        "active_symbols": universe.get("active_symbols"),
        "delisted_symbols": universe.get("delisted_symbols"),
        "survivorship_bias_blocked": universe.get("survivorship_bias_blocked"),
        "train_validation_test": trades.get("periods"),
        "portfolio_test": portfolio.get("test_max_positions_4"),
        "cost_25_bps_test": costs.get("per_side_bps", {}).get("25"),
        "delisting_stress": trades.get("delisting_stress"),
        "parameter_plateau": robust.get("parameter_plateau"),
        "Shariah_coverage_ratio": shariah.get("Shariah_coverage_ratio"),
        "Shariah_cohort_status": shariah.get("status"),
        "bootstrap_confidence": {"expectancy": boot.get("expectancy_95_confidence_interval"), "PF": boot.get("PF_95_confidence_interval")},
        "DSR": robust.get("Deflated_Sharpe_Ratio"),
        "PBO": robust.get("Probability_of_Backtest_Overfitting"),
        "concentration_warnings": concentration_result.get("warnings"),
        "corporate_action_audit": {
            "status": corporate_actions.get("status"),
            "conflicted_security_count": corporate_actions.get("conflicted_security_count"),
        },
        "currency_normalization": {
            "status": currency.get("status"),
            "source": currency.get("source"),
            "missing_date_count": currency.get("missing_date_count_after_backward_asof"),
        },
        "benchmarks": portfolio.get("benchmarks"),
        "yfinance_independent_validation": {
            "status": yahoo.get("status"),
            "current_active_symbol_overlap": yahoo.get("current_active_symbol_overlap"),
            "candidate_gate_used": yahoo.get("candidate_gate_used"),
        },
        "open_blockers": _blockers(universe, shariah, concentration_result, currency),
        "private_database": str(layout.database),
        "private_store_counts": {
            **Phase114Store(layout.database).counts(),
            "decisions": Phase114Store(layout.database).counts()["decisions"] + 1,
        },
        "frozen_dependency_integrity": _dependency_integrity(root),
    }
    payload = _write(layout.artifact("status.json"), payload)
    Phase114Store(layout.database).append("decisions", payload["content_hash"], [(decision, payload)])
    _write_report(layout, payload)
    return payload


def freeze(root: Path) -> dict[str, Any]:
    current = status(root)
    layout = Layout(root)
    artifacts = [
        "preregistration.json", "universe-audit.json", "trade-results.json", "portfolio-results.json",
        "cost-sensitivity.json", "shariah-cohort.json", "robustness.json", "bootstrap.json",
        "concentration.json", "status.json", "report.md",
        "yfinance-validation.json",
        "corporate-action-audit.json", "currency-normalization.json", "benchmarks.json",
    ]
    hashes = {name: _file_hash(layout.artifact(name)) for name in artifacts}
    payload = {
        **_base("phase11_4_freeze_status_v1"),
        "freeze_status": "PHASE11_4_RSI_MEAN_REVERSION_PIT_VALIDATION_EVIDENCE_FROZEN_GO",
        "financial_outcome": current["candidate_decision"],
        "PIT_VALIDATED_RESEARCH_CANDIDATE": current["candidate_decision"] in {"PIT_VALIDATED_RESEARCH_CANDIDATE", "FORWARD_SHADOW_ELIGIBLE"},
        "FORWARD_SHADOW_ELIGIBLE": current["candidate_decision"] == "FORWARD_SHADOW_ELIGIBLE",
        "artifact_hashes": hashes,
        "frozen_dependency_integrity": current["frozen_dependency_integrity"],
    }
    return _write(layout.artifact("freeze-status.json"), payload)


def phase11_4_command(root: Path, command: str) -> dict[str, Any]:
    operations = {
        "preregister": preregister,
        "build-universe": build_universe,
        "trade-backtest": trade_backtest,
        "portfolio-backtest": portfolio_backtest,
        "cost-sensitivity": cost_sensitivity,
        "shariah-cohort": shariah_cohort,
        "robustness": robustness,
        "bootstrap": bootstrap,
        "concentration": concentration,
        "status": status,
        "freeze": freeze,
    }
    if command not in operations:
        raise ValueError(f"unknown rsi-pit command: {command}")
    return operations[command](root)


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _grouped_trade_summaries(trades: list[PitTrade], key: Callable[[PitTrade], str]) -> dict[str, Any]:
    groups: dict[str, list[PitTrade]] = defaultdict(list)
    for trade in trades:
        groups[key(trade)].append(trade)
    return {name: trade_summary(rows) for name, rows in sorted(groups.items())}


def _rolling_years(trades: list[PitTrade]) -> list[dict[str, Any]]:
    rows = []
    for start in range(2000, 2024):
        subset = [trade for trade in trades if str(start) <= trade.entry_date[:4] <= str(start + 2)]
        rows.append({"start_year": start, "end_year": start + 2, **trade_summary(subset)})
    return rows


def _distribution(values: list[float | None]) -> dict[str, Any]:
    clean = np.array([float(value) for value in values if value is not None], dtype=float)
    return {"count": len(clean), "p05": float(np.quantile(clean, 0.05)) if len(clean) else None, "median": float(np.median(clean)) if len(clean) else None, "p95": float(np.quantile(clean, 0.95)) if len(clean) else None}


def _delisting_stress(trades: list[PitTrade], recovery: float) -> dict[str, Any]:
    stressed = [trade for trade in trades if trade.net_return is not None]
    uncertain = [trade for trade in trades if trade.status == "DELISTING_EXECUTION_UNCERTAIN"]
    values = [
        value
        for trade in stressed
        if (value := trade.net_return) is not None
    ] + [recovery] * len(uncertain)
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value < 0))
    return {"recovery_return": recovery, "stressed_positions": len(uncertain), "trade_profit_factor": None if losses == 0 else gains / losses, "expectancy": sum(values) / len(values) if values else None}


def _equity_yearly_returns(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(frame["date"])
    result: dict[str, float] = {}
    for year, group in frame.groupby(frame.index.year):
        result[str(year)] = float(group["nav"].iloc[-1] / group["nav"].iloc[0] - 1.0)
    return result


def _positive_year_ratio(rows: dict[str, float], minimum_year: int) -> float | None:
    values = [value for year, value in rows.items() if int(year) >= minimum_year]
    return sum(value > 0 for value in values) / len(values) if values else None


def _market_benchmark(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    test_start = pd.Timestamp(SPLITS["test"][0])
    test_end = pd.Timestamp(SPLITS["test"][1])
    covered = {
        symbol: frame.loc[
            (frame.index >= test_start) & (frame.index <= test_end),
            "close",
        ]
        for symbol, frame in frames.items()
    }
    covered = {symbol: values for symbol, values in covered.items() if len(values) >= 2}
    symbol = next((item for item in ("SPY", "VTI", "QQQ") if item in covered), None)
    if symbol is None:
        return {"status": "BLOCKED_MISSING"}
    values = covered[symbol]
    years = (values.index[-1] - values.index[0]).days / 365.25
    return {"status": "GO", "symbol": symbol, "CAGR": float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) if years > 0 else None}


def _liquidity_slippage(trades: list[PitTrade]) -> dict[str, Any]:
    test = [trade for trade in trades if SPLITS["test"][0] <= trade.entry_date <= SPLITS["test"][1] and trade.gross_return is not None]
    values: list[float] = []
    classes: Counter[str] = Counter()
    for trade in test:
        if trade.historical_dollar_volume >= 50_000_000:
            extra, name = 2.5, "large_liquid"
        elif trade.historical_dollar_volume >= 10_000_000:
            extra, name = 10.0, "medium_liquid"
        else:
            extra, name = 25.0, "lower_liquid"
        classes[name] += 1
        total = (10.0 + extra) / 10_000.0
        gross_return = trade.gross_return
        if gross_return is None:
            continue
        values.append((1 + gross_return) * (1 - total) / (1 + total) - 1)
    gains, losses = sum(v for v in values if v > 0), abs(sum(v for v in values if v < 0))
    return {"class_counts": dict(classes), "trade_count": len(values), "profit_factor": None if losses == 0 else gains / losses, "expectancy": float(np.mean(values)) if values else None}


def _repriced_summary(trades: list[PitTrade], cost_bps: float, start: str, end: str) -> dict[str, Any]:
    cost = cost_bps / 10_000.0
    values = [
        (1 + float(trade.gross_return)) * (1 - cost) / (1 + cost) - 1
        for trade in trades
        if start <= trade.entry_date <= end and trade.gross_return is not None
    ]
    array = np.array(values, dtype=float)
    gains = float(array[array > 0].sum()) if len(array) else 0.0
    losses = abs(float(array[array < 0].sum())) if len(array) else 0.0
    return {
        "trade_count": len(array),
        "win_rate": float((array > 0).mean()) if len(array) else None,
        "trade_profit_factor": None if losses == 0 else gains / losses,
        "expectancy": float(array.mean()) if len(array) else None,
        "median_trade": float(np.median(array)) if len(array) else None,
    }


def _stored_trades(layout: Layout) -> list[PitTrade]:
    return [PitTrade(**row) for row in Phase114Store(layout.database).latest("trades")]


def _eligible_on(screens: list[dict[str, Any]], symbol: str, signal_date: str) -> bool:
    for row in screens:
        if row.get("symbol") == symbol and str(row.get("screened_at", ""))[:10] <= signal_date <= str(row.get("expiry", ""))[:10]:
            return True
    return False


def _trade_daily_series(trades: list[PitTrade], start: str, end: str) -> pd.Series:
    values: defaultdict[str, float] = defaultdict(float)
    for trade in trades:
        if start <= trade.entry_date <= end and trade.net_return is not None and trade.exit_date:
            values[trade.exit_date] += float(trade.net_return)
    return pd.Series(values, dtype=float).sort_index()


def _deflated_sharpe(values: pd.Series, trials: int) -> float | None:
    if len(values) < 3 or values.std(ddof=0) == 0:
        return None
    sharpe = float(values.mean() / values.std(ddof=0) * math.sqrt(252))
    threshold = math.sqrt(2 * math.log(trials)) / math.sqrt(max(len(values) - 1, 1)) * math.sqrt(252)
    z = (sharpe - threshold) * math.sqrt(max(len(values) - 1, 1) / 252)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _pbo(series: list[pd.Series]) -> float | None:
    if not series:
        return None
    frame = pd.concat(series, axis=1).fillna(0.0)
    blocks = np.array_split(frame, 10)
    failures = 0
    evaluated = 0
    for index in range(0, 10, 2):
        train = pd.concat([block for i, block in enumerate(blocks) if i != index])
        test = blocks[index]
        train_scores = train.mean() / train.std(ddof=0).replace(0, np.nan)
        if train_scores.dropna().empty:
            continue
        winner = train_scores.idxmax()
        test_scores = test.mean() / test.std(ddof=0).replace(0, np.nan)
        if pd.isna(test_scores.get(winner)):
            continue
        failures += bool(test_scores[winner] < test_scores.median())
        evaluated += 1
    return failures / evaluated if evaluated else None


def _white_like(series: list[pd.Series]) -> dict[str, Any]:
    if not series:
        return {"status": "INSUFFICIENT_SAMPLE"}
    frame = pd.concat(series, axis=1).fillna(0.0)
    observed = float(frame.mean().max())
    rng = np.random.default_rng(114)
    maxima = []
    centered = frame - frame.mean()
    for _ in range(500):
        sample = centered.iloc[rng.integers(0, len(centered), len(centered))]
        maxima.append(float(sample.mean().max()))
    return {"status": "GO", "observed_best_mean": observed, "p_value": sum(value >= observed for value in maxima) / len(maxima)}


def _profit_factor(values: np.ndarray) -> float:
    losses = abs(float(values[values < 0].sum()))
    return math.inf if losses == 0 and bool((values > 0).any()) else (float(values[values > 0].sum()) / losses if losses else 0.0)


def _value_bootstrap(values: np.ndarray, rng: np.random.Generator, runs: int = 1000) -> dict[str, Any]:
    if len(values) == 0:
        return {"status": "INSUFFICIENT_SAMPLE", "expectancy_ci_95": [None, None], "profit_factor_ci_95": [None, None], "probability_expectancy_positive": 0.0}
    means, pfs = [], []
    for _ in range(runs):
        sample = rng.choice(values, len(values), replace=True)
        means.append(float(sample.mean()))
        pfs.append(_profit_factor(sample))
    finite_pf = np.array([value for value in pfs if math.isfinite(value)])
    return {"status": "GO", "expectancy_ci_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "profit_factor_ci_95": [float(np.quantile(finite_pf, 0.025)), float(np.quantile(finite_pf, 0.975))] if len(finite_pf) else [None, None], "probability_expectancy_positive": sum(value > 0 for value in means) / runs}


def _cluster_bootstrap(clusters: dict[str, list[float]], rng: np.random.Generator) -> dict[str, Any]:
    keys = list(clusters)
    if len(keys) < 2:
        return {"status": "INSUFFICIENT_CLUSTERS", "cluster_count": len(keys)}
    outcomes = []
    for _ in range(1000):
        selected = rng.choice(keys, len(keys), replace=True)
        sample = [value for key in selected for value in clusters[str(key)]]
        outcomes.append(float(np.mean(sample)))
    return {"status": "GO", "cluster_count": len(keys), "expectancy_ci_95": [float(np.quantile(outcomes, 0.025)), float(np.quantile(outcomes, 0.975))], "probability_expectancy_positive": sum(value > 0 for value in outcomes) / len(outcomes)}


def _stationary_bootstrap(values: np.ndarray, rng: np.random.Generator, average_block: int = 20) -> dict[str, Any]:
    if len(values) == 0:
        return {"status": "INSUFFICIENT_SAMPLE", "probability_cumulative_return_positive": 0.0}
    totals = []
    restart_probability = 1 / average_block
    for _ in range(1000):
        index = int(rng.integers(0, len(values)))
        sample = []
        for _ in range(len(values)):
            if rng.random() < restart_probability:
                index = int(rng.integers(0, len(values)))
            sample.append(values[index])
            index = (index + 1) % len(values)
        totals.append(float(np.prod(1 + np.array(sample)) - 1))
    return {"status": "GO", "average_block_length": average_block, "cumulative_return_ci_95": [float(np.quantile(totals, 0.025)), float(np.quantile(totals, 0.975))], "probability_cumulative_return_positive": sum(value > 0 for value in totals) / len(totals)}


def _positive_contribution(trades: list[PitTrade], key: Callable[[PitTrade], str]) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for trade in trades:
        result[key(trade)] += max(float(trade.net_return or 0), 0.0)
    return dict(result)


def _top(values: dict[str, float]) -> tuple[str | None, float]:
    return max(values.items(), key=lambda item: item[1]) if values else (None, 0.0)


def _leave_one_out(trades: list[PitTrade], key: Callable[[PitTrade], str]) -> dict[str, Any]:
    groups = sorted({key(trade) for trade in trades})
    return {group: trade_summary([trade for trade in trades if key(trade) != group]) for group in groups}


def _liquidity_groups(trades: list[PitTrade]) -> dict[str, Any]:
    groups = {
        "large": [trade for trade in trades if trade.historical_dollar_volume >= 50_000_000],
        "medium": [trade for trade in trades if 10_000_000 <= trade.historical_dollar_volume < 50_000_000],
        "lower": [trade for trade in trades if trade.historical_dollar_volume < 10_000_000],
    }
    return {name: trade_summary(rows) for name, rows in groups.items()}


def _decision(universe: dict[str, Any], trades: dict[str, Any], portfolio: dict[str, Any], costs: dict[str, Any], shariah: dict[str, Any], robust: dict[str, Any], boot: dict[str, Any], concentration_result: dict[str, Any]) -> str:
    if not universe.get("survivorship_bias_blocked"):
        return "REJECTED_SURVIVORSHIP_DEPENDENT"
    test_pf = trades.get("periods", {}).get("test", {}).get("trade_profit_factor") or 0
    if test_pf < 1.20:
        return "REJECTED_NO_NET_ALPHA"
    if (costs.get("per_side_bps", {}).get("25", {}).get("trade_profit_factor") or 0) <= 1:
        return "REJECTED_COST_SENSITIVE"
    warnings = concentration_result.get("warnings", {})
    if warnings.get("single_year_over_25_percent") or warnings.get("single_sector_over_25_percent") or warnings.get("single_symbol_over_15_percent"):
        return "REJECTED_CONCENTRATION"
    if shariah.get("status") != "SHARIAH_COHORT_GO":
        return "REJECTED_SHARIAH_SAMPLE"
    test_portfolio = portfolio.get("test_max_positions_4", {})
    candidate = (test_portfolio.get("CAGR") or 0) > 0 and abs(test_portfolio.get("maximum_drawdown") or -1) <= 0.25 and (boot.get("probability_expectancy_positive") or 0) >= 0.95 and (robust.get("Deflated_Sharpe_Ratio") or 0) >= 0.95 and (robust.get("Probability_of_Backtest_Overfitting") or 1) <= 0.20
    return "FORWARD_SHADOW_ELIGIBLE" if candidate else "PROMISING_RESEARCH_CANDIDATE"


def _blockers(
    universe: dict[str, Any],
    shariah: dict[str, Any],
    concentration_result: dict[str, Any],
    currency: dict[str, Any],
) -> list[str]:
    blockers = list(universe.get("blocking_reasons", []))
    if shariah.get("status") != "SHARIAH_COHORT_GO":
        blockers.append("SHARIAH_SAMPLE_INSUFFICIENT")
    if concentration_result.get("cap_vol_regime_breadth_liquidity", {}).get("status", "").startswith("BLOCKED"):
        blockers.append("PIT_CAP_SECTOR_REGIME_BREADTH_CLASSIFICATIONS_MISSING")
    if currency.get("status") != "GO":
        blockers.append("HISTORICAL_USD_EUR_NORMALIZATION_MISSING")
    return blockers


def _dependency_integrity(root: Path) -> dict[str, Any]:
    paths = {
        "phase1": root / "output" / "ibkr" / "phase1-freeze-status.json",
        "phase6_4": root / "output" / "research" / "phase6_4" / "freeze-status.json",
        "phase7": root / "output" / "execution" / "phase7" / "freeze-status.json",
        "phase8": root / "output" / "ibkr" / "phase8" / "freeze-status.json",
        "phase11_3": root / "output" / "ibkr" / "phase11_3" / "freeze-status.json",
    }
    return {name: {"exists": path.is_file(), "hash": _file_hash(path)} for name, path in paths.items()}


def _write_report(layout: Layout, result: dict[str, Any]) -> None:
    test = result.get("train_validation_test", {}).get("test", {})
    portfolio = result.get("portfolio_test", {})
    lines = [
        "# Phase 11.4 RSI Mean Reversion PIT Validation",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Decision",
        "",
        f"- Candidate decision: `{result['candidate_decision']}`",
        f"- Forward shadow eligible: `{result['forward_shadow_readiness']}`",
        f"- Survivorship bias blocked: `{result['survivorship_bias_blocked']}`",
        "",
        "## Evidence",
        "",
        f"- Universe: {result.get('PIT_universe_size')} symbols ({result.get('active_symbols')} active, {result.get('delisted_symbols')} delisted/inactive).",
        f"- Test trades: {test.get('trade_count')}; provisional PF: {test.get('trade_profit_factor')}.",
        f"- Portfolio test CAGR: {portfolio.get('CAGR')}; maximum drawdown: {portfolio.get('maximum_drawdown')}.",
        f"- Shariah coverage: {result.get('Shariah_coverage_ratio')}; status: `{result.get('Shariah_cohort_status')}`.",
        f"- DSR: {result.get('DSR')}; PBO: {result.get('PBO')}.",
        "",
        "## Blockers",
        "",
        *[f"- `{item}`" for item in result.get("open_blockers", [])],
        "",
        "All strategy and execution authority remains `NONE`. Provisional metrics are not valid PIT evidence while the historical-membership gate is blocked.",
        "",
    ]
    layout.artifact("report.md").write_text("\n".join(lines), encoding="utf-8")
