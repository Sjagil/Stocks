from __future__ import annotations

import os
import tempfile
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.p3.io import atomic_write_json
from stocks.rl.data import (
    build_causal_multitimeframe_context,
    load_multitimeframe_frames,
)


PANEL_PATH = Path("data/ai/private/canonical-ml-panel.parquet")
PANEL_STATUS_PATH = Path("output/ai/decision-intelligence/panel-status.json")
FILLS_PATH = Path("output/research/phase11_14/fills.parquet")
BARS_PATH = Path("data/research/phase11_4/private/pit-bars.parquet")
PRICE_ROOT = Path("data/research/critical_trading/yfinance")
SECURITY_MASTER_PATH = Path(
    "data/research/phase11_4/private/security-master.parquet"
)
QUALIFICATION_PATH = Path("output/research/phase11_14/qualification.json")
HORIZONS = (1, 3, 5, 10, 20)


DAILY_NUMERIC_FEATURES = (
    "return_1d",
    "return_5d",
    "return_20d",
    "return_63d",
    "volatility_20d",
    "volatility_63d",
    "atr_pct_14d",
    "rsi_14d",
    "distance_sma_20d",
    "distance_sma_50d",
    "breakout_distance_20d",
    "drawdown_63d",
    "volume_z_20d",
    "relative_volume_20d",
    "log_dollar_volume_1d",
    "return_20d_cross_sectional_rank",
    "volatility_20d_cross_sectional_rank",
    "market_breadth_20d",
    "estimated_round_trip_cost_rate",
)
MULTITIMEFRAME_RETURN_FEATURES = (
    "return_15m",
    "return_1h",
    "return_2h",
    "return_4h",
)
MULTITIMEFRAME_MISSING_FEATURES = tuple(
    f"missing__{name}" for name in MULTITIMEFRAME_RETURN_FEATURES
)
NUMERIC_FEATURES = (
    *DAILY_NUMERIC_FEATURES,
    *MULTITIMEFRAME_RETURN_FEATURES,
    *MULTITIMEFRAME_MISSING_FEATURES,
    "missing_feature_fraction",
)

CATEGORICAL_FEATURES = (
    "symbol",
    "strategy_id",
    "strategy_family",
    "entry_timeframe",
    "asset_class",
    "sector",
    "currency",
    "regime",
)


def build_canonical_ml_panel(
    project_root: Path,
    *,
    cost_bps: float = 10.0,
    publish: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = project_root.resolve()
    fills_path = root / FILLS_PATH
    if not fills_path.is_file():
        raise ValueError("canonical Phase 11.14 fills are missing")
    fills = pd.read_parquet(fills_path)
    fills = fills.loc[pd.to_numeric(fills["cost_bps"]).eq(float(cost_bps))]
    episodes = _closed_trade_episodes(fills)
    if episodes.empty:
        raise ValueError("no closed deterministic candidate episodes are available")
    bars, price_source_hashes = _load_candidate_price_histories(
        root, sorted(episodes["security_id"].unique())
    )
    featured = build_causal_bar_features(bars)
    strategy_metadata = _strategy_metadata(root / QUALIFICATION_PATH)
    panel = _attach_causal_context_and_targets(
        episodes,
        featured,
        strategy_metadata,
        cost_bps=float(cost_bps),
    )
    panel, multitimeframe_status = _attach_multitimeframe_context(root, panel)
    if panel.empty:
        raise ValueError("no causally aligned ML panel rows are available")
    panel = panel.sort_values(
        ["decision_timestamp", "strategy_id", "security_id"]
    ).reset_index(drop=True)
    if not (
        pd.to_datetime(panel["feature_timestamp"], utc=True)
        < pd.to_datetime(panel["decision_timestamp"], utc=True)
    ).all():
        raise ValueError("canonical ML panel contains feature lookahead")
    if not (
        pd.to_datetime(panel["label_available_at"], utc=True)
        >= pd.to_datetime(panel["decision_timestamp"], utc=True)
    ).all():
        raise ValueError("canonical ML panel contains premature labels")
    duplicate_columns = [
        "strategy_id",
        "fold_id",
        "security_id",
        "decision_timestamp",
    ]
    if panel.duplicated(duplicate_columns).any():
        raise ValueError("canonical ML panel contains duplicate decision identities")

    status: dict[str, Any] = {
        "schema": "canonical_decision_ml_panel_status_v1",
        "status": "RESEARCH_PANEL_GO" if len(panel) >= 500 else "INSUFFICIENT_DATA",
        "generated_at": datetime.now(UTC).isoformat(),
        "panel_path": PANEL_PATH.as_posix(),
        "row_count": len(panel),
        "decision_timestamp_count": int(panel["decision_timestamp"].nunique()),
        "symbol_count": int(panel["symbol"].nunique()),
        "security_count": int(panel["security_id"].nunique()),
        "strategy_count": int(panel["strategy_id"].nunique()),
        "class_balance_positive": float(panel["positive_net_trade"].mean()),
        "decision_start": str(panel["decision_timestamp"].min()),
        "decision_end": str(panel["decision_timestamp"].max()),
        "label_available_end": str(panel["label_available_at"].max()),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "targets": [
            "native_exit_net_return",
            "positive_net_trade",
            "native_exit_risk_adjusted_return",
            "maximum_adverse_excursion",
            "maximum_favorable_excursion",
            *[f"forward_net_return_{h}d" for h in HORIZONS],
        ],
        "cost_bps_scenario": float(cost_bps),
        "candidate_source": "PHASE11_14_NATURAL_DETERMINISTIC_OOS_BUY_FILLS",
        "candidate_unit": "ONE_EXECUTED_OOS_TRADE_EPISODE",
        "active_swing_required_candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "active_swing_candidate_unit_go": False,
        "candidate_identity_deduplicated": True,
        "feature_rule": "LATEST_CLOSED_DAILY_BAR_STRICTLY_BEFORE_ENTRY",
        "multitimeframe_feature_rule": (
            "LATEST_FULLY_CLOSED_BAR_AVAILABLE_AT_OR_BEFORE_DECISION"
        ),
        "multitimeframe": multitimeframe_status,
        "price_history_semantics": "CURRENTLY_ADJUSTED_RESEARCH_BACKFILL_NOT_VENDOR_PIT",
        "target_rule": "ATTACHED_ONLY_AFTER_NATIVE_EXIT_OR_HORIZON_CLOSE",
        "trade_matching": "FIFO_APPEND_ONLY_FILL_RECONSTRUCTION",
        "selection_conditioned_history": True,
        "point_in_time_universe_complete": False,
        "shariah_history_complete": False,
        "production_training_allowed": False,
        "research_only": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "source_hashes": {
            FILLS_PATH.as_posix(): sha256_file(fills_path).upper(),
            **price_source_hashes,
        },
    }
    if publish:
        destination = root / PANEL_PATH
        _atomic_parquet(destination, panel)
        status["panel_sha256"] = sha256_file(destination).upper()
        status["content_hash"] = stable_hash(status)
        atomic_write_json(root / PANEL_STATUS_PATH, status)
    return panel, status


def build_causal_bar_features(bars: pd.DataFrame) -> pd.DataFrame:
    required = {
        "security_id",
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sector",
        "currency",
    }
    if not required.issubset(bars.columns):
        raise ValueError(f"PIT bars missing columns: {sorted(required - set(bars))}")
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["date", "security_id", "close"])
    frame = frame.sort_values(["security_id", "date"])
    groups: list[pd.DataFrame] = []
    for _, group in frame.groupby("security_id", sort=False):
        value = group.copy()
        close = value["close"]
        returns = close.pct_change(fill_method=None)
        previous = close.shift(1)
        true_range = pd.concat(
            [
                value["high"] - value["low"],
                (value["high"] - previous).abs(),
                (value["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        delta = close.diff()
        gains = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        losses = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
        relative_strength = gains / losses.replace(0.0, np.nan)
        volume_mean = value["volume"].rolling(20, min_periods=20).mean()
        volume_std = value["volume"].rolling(20, min_periods=20).std(ddof=0)
        value["return_1d"] = returns
        for horizon in (5, 20, 63):
            value[f"return_{horizon}d"] = close.pct_change(
                horizon, fill_method=None
            )
        value["volatility_20d"] = returns.rolling(20, min_periods=20).std(ddof=0)
        value["volatility_63d"] = returns.rolling(63, min_periods=40).std(ddof=0)
        value["atr_pct_14d"] = true_range.rolling(14, min_periods=14).mean() / close
        value["rsi_14d"] = 100.0 - 100.0 / (1.0 + relative_strength)
        value["distance_sma_20d"] = close / close.rolling(20, min_periods=20).mean() - 1.0
        value["distance_sma_50d"] = close / close.rolling(50, min_periods=40).mean() - 1.0
        value["breakout_distance_20d"] = close / value["high"].rolling(20, min_periods=20).max() - 1.0
        value["drawdown_63d"] = close / close.rolling(63, min_periods=40).max() - 1.0
        value["volume_z_20d"] = (value["volume"] - volume_mean) / volume_std.replace(0.0, np.nan)
        value["relative_volume_20d"] = value["volume"] / volume_mean.replace(0.0, np.nan)
        value["log_dollar_volume_1d"] = np.log1p((value["close"] * value["volume"]).clip(lower=0.0))
        groups.append(value)
    result = pd.concat(groups, ignore_index=True)
    result["return_20d_cross_sectional_rank"] = result.groupby("date")[
        "return_20d"
    ].rank(pct=True)
    result["volatility_20d_cross_sectional_rank"] = result.groupby("date")[
        "volatility_20d"
    ].rank(pct=True)
    result["market_breadth_20d"] = result["return_20d"].gt(0).groupby(
        result["date"]
    ).transform("mean")
    result["regime"] = np.select(
        [
            result["market_breadth_20d"].ge(0.60)
            & result["return_20d"].ge(0),
            result["market_breadth_20d"].le(0.40)
            & result["return_20d"].lt(0),
            result["volatility_20d_cross_sectional_rank"].ge(0.80),
        ],
        ["RISK_ON", "RISK_OFF", "HIGH_VOLATILITY"],
        default="MIXED",
    )
    return result.sort_values(["security_id", "date"]).reset_index(drop=True)


def _closed_trade_episodes(fills: pd.DataFrame) -> pd.DataFrame:
    required = {
        "strategy_id",
        "fold_id",
        "cost_bps",
        "date",
        "security_id",
        "side",
        "shares",
        "price_eur",
        "fee_eur",
    }
    if not required.issubset(fills.columns):
        raise ValueError(f"fills missing columns: {sorted(required - set(fills))}")
    frame = fills.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["side_order"] = frame["side"].eq("BUY").map({True: 0, False: 1})
    frame = frame.sort_values(
        ["strategy_id", "fold_id", "security_id", "date", "side_order"]
    )
    closed: list[dict[str, Any]] = []
    key_columns = ["strategy_id", "fold_id", "cost_bps", "security_id"]
    for key, group in frame.groupby(key_columns, sort=False):
        lots: deque[dict[str, Any]] = deque()
        for row in group.itertuples(index=False):
            shares = int(row.shares)
            if shares <= 0:
                raise ValueError("fill shares must be positive whole numbers")
            if str(row.side) == "BUY":
                lots.append(
                    {
                        "entry_timestamp": row.date,
                        "entry_price": float(row.price_eur),
                        "entry_fee": float(row.fee_eur),
                        "shares": shares,
                        "remaining": shares,
                        "exit_notional": 0.0,
                        "exit_fee": 0.0,
                        "exit_timestamp": None,
                        "exit_sides": set(),
                    }
                )
                continue
            remaining_sale = shares
            while remaining_sale > 0 and lots:
                lot = lots[0]
                consumed = min(remaining_sale, lot["remaining"])
                sale_fraction = consumed / shares
                lot["exit_notional"] += consumed * float(row.price_eur)
                lot["exit_fee"] += sale_fraction * float(row.fee_eur)
                lot["exit_timestamp"] = row.date
                lot["exit_sides"].add(str(row.side))
                lot["remaining"] -= consumed
                remaining_sale -= consumed
                if lot["remaining"] == 0:
                    entry_notional = lot["shares"] * lot["entry_price"]
                    net_pnl = (
                        lot["exit_notional"]
                        - lot["exit_fee"]
                        - entry_notional
                        - lot["entry_fee"]
                    )
                    closed.append(
                        {
                            "strategy_id": key[0],
                            "fold_id": key[1],
                            "cost_bps": float(key[2]),
                            "security_id": key[3],
                            "entry_timestamp": lot["entry_timestamp"],
                            "exit_timestamp": lot["exit_timestamp"],
                            "entry_price": lot["entry_price"],
                            "exit_price": lot["exit_notional"] / lot["shares"],
                            "shares": lot["shares"],
                            "entry_fee_eur": lot["entry_fee"],
                            "exit_fee_eur": lot["exit_fee"],
                            "native_exit_net_pnl_eur": net_pnl,
                            "native_exit_net_return": net_pnl / entry_notional,
                            "exit_reason": "+".join(sorted(lot["exit_sides"])),
                        }
                    )
                    lots.popleft()
            if remaining_sale:
                raise ValueError("sell fill exceeds reconstructed open inventory")
    return pd.DataFrame(closed)


def _attach_causal_context_and_targets(
    episodes: pd.DataFrame,
    featured_bars: pd.DataFrame,
    strategy_metadata: dict[str, dict[str, Any]],
    *,
    cost_bps: float,
) -> pd.DataFrame:
    histories = {
        str(ticker): group.sort_values("date").reset_index(drop=True)
        for ticker, group in featured_bars.groupby("ticker", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        history = histories.get(str(episode.security_id))
        if history is None or history.empty:
            continue
        decision_timestamp = pd.Timestamp(episode.entry_timestamp)
        decision_day = decision_timestamp.normalize()
        dates = pd.DatetimeIndex(history["date"])
        feature_position = int(dates.searchsorted(decision_day, side="left")) - 1
        if feature_position < 63:
            continue
        context = history.iloc[feature_position]
        feature_timestamp = pd.Timestamp(context["date"])
        future = history.loc[history["date"].ge(decision_day)].head(max(HORIZONS))
        if len(future) < max(HORIZONS):
            continue
        exit_day = pd.Timestamp(episode.exit_timestamp).normalize()
        path = history.loc[
            history["date"].ge(decision_day) & history["date"].le(exit_day)
        ]
        if path.empty:
            continue
        entry_price = float(episode.entry_price)
        approximate_round_trip_cost = max(
            2.0 * float(episode.entry_fee_eur) / (episode.shares * entry_price),
            2.0 * cost_bps / 10_000.0,
        )
        metadata = strategy_metadata.get(str(episode.strategy_id), {})
        observed_feature_names = tuple(
            name
            for name in DAILY_NUMERIC_FEATURES
            if name
            not in {"estimated_round_trip_cost_rate", "missing_feature_fraction"}
        )
        numeric_values = {
            name: _finite_or_nan(context.get(name))
            for name in observed_feature_names
        }
        numeric_values["estimated_round_trip_cost_rate"] = (
            approximate_round_trip_cost
        )
        missing = sum(
            pd.isna(numeric_values[name]) for name in observed_feature_names
        )
        numeric_values["missing_feature_fraction"] = missing / len(
            observed_feature_names
        )
        volatility = _finite_or_nan(context.get("volatility_20d"))
        native_return = float(episode.native_exit_net_return)
        row: dict[str, Any] = {
            "candidate_unit": "ONE_EXECUTED_OOS_TRADE_EPISODE",
            "candidate_identity": stable_hash(
                {
                    "strategy_id": str(episode.strategy_id),
                    "fold_id": str(episode.fold_id),
                    "security_id": str(episode.security_id),
                    "decision_timestamp": decision_timestamp.isoformat(),
                }
            ),
            "natural_strategy_candidate": False,
            "symbol": str(context.get("ticker") or episode.security_id),
            "security_id": str(
                context.get("security_id") or f"SYMBOL:{episode.security_id}"
            ),
            "decision_timestamp": decision_timestamp,
            "feature_timestamp": feature_timestamp,
            "available_at": feature_timestamp + pd.Timedelta(hours=23, minutes=59),
            "label_available_at": pd.Timestamp(episode.exit_timestamp),
            "strategy_id": str(episode.strategy_id),
            "strategy_family": str(metadata.get("formula") or "UNKNOWN"),
            "entry_timeframe": str(metadata.get("timeframe") or "UNKNOWN"),
            "setup_timeframe": str(metadata.get("timeframe") or "UNKNOWN"),
            "context_timeframes": "1D",
            "fold_id": str(episode.fold_id),
            "asset_class": str(metadata.get("asset_class") or "UNKNOWN"),
            "sector": str(context.get("sector") or "UNKNOWN"),
            "industry": "UNKNOWN",
            "current_strategy_score": np.nan,
            "current_strategy_score_missing": 1,
            "currency": str(context.get("currency") or "UNKNOWN"),
            "regime": str(context.get("regime") or "UNKNOWN"),
            "entry_price": entry_price,
            "exit_price": float(episode.exit_price),
            "shares": int(episode.shares),
            "exit_reason": str(episode.exit_reason),
            "native_exit_net_pnl_eur": float(episode.native_exit_net_pnl_eur),
            "native_exit_net_return": native_return,
            "positive_net_trade": int(native_return > 0.0),
            "native_exit_risk_adjusted_return": (
                native_return / volatility
                if np.isfinite(volatility) and volatility > 0
                else np.nan
            ),
            "maximum_adverse_excursion": float(path["low"].min() / entry_price - 1.0),
            "maximum_favorable_excursion": float(path["high"].max() / entry_price - 1.0),
            "holding_days": float(
                (pd.Timestamp(episode.exit_timestamp) - decision_timestamp)
                / pd.Timedelta(days=1)
            ),
            "selection_conditioned_history": True,
            "point_in_time_universe_verified": False,
            "shariah_point_in_time_verified": False,
            **numeric_values,
        }
        for horizon in HORIZONS:
            horizon_close = float(future.iloc[horizon - 1]["close"])
            row[f"forward_net_return_{horizon}d"] = (
                horizon_close / entry_price - 1.0 - approximate_round_trip_cost
            )
            row[f"forward_target_available_at_{horizon}d"] = pd.Timestamp(
                future.iloc[horizon - 1]["date"]
            ) + pd.Timedelta(hours=23, minutes=59)
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_multitimeframe_context(
    project_root: Path, panel: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = panel.copy()
    audit_columns = tuple(
        f"source_available_at_{timeframe}"
        for timeframe in ("15m", "1h", "2h", "4h")
    )
    for name in (*MULTITIMEFRAME_RETURN_FEATURES, *audit_columns):
        if name.startswith("source_available_at_"):
            result[name] = pd.Series(
                pd.NaT,
                index=result.index,
                dtype="datetime64[ns, UTC]",
            )
        else:
            result[name] = np.nan

    symbols_with_any_context = 0
    source_hashes: dict[str, str] = {}
    for symbol, indices in result.groupby("symbol", sort=False).groups.items():
        frames = load_multitimeframe_frames(project_root, str(symbol))
        if frames:
            symbols_with_any_context += 1
        for source in frames.values():
            relative = str(source.attrs.get("source_path") or "")
            path = project_root / relative
            if relative and relative not in source_hashes and path.is_file():
                source_hashes[relative] = sha256_file(path)
        decisions = result.loc[indices, "decision_timestamp"].reset_index(drop=True)
        context = build_causal_multitimeframe_context(decisions, frames)
        for name in (*MULTITIMEFRAME_RETURN_FEATURES, *audit_columns):
            values = (
                pd.to_datetime(context[name], utc=True).array
                if name.startswith("source_available_at_")
                else context[name].to_numpy()
            )
            result.loc[indices, name] = values

    for name in MULTITIMEFRAME_RETURN_FEATURES:
        result[f"missing__{name}"] = result[name].isna().astype(int)
    feature_values = result.loc[
        :, [*DAILY_NUMERIC_FEATURES, *MULTITIMEFRAME_RETURN_FEATURES]
    ]
    result["missing_feature_fraction"] = feature_values.isna().mean(axis=1)

    causality_violations = 0
    for name in audit_columns:
        available = pd.to_datetime(result[name], utc=True, errors="coerce")
        decision = pd.to_datetime(
            result["decision_timestamp"], utc=True, errors="coerce"
        )
        causality_violations += int((available.notna() & available.gt(decision)).sum())
    if causality_violations:
        raise ValueError("multitimeframe panel contains feature lookahead")

    coverage = {
        name: {
            "available_rows": int(result[name].notna().sum()),
            "missing_rows": int(result[name].isna().sum()),
            "coverage_ratio": round(float(result[name].notna().mean()), 8),
        }
        for name in MULTITIMEFRAME_RETURN_FEATURES
    }
    return result, {
        "status": (
            "RESEARCH_CONTEXT_PARTIAL"
            if any(row["missing_rows"] for row in coverage.values())
            else "RESEARCH_CONTEXT_COMPLETE"
        ),
        "symbols_with_any_context": symbols_with_any_context,
        "symbol_count": int(result["symbol"].nunique()),
        "coverage": coverage,
        "source_hashes": dict(sorted(source_hashes.items())),
        "partial_bars_excluded": True,
        "availability_rule": "source_available_at <= decision_timestamp",
        "causality_violation_count": causality_violations,
        "production_training_allowed": False,
    }


def _strategy_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row.get("strategy_id")): row
        for row in payload.get("strategies", [])
        if row.get("strategy_id")
    }


def _load_candidate_price_histories(
    project_root: Path, symbols: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    master_path = project_root / SECURITY_MASTER_PATH
    identity_by_symbol: dict[str, dict[str, Any]] = {}
    if master_path.is_file():
        master = pd.read_parquet(
            master_path,
            columns=["security_id", "ticker", "sector", "currency"],
        )
        identity_by_symbol = {
            str(row["ticker"]).upper(): row.to_dict()
            for _, row in master.iterrows()
        }
    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for symbol in symbols:
        path = project_root / PRICE_ROOT / f"{symbol}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path).rename(columns={"session_date": "date"})
        if not {"date", "open", "high", "low", "close", "volume"}.issubset(
            frame.columns
        ):
            continue
        identity = identity_by_symbol.get(symbol, {})
        frame["security_id"] = str(
            identity.get("security_id") or f"SYMBOL:{symbol}"
        )
        frame["ticker"] = symbol
        frame["sector"] = str(identity.get("sector") or "UNKNOWN")
        frame["currency"] = str(identity.get("currency") or "USD")
        frame["source"] = "YFINANCE_CAUSAL_RESEARCH_CACHE"
        frame["price_basis"] = "CURRENTLY_ADJUSTED_RESEARCH_ONLY"
        frames.append(frame)
        relative = path.relative_to(project_root).as_posix()
        source_hashes[relative] = sha256_file(path).upper()
    if not frames:
        raise ValueError("candidate price histories are unavailable")
    return pd.concat(frames, ignore_index=True), source_hashes


def _finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=path.stem + "-", suffix=".parquet.tmp", dir=path.parent, delete=False
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "BARS_PATH",
    "CATEGORICAL_FEATURES",
    "FILLS_PATH",
    "HORIZONS",
    "NUMERIC_FEATURES",
    "PANEL_PATH",
    "PANEL_STATUS_PATH",
    "PRICE_ROOT",
    "build_canonical_ml_panel",
    "build_causal_bar_features",
]
