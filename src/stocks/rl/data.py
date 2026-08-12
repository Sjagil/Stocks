from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from stocks.rl.contracts import stable_hash


MARKET_FEATURES = (
    "return_15m",
    "return_1h",
    "return_2h",
    "return_4h",
    "return_1d",
    "return_1w",
    "primary_return_1bar",
    "primary_return_5bars",
    "primary_return_20bars",
    "primary_volatility_20bars",
    "primary_atr_pct_14bars",
    "primary_rsi_14bars",
    "primary_momentum_20bars",
    "primary_trend_strength",
    "primary_distance_ema_20bars",
    "primary_breakout_distance_20bars",
    "primary_volume_log",
    "primary_relative_volume_20bars",
    "spread_bps",
    "primary_liquidity_score_63bars",
    "primary_volatility_percentile_252bars",
)
REGIME_FEATURES = (
    "market_regime",
    "volatility_regime",
    "trend_regime",
    "risk_on",
    "macro_regime",
    "market_breadth",
    "sector_regime",
)
SIGNAL_FEATURES = (
    "strategy_code",
    "signal_direction",
    "setup_score",
    "expected_return",
    "expected_risk",
    "expected_rr",
    "historical_expectancy",
    "regime_expectancy",
    "signal_confidence",
    "signal_age",
)
CONTEXT_FEATURES = (
    "news_sentiment",
    "news_novelty",
    "sec_event_score",
    "options_gex_context",
    "commodity_context",
    "rates_context",
    "dxy_context",
    "vix_context",
    "btc_regime_context",
)
STATIC_FEATURES = MARKET_FEATURES + REGIME_FEATURES + SIGNAL_FEATURES + CONTEXT_FEATURES

DYNAMIC_FEATURES = (
    "cash_pct",
    "current_exposure",
    "open_risk",
    "number_positions",
    "unrealized_pnl",
    "realized_daily_pnl",
    "portfolio_drawdown",
    "correlation_concentration",
    "sector_concentration",
    "asset_class_concentration",
    "position_entry_price_ratio",
    "position_unrealized_return",
    "position_atr_since_entry",
    "position_mfe",
    "position_mae",
    "position_holding_duration",
    "position_distance_stop",
    "position_distance_target",
    "position_trailing_stop_status",
)

MISSINGNESS_FEATURES = tuple(f"missing__{name}" for name in STATIC_FEATURES)
OBSERVATION_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES + MISSINGNESS_FEATURES

OUTCOME_COLUMNS = (
    "outcome_next_return",
    "outcome_next_high_return",
    "outcome_next_low_return",
)


@dataclass(frozen=True)
class DatasetContract:
    symbol: str
    asset_class: str = "STOCK"
    strategy_id: str = "RL_CAUSAL_SWING_BASELINE"
    timeframe: str = "1d"
    evidence_scope: str = "UNRESTRICTED_RESEARCH_ONLY"
    shariah_point_in_time_verified: bool = False
    point_in_time_universe_verified: bool = False
    survivorship_verified: bool = False
    source_name: str = "LOCAL_PARQUET"
    source_license: str = "UNKNOWN"
    source_version: str = "UNVERSIONED"
    exchange_timezone: str = "America/New_York"
    decision_clock: str = "BAR_CLOSE"
    regular_session_close: str = "16:00"

    def promotion_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not self.point_in_time_universe_verified:
            blockers.append("PIT_DATA_NOT_VERIFIED")
        if not self.survivorship_verified:
            blockers.append("SURVIVORSHIP_NOT_VERIFIED")
        if not self.shariah_point_in_time_verified:
            blockers.append("SHARIAH_PIT_NOT_VERIFIED")
        if self.evidence_scope != "PRODUCTION_EVIDENCE":
            blockers.append("RESEARCH_ONLY_DATASET")
        return blockers


@dataclass(frozen=True)
class CausalFeatureScaler:
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]
    fitted_start: str
    fitted_end: str
    fitted_rows: int

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        feature_names: Iterable[str] = STATIC_FEATURES,
    ) -> CausalFeatureScaler:
        names = tuple(feature_names)
        if frame.empty:
            raise ValueError("cannot fit RL feature scaler on an empty frame")
        missing = [name for name in names if name not in frame]
        if missing:
            raise ValueError(f"feature scaler columns missing: {missing}")
        medians: list[float] = []
        scales: list[float] = []
        for name in names:
            values = pd.to_numeric(frame[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            median = float(values.median()) if values.notna().any() else 0.0
            q25 = float(values.quantile(0.25)) if values.notna().any() else 0.0
            q75 = float(values.quantile(0.75)) if values.notna().any() else 0.0
            scale = q75 - q25
            if not math.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            medians.append(median if math.isfinite(median) else 0.0)
            scales.append(scale)
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        return cls(
            feature_names=names,
            medians=tuple(medians),
            scales=tuple(scales),
            fitted_start=timestamps.iloc[0].isoformat(),
            fitted_end=timestamps.iloc[-1].isoformat(),
            fitted_rows=len(frame),
        )

    def transform_row(self, row: pd.Series | dict[str, Any]) -> tuple[list[float], list[float]]:
        scaled: list[float] = []
        missingness: list[float] = []
        for name, median, scale in zip(
            self.feature_names, self.medians, self.scales, strict=True
        ):
            raw = row.get(name)
            missing = raw is None or not _finite(raw)
            missingness.append(1.0 if missing else 0.0)
            value = median if missing else float(raw)
            scaled.append(float(np.clip((value - median) / scale, -10.0, 10.0)))
        return scaled, missingness

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["medians"] = list(self.medians)
        payload["scales"] = list(self.scales)
        payload["schema"] = "rl_causal_feature_scaler_v1"
        payload["content_hash"] = stable_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CausalFeatureScaler:
        return cls(
            feature_names=tuple(str(x) for x in payload["feature_names"]),
            medians=tuple(float(x) for x in payload["medians"]),
            scales=tuple(float(x) for x in payload["scales"]),
            fitted_start=str(payload["fitted_start"]),
            fitted_end=str(payload["fitted_end"]),
            fitted_rows=int(payload["fitted_rows"]),
        )


def load_price_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"session_date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"price history schema incomplete: {path}")
    result = frame.loc[:, sorted(required)].copy()
    result["session_date"] = pd.to_datetime(result["session_date"], utc=True)
    result = result.sort_values("session_date").drop_duplicates(
        "session_date", keep="last"
    )
    numeric = sorted(required - {"session_date"})
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=numeric)
    if result.empty or (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"invalid positive OHLC history: {path}")
    return result.reset_index(drop=True)


def build_causal_swing_frame(
    prices: pd.DataFrame,
    contract: DatasetContract,
    *,
    spread_bps: float | None = None,
    minimum_history: int = 63,
    retain_unresolved_outcome: bool = False,
    multitimeframe_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build close-of-bar observations; outcomes are stored in reserved columns.

    All observation features at row *t* use rows no later than *t*. Reserved
    ``outcome_*`` columns may only be read by environment transitions and
    evaluators, never by the observation encoder.
    """

    required = {"session_date", "open", "high", "low", "close", "volume"}
    if not required.issubset(prices.columns):
        raise ValueError("causal swing builder requires canonical OHLCV columns")
    frame = prices.copy().sort_values("session_date").reset_index(drop=True)
    decision_timestamps = _primary_decision_timestamps(frame, contract)
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float).clip(lower=0)
    returns = close.pct_change()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    volatility = returns.rolling(20, min_periods=20).std(ddof=0)
    rolling_high = close.rolling(20, min_periods=20).max()
    volume_mean = volume.rolling(20, min_periods=20).mean()
    dollar_volume = close * volume
    liquidity_median = dollar_volume.rolling(63, min_periods=20).median()
    volatility_rank = volatility.rolling(252, min_periods=63).apply(
        _last_percentile_rank, raw=True
    )
    trend = (ema20 - ema50) / atr.replace(0, np.nan)
    momentum_20 = close.pct_change(20)
    direction = ((ema20 > ema50) & (momentum_20 > 0)).astype(float)
    expected_return = returns.rolling(63, min_periods=40).mean() * 5.0
    expected_risk = volatility * math.sqrt(5.0)
    expected_rr = expected_return / expected_risk.replace(0, np.nan)
    historical_expectancy = close.pct_change(5).rolling(126, min_periods=63).mean()
    risk_on = ((close > ema50) & (volatility_rank < 0.80)).astype(float)
    market_regime = np.select(
        [trend > 0.75, trend < -0.75], [1.0, -1.0], default=0.0
    )
    volatility_regime = np.select(
        [volatility_rank >= 0.75, volatility_rank <= 0.25],
        [1.0, -1.0],
        default=0.0,
    )
    setup_score = (
        0.40 * direction
        + 0.25 * momentum_20.clip(-0.25, 0.25).add(0.25).div(0.50)
        + 0.20 * (1.0 - volatility_rank.clip(0, 1))
        + 0.15 * (close / rolling_high).clip(0, 1)
    ).clip(0, 1)
    signal_age = _consecutive_age(direction.astype(bool))
    strategy_code = int(stable_hash(contract.strategy_id)[:8], 16) / 0xFFFFFFFF
    mtf = build_causal_multitimeframe_context(
        decision_timestamps,
        multitimeframe_frames or {},
    )
    primary_timeframe = contract.timeframe.lower()
    explicit_returns: dict[str, pd.Series] = {
        "return_15m": mtf["return_15m"],
        "return_1h": mtf["return_1h"],
        "return_2h": mtf["return_2h"],
        "return_4h": mtf["return_4h"],
        "return_1d": pd.Series(np.nan, index=frame.index, dtype=float),
        "return_1w": pd.Series(np.nan, index=frame.index, dtype=float),
    }
    primary_return_name = f"return_{primary_timeframe}"
    if primary_return_name in explicit_returns:
        explicit_returns[primary_return_name] = returns
    if primary_timeframe == "1d":
        explicit_returns["return_1w"] = close.pct_change(5)

    result = pd.DataFrame(
        {
            "timestamp": decision_timestamps,
            "asset": contract.symbol.upper(),
            "asset_class": contract.asset_class.upper(),
            "strategy_id": contract.strategy_id,
            "timeframe": contract.timeframe,
            "close": close,
            "high": high,
            "low": low,
            "atr": atr,
            **explicit_returns,
            "primary_return_1bar": returns,
            "primary_return_5bars": close.pct_change(5),
            "primary_return_20bars": momentum_20,
            "primary_volatility_20bars": volatility,
            "primary_atr_pct_14bars": atr / close,
            "primary_rsi_14bars": rsi / 100.0,
            "primary_momentum_20bars": momentum_20,
            "primary_trend_strength": trend.clip(-5, 5) / 5.0,
            "primary_distance_ema_20bars": close / ema20 - 1.0,
            "primary_breakout_distance_20bars": close / rolling_high - 1.0,
            "primary_volume_log": np.log1p(volume),
            "primary_relative_volume_20bars": volume
            / volume_mean.replace(0, np.nan),
            "spread_bps": float(spread_bps) if spread_bps is not None else np.nan,
            "primary_liquidity_score_63bars": np.log1p(liquidity_median),
            "primary_volatility_percentile_252bars": volatility_rank,
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "trend_regime": np.sign(trend),
            "risk_on": risk_on,
            "macro_regime": np.nan,
            "market_breadth": np.nan,
            "sector_regime": np.nan,
            "strategy_code": strategy_code,
            "signal_direction": direction,
            "setup_score": setup_score,
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "expected_rr": expected_rr,
            "historical_expectancy": historical_expectancy,
            "regime_expectancy": expected_return * (0.5 + 0.5 * risk_on),
            "signal_confidence": setup_score,
            "signal_age": signal_age / 20.0,
            "news_sentiment": np.nan,
            "news_novelty": np.nan,
            "sec_event_score": np.nan,
            "options_gex_context": np.nan,
            "commodity_context": np.nan,
            "rates_context": np.nan,
            "dxy_context": np.nan,
            "vix_context": np.nan,
            "btc_regime_context": np.nan,
            "gate_risk_budget": True,
            "gate_shariah": bool(contract.shariah_point_in_time_verified),
            "gate_tradeable": True,
            "gate_liquidity": liquidity_median.gt(0),
            "gate_portfolio_capacity": True,
            "gate_data_fresh": True,
            "gate_quote_valid": spread_bps is not None,
            "outcome_next_return": close.shift(-1) / close - 1.0,
            "outcome_next_high_return": high.shift(-1) / close - 1.0,
            "outcome_next_low_return": low.shift(-1) / close - 1.0,
        }
    )
    for timeframe in ("15m", "1h", "2h", "4h"):
        audit_column = f"source_available_at_{timeframe}"
        result[audit_column] = mtf[audit_column]
    result = result.iloc[max(0, minimum_history - 1) :].copy()
    if not retain_unresolved_outcome:
        result = result.dropna(subset=list(OUTCOME_COLUMNS))
    result = result.reset_index(drop=True)
    result.attrs["dataset_contract"] = asdict(contract)
    result.attrs["promotion_blockers"] = contract.promotion_blockers()
    result.attrs["observation_features"] = list(OBSERVATION_FEATURES)
    result.attrs["outcome_columns_excluded_from_observation"] = list(OUTCOME_COLUMNS)
    return result


def load_causal_dataset(
    path: Path,
    contract: DatasetContract,
    *,
    spread_bps: float | None = None,
    retain_unresolved_outcome: bool = False,
    multitimeframe_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return build_causal_swing_frame(
        load_price_frame(path),
        contract,
        spread_bps=spread_bps,
        retain_unresolved_outcome=retain_unresolved_outcome,
        multitimeframe_frames=multitimeframe_frames,
    )


def load_multitimeframe_frames(
    project_root: Path, symbol: str
) -> dict[str, pd.DataFrame]:
    root = project_root.resolve()
    result: dict[str, pd.DataFrame] = {}
    allowed_source_intervals = {
        "15m": {"15m"},
        "1h": {"1h"},
        "2h": {"1h", "2h"},
        "4h": {"1h", "2h", "4h"},
    }
    for timeframe in ("15m", "1h", "2h", "4h"):
        candidates = sorted(
            [
                path
                for path in (
                root
                / "data"
                / "research"
                / "multitimeframe"
                / "private"
                ).glob(
                    f"provider=*/symbol={symbol.upper()}/interval={timeframe}/"
                    "source_interval=*/bars.parquet"
                )
                if path.parent.name.removeprefix("source_interval=")
                in allowed_source_intervals[timeframe]
            ],
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if candidates:
            frame = pd.read_parquet(candidates[0])
            frame.attrs["source_path"] = candidates[0].relative_to(root).as_posix()
            result[timeframe] = frame
    return result


def build_causal_multitimeframe_context(
    decision_timestamps: pd.Series,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Backward-as-of join closed lower-timeframe bars without lookahead.

    Missing intervals remain NaN and therefore receive explicit missingness
    indicators in the canonical observation. No higher-frequency bar with a
    timestamp after the decision timestamp can enter the state.
    """

    decisions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                decision_timestamps, utc=True, errors="coerce"
            )
        }
    )
    if decisions["timestamp"].isna().any():
        raise ValueError("multi-timeframe decision timestamps are invalid")
    decisions["_order"] = np.arange(len(decisions))
    ordered = decisions.sort_values("timestamp")
    result = pd.DataFrame(index=ordered.index)
    maximum_age = {
        "15m": pd.Timedelta(days=4),
        "1h": pd.Timedelta(days=4),
        "2h": pd.Timedelta(days=4),
        "4h": pd.Timedelta(days=7),
    }
    duration = {
        "15m": pd.Timedelta(minutes=15),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
    }
    for timeframe in ("15m", "1h", "2h", "4h"):
        column = f"return_{timeframe}"
        audit_column = f"source_available_at_{timeframe}"
        source = frames.get(timeframe)
        result[column] = np.nan
        result[audit_column] = pd.Series(
            pd.NaT,
            index=ordered.index,
            dtype="datetime64[ns, UTC]",
        )
        if source is None or source.empty:
            continue
        timestamp_name = next(
            (
                name
                for name in ("timestamp_utc", "timestamp", "session_date")
                if name in source.columns
            ),
            "",
        )
        if timestamp_name not in source.columns or "close" not in source.columns:
            raise ValueError(f"{timeframe} context needs timestamp and close")
        optional = [
            name
            for name in ("available_at", "bar_close_utc", "partial_bucket", "is_partial")
            if name in source.columns
        ]
        bars = source.loc[:, [timestamp_name, "close", *optional]].copy()
        bars["bar_open"] = pd.to_datetime(
            bars[timestamp_name], utc=True, errors="coerce"
        )
        bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
        partial = pd.Series(False, index=bars.index)
        for name in ("partial_bucket", "is_partial"):
            if name in bars:
                partial |= bars[name].map(
                    lambda value: bool(value) if pd.notna(value) else False
                )
        explicit_available = next(
            (name for name in ("available_at", "bar_close_utc") if name in bars),
            "",
        )
        if explicit_available:
            bars["source_available_at"] = pd.to_datetime(
                bars[explicit_available], utc=True, errors="coerce"
            )
        else:
            bars["source_available_at"] = bars["bar_open"] + duration[timeframe]
        bars = (
            bars.loc[~partial]
            .dropna(subset=["bar_open", "source_available_at", "close"])
            .sort_values("source_available_at")
            .drop_duplicates("source_available_at", keep="last")
        )
        bars[column] = bars["close"].pct_change()
        joined = pd.merge_asof(
            ordered.loc[:, ["timestamp"]],
            bars.loc[:, ["source_available_at", column]],
            left_on="timestamp",
            right_on="source_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        stale = (
            joined["source_available_at"].isna()
            | (
                (joined["timestamp"] - joined["source_available_at"])
                > maximum_age[timeframe]
            )
        )
        result[column] = joined[column].mask(stale).to_numpy()
        result[audit_column] = joined["source_available_at"].mask(stale).to_numpy()
    return result.sort_index().reset_index(drop=True)


def dataset_manifest(frame: pd.DataFrame, contract: DatasetContract) -> dict[str, Any]:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    payload: dict[str, Any] = {
        "schema": "rl_causal_dataset_manifest_v2",
        "contract": asdict(contract),
        "rows": len(frame),
        "start": timestamps.iloc[0].isoformat() if len(frame) else None,
        "end": timestamps.iloc[-1].isoformat() if len(frame) else None,
        "observation_features": list(OBSERVATION_FEATURES),
        "reserved_outcome_columns": list(OUTCOME_COLUMNS),
        "promotion_blockers": contract.promotion_blockers(),
        "causality": (
            "DECISION_AT_PRIMARY_BAR_CLOSE;CONTEXT_AVAILABLE_AT_LE_DECISION;"
            "PARTIAL_BARS_EXCLUDED;OUTCOME_T_PLUS_1_RESERVED"
        ),
        "random_split_allowed": False,
        "execution_authority": "NONE",
        "money_control": False,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def save_scaler(path: Path, scaler: CausalFeatureScaler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _last_percentile_rank(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite <= finite[-1]))


def _consecutive_age(condition: pd.Series) -> pd.Series:
    groups = condition.ne(condition.shift()).cumsum()
    age = condition.groupby(groups).cumcount() + 1
    return age.where(condition, 0).astype(float)


def _primary_decision_timestamps(
    frame: pd.DataFrame, contract: DatasetContract
) -> pd.Series:
    """Return an auditable decision clock for the primary bar.

    Legacy daily research files carry date-only session labels. Those labels
    are not availability timestamps, so using midnight would either leak the
    same day's close or unnecessarily lag intraday context. Daily decisions
    are therefore stamped at the configured regular-session close. Intraday
    callers must provide actual timestamps and are advanced by their bar
    duration.
    """

    raw = pd.to_datetime(frame["session_date"], utc=True, errors="coerce")
    if raw.isna().any():
        raise ValueError("primary decision timestamps are invalid")
    timeframe = contract.timeframe.lower()
    if timeframe == "1d":
        date_labels = raw.dt.strftime("%Y-%m-%d") + f" {contract.regular_session_close}"
        local = pd.to_datetime(date_labels, errors="coerce").dt.tz_localize(
            contract.exchange_timezone,
            ambiguous="raise",
            nonexistent="raise",
        )
        return local.dt.tz_convert("UTC")
    duration = {
        "15m": pd.Timedelta(minutes=15),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "6h": pd.Timedelta(hours=6),
        "12h": pd.Timedelta(hours=12),
        "1w": pd.Timedelta(days=7),
    }.get(timeframe)
    if duration is None:
        raise ValueError(f"unsupported primary decision timeframe: {contract.timeframe}")
    return raw + duration


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "CONTEXT_FEATURES",
    "CausalFeatureScaler",
    "DYNAMIC_FEATURES",
    "DatasetContract",
    "MARKET_FEATURES",
    "MISSINGNESS_FEATURES",
    "OBSERVATION_FEATURES",
    "OUTCOME_COLUMNS",
    "REGIME_FEATURES",
    "SIGNAL_FEATURES",
    "STATIC_FEATURES",
    "build_causal_swing_frame",
    "build_causal_multitimeframe_context",
    "dataset_manifest",
    "load_causal_dataset",
    "load_multitimeframe_frames",
    "load_price_frame",
    "save_scaler",
]
