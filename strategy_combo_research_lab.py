#!/usr/bin/env python3
"""
Strategy Combo Research Lab v2
================================

One-file, research-only backtest laboratory for the Stocks repository.

What it does
------------
1. Reads local daily OHLCV data from Parquet/CSV, with a fail-closed corporate-action discontinuity gate (no broker calls or secret-file reads).
2. Backtests a broad registry of mechanically specified long-only strategies from
   the supplied Critical Trading transcripts.
3. Searches parameter values per strategy using train/validation only.
4. Freezes one validation-selected winner per strategy.
5. Re-runs those winners with a bounded portfolio model.
6. Exhaustively evaluates every pair, triple and four-strategy portfolio.
7. Writes CSV/JSON/Markdown artifacts, checkpoints and top equity curves.

Important research semantics
----------------------------
* Signals are formed after the daily close unless a strategy explicitly states
  otherwise. Entries and ordinary exits execute at the next available open.
* Parameter selection does NOT use the test period.
* Combinations are portfolios of independently running strategy sleeves; they
  are not hindsight-created AND-filters.
* The default policy is long-only and excludes short/futures/bond strategies.
* This program never imports a broker SDK and contains no order, market-data or account
  methods. It is an offline research tool only.
* A single positive result is not authority for paper or live trading.

Quick start (from C:\\Users\\alhar\\Documents\\Stocks)
---------------------------------------------------------
    .\\.venv-ibkr\\Scripts\\python.exe .\\strategy_combo_research_lab.py run --preset long

Smoke test:
    python .\\strategy_combo_research_lab.py run --preset smoke --max-symbols 50

Explicit data path:
    python .\\strategy_combo_research_lab.py run ^
      --data .\\data\\research\\phase11_4\\private\\pit-bars.parquet ^
      --preset long --fixed-fee-eur 3 --combo-sizes 2,3,4

The default auto-discovery path is:
    data/research/phase11_4/private/pit-bars.parquet

Required columns (aliases accepted):
    security_id/ticker/symbol, date, open, high, low, close, volume

Dependencies already expected in the Stocks environment:
    numpy, pandas, pyarrow, duckdb
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import heapq
import itertools
import json
import math
import pickle
import random
import statistics
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import duckdb
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - user environment error
    raise SystemExit(
        "Missing dependency. Activate the Stocks virtual environment and install "
        "numpy pandas pyarrow duckdb. Original error: " + str(exc)
    ) from exc


PROGRAM_VERSION = "2.1.0"
SCHEMA = "strategy_combo_research_lab_v2"
PROGRAM_SCHEMA = SCHEMA
DEFAULT_DATA_CANDIDATES = (
    Path("data/research/phase11_4/private/pit-bars.parquet"),
    Path("data/research/phase11_4/private/pit_bars.parquet"),
    Path("data/research/critical_trading/yfinance"),
)
PERIOD_NAMES = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def safe_divide(
    numerator: Any,
    denominator: Any,
    *,
    min_abs_denominator: float = 1e-12,
) -> np.ndarray:
    """Elementwise float division without divide-by-zero RuntimeWarnings.

    Invalid, non-finite or effectively zero denominators produce NaN. This is
    deliberate: indicator warm-up bars and zero-ATR bars must never create a
    trading signal.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    num, den = np.broadcast_arrays(num, den)
    out = np.full(num.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(num) & np.isfinite(den) & (np.abs(den) > float(min_abs_denominator))
    )
    np.divide(num, den, out=out, where=valid)
    return out


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def parse_date(value: str | dt.date | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def annualization_factor(index: pd.Index | None = None) -> float:
    """Infer observations per year; retain 252 for index-free legacy callers."""
    if index is None or len(index) < 2:
        return 252.0
    timestamps = pd.DatetimeIndex(index).sort_values().unique()
    if len(timestamps) < 2:
        return 252.0
    elapsed_years = max((timestamps[-1] - timestamps[0]).days / 365.2425, 1 / 365.2425)
    empirical = (len(timestamps) - 1) / elapsed_years
    return float(max(empirical, 1.0))


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def profit_factor(values: Sequence[float] | np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    gains = float(array[array > 0].sum())
    losses = float(-array[array < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Bailey/Lopez de Prado style PSR approximation."""
    if observations < 3 or not math.isfinite(observed_sharpe):
        return float("nan")
    denominator_term = (
        1.0 - skewness * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    )
    if denominator_term <= 0:
        return float("nan")
    z = (
        (observed_sharpe - benchmark_sharpe)
        * math.sqrt(observations - 1)
        / math.sqrt(denominator_term)
    )
    return normal_cdf(z)


def deflated_sharpe_probability(
    observed_sharpe: float,
    observations: int,
    trial_count: int,
    sharpe_std_across_trials: float,
    skewness: float,
    kurtosis: float,
) -> float:
    """Conservative, transparent DSR approximation for multiple trials."""
    if trial_count <= 1 or not math.isfinite(sharpe_std_across_trials):
        benchmark = 0.0
    else:
        # Approximate expected maximum of N standard normal variables.
        euler_gamma = 0.5772156649015329
        z1 = statistics.NormalDist().inv_cdf(1.0 - 1.0 / trial_count)
        z2 = statistics.NormalDist().inv_cdf(1.0 - 1.0 / (trial_count * math.e))
        benchmark = sharpe_std_across_trials * (
            (1.0 - euler_gamma) * z1 + euler_gamma * z2
        )
    return probabilistic_sharpe_ratio(
        observed_sharpe,
        benchmark,
        observations,
        skewness,
        kurtosis,
    )


def daily_metrics(
    returns: pd.Series,
    initial_capital: float,
    trial_count: int = 1,
    sharpe_std_across_trials: float = 0.0,
) -> dict[str, Any]:
    series = returns.fillna(0.0).astype(float)
    if series.empty:
        return {
            "observations": 0,
            "total_return": float("nan"),
            "CAGR": float("nan"),
            "Sharpe": float("nan"),
            "Sortino": float("nan"),
            "maximum_drawdown": float("nan"),
            "Calmar": float("nan"),
            "daily_profit_factor": float("nan"),
            "positive_day_ratio": float("nan"),
            "positive_year_ratio": float("nan"),
            "terminal_equity": float("nan"),
            "PSR": float("nan"),
            "DSR_probability": float("nan"),
        }
    equity = (1.0 + series).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    elapsed_days = max((series.index[-1] - series.index[0]).days, 1)
    years = elapsed_days / 365.2425
    cagr = (
        float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    )
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    periods_per_year = annualization_factor(series.index)
    sharpe = mean / std * math.sqrt(periods_per_year) if std > 0 else float("nan")
    downside = series[series < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = (
        mean / downside_std * math.sqrt(periods_per_year)
        if downside_std > 0
        else float("nan")
    )
    mdd = max_drawdown(series)
    calmar = cagr / abs(mdd) if math.isfinite(mdd) and mdd < 0 else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    underwater = drawdown < 0
    drawdown_groups = (~underwater).cumsum()
    maximum_drawdown_duration = (
        int(underwater.groupby(drawdown_groups).sum().max()) if len(underwater) else 0
    )
    monthly = series.groupby(series.index.to_period("M")).apply(
        lambda values: (1.0 + values).prod() - 1.0
    )
    yearly = series.groupby(series.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    skewness = float(series.skew()) if len(series) > 2 else 0.0
    kurtosis = float(series.kurt() + 3.0) if len(series) > 3 else 3.0
    psr = probabilistic_sharpe_ratio(sharpe, 0.0, len(series), skewness, kurtosis)
    dsr = deflated_sharpe_probability(
        sharpe,
        len(series),
        max(trial_count, 1),
        max(sharpe_std_across_trials, 0.0),
        skewness,
        kurtosis,
    )
    return {
        "observations": int(len(series)),
        "start": str(series.index[0].date()),
        "end": str(series.index[-1].date()),
        "total_return": total_return,
        "CAGR": cagr,
        "periods_per_year": periods_per_year,
        "annualized_volatility": std * math.sqrt(periods_per_year)
        if std > 0
        else 0.0,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "maximum_drawdown": mdd,
        "maximum_drawdown_duration_observations": maximum_drawdown_duration,
        "Calmar": calmar,
        "daily_profit_factor": profit_factor(series.to_numpy()),
        "positive_day_ratio": float((series > 0).mean()),
        "positive_year_ratio": float((yearly > 0).mean())
        if len(yearly)
        else float("nan"),
        "year_count": int(len(yearly)),
        "monthly_returns": {str(key): float(value) for key, value in monthly.items()},
        "terminal_equity": float(initial_capital * equity.iloc[-1]),
        "PSR": psr,
        "DSR_probability": dsr,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "yearly_returns": {str(int(k)): float(v) for k, v in yearly.items()},
    }


def block_bootstrap(
    returns: pd.Series,
    runs: int,
    block_size: int,
    seed: int,
) -> dict[str, Any]:
    values = returns.fillna(0.0).to_numpy(dtype=float)
    if runs <= 0 or len(values) < 5:
        return {"runs": 0, "status": "SKIPPED"}
    rng = np.random.default_rng(seed)
    n = len(values)
    periods_per_year = annualization_factor(returns.index)
    block_size = max(1, min(block_size, n))
    totals: list[float] = []
    cagrs: list[float] = []
    mdds: list[float] = []
    for _ in range(runs):
        sampled: list[float] = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            indices: np.ndarray = (start + np.arange(block_size)) % n
            sampled.extend(values[indices].tolist())
        arr = np.asarray(sampled[:n], dtype=float)
        equity = np.cumprod(1.0 + arr)
        terminal = float(equity[-1])
        totals.append(terminal - 1.0)
        years = n / periods_per_year
        cagrs.append(terminal ** (1.0 / years) - 1.0 if terminal > 0 else -1.0)
        peaks = np.maximum.accumulate(equity)
        mdds.append(float(np.min(equity / peaks - 1.0)))
    return {
        "runs": runs,
        "block_size": block_size,
        "probability_total_return_gt_0": float(np.mean(np.asarray(totals) > 0)),
        "total_return_ci95": [
            float(np.quantile(totals, 0.025)),
            float(np.quantile(totals, 0.975)),
        ],
        "CAGR_ci95": [
            float(np.quantile(cagrs, 0.025)),
            float(np.quantile(cagrs, 0.975)),
        ],
        "max_drawdown_ci95": [
            float(np.quantile(mdds, 0.025)),
            float(np.quantile(mdds, 0.975)),
        ],
    }


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "security_id": ("security_id", "securityid", "perm_id", "id", "ticker", "symbol"),
    "symbol": ("symbol", "ticker", "code", "security_id"),
    "date": ("date", "datetime", "timestamp", "time"),
    "open": ("open", "adj_open", "adjusted_open"),
    "high": ("high", "adj_high", "adjusted_high"),
    "low": ("low", "adj_low", "adjusted_low"),
    "close": ("close", "adj_close", "adjusted_close"),
    "volume": ("volume", "adj_volume", "adjusted_volume"),
    "sector": ("sector", "sector_name", "gics_sector"),
    "industry": ("industry", "industry_name", "gics_industry"),
    "currency": ("currency", "currency_code"),
}


def sql_quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def discover_default_data() -> Path:
    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No market-data file was supplied and no default was found. Pass --data PATH. "
        "Expected first choice: data/research/phase11_4/private/pit-bars.parquet"
    )


@dataclass
class DataColumns:
    security_id: str
    symbol: str
    date: str
    open: str
    high: str
    low: str
    close: str
    volume: str | None
    sector: str | None
    industry: str | None
    currency: str | None


class MarketDataSource:
    """DuckDB-backed, chunked reader that avoids loading the whole PIT file."""

    def __init__(
        self,
        path: Path,
        start: pd.Timestamp,
        end: pd.Timestamp,
        min_bars: int,
        max_symbols: int | None,
        seed: int,
        database_path: Path,
        corporate_action_gate: bool = True,
        overnight_ratio_min: float = 0.25,
        overnight_ratio_max: float = 4.0,
    ) -> None:
        self.path = path
        self.start = start
        self.end = end
        self.min_bars = min_bars
        self.max_symbols = max_symbols
        self.seed = seed
        self.corporate_action_gate = bool(corporate_action_gate)
        self.overnight_ratio_min = float(overnight_ratio_min)
        self.overnight_ratio_max = float(overnight_ratio_max)
        if not (0.0 < self.overnight_ratio_min < 1.0 < self.overnight_ratio_max):
            raise ValueError(
                "Corporate-action ratio bounds must satisfy 0 < min < 1 < max"
            )
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(database_path))
        self.source_sql = self._build_source_sql(path)
        self.columns = self._resolve_columns()
        self.standardized_sql = self._build_standardized_sql()
        self._create_source_view()
        self.identities = self._load_identities()

    @staticmethod
    def _build_source_sql(path: Path) -> str:
        if path.is_dir():
            parquet_files = sorted(path.rglob("*.parquet"))
            csv_files = sorted(path.rglob("*.csv"))
            if parquet_files:
                paths = ",".join(
                    sql_quote_literal(str(p.resolve())) for p in parquet_files
                )
                return f"read_parquet([{paths}], union_by_name=true)"
            if csv_files:
                # DuckDB can read a list/glob of CSVs via read_csv_auto.
                paths = ",".join(sql_quote_literal(str(p.resolve())) for p in csv_files)
                return f"read_csv_auto([{paths}], union_by_name=true, header=true)"
            raise FileNotFoundError(f"No Parquet or CSV files found under {path}")
        suffix = path.suffix.lower()
        literal = sql_quote_literal(str(path.resolve()))
        if suffix in {".parquet", ".pq"}:
            return f"read_parquet({literal}, union_by_name=true)"
        if suffix in {".csv", ".txt"}:
            return f"read_csv_auto({literal}, header=true)"
        raise ValueError(f"Unsupported data format: {path}")

    def _available_columns(self) -> list[str]:
        description = self.con.execute(
            f"DESCRIBE SELECT * FROM {self.source_sql}"
        ).fetchdf()
        return description["column_name"].astype(str).tolist()

    def _resolve_columns(self) -> DataColumns:
        available = self._available_columns()
        lower_map = {name.lower(): name for name in available}

        def find(canonical: str, required: bool = True) -> str | None:
            for alias in COLUMN_ALIASES[canonical]:
                if alias.lower() in lower_map:
                    return lower_map[alias.lower()]
            if required:
                raise ValueError(
                    f"Required column {canonical!r} not found. Available columns: {available}"
                )
            return None

        security_id = find("security_id")
        if security_id is None:
            raise ValueError("Required column 'security_id' not found")
        symbol = find("symbol", required=False) or security_id
        date_column = find("date")
        open_column = find("open")
        high_column = find("high")
        low_column = find("low")
        close_column = find("close")
        if any(
            column is None
            for column in (
                date_column,
                open_column,
                high_column,
                low_column,
                close_column,
            )
        ):
            raise ValueError("Required OHLC date columns are incomplete")
        assert date_column is not None
        assert open_column is not None
        assert high_column is not None
        assert low_column is not None
        assert close_column is not None
        return DataColumns(
            security_id=security_id,
            symbol=symbol,
            date=date_column,
            open=open_column,
            high=high_column,
            low=low_column,
            close=close_column,
            volume=find("volume", required=False),
            sector=find("sector", required=False),
            industry=find("industry", required=False),
            currency=find("currency", required=False),
        )

    def _build_standardized_sql(self) -> str:
        c = self.columns
        volume_expr = (
            f"TRY_CAST(src.{sql_quote_identifier(c.volume)} AS DOUBLE)"
            if c.volume
            else "NULL::DOUBLE"
        )
        sector_expr = (
            f"CAST(src.{sql_quote_identifier(c.sector)} AS VARCHAR)"
            if c.sector
            else "NULL::VARCHAR"
        )
        industry_expr = (
            f"CAST(src.{sql_quote_identifier(c.industry)} AS VARCHAR)"
            if c.industry
            else "NULL::VARCHAR"
        )
        currency_expr = (
            f"CAST(src.{sql_quote_identifier(c.currency)} AS VARCHAR)"
            if c.currency
            else "'USD'::VARCHAR"
        )
        return f"""
            SELECT
                CAST(src.{sql_quote_identifier(c.security_id)} AS VARCHAR) AS security_id,
                CAST(src.{sql_quote_identifier(c.symbol)} AS VARCHAR) AS symbol,
                TRY_CAST(src.{sql_quote_identifier(c.date)} AS DATE) AS date,
                TRY_CAST(src.{sql_quote_identifier(c.open)} AS DOUBLE) AS open,
                TRY_CAST(src.{sql_quote_identifier(c.high)} AS DOUBLE) AS high,
                TRY_CAST(src.{sql_quote_identifier(c.low)} AS DOUBLE) AS low,
                TRY_CAST(src.{sql_quote_identifier(c.close)} AS DOUBLE) AS close,
                {volume_expr} AS volume,
                {sector_expr} AS sector,
                {industry_expr} AS industry
                ,{currency_expr} AS currency
            FROM {self.source_sql} AS src
        """

    def _create_source_view(self) -> None:
        """Create a cleaned market view and an explicit identity-level QA audit.

        The source file used by Phase 11.4 still contains identities that were
        later rejected by the corporate-action audit. A single missed split or
        reused-ticker discontinuity can manufacture returns of hundreds or
        thousands of percent. Therefore the default is deliberately fail-closed:
        an identity is excluded in full when either its next open or next close,
        relative to the preceding close, falls outside the configured ratio
        interval.
        """
        self.con.execute("DROP VIEW IF EXISTS market_bars")
        self.con.execute("DROP VIEW IF EXISTS _market_bars_base")
        self.con.execute("DROP TABLE IF EXISTS data_quality_exclusions")

        start_literal = sql_quote_literal(str(self.start.date()))
        end_literal = sql_quote_literal(str(self.end.date()))
        self.con.execute(
            f"""
            CREATE TEMP VIEW _market_bars_base AS
            SELECT * FROM ({self.standardized_sql})
            WHERE date BETWEEN CAST({start_literal} AS DATE) AND CAST({end_literal} AS DATE)
              AND date IS NOT NULL
              AND open > 0 AND high > 0 AND low > 0 AND close > 0
              AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
              AND high >= GREATEST(open, close, low)
              AND low <= LEAST(open, close, high)
            """
        )

        if self.corporate_action_gate:
            ratio_min = float(self.overnight_ratio_min)
            ratio_max = float(self.overnight_ratio_max)
            self.con.execute(
                f"""
                CREATE TEMP TABLE data_quality_exclusions AS
                WITH ordered AS (
                    SELECT
                        security_id,
                        symbol,
                        date,
                        open,
                        close,
                        LAG(close) OVER (
                            PARTITION BY security_id
                            ORDER BY date
                        ) AS previous_close
                    FROM _market_bars_base
                ), ratios AS (
                    SELECT
                        *,
                        open / NULLIF(previous_close, 0) AS open_to_previous_close,
                        close / NULLIF(previous_close, 0) AS close_to_previous_close
                    FROM ordered
                    WHERE previous_close IS NOT NULL
                      AND previous_close > 0
                )
                SELECT
                    security_id,
                    ANY_VALUE(symbol) AS symbol,
                    COUNT(*) FILTER (
                        WHERE open_to_previous_close < {ratio_min}
                           OR open_to_previous_close > {ratio_max}
                    ) AS overnight_open_conflict_count,
                    COUNT(*) FILTER (
                        WHERE close_to_previous_close < {ratio_min}
                           OR close_to_previous_close > {ratio_max}
                    ) AS close_jump_conflict_count,
                    MIN(open_to_previous_close) AS minimum_open_ratio,
                    MAX(open_to_previous_close) AS maximum_open_ratio,
                    MIN(close_to_previous_close) AS minimum_close_ratio,
                    MAX(close_to_previous_close) AS maximum_close_ratio,
                    MIN(date) FILTER (
                        WHERE open_to_previous_close < {ratio_min}
                           OR open_to_previous_close > {ratio_max}
                           OR close_to_previous_close < {ratio_min}
                           OR close_to_previous_close > {ratio_max}
                    ) AS first_conflict_date
                FROM ratios
                GROUP BY security_id
                HAVING
                    COUNT(*) FILTER (
                        WHERE open_to_previous_close < {ratio_min}
                           OR open_to_previous_close > {ratio_max}
                           OR close_to_previous_close < {ratio_min}
                           OR close_to_previous_close > {ratio_max}
                    ) > 0
                """
            )
            self.con.execute(
                """
                CREATE TEMP VIEW market_bars AS
                SELECT b.*
                FROM _market_bars_base b
                LEFT JOIN data_quality_exclusions q USING (security_id)
                WHERE q.security_id IS NULL
                """
            )
        else:
            self.con.execute(
                """
                CREATE TEMP TABLE data_quality_exclusions AS
                SELECT
                    NULL::VARCHAR AS security_id,
                    NULL::VARCHAR AS symbol,
                    0::BIGINT AS overnight_open_conflict_count,
                    0::BIGINT AS close_jump_conflict_count,
                    NULL::DOUBLE AS minimum_open_ratio,
                    NULL::DOUBLE AS maximum_open_ratio,
                    NULL::DOUBLE AS minimum_close_ratio,
                    NULL::DOUBLE AS maximum_close_ratio,
                    NULL::DATE AS first_conflict_date
                WHERE FALSE
                """
            )
            self.con.execute(
                "CREATE TEMP VIEW market_bars AS SELECT * FROM _market_bars_base"
            )

    def _load_identities(self) -> pd.DataFrame:
        df = self.con.execute(
            """
            SELECT security_id,
                   ANY_VALUE(symbol) AS symbol,
                   COUNT(*) AS bar_count,
                   MIN(date) AS first_date,
                   MAX(date) AS last_date
            FROM market_bars
            GROUP BY security_id
            HAVING COUNT(*) >= ?
            ORDER BY security_id
            """,
            [self.min_bars],
        ).fetchdf()
        if self.max_symbols and len(df) > self.max_symbols:
            # Deterministic sample; useful for smoke runs without selecting on outcomes.
            rng = random.Random(self.seed)
            indices = list(range(len(df)))
            rng.shuffle(indices)
            df = df.iloc[sorted(indices[: self.max_symbols])].reset_index(drop=True)
        return df

    def iter_batches(self, batch_size: int) -> Iterator[pd.DataFrame]:
        ids = self.identities["security_id"].astype(str).tolist()
        for start in range(0, len(ids), batch_size):
            chunk = ids[start : start + batch_size]
            id_df = pd.DataFrame({"security_id": chunk})
            self.con.register("_selected_ids", id_df)
            try:
                bars = self.con.execute(
                    """
                    SELECT b.*
                    FROM market_bars b
                    INNER JOIN _selected_ids i USING (security_id)
                    ORDER BY b.security_id, b.date
                    """
                ).fetchdf()
            finally:
                self.con.unregister("_selected_ids")
            bars["date"] = pd.to_datetime(bars["date"])
            yield bars

    def quality_exclusions(self) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT *
            FROM data_quality_exclusions
            ORDER BY security_id
            """
        ).fetchdf()

    def metadata(self) -> dict[str, Any]:
        raw_counts = self.con.execute(
            """
            SELECT
                COUNT(DISTINCT security_id) AS identity_count,
                COUNT(*) AS bar_count
            FROM _market_bars_base
            """
        ).fetchone()
        exclusion_row = self.con.execute(
            "SELECT COUNT(*) FROM data_quality_exclusions"
        ).fetchone()
        if raw_counts is None or exclusion_row is None:
            raise RuntimeError("Market-data metadata query returned no row")
        exclusion_count = int(exclusion_row[0])
        return {
            "path": str(self.path.resolve()),
            "columns": dataclasses.asdict(self.columns),
            "raw_identity_count": int(raw_counts[0] or 0),
            "raw_bar_count": int(raw_counts[1] or 0),
            "quality_excluded_identity_count": exclusion_count,
            "identity_count": int(len(self.identities)),
            "bar_count": int(self.identities["bar_count"].sum()),
            "corporate_action_gate": self.corporate_action_gate,
            "overnight_ratio_min": self.overnight_ratio_min,
            "overnight_ratio_max": self.overnight_ratio_max,
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "min_bars": self.min_bars,
            "max_symbols": self.max_symbols,
        }


# ---------------------------------------------------------------------------
# Indicators and signal/trade engine
# ---------------------------------------------------------------------------


class FeatureCache:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self._cache: dict[tuple[Any, ...], np.ndarray] = {}

    def arr(self, column: str) -> np.ndarray:
        return self.frame[column].to_numpy(dtype=float)

    def series(self, column: str) -> pd.Series:
        return self.frame[column].astype(float)

    def sma(self, column: str, period: int) -> np.ndarray:
        key = ("sma", column, period)
        if key not in self._cache:
            self._cache[key] = (
                self.series(column)
                .rolling(period, min_periods=period)
                .mean()
                .to_numpy()
            )
        return self._cache[key]

    def ema(self, column: str, period: int) -> np.ndarray:
        key = ("ema", column, period)
        if key not in self._cache:
            self._cache[key] = (
                self.series(column)
                .ewm(span=period, adjust=False, min_periods=period)
                .mean()
                .to_numpy()
            )
        return self._cache[key]

    def rolling_min(self, column: str, period: int, shift: int = 0) -> np.ndarray:
        key = ("rmin", column, period, shift)
        if key not in self._cache:
            series = self.series(column).shift(shift) if shift else self.series(column)
            self._cache[key] = (
                series.rolling(period, min_periods=period).min().to_numpy()
            )
        return self._cache[key]

    def rolling_max(self, column: str, period: int, shift: int = 0) -> np.ndarray:
        key = ("rmax", column, period, shift)
        if key not in self._cache:
            series = self.series(column).shift(shift) if shift else self.series(column)
            self._cache[key] = (
                series.rolling(period, min_periods=period).max().to_numpy()
            )
        return self._cache[key]

    def rolling_std(self, column: str, period: int) -> np.ndarray:
        key = ("rstd", column, period)
        if key not in self._cache:
            self._cache[key] = (
                self.series(column)
                .rolling(period, min_periods=period)
                .std(ddof=0)
                .to_numpy()
            )
        return self._cache[key]

    def rsi(self, period: int) -> np.ndarray:
        key = ("rsi", period)
        if key not in self._cache:
            close = self.series("close")
            delta = close.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(
                alpha=1.0 / period, adjust=False, min_periods=period
            ).mean()
            avg_loss = loss.ewm(
                alpha=1.0 / period, adjust=False, min_periods=period
            ).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            values = 100.0 - 100.0 / (1.0 + rs)
            values = values.where(avg_loss != 0.0, 100.0)
            values = values.where(avg_gain != 0.0, 0.0)
            self._cache[key] = values.to_numpy()
        return self._cache[key]

    def true_range(self) -> np.ndarray:
        key = ("tr",)
        if key not in self._cache:
            high = self.series("high")
            low = self.series("low")
            prev_close = self.series("close").shift(1)
            tr = pd.concat(
                [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            self._cache[key] = tr.to_numpy()
        return self._cache[key]

    def atr(self, period: int) -> np.ndarray:
        key = ("atr", period)
        if key not in self._cache:
            tr = pd.Series(self.true_range())
            self._cache[key] = (
                tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period)
                .mean()
                .to_numpy()
            )
        return self._cache[key]

    def adx(self, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("adx", period)
        if key not in self._cache:
            high = self.series("high")
            low = self.series("low")
            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
            atr = pd.Series(self.atr(period))
            plus_di = (
                100.0
                * plus_dm.ewm(
                    alpha=1.0 / period, adjust=False, min_periods=period
                ).mean()
                / atr
            )
            minus_di = (
                100.0
                * minus_dm.ewm(
                    alpha=1.0 / period, adjust=False, min_periods=period
                ).mean()
                / atr
            )
            dx = (
                100.0
                * (plus_di - minus_di).abs()
                / (plus_di + minus_di).replace(0.0, np.nan)
            )
            adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
            self._cache[key] = np.vstack(
                [adx.to_numpy(), plus_di.to_numpy(), minus_di.to_numpy()]
            )
        values = self._cache[key]
        return values[0], values[1], values[2]

    def williams_r(self, period: int) -> np.ndarray:
        key = ("willr", period)
        if key not in self._cache:
            highest = self.rolling_max("high", period)
            lowest = self.rolling_min("low", period)
            close = self.arr("close")
            denominator = highest - lowest
            result = (
                -100.0
                * (highest - close)
                / np.where(denominator == 0, np.nan, denominator)
            )
            self._cache[key] = result
        return self._cache[key]

    def macd_hist(self, fast: int, slow: int, signal: int) -> np.ndarray:
        key = ("macd", fast, slow, signal)
        if key not in self._cache:
            fast_ema = self.ema("close", fast)
            slow_ema = self.ema("close", slow)
            macd = fast_ema - slow_ema
            signal_line = (
                pd.Series(macd)
                .ewm(span=signal, adjust=False, min_periods=signal)
                .mean()
                .to_numpy()
            )
            self._cache[key] = macd - signal_line
        return self._cache[key]

    def weekly_macd_hist(self, fast: int, slow: int, signal: int) -> np.ndarray:
        key = ("weekly_macd", fast, slow, signal)
        if key not in self._cache:
            temp = self.frame[["date", "close"]].copy().set_index("date")
            weekly = temp["close"].resample("W-FRI").last().dropna()
            fast_ema = weekly.ewm(span=fast, adjust=False, min_periods=fast).mean()
            slow_ema = weekly.ewm(span=slow, adjust=False, min_periods=slow).mean()
            macd = fast_ema - slow_ema
            signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
            hist = (macd - signal_line).rename("hist")
            mapped = hist.reindex(temp.index, method="ffill")
            self._cache[key] = mapped.to_numpy()
        return self._cache[key]


@dataclass
class TradeBatch:
    entry_dates: np.ndarray
    exit_dates: np.ndarray
    entry_prices: np.ndarray
    exit_prices: np.ndarray
    gross_returns: np.ndarray
    scores: np.ndarray
    durations: np.ndarray
    forced: np.ndarray

    @classmethod
    def empty(cls) -> "TradeBatch":
        return cls(
            entry_dates=np.asarray([], dtype="datetime64[ns]"),
            exit_dates=np.asarray([], dtype="datetime64[ns]"),
            entry_prices=np.asarray([], dtype=float),
            exit_prices=np.asarray([], dtype=float),
            gross_returns=np.asarray([], dtype=float),
            scores=np.asarray([], dtype=float),
            durations=np.asarray([], dtype=int),
            forced=np.asarray([], dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.gross_returns)


def _finalize_trade_lists(lists: tuple[list[Any], ...]) -> TradeBatch:
    if not lists[0]:
        return TradeBatch.empty()
    return TradeBatch(
        entry_dates=np.asarray(lists[0], dtype="datetime64[ns]"),
        exit_dates=np.asarray(lists[1], dtype="datetime64[ns]"),
        entry_prices=np.asarray(lists[2], dtype=float),
        exit_prices=np.asarray(lists[3], dtype=float),
        gross_returns=np.asarray(lists[4], dtype=float),
        scores=np.asarray(lists[5], dtype=float),
        durations=np.asarray(lists[6], dtype=int),
        forced=np.asarray(lists[7], dtype=bool),
    )


def trades_from_signals(
    frame: pd.DataFrame,
    entry_signal: np.ndarray,
    exit_signal: np.ndarray,
    score: np.ndarray | None = None,
    stop_distance: np.ndarray | None = None,
    target_distance: np.ndarray | None = None,
    max_hold: int | None = None,
    force_close_end: bool = True,
) -> TradeBatch:
    """Long-only, non-overlapping, next-open signal execution."""
    n = len(frame)
    if n < 3:
        return TradeBatch.empty()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    entry_signal = np.asarray(entry_signal, dtype=bool)
    exit_signal = np.asarray(exit_signal, dtype=bool)
    score = (
        np.zeros(n, dtype=float) if score is None else np.nan_to_num(score, nan=-1e12)
    )
    entry_signal &= np.isfinite(opens) & np.isfinite(closes)
    exit_signal &= np.isfinite(opens) & np.isfinite(closes)
    entry_indices = np.flatnonzero(entry_signal[:-1])
    exit_indices = np.flatnonzero(exit_signal[:-1])
    if entry_indices.size == 0:
        return TradeBatch.empty()

    lists: tuple[list[Any], ...] = tuple([] for _ in range(8))  # type: ignore[assignment]
    last_exit_exec = -1
    for signal_idx in entry_indices:
        if signal_idx < last_exit_exec:
            continue
        entry_exec = signal_idx + 1
        if (
            entry_exec >= n
            or not math.isfinite(opens[entry_exec])
            or opens[entry_exec] <= 0
        ):
            continue
        search_at = int(np.searchsorted(exit_indices, entry_exec, side="left"))
        forced = False
        if search_at < len(exit_indices):
            exit_exec = int(exit_indices[search_at] + 1)
        elif force_close_end:
            exit_exec = n - 1
            forced = True
        else:
            continue
        if max_hold is not None:
            max_exit = min(n - 1, entry_exec + max_hold)
            if max_exit < exit_exec:
                exit_exec = max_exit
                forced = True
        entry_price = float(opens[entry_exec])
        exit_price = float(opens[exit_exec])

        stop_price = None
        target_price = None
        if stop_distance is not None and math.isfinite(
            float(stop_distance[signal_idx])
        ):
            stop_price = entry_price - float(stop_distance[signal_idx])
        if target_distance is not None and math.isfinite(
            float(target_distance[signal_idx])
        ):
            target_price = entry_price + float(target_distance[signal_idx])

        if stop_price is not None or target_price is not None:
            for j in range(entry_exec, exit_exec + 1):
                stop_hit = stop_price is not None and lows[j] <= stop_price
                target_hit = target_price is not None and highs[j] >= target_price
                if stop_hit and target_hit:
                    assert stop_price is not None
                    # Conservative same-bar assumption.
                    exit_exec = j
                    exit_price = (
                        min(opens[j], stop_price)
                        if opens[j] < stop_price
                        else stop_price
                    )
                    forced = False
                    break
                if stop_hit:
                    assert stop_price is not None
                    exit_exec = j
                    exit_price = (
                        min(opens[j], stop_price)
                        if opens[j] < stop_price
                        else stop_price
                    )
                    forced = False
                    break
                if target_hit:
                    assert target_price is not None
                    exit_exec = j
                    exit_price = (
                        max(opens[j], target_price)
                        if opens[j] > target_price
                        else target_price
                    )
                    forced = False
                    break

        if exit_exec <= entry_exec and exit_price <= 0:
            continue
        gross_return = exit_price / entry_price - 1.0
        if not math.isfinite(gross_return):
            continue
        lists[0].append(dates[entry_exec])
        lists[1].append(dates[exit_exec])
        lists[2].append(entry_price)
        lists[3].append(exit_price)
        lists[4].append(gross_return)
        lists[5].append(float(score[signal_idx]))
        lists[6].append(int(max(exit_exec - entry_exec, 0)))
        lists[7].append(bool(forced))
        last_exit_exec = exit_exec
    return _finalize_trade_lists(lists)


def first_profitable_exit_trades(
    frame: pd.DataFrame,
    entry_signal: np.ndarray,
    score: np.ndarray,
    max_hold: int | None,
) -> TradeBatch:
    n = len(frame)
    if n < 3:
        return TradeBatch.empty()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    opens = frame["open"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    entries = np.flatnonzero(np.asarray(entry_signal, dtype=bool)[:-1])
    lists: tuple[list[Any], ...] = tuple([] for _ in range(8))  # type: ignore[assignment]
    last_exit = -1
    for s in entries:
        if s < last_exit:
            continue
        entry_exec = s + 1
        entry_price = float(opens[entry_exec])
        limit = n - 2
        if max_hold is not None:
            limit = min(limit, entry_exec + max_hold - 1)
        profitable_signal = None
        for j in range(entry_exec, limit + 1):
            if closes[j] > entry_price:
                profitable_signal = j
                break
        forced = False
        if profitable_signal is None:
            exit_exec = min(n - 1, limit + 1)
            forced = True
        else:
            exit_exec = profitable_signal + 1
        exit_price = float(opens[exit_exec])
        gross = exit_price / entry_price - 1.0
        lists[0].append(dates[entry_exec])
        lists[1].append(dates[exit_exec])
        lists[2].append(entry_price)
        lists[3].append(exit_price)
        lists[4].append(gross)
        lists[5].append(float(score[s]))
        lists[6].append(int(exit_exec - entry_exec))
        lists[7].append(forced)
        last_exit = exit_exec
    return _finalize_trade_lists(lists)


def trailing_stop_target_trades(
    frame: pd.DataFrame,
    entry_signal: np.ndarray,
    score: np.ndarray,
    trailing_stop_pct: float,
    target_pct: float,
    max_hold: int,
) -> TradeBatch:
    """Next-open entry with a per-trade trailing stop and fixed percentage target."""
    n = len(frame)
    if n < 3:
        return TradeBatch.empty()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    entries = np.flatnonzero(np.asarray(entry_signal, dtype=bool)[:-1])
    lists: tuple[list[Any], ...] = tuple([] for _ in range(8))  # type: ignore[assignment]
    last_exit = -1
    for signal_idx in entries:
        if signal_idx < last_exit:
            continue
        entry_exec = signal_idx + 1
        entry_price = float(opens[entry_exec])
        if not math.isfinite(entry_price) or entry_price <= 0:
            continue
        highest = entry_price
        target = entry_price * (1.0 + target_pct) if target_pct > 0 else None
        exit_exec = min(n - 1, entry_exec + max_hold)
        exit_price = float(opens[exit_exec])
        forced = True
        for j in range(entry_exec, exit_exec + 1):
            highest = max(highest, float(highs[j]))
            trailing = (
                highest * (1.0 - trailing_stop_pct) if trailing_stop_pct > 0 else None
            )
            stop_hit = trailing is not None and lows[j] <= trailing
            target_hit = target is not None and highs[j] >= target
            if stop_hit and target_hit:
                assert trailing is not None
                exit_exec = j
                exit_price = (
                    min(opens[j], trailing) if opens[j] < trailing else trailing
                )
                forced = False
                break
            if stop_hit:
                assert trailing is not None
                exit_exec = j
                exit_price = (
                    min(opens[j], trailing) if opens[j] < trailing else trailing
                )
                forced = False
                break
            if target_hit:
                assert target is not None
                exit_exec = j
                exit_price = max(opens[j], target) if opens[j] > target else target
                forced = False
                break
        gross = exit_price / entry_price - 1.0
        if not math.isfinite(gross):
            continue
        lists[0].append(dates[entry_exec])
        lists[1].append(dates[exit_exec])
        lists[2].append(entry_price)
        lists[3].append(float(exit_price))
        lists[4].append(float(gross))
        lists[5].append(float(score[signal_idx]))
        lists[6].append(int(exit_exec - entry_exec))
        lists[7].append(bool(forced))
        last_exit = exit_exec
    return _finalize_trade_lists(lists)


def market_structure_pullback_trades(
    frame: pd.DataFrame,
    cache: FeatureCache,
    breakout_lookback: int,
    atr_period: int,
    pullback_atr: float,
    valid_days: int,
    stop_atr: float,
    exit_lookback: int,
) -> TradeBatch:
    n = len(frame)
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    opens = cache.arr("open")
    lows = cache.arr("low")
    closes = cache.arr("close")
    atr = cache.atr(atr_period)
    prior_high = cache.rolling_max("high", breakout_lookback, shift=1)
    exit_high = cache.rolling_max("high", exit_lookback, shift=1)
    breakout_indices = np.flatnonzero((closes > prior_high) & np.isfinite(atr))
    lists: tuple[list[Any], ...] = tuple([] for _ in range(8))  # type: ignore[assignment]
    last_exit = -1
    for b in breakout_indices:
        if b < last_exit:
            continue
        level = lows[b] - pullback_atr * atr[b]
        entry_exec = None
        entry_price = None
        for j in range(b + 1, min(n, b + 1 + valid_days)):
            if lows[j] <= level:
                entry_exec = j
                entry_price = min(opens[j], level) if opens[j] < level else level
                break
        if entry_exec is None or entry_price is None or entry_price <= 0:
            continue
        stop_price = entry_price - stop_atr * atr[b]
        exit_exec = n - 1
        exit_price = float(closes[-1])
        forced = True
        for j in range(entry_exec, n - 1):
            if lows[j] <= stop_price:
                exit_exec = j
                exit_price = (
                    min(opens[j], stop_price) if opens[j] < stop_price else stop_price
                )
                forced = False
                break
            if closes[j] > exit_high[j]:
                exit_exec = j + 1
                exit_price = float(opens[exit_exec])
                forced = False
                break
        gross = exit_price / entry_price - 1.0
        if not math.isfinite(gross):
            continue
        lists[0].append(dates[entry_exec])
        lists[1].append(dates[exit_exec])
        lists[2].append(float(entry_price))
        lists[3].append(float(exit_price))
        lists[4].append(float(gross))
        lists[5].append(float(closes[b] / prior_high[b] - 1.0))
        lists[6].append(int(exit_exec - entry_exec))
        lists[7].append(bool(forced))
        last_exit = exit_exec
    return _finalize_trade_lists(lists)


# ---------------------------------------------------------------------------
# Strategy registry and parameter grids
# ---------------------------------------------------------------------------


SignalBuilder = Callable[[pd.DataFrame, FeatureCache, Mapping[str, Any]], TradeBatch]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    horizon: str
    description: str
    default_params: Mapping[str, Any]
    choices: Mapping[str, Sequence[Any]]
    builder: SignalBuilder
    policy_status: str = "LONG_ONLY_GO"
    notes: str = ""


def one_factor_and_sampled_grid(
    default: Mapping[str, Any],
    choices: Mapping[str, Sequence[Any]],
    preset: str,
    full_cartesian: bool,
    max_variants: int,
    seed: int,
) -> list[dict[str, Any]]:
    keys = list(choices)
    if full_cartesian:
        values = [list(choices[k]) for k in keys]
        result = [
            dict(zip(keys, combination)) for combination in itertools.product(*values)
        ]
        if len(result) > max_variants:
            raise ValueError(
                f"Full Cartesian grid has {len(result)} variants, above --max-variants-per-strategy={max_variants}. "
                "Increase the cap explicitly if you really want this run."
            )
        return result

    variants: list[dict[str, Any]] = [dict(default)]
    if preset == "smoke":
        return variants

    # Every configured value is tested at least once while all other values stay default.
    for key in keys:
        for value in choices[key]:
            candidate = dict(default)
            candidate[key] = value
            variants.append(candidate)

    if preset in {"long", "extreme"}:
        rng = random.Random(seed)
        target = (
            max_variants
            if preset == "extreme"
            else min(max_variants, max(20, len(variants) * 2))
        )
        seen = {stable_json(v) for v in variants}
        attempts = 0
        while len(variants) < target and attempts < target * 100:
            candidate = {key: rng.choice(list(choices[key])) for key in keys}
            marker = stable_json(candidate)
            if marker not in seen:
                seen.add(marker)
                variants.append(candidate)
            attempts += 1
    unique: dict[str, dict[str, Any]] = {}
    for variant in variants:
        unique[stable_json(variant)] = variant
    return list(unique.values())[:max_variants]


def previous_high_exit(cache: FeatureCache) -> np.ndarray:
    close = cache.arr("close")
    prev_high = np.roll(cache.arr("high"), 1)
    prev_high[0] = np.nan
    return close > prev_high


def first_up_close_exit(cache: FeatureCache) -> np.ndarray:
    close = cache.arr("close")
    prev = np.roll(close, 1)
    prev[0] = np.nan
    return close > prev


def build_rsi_oversold(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    rsi = cache.rsi(int(p["rsi_period"]))
    entry = rsi < float(p["entry_threshold"])
    exit_mode = p["exit_mode"]
    if exit_mode == "previous_high":
        exit_signal = previous_high_exit(cache)
    elif exit_mode == "up_close":
        exit_signal = first_up_close_exit(cache)
    else:
        exit_signal = rsi > float(p["exit_rsi"])
    return trades_from_signals(
        frame,
        entry,
        exit_signal,
        score=-rsi,
        max_hold=int(p["max_hold"]),
    )


def build_rsi_adx(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    rsi = cache.rsi(int(p["rsi_period"]))
    adx, plus_di, minus_di = cache.adx(int(p["adx_period"]))
    entry = (rsi < float(p["entry_threshold"])) & (adx > float(p["adx_threshold"]))
    direction = p["direction_filter"]
    if direction == "plus_di":
        entry &= plus_di > minus_di
    elif direction == "sma200":
        entry &= cache.arr("close") > cache.sma("close", 200)
    exit_signal = (
        previous_high_exit(cache)
        if p["exit_mode"] == "previous_high"
        else first_up_close_exit(cache)
    )
    score = (float(p["entry_threshold"]) - rsi) + np.maximum(
        adx - float(p["adx_threshold"]), 0.0
    ) / 10.0
    return trades_from_signals(
        frame, entry, exit_signal, score=score, max_hold=int(p["max_hold"])
    )


def build_rsi_first_profitable(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    rsi = cache.rsi(int(p["rsi_period"]))
    entry = rsi < float(p["entry_threshold"])
    return first_profitable_exit_trades(frame, entry, -rsi, int(p["max_hold"]))


def build_rsi_threshold_exit(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    rsi = cache.rsi(int(p["rsi_period"]))
    entry = rsi < float(p["entry_threshold"])
    exit_signal = rsi > float(p["exit_threshold"])
    return trades_from_signals(
        frame, entry, exit_signal, score=-rsi, max_hold=int(p["max_hold"])
    )


def build_rsi_hist_flip(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    rsi = cache.rsi(int(p["rsi_period"]))
    ema = (
        pd.Series(rsi)
        .ewm(span=int(p["ema_period"]), adjust=False, min_periods=int(p["ema_period"]))
        .mean()
        .to_numpy()
    )
    hist = rsi - ema
    prev = np.roll(hist, 1)
    prev[0] = np.nan
    entry = (hist > 0) & (prev <= 0)
    exit_signal = (hist < 0) & (prev >= 0)
    return trades_from_signals(
        frame, entry, exit_signal, score=hist, max_hold=int(p["max_hold"])
    )


def build_cumulative_rsi(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    rsi = cache.rsi(int(p["rsi_period"]))
    cumulative = (
        pd.Series(rsi)
        .rolling(int(p["sum_days"]), min_periods=int(p["sum_days"]))
        .sum()
        .to_numpy()
    )
    entry = cumulative < float(p["threshold"])
    exit_signal = (
        previous_high_exit(cache)
        if p["exit_mode"] == "previous_high"
        else first_up_close_exit(cache)
    )
    return trades_from_signals(
        frame, entry, exit_signal, score=-cumulative, max_hold=int(p["max_hold"])
    )


def build_consecutive_lows(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    low = cache.arr("low")
    days = int(p["days"])
    entry: np.ndarray = np.ones(len(low), dtype=bool)
    for lag in range(1, days):
        entry &= np.roll(low, lag - 1) < np.roll(low, lag)
    entry[:days] = False
    return trades_from_signals(
        frame,
        entry,
        previous_high_exit(cache),
        score=np.arange(len(low), dtype=float),
        max_hold=int(p["max_hold"]),
    )


def build_consecutive_closes(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    close = cache.arr("close")
    days = int(p["days"])
    entry: np.ndarray = np.ones(len(close), dtype=bool)
    for lag in range(1, days):
        entry &= np.roll(close, lag - 1) < np.roll(close, lag)
    entry[:days] = False
    return trades_from_signals(
        frame,
        entry,
        previous_high_exit(cache),
        score=np.arange(len(close), dtype=float),
        max_hold=int(p["max_hold"]),
    )


def build_return_atr_filter(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    close = cache.arr("close")
    prev = np.roll(close, 1)
    prev[0] = np.nan
    daily_return = close / prev - 1.0
    fast = cache.atr(int(p["fast_atr"]))
    slow = cache.atr(int(p["slow_atr"]))
    entry = daily_return <= -float(p["down_return"])
    if p["regime"] == "compression":
        entry &= fast < slow
    else:
        entry &= fast > slow
    return trades_from_signals(
        frame,
        entry,
        previous_high_exit(cache),
        score=-daily_return,
        max_hold=int(p["max_hold"]),
    )


def build_ema_stretch(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    close = cache.arr("close")
    ema = cache.ema("close", int(p["ema_period"]))
    if p["mode"] == "percent":
        stretch = safe_divide(ema - close, ema)
        entry = np.isfinite(stretch) & (stretch > float(p["threshold"]))
        score = stretch
    else:
        atr = cache.atr(int(p["atr_period"]))
        stretch = safe_divide(ema - close, atr)
        entry = np.isfinite(stretch) & (stretch > float(p["threshold"]))
        score = stretch
    exit_signal = close > ema if p["exit_mode"] == "ema" else previous_high_exit(cache)
    return trades_from_signals(
        frame, entry, exit_signal, score=score, max_hold=int(p["max_hold"])
    )


def build_ema_decline(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    ema = cache.ema("close", int(p["ema_period"]))
    previous = np.roll(ema, 1)
    previous[0] = np.nan
    change = ema / previous - 1.0
    entry = change <= -float(p["decline_threshold"])
    exit_signal = (
        previous_high_exit(cache)
        if p["exit_mode"] == "previous_high"
        else cache.arr("close") > ema
    )
    return trades_from_signals(
        frame, entry, exit_signal, score=-change, max_hold=int(p["max_hold"])
    )


def build_close_below_n_day_low(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    close = cache.arr("close")
    reference = cache.rolling_min("low", int(p["lookback"]), shift=1)
    entry = close < reference
    exit_signal = (
        previous_high_exit(cache)
        if p["exit_mode"] == "previous_high"
        else first_up_close_exit(cache)
    )
    return trades_from_signals(
        frame,
        entry,
        exit_signal,
        score=reference / close - 1.0,
        max_hold=int(p["max_hold"]),
    )


def build_five_bar_contrarian(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    lookback = int(p["lookback"])
    entry = cache.arr("low") < cache.rolling_min("low", lookback, shift=1)
    exit_signal = cache.arr("high") > cache.rolling_max(
        "high", int(p["exit_lookback"]), shift=1
    )
    score = cache.rolling_min("low", lookback, shift=1) / cache.arr("low") - 1.0
    return trades_from_signals(
        frame, entry, exit_signal, score=score, max_hold=int(p["max_hold"])
    )


def build_seven_day_pullback(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    close = cache.arr("close")
    entry = (close < cache.rolling_min("low", int(p["entry_lookback"]), shift=1)) & (
        close > cache.sma("close", int(p["trend_ma"]))
    )
    exit_signal = close > cache.rolling_max("high", int(p["exit_lookback"]), shift=1)
    atr = cache.atr(int(p["atr_period"]))
    score = cache.rolling_min("low", int(p["entry_lookback"]), shift=1) / close - 1.0
    return trades_from_signals(
        frame,
        entry,
        exit_signal,
        score=score,
        stop_distance=atr * float(p["stop_atr"]),
        max_hold=int(p["max_hold"]),
    )


def build_williams_r(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    willr = cache.williams_r(int(p["period"]))
    close = cache.arr("close")
    entry = willr < float(p["entry_threshold"])
    if int(p["trend_ma"]) > 0:
        entry &= close > cache.sma("close", int(p["trend_ma"]))
    exit_signal = willr > float(p["exit_threshold"])
    atr = cache.atr(int(p["atr_period"]))
    return trades_from_signals(
        frame,
        entry,
        exit_signal,
        score=-willr,
        stop_distance=atr * float(p["stop_atr"]),
        max_hold=int(p["max_hold"]),
    )


def build_williams_macd(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    willr = cache.williams_r(int(p["willr_period"]))
    hist = cache.macd_hist(
        int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"])
    )
    hist_low = (
        pd.Series(hist)
        .rolling(int(p["hist_lookback"]), min_periods=int(p["hist_lookback"]))
        .min()
        .to_numpy()
    )
    entry = (willr < float(p["entry_threshold"])) & (hist <= hist_low)
    prev_hist = np.roll(hist, 1)
    prev_hist[0] = np.nan
    exit_signal = hist > prev_hist
    return trades_from_signals(
        frame, entry, exit_signal, score=-willr - hist, max_hold=int(p["max_hold"])
    )


def build_five_bar_breakout(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    entry = cache.arr("high") > cache.rolling_max(
        "high", int(p["entry_lookback"]), shift=1
    )
    exit_signal = cache.arr("low") < cache.rolling_min(
        "low", int(p["exit_lookback"]), shift=1
    )
    score = (
        cache.arr("high") / cache.rolling_max("high", int(p["entry_lookback"]), shift=1)
        - 1.0
    )
    return trades_from_signals(
        frame, entry, exit_signal, score=score, max_hold=int(p["max_hold"])
    )


def build_ma_crossover(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    fast = cache.sma("close", int(p["fast"]))
    slow = cache.sma("close", int(p["slow"]))
    prev_fast = np.roll(fast, 1)
    prev_slow = np.roll(slow, 1)
    prev_fast[0] = prev_slow[0] = np.nan
    entry = (fast > slow) & (prev_fast <= prev_slow)
    exit_signal = (fast < slow) & (prev_fast >= prev_slow)
    return trades_from_signals(
        frame,
        entry,
        exit_signal,
        score=(fast / slow - 1.0),
        max_hold=int(p["max_hold"]),
    )


def build_asym_ma(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    ef = cache.sma("close", int(p["entry_fast"]))
    es = cache.sma("close", int(p["entry_slow"]))
    xf = cache.sma("close", int(p["exit_fast"]))
    xs = cache.sma("close", int(p["exit_slow"]))
    pef = np.roll(ef, 1)
    pes = np.roll(es, 1)
    pxf = np.roll(xf, 1)
    pxs = np.roll(xs, 1)
    for array in (pef, pes, pxf, pxs):
        array[0] = np.nan
    entry = (ef > es) & (pef <= pes)
    exit_signal = (xf < xs) & (pxf >= pxs)
    return trades_from_signals(
        frame, entry, exit_signal, score=(ef / es - 1.0), max_hold=int(p["max_hold"])
    )


def build_ma_channel(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    period = int(p["period"])
    bars = int(p["confirm_bars"])
    upper = cache.sma("high", period)
    lower = cache.sma("low", period)
    low = cache.arr("low")
    high = cache.arr("high")
    above = low > upper
    below = high < lower
    entry = (
        pd.Series(above.astype(int)).rolling(bars, min_periods=bars).sum().to_numpy()
        == bars
    )
    exit_signal = (
        pd.Series(below.astype(int)).rolling(bars, min_periods=bars).sum().to_numpy()
        == bars
    )
    return trades_from_signals(
        frame,
        entry,
        exit_signal,
        score=(low / upper - 1.0),
        max_hold=int(p["max_hold"]),
    )


def build_bollinger_breakout(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    period = int(p["period"])
    close = cache.arr("close")
    mean = cache.sma("close", period)
    std = cache.rolling_std("close", period)
    upper = mean + float(p["entry_sigma"]) * std
    lower = mean - float(p["exit_sigma"]) * std
    entry = close > upper
    exit_signal = close < lower
    atr = cache.atr(int(p["atr_period"]))
    return trades_from_signals(
        frame,
        entry,
        exit_signal,
        score=(close - upper) / np.where(std == 0, np.nan, std),
        stop_distance=atr * float(p["stop_atr"]),
        max_hold=int(p["max_hold"]),
    )


def build_flipper(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    close = cache.arr("close")
    lookback = int(p["lookback"])
    threshold = float(p["threshold"])
    low_ref = cache.rolling_min("low", lookback, shift=1)
    high_ref = cache.rolling_max("high", lookback, shift=1)
    entry = close >= low_ref * (1.0 + threshold)
    exit_signal = close <= high_ref * (1.0 - threshold)
    # Use crossings to avoid repeated signals while already flat.
    prev_entry = np.roll(entry, 1)
    prev_exit = np.roll(exit_signal, 1)
    prev_entry[0] = prev_exit[0] = False
    return trades_from_signals(
        frame,
        entry & ~prev_entry,
        exit_signal & ~prev_exit,
        score=close / low_ref - 1.0,
        max_hold=int(p["max_hold"]),
    )


def build_triple_screen(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    weekly_hist = cache.weekly_macd_hist(
        int(p["macd_fast"]), int(p["macd_slow"]), int(p["macd_signal"])
    )
    ema = cache.ema("close", int(p["ema_period"]))
    bear_power = cache.arr("low") - ema
    prev_bear = np.roll(bear_power, 1)
    prev_high = np.roll(cache.arr("high"), 1)
    prev_bear[0] = prev_high[0] = np.nan
    entry = (
        (weekly_hist > 0)
        & (bear_power < 0)
        & (prev_bear < 0)
        & (bear_power > prev_bear)
        & (cache.arr("high") > prev_high)
    )
    return trailing_stop_target_trades(
        frame,
        entry,
        bear_power - prev_bear,
        float(p["trailing_stop_pct"]),
        float(p["target_pct"]),
        int(p["max_hold"]),
    )


def build_market_structure(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    return market_structure_pullback_trades(
        frame,
        cache,
        int(p["breakout_lookback"]),
        int(p["atr_period"]),
        float(p["pullback_atr"]),
        int(p["valid_days"]),
        float(p["stop_atr"]),
        int(p["exit_lookback"]),
    )


def bearish_engulfing(cache: FeatureCache) -> np.ndarray:
    o = cache.arr("open")
    c = cache.arr("close")
    po = np.roll(o, 1)
    pc = np.roll(c, 1)
    po[0] = pc[0] = np.nan
    return (pc > po) & (c < o) & (o > pc) & (c < po)


def bullish_engulfing(cache: FeatureCache) -> np.ndarray:
    o = cache.arr("open")
    c = cache.arr("close")
    po = np.roll(o, 1)
    pc = np.roll(c, 1)
    po[0] = pc[0] = np.nan
    return (pc < po) & (c > o) & (o < pc) & (c > po)


def build_engulfing(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    entry = (
        bearish_engulfing(cache)
        if p["pattern"] == "bearish_contrarian"
        else bullish_engulfing(cache)
    )
    exit_signal = (
        previous_high_exit(cache)
        if p["exit_mode"] == "previous_high"
        else first_up_close_exit(cache)
    )
    atr = cache.atr(int(p["atr_period"]))
    stop = atr * float(p["stop_atr"]) if float(p["stop_atr"]) > 0 else None
    target = atr * float(p["target_atr"]) if float(p["target_atr"]) > 0 else None
    body = np.abs(cache.arr("close") - cache.arr("open"))
    return trades_from_signals(
        frame,
        entry,
        exit_signal,
        score=body,
        stop_distance=stop,
        target_distance=target,
        max_hold=int(p["max_hold"]),
    )


def build_harami(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    o = cache.arr("open")
    c = cache.arr("close")
    po = np.roll(o, 1)
    pc = np.roll(c, 1)
    po[0] = pc[0] = np.nan
    current_body_high = np.maximum(o, c)
    current_body_low = np.minimum(o, c)
    previous_body_high = np.maximum(po, pc)
    previous_body_low = np.minimum(po, pc)
    entry = (
        (pc < po)
        & (c > o)
        & (current_body_high < previous_body_high)
        & (current_body_low > previous_body_low)
    )
    atr = cache.atr(int(p["atr_period"]))
    return trades_from_signals(
        frame,
        entry,
        previous_high_exit(cache),
        score=(previous_body_high - previous_body_low)
        - (current_body_high - current_body_low),
        stop_distance=atr * float(p["stop_atr"]),
        target_distance=atr * float(p["target_atr"]),
        max_hold=int(p["max_hold"]),
    )


def build_piercing(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    o = cache.arr("open")
    c = cache.arr("close")
    low = cache.arr("low")
    po = np.roll(o, 1)
    pc = np.roll(c, 1)
    plow = np.roll(low, 1)
    for arr in (po, pc, plow):
        arr[0] = np.nan
    midpoint = (po + pc) / 2.0
    entry = (pc < po) & (o < plow) & (c > midpoint) & (c < po)
    atr = cache.atr(int(p["atr_period"]))
    return trades_from_signals(
        frame,
        entry,
        previous_high_exit(cache),
        score=(c - midpoint) / np.where(atr == 0, np.nan, atr),
        stop_distance=atr * float(p["stop_atr"]),
        target_distance=atr * float(p["target_atr"]),
        max_hold=int(p["max_hold"]),
    )


def build_month_end(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    symbol = str(frame["symbol"].iloc[-1]).upper()
    allowed = {str(x).upper() for x in p["symbols"]}
    if symbol not in allowed:
        return TradeBatch.empty()
    dates = frame["date"]
    next_month = dates.shift(-1).dt.month
    month_end = dates.dt.month != next_month
    close = cache.arr("close")
    entry = month_end.to_numpy() & (close > cache.sma("close", int(p["trend_ma"])))
    return trades_from_signals(
        frame,
        entry,
        previous_high_exit(cache),
        score=np.ones(len(frame)),
        max_hold=int(p["max_hold"]),
    )


def build_thursday_dip(
    frame: pd.DataFrame, cache: FeatureCache, p: Mapping[str, Any]
) -> TradeBatch:
    symbol = str(frame["symbol"].iloc[-1]).upper()
    allowed = {str(x).upper() for x in p["symbols"]}
    if symbol not in allowed:
        return TradeBatch.empty()
    close = cache.arr("close")
    weekday = frame["date"].dt.weekday.to_numpy()
    entry = (weekday == int(p["weekday"])) & (
        close < cache.sma("close", int(p["ma_period"]))
    )
    return trades_from_signals(
        frame,
        entry,
        previous_high_exit(cache),
        score=cache.sma("close", int(p["ma_period"])) - close,
        max_hold=int(p["max_hold"]),
    )


def params_valid(strategy: str, p: Mapping[str, Any]) -> bool:
    """Reject structurally nonsensical sampled combinations before any backtest."""
    if strategy == "ma_crossover":
        return int(p["fast"]) < int(p["slow"])
    if strategy == "asymmetric_ma_crossover":
        return int(p["entry_fast"]) < int(p["entry_slow"]) and int(
            p["exit_fast"]
        ) < int(p["exit_slow"])
    if strategy in {"williams_macd", "triple_screen"}:
        return int(p["macd_fast"]) < int(p["macd_slow"])
    if strategy in {"return_atr_compression", "return_atr_expansion"}:
        return int(p["fast_atr"]) < int(p["slow_atr"])
    if strategy == "rsi_threshold_exit":
        return float(p["entry_threshold"]) < float(p["exit_threshold"])
    if strategy == "williams_r":
        return float(p["entry_threshold"]) < float(p["exit_threshold"])
    if strategy == "ema_stretch_percent":
        return p["mode"] == "percent" and 0 < float(p["threshold"]) <= 0.10
    if strategy == "ema_stretch_atr":
        return p["mode"] == "atr" and 0 < float(p["threshold"]) <= 5.0
    return True


def strategy_registry() -> list[StrategySpec]:
    """Mechanically specified long-only strategies from the supplied transcripts."""
    return [
        StrategySpec(
            "rsi_oversold",
            "mean_reversion",
            "short",
            "RSI oversold entry with next-open execution and fast recovery exit.",
            {
                "rsi_period": 2,
                "entry_threshold": 15,
                "exit_mode": "previous_high",
                "exit_rsi": 75,
                "max_hold": 20,
            },
            {
                "rsi_period": [2, 3, 5, 7],
                "entry_threshold": [5, 10, 15, 20, 30],
                "exit_mode": ["previous_high", "up_close", "rsi"],
                "exit_rsi": [60, 70, 75, 80],
                "max_hold": [5, 10, 20, 40],
            },
            build_rsi_oversold,
        ),
        StrategySpec(
            "rsi_adx",
            "mean_reversion_filtered",
            "short",
            "RSI oversold plus ADX trend-strength filter.",
            {
                "rsi_period": 2,
                "entry_threshold": 15,
                "adx_period": 5,
                "adx_threshold": 35,
                "direction_filter": "none",
                "exit_mode": "previous_high",
                "max_hold": 20,
            },
            {
                "rsi_period": [2, 3, 5],
                "entry_threshold": [5, 10, 15, 20, 30],
                "adx_period": [3, 5, 7, 10],
                "adx_threshold": [20, 25, 30, 35, 40, 45],
                "direction_filter": ["none", "plus_di", "sma200"],
                "exit_mode": ["previous_high", "up_close"],
                "max_hold": [5, 10, 20, 40],
            },
            build_rsi_adx,
        ),
        StrategySpec(
            "rsi_first_profitable_close",
            "mean_reversion",
            "short_to_unbounded",
            "Oversold entry held until the first close above entry; max-hold makes censoring explicit.",
            {"rsi_period": 2, "entry_threshold": 20, "max_hold": 60},
            {
                "rsi_period": [2, 3, 5],
                "entry_threshold": [5, 10, 15, 20, 30],
                "max_hold": [10, 20, 40, 60, 120, 252],
            },
            build_rsi_first_profitable,
            notes="Historically prone to hidden long-duration losers; treat forced exits as a red flag.",
        ),
        StrategySpec(
            "rsi_threshold_exit",
            "mean_reversion",
            "short_to_medium",
            "Oversold RSI entry; exit only after RSI reaches a high threshold.",
            {
                "rsi_period": 2,
                "entry_threshold": 15,
                "exit_threshold": 75,
                "max_hold": 120,
            },
            {
                "rsi_period": [2, 3, 5],
                "entry_threshold": [5, 10, 15, 20, 30],
                "exit_threshold": [60, 70, 75, 80, 90],
                "max_hold": [20, 40, 60, 120, 252],
            },
            build_rsi_threshold_exit,
        ),
        StrategySpec(
            "rsi_ema_hist_flip",
            "oscillator_flip",
            "short_to_medium",
            "Long-only interpretation of RSI minus its EMA crossing above/below zero.",
            {"rsi_period": 3, "ema_period": 5, "max_hold": 60},
            {
                "rsi_period": [2, 3, 5, 7],
                "ema_period": [3, 5, 7, 10],
                "max_hold": [20, 40, 60, 120],
            },
            build_rsi_hist_flip,
            notes="Transcript comment was not a complete strategy; this is an explicit long/cash interpretation.",
        ),
        StrategySpec(
            "cumulative_rsi",
            "mean_reversion",
            "short",
            "Cumulative short-RSI weakness over multiple days.",
            {
                "rsi_period": 2,
                "sum_days": 2,
                "threshold": 20,
                "exit_mode": "previous_high",
                "max_hold": 20,
            },
            {
                "rsi_period": [2, 3, 5],
                "sum_days": [2, 3, 4],
                "threshold": [10, 15, 20, 25, 30, 40],
                "exit_mode": ["previous_high", "up_close"],
                "max_hold": [5, 10, 20, 40],
            },
            build_cumulative_rsi,
        ),
        StrategySpec(
            "consecutive_lower_lows",
            "price_action_mean_reversion",
            "short",
            "Buy after several consecutively lower lows.",
            {"days": 3, "max_hold": 20},
            {"days": [2, 3, 4, 5], "max_hold": [5, 10, 20, 40]},
            build_consecutive_lows,
        ),
        StrategySpec(
            "consecutive_lower_closes",
            "price_action_mean_reversion",
            "short",
            "Buy after several consecutively lower closes.",
            {"days": 3, "max_hold": 20},
            {"days": [2, 3, 4, 5], "max_hold": [5, 10, 20, 40]},
            build_consecutive_closes,
        ),
        StrategySpec(
            "return_atr_compression",
            "volatility_mean_reversion",
            "short",
            "Buy a negative day only when short ATR is below slow ATR.",
            {
                "down_return": 0.01,
                "fast_atr": 5,
                "slow_atr": 10,
                "regime": "compression",
                "max_hold": 20,
            },
            {
                "down_return": [0.005, 0.01, 0.015, 0.02, 0.03],
                "fast_atr": [3, 5, 7],
                "slow_atr": [10, 14, 20],
                "regime": ["compression"],
                "max_hold": [5, 10, 20, 40],
            },
            build_return_atr_filter,
        ),
        StrategySpec(
            "return_atr_expansion",
            "volatility_mean_reversion",
            "short",
            "Buy a negative day while short ATR exceeds slow ATR.",
            {
                "down_return": 0.01,
                "fast_atr": 5,
                "slow_atr": 10,
                "regime": "expansion",
                "max_hold": 20,
            },
            {
                "down_return": [0.005, 0.01, 0.015, 0.02, 0.03],
                "fast_atr": [3, 5, 7],
                "slow_atr": [10, 14, 20],
                "regime": ["expansion"],
                "max_hold": [5, 10, 20, 40],
            },
            build_return_atr_filter,
        ),
        StrategySpec(
            "ema_stretch_percent",
            "mean_reversion",
            "short",
            "Buy a fixed percentage stretch below an EMA.",
            {
                "ema_period": 5,
                "mode": "percent",
                "threshold": 0.01,
                "atr_period": 5,
                "exit_mode": "previous_high",
                "max_hold": 20,
            },
            {
                "ema_period": [3, 5, 10, 20],
                "mode": ["percent"],
                "threshold": [0.005, 0.006, 0.01, 0.015, 0.02, 0.03],
                "atr_period": [5],
                "exit_mode": ["previous_high", "ema"],
                "max_hold": [5, 10, 20, 40],
            },
            build_ema_stretch,
        ),
        StrategySpec(
            "ema_stretch_atr",
            "mean_reversion",
            "short",
            "Buy an ATR-normalized stretch below an EMA.",
            {
                "ema_period": 5,
                "mode": "atr",
                "threshold": 0.5,
                "atr_period": 5,
                "exit_mode": "previous_high",
                "max_hold": 20,
            },
            {
                "ema_period": [3, 5, 10, 20],
                "mode": ["atr"],
                "threshold": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                "atr_period": [3, 5, 10, 14],
                "exit_mode": ["previous_high", "ema"],
                "max_hold": [5, 10, 20, 40],
            },
            build_ema_stretch,
        ),
        StrategySpec(
            "ema_decline",
            "mean_reversion",
            "short",
            "Buy after a short EMA declines by a specified one-day percentage.",
            {
                "ema_period": 5,
                "decline_threshold": 0.005,
                "exit_mode": "previous_high",
                "max_hold": 20,
            },
            {
                "ema_period": [3, 5, 10, 20],
                "decline_threshold": [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02],
                "exit_mode": ["previous_high", "ema"],
                "max_hold": [5, 10, 20, 40],
            },
            build_ema_decline,
        ),
        StrategySpec(
            "close_below_n_day_low",
            "price_action_mean_reversion",
            "short",
            "Buy when the close falls below the previous N-day low.",
            {"lookback": 5, "exit_mode": "previous_high", "max_hold": 20},
            {
                "lookback": [3, 5, 7, 10, 14, 20],
                "exit_mode": ["previous_high", "up_close"],
                "max_hold": [5, 10, 20, 40, 60],
            },
            build_close_below_n_day_low,
        ),
        StrategySpec(
            "five_bar_contrarian",
            "price_action_mean_reversion",
            "short",
            "Buy a downside break of the prior N-bar low; exit on upside break.",
            {"lookback": 5, "exit_lookback": 5, "max_hold": 40},
            {
                "lookback": [3, 5, 7, 10, 14],
                "exit_lookback": [3, 5, 7, 10, 14],
                "max_hold": [10, 20, 40, 60],
            },
            build_five_bar_contrarian,
        ),
        StrategySpec(
            "seven_day_pullback",
            "trend_filtered_mean_reversion",
            "short",
            "Close below prior N-day low while above a long moving average; exit above prior high.",
            {
                "entry_lookback": 7,
                "exit_lookback": 7,
                "trend_ma": 200,
                "atr_period": 20,
                "stop_atr": 2.0,
                "max_hold": 60,
            },
            {
                "entry_lookback": [5, 7, 10, 14],
                "exit_lookback": [5, 7, 10, 14],
                "trend_ma": [100, 150, 200, 250],
                "atr_period": [10, 14, 20],
                "stop_atr": [1.0, 1.5, 2.0, 2.5, 3.0],
                "max_hold": [20, 40, 60, 120],
            },
            build_seven_day_pullback,
        ),
        StrategySpec(
            "williams_r",
            "trend_filtered_mean_reversion",
            "short",
            "Williams %R oversold entry with optional long-term trend filter.",
            {
                "period": 5,
                "entry_threshold": -90,
                "exit_threshold": -20,
                "trend_ma": 200,
                "atr_period": 20,
                "stop_atr": 2.0,
                "max_hold": 60,
            },
            {
                "period": [3, 5, 7, 10, 14],
                "entry_threshold": [-95, -90, -85, -80],
                "exit_threshold": [-30, -20, -10],
                "trend_ma": [0, 100, 150, 200, 250],
                "atr_period": [10, 14, 20],
                "stop_atr": [1.0, 1.5, 2.0, 2.5],
                "max_hold": [20, 40, 60, 120],
            },
            build_williams_r,
        ),
        StrategySpec(
            "williams_macd",
            "filtered_mean_reversion",
            "short",
            "Williams %R oversold plus a locally depressed MACD histogram; exit on histogram improvement.",
            {
                "willr_period": 5,
                "entry_threshold": -90,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "hist_lookback": 5,
                "max_hold": 60,
            },
            {
                "willr_period": [3, 5, 7, 10],
                "entry_threshold": [-95, -90, -85, -80],
                "macd_fast": [8, 12],
                "macd_slow": [20, 26, 35],
                "macd_signal": [5, 9],
                "hist_lookback": [3, 5, 7, 10],
                "max_hold": [20, 40, 60, 120],
            },
            build_williams_macd,
        ),
        StrategySpec(
            "n_bar_breakout",
            "trend_following",
            "medium",
            "Buy an upside N-bar breakout; exit on downside N-bar break.",
            {"entry_lookback": 5, "exit_lookback": 5, "max_hold": 504},
            {
                "entry_lookback": [3, 5, 10, 20, 50],
                "exit_lookback": [3, 5, 10, 20, 50],
                "max_hold": [60, 120, 252, 504],
            },
            build_five_bar_breakout,
        ),
        StrategySpec(
            "ma_crossover",
            "trend_following",
            "long",
            "Symmetric simple moving-average crossover.",
            {"fast": 50, "slow": 200, "max_hold": 2520},
            {
                "fast": [20, 30, 50, 70, 100, 110],
                "slow": [100, 150, 200, 210, 230, 250],
                "max_hold": [504, 1260, 2520],
            },
            build_ma_crossover,
        ),
        StrategySpec(
            "asymmetric_ma_crossover",
            "trend_following",
            "long",
            "Entry and exit use different MA pairs to hold winners longer.",
            {
                "entry_fast": 70,
                "entry_slow": 210,
                "exit_fast": 110,
                "exit_slow": 230,
                "max_hold": 2520,
            },
            {
                "entry_fast": [30, 50, 70, 100],
                "entry_slow": [150, 200, 210, 250],
                "exit_fast": [70, 90, 110, 130],
                "exit_slow": [200, 230, 250, 300],
                "max_hold": [504, 1260, 2520],
            },
            build_asym_ma,
        ),
        StrategySpec(
            "ma_channel",
            "trend_following",
            "medium",
            "Require several bars fully above/below a moving-average high/low channel.",
            {"period": 10, "confirm_bars": 5, "max_hold": 1260},
            {
                "period": [5, 10, 15, 20, 30],
                "confirm_bars": [2, 3, 5, 7, 10],
                "max_hold": [120, 252, 504, 1260],
            },
            build_ma_channel,
        ),
        StrategySpec(
            "bollinger_breakout",
            "trend_following",
            "medium",
            "Buy a rare upper-band breakout; exit on a lower-band break.",
            {
                "period": 100,
                "entry_sigma": 3.0,
                "exit_sigma": 1.0,
                "atr_period": 20,
                "stop_atr": 3.0,
                "max_hold": 1260,
            },
            {
                "period": [50, 75, 100, 150, 200],
                "entry_sigma": [1.5, 2.0, 2.5, 3.0],
                "exit_sigma": [0.5, 1.0, 1.5, 2.0],
                "atr_period": [14, 20, 30],
                "stop_atr": [0.0, 2.0, 3.0, 4.0],
                "max_hold": [120, 252, 504, 1260],
            },
            build_bollinger_breakout,
        ),
        StrategySpec(
            "percent_flipper",
            "trend_following",
            "medium",
            "Buy X% above a rolling low and exit X% below a rolling high.",
            {"lookback": 50, "threshold": 0.20, "max_hold": 1260},
            {
                "lookback": [20, 30, 50, 75, 100],
                "threshold": [0.10, 0.15, 0.20, 0.25, 0.30],
                "max_hold": [120, 252, 504, 1260],
            },
            build_flipper,
        ),
        StrategySpec(
            "triple_screen",
            "multi_timeframe_trend",
            "medium_to_long",
            "Weekly MACD trend, daily Bear Power exhaustion, price confirmation.",
            {
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "ema_period": 13,
                "trailing_stop_pct": 0.20,
                "target_pct": 0.30,
                "max_hold": 504,
            },
            {
                "macd_fast": [8, 12],
                "macd_slow": [20, 26, 35],
                "macd_signal": [5, 9],
                "ema_period": [8, 13, 21],
                "trailing_stop_pct": [0.10, 0.15, 0.20, 0.25],
                "target_pct": [0.15, 0.20, 0.30, 0.40],
                "max_hold": [120, 252, 504, 756],
            },
            build_triple_screen,
        ),
        StrategySpec(
            "market_structure_atr_pullback",
            "breakout_pullback",
            "medium",
            "After a major breakout, buy a volatility-scaled retracement for a limited time.",
            {
                "breakout_lookback": 63,
                "atr_period": 20,
                "pullback_atr": 2.0,
                "valid_days": 10,
                "stop_atr": 2.0,
                "exit_lookback": 5,
            },
            {
                "breakout_lookback": [40, 50, 63, 90, 126],
                "atr_period": [10, 14, 20, 30],
                "pullback_atr": [0.5, 1.0, 1.5, 2.0, 2.5],
                "valid_days": [5, 10, 15, 20],
                "stop_atr": [1.0, 1.5, 2.0, 2.5, 3.0],
                "exit_lookback": [3, 5, 7, 10],
            },
            build_market_structure,
        ),
        StrategySpec(
            "bearish_engulfing_contrarian_long",
            "candlestick_reversal",
            "short",
            "Buy, rather than short, a textbook bearish engulfing pattern.",
            {
                "pattern": "bearish_contrarian",
                "exit_mode": "previous_high",
                "atr_period": 20,
                "stop_atr": 0.0,
                "target_atr": 0.0,
                "max_hold": 20,
            },
            {
                "pattern": ["bearish_contrarian"],
                "exit_mode": ["previous_high", "up_close"],
                "atr_period": [10, 14, 20],
                "stop_atr": [0.0, 1.0, 1.5, 2.0],
                "target_atr": [0.0, 1.0, 1.5, 2.0, 3.0],
                "max_hold": [5, 10, 20, 40],
            },
            build_engulfing,
        ),
        StrategySpec(
            "bullish_engulfing_long",
            "candlestick_reversal",
            "short",
            "Conventional bullish engulfing reversal entry.",
            {
                "pattern": "bullish",
                "exit_mode": "previous_high",
                "atr_period": 20,
                "stop_atr": 0.0,
                "target_atr": 0.0,
                "max_hold": 20,
            },
            {
                "pattern": ["bullish"],
                "exit_mode": ["previous_high", "up_close"],
                "atr_period": [10, 14, 20],
                "stop_atr": [0.0, 1.0, 1.5, 2.0],
                "target_atr": [0.0, 1.0, 1.5, 2.0, 3.0],
                "max_hold": [5, 10, 20, 40],
            },
            build_engulfing,
        ),
        StrategySpec(
            "bullish_harami",
            "candlestick_reversal",
            "short",
            "Bullish body contained inside the previous bearish body.",
            {"atr_period": 20, "stop_atr": 2.0, "target_atr": 2.0, "max_hold": 40},
            {
                "atr_period": [10, 14, 20, 30],
                "stop_atr": [0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
                "target_atr": [0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
                "max_hold": [5, 10, 20, 40, 60],
            },
            build_harami,
        ),
        StrategySpec(
            "piercing_line",
            "candlestick_reversal",
            "short",
            "Gap-down bullish recovery closing above the prior bearish-body midpoint.",
            {"atr_period": 20, "stop_atr": 2.0, "target_atr": 2.0, "max_hold": 40},
            {
                "atr_period": [10, 14, 20, 30],
                "stop_atr": [0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
                "target_atr": [0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
                "max_hold": [5, 10, 20, 40, 60],
            },
            build_piercing,
        ),
        StrategySpec(
            "spy_month_end",
            "seasonality",
            "short",
            "SPY last-trading-day-of-month entry above its long MA.",
            {"symbols": ["SPY"], "trend_ma": 200, "max_hold": 10},
            {
                "symbols": [["SPY"]],
                "trend_ma": [100, 150, 200, 250],
                "max_hold": [3, 5, 10, 20],
            },
            build_month_end,
            notes="Runs only when SPY exists in the supplied dataset.",
        ),
        StrategySpec(
            "tlt_thursday_dip",
            "seasonality_bond",
            "short",
            "TLT Thursday dip below a short MA.",
            {"symbols": ["TLT"], "weekday": 3, "ma_period": 5, "max_hold": 10},
            {
                "symbols": [["TLT"]],
                "weekday": [0, 1, 2, 3, 4],
                "ma_period": [3, 5, 7, 10],
                "max_hold": [3, 5, 10, 20],
            },
            build_thursday_dip,
            policy_status="BLOCKED_SHARIAH_BOND_PRODUCT",
            notes="Disabled under the default Shariah policy; research-only under --policy all.",
        ),
    ]


BLOCKED_TRANSCRIPT_ITEMS = [
    {
        "name": "bearish_engulfing_short",
        "status": "BLOCKED_LONG_ONLY_POLICY",
        "reason": "Short selling is outside the user's contract.",
    },
    {
        "name": "supply_demand_discretionary",
        "status": "SPECIFICATION_BLOCKED",
        "reason": "Impulse, consolidation, swing validation and zone expiry were not mechanical enough.",
    },
    {
        "name": "intraday_breakout_long_short",
        "status": "SPECIFICATION_AND_POLICY_BLOCKED",
        "reason": "Incomplete rules and includes short/futures execution.",
    },
    {
        "name": "inverse_volatility_multi_asset",
        "status": "SPECIAL_PORTFOLIO_DATA_REQUIRED",
        "reason": "Requires a separate, explicitly reviewed multi-asset ETF universe; TLT/bonds are excluded by default.",
    },
]


# ---------------------------------------------------------------------------
# Cross-sectional rotational momentum
# ---------------------------------------------------------------------------


ROTATIONAL_DEFAULT: dict[str, Any] = {
    "lookback": 252,
    "skip_days": 0,
    "trend_ma": 200,
    "top_n": 15,
    "require_positive_momentum": True,
}
ROTATIONAL_CHOICES: dict[str, Sequence[Any]] = {
    "lookback": [126, 189, 252],
    "skip_days": [0, 21],
    "trend_ma": [100, 150, 200, 250],
    "top_n": [5, 10, 15, 20],
    "require_positive_momentum": [True, False],
}


def rebuild_monthly_feature_table(source: MarketDataSource) -> int:
    """Build one row per security/month with causal next-open holding endpoints."""
    con = source.con
    con.execute("DROP TABLE IF EXISTS monthly_features")
    momentum_columns = [
        f"mom_{lookback}_{skip} DOUBLE"
        for lookback in ROTATIONAL_CHOICES["lookback"]
        for skip in ROTATIONAL_CHOICES["skip_days"]
    ]
    sma_columns = [f"sma_{period} DOUBLE" for period in ROTATIONAL_CHOICES["trend_ma"]]
    con.execute(
        f"""
        CREATE TABLE monthly_features (
            security_id VARCHAR,
            symbol VARCHAR,
            sector VARCHAR,
            signal_date DATE,
            entry_date DATE,
            exit_date DATE,
            entry_open DOUBLE,
            exit_open DOUBLE,
            signal_close DOUBLE,
            dollar_volume_20 DOUBLE,
            {", ".join(momentum_columns + sma_columns)}
        )
        """
    )
    inserted = 0
    for bars in source.iter_batches(100):
        records: list[dict[str, Any]] = []
        for security_id, frame in bars.groupby("security_id", sort=False):
            frame = (
                frame.sort_values("date")
                .drop_duplicates("date", keep="last")
                .reset_index(drop=True)
            )
            if len(frame) < 260:
                continue
            cache = FeatureCache(frame)
            dates = frame["date"].to_numpy(dtype="datetime64[ns]")
            opens = cache.arr("open")
            closes = cache.arr("close")
            volume = (
                cache.arr("volume")
                if frame["volume"].notna().any()
                else np.full(len(frame), np.nan)
            )
            dollar_volume = (
                pd.Series(closes * volume)
                .rolling(20, min_periods=20)
                .median()
                .to_numpy()
            )
            month_codes = frame["date"].dt.to_period("M").astype(str).to_numpy()
            month_end_indices = np.flatnonzero(month_codes[:-1] != month_codes[1:])
            if len(month_end_indices) < 2:
                continue
            sma_values = {
                period: cache.sma("close", period)
                for period in ROTATIONAL_CHOICES["trend_ma"]
            }
            symbol = str(frame["symbol"].iloc[-1])
            sector = (
                None
                if frame["sector"].isna().all()
                else str(frame["sector"].dropna().iloc[-1])
            )
            for position in range(len(month_end_indices) - 1):
                signal_idx = int(month_end_indices[position])
                next_signal_idx = int(month_end_indices[position + 1])
                entry_idx = signal_idx + 1
                exit_idx = next_signal_idx + 1
                if exit_idx >= len(frame):
                    continue
                row: dict[str, Any] = {
                    "security_id": str(security_id),
                    "symbol": symbol,
                    "sector": sector,
                    "signal_date": pd.Timestamp(dates[signal_idx]).date(),
                    "entry_date": pd.Timestamp(dates[entry_idx]).date(),
                    "exit_date": pd.Timestamp(dates[exit_idx]).date(),
                    "entry_open": float(opens[entry_idx]),
                    "exit_open": float(opens[exit_idx]),
                    "signal_close": float(closes[signal_idx]),
                    "dollar_volume_20": float(dollar_volume[signal_idx])
                    if math.isfinite(float(dollar_volume[signal_idx]))
                    else None,
                }
                for lookback in ROTATIONAL_CHOICES["lookback"]:
                    for skip in ROTATIONAL_CHOICES["skip_days"]:
                        numerator_idx = signal_idx - skip
                        denominator_idx = signal_idx - lookback
                        value = None
                        if (
                            denominator_idx >= 0
                            and numerator_idx >= 0
                            and closes[denominator_idx] > 0
                        ):
                            candidate = (
                                closes[numerator_idx] / closes[denominator_idx] - 1.0
                            )
                            value = (
                                float(candidate)
                                if math.isfinite(float(candidate))
                                else None
                            )
                        row[f"mom_{lookback}_{skip}"] = value
                for period, values in sma_values.items():
                    candidate = float(values[signal_idx])
                    row[f"sma_{period}"] = (
                        candidate if math.isfinite(candidate) else None
                    )
                records.append(row)
        if records:
            batch = pd.DataFrame.from_records(records)
            con.register("_monthly_append", batch)
            try:
                con.execute(
                    "INSERT INTO monthly_features SELECT * FROM _monthly_append"
                )
            finally:
                con.unregister("_monthly_append")
            inserted += len(batch)
            print(f"[monthly] features={inserted:,}")
    return inserted


def rotational_grid(config: "RunConfig | V2Config") -> list[dict[str, Any]]:
    return one_factor_and_sampled_grid(
        ROTATIONAL_DEFAULT,
        ROTATIONAL_CHOICES,
        config.preset,
        config.full_cartesian,
        config.max_variants_per_strategy,
        config.seed + 770_001,
    )


def evaluate_rotational_variants(
    source: MarketDataSource,
    config: "RunConfig",
    side_cost: float,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[list[VariantState], VariantState | None, pd.DataFrame]:
    states: list[VariantState] = []
    selected_by_variant: dict[str, pd.DataFrame] = {}
    for params in rotational_grid(config):
        variant_id = make_variant_id("rotational_momentum", params)
        state = VariantState(
            strategy="rotational_momentum",
            family="cross_sectional_momentum",
            horizon="long",
            params=dict(params),
            variant_id=variant_id,
        )
        momentum_col = f"mom_{int(params['lookback'])}_{int(params['skip_days'])}"
        sma_col = f"sma_{int(params['trend_ma'])}"
        positive_clause = (
            f"AND {momentum_col} > 0"
            if bool(params["require_positive_momentum"])
            else ""
        )
        selected = source.con.execute(
            f"""
            WITH eligible AS (
                SELECT *, {momentum_col} AS momentum
                FROM monthly_features
                WHERE {momentum_col} IS NOT NULL
                  AND {sma_col} IS NOT NULL
                  AND signal_close > {sma_col}
                  AND entry_open > 0 AND exit_open > 0
                  {positive_clause}
            ), ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY signal_date
                           ORDER BY momentum DESC, dollar_volume_20 DESC NULLS LAST, security_id
                       ) AS rank_number
                FROM eligible
            )
            SELECT *
            FROM ranked
            WHERE rank_number <= ?
            ORDER BY signal_date, rank_number
            """,
            [int(params["top_n"])],
        ).fetchdf()
        if not selected.empty:
            selected["entry_date"] = pd.to_datetime(selected["entry_date"])
            gross = (
                selected["exit_open"].to_numpy(float)
                / selected["entry_open"].to_numpy(float)
                - 1.0
            )
            position_notional = config.initial_capital / int(params["top_n"])
            fixed_round_trip = 2.0 * config.fixed_fee_eur / position_notional
            net = gross - 2.0 * side_cost - fixed_round_trip
            periods = np.asarray(
                [
                    period_for_date(value, train_end, validation_end)
                    for value in selected["entry_date"]
                ],
                dtype=object,
            )
            durations = (
                pd.to_datetime(selected["exit_date"])
                - pd.to_datetime(selected["entry_date"])
            ).dt.days.to_numpy(int)
            forced: np.ndarray = np.zeros(len(selected), dtype=bool)
            dates = selected["entry_date"].to_numpy(dtype="datetime64[ns]")
            for period in PERIOD_NAMES:
                mask = periods == period
                if np.any(mask):
                    state.periods[period].update(
                        dates[mask], net[mask], durations[mask], forced[mask]
                    )
        states.append(state)
        selected_by_variant[variant_id] = selected
    if not states:
        return [], None, pd.DataFrame()
    states.sort(
        key=lambda state: variant_selection_score(
            state, max(12, config.min_validation_trades)
        ),
        reverse=True,
    )
    winner = states[0]
    return states, winner, selected_by_variant[winner.variant_id]


def merge_rotational_holding_episodes(
    selected: pd.DataFrame, top_n: int, variant_id: str
) -> pd.DataFrame:
    """Keep names that remain selected in consecutive months instead of churning them."""
    if selected.empty:
        return pd.DataFrame()
    episodes: list[dict[str, Any]] = []
    for security_id, group in selected.sort_values(
        ["security_id", "entry_date"]
    ).groupby("security_id"):
        current: dict[str, Any] | None = None
        for _, row in group.iterrows():
            entry_date = pd.Timestamp(row["entry_date"])
            exit_date = pd.Timestamp(row["exit_date"])
            if current is not None and entry_date <= current["exit_date"]:
                current["exit_date"] = max(current["exit_date"], exit_date)
                current["exit_price"] = float(row["exit_open"])
                current["score"] = max(current["score"], float(row["momentum"]))
                continue
            if current is not None:
                episodes.append(current)
            current = {
                "strategy": "rotational_momentum",
                "variant_id": variant_id,
                "security_id": str(security_id),
                "symbol": str(row["symbol"]),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(row["entry_open"]),
                "exit_price": float(row["exit_open"]),
                "score": float(row["momentum"]),
                "forced": False,
                "slot_weight": 1.0 / top_n,
            }
        if current is not None:
            episodes.append(current)
    result = pd.DataFrame(episodes)
    if result.empty:
        return result
    result["gross_return"] = result["exit_price"] / result["entry_price"] - 1.0
    result["duration"] = (result["exit_date"] - result["entry_date"]).dt.days.astype(
        int
    )
    result.insert(0, "trade_id", np.arange(1, len(result) + 1, dtype=np.int64))
    return result[
        [
            "trade_id",
            "strategy",
            "variant_id",
            "security_id",
            "symbol",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "gross_return",
            "score",
            "duration",
            "forced",
            "slot_weight",
        ]
    ]


# ---------------------------------------------------------------------------
# Screening accumulators
# ---------------------------------------------------------------------------


@dataclass
class MetricAccumulator:
    trade_count: int = 0
    win_count: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    sum_return: float = 0.0
    sum_sq_return: float = 0.0
    duration_sum: int = 0
    max_duration: int = 0
    forced_count: int = 0
    yearly_sum: dict[int, float] = field(default_factory=dict)
    yearly_count: dict[int, int] = field(default_factory=dict)

    def update(
        self,
        dates: np.ndarray,
        returns: np.ndarray,
        durations: np.ndarray,
        forced: np.ndarray,
    ) -> None:
        if len(returns) == 0:
            return
        valid = np.isfinite(returns)
        values = returns[valid]
        if values.size == 0:
            return
        valid_dates = pd.to_datetime(dates[valid])
        valid_durations = durations[valid]
        valid_forced = forced[valid]
        self.trade_count += int(values.size)
        self.win_count += int(np.sum(values > 0))
        self.gross_profit += float(values[values > 0].sum())
        self.gross_loss += float(-values[values < 0].sum())
        self.sum_return += float(values.sum())
        self.sum_sq_return += float(np.square(values).sum())
        self.duration_sum += int(valid_durations.sum())
        self.max_duration = max(self.max_duration, int(valid_durations.max(initial=0)))
        self.forced_count += int(valid_forced.sum())
        years = valid_dates.year.to_numpy()
        for year in np.unique(years):
            mask = years == year
            self.yearly_sum[int(year)] = self.yearly_sum.get(int(year), 0.0) + float(
                values[mask].sum()
            )
            self.yearly_count[int(year)] = self.yearly_count.get(int(year), 0) + int(
                mask.sum()
            )

    def merge(self, other: "MetricAccumulator") -> None:
        self.trade_count += other.trade_count
        self.win_count += other.win_count
        self.gross_profit += other.gross_profit
        self.gross_loss += other.gross_loss
        self.sum_return += other.sum_return
        self.sum_sq_return += other.sum_sq_return
        self.duration_sum += other.duration_sum
        self.max_duration = max(self.max_duration, other.max_duration)
        self.forced_count += other.forced_count
        for year, value in other.yearly_sum.items():
            self.yearly_sum[year] = self.yearly_sum.get(year, 0.0) + value
        for year, value in other.yearly_count.items():
            self.yearly_count[year] = self.yearly_count.get(year, 0) + value

    def metrics(self) -> dict[str, Any]:
        n = self.trade_count
        if n == 0:
            return {
                "trade_count": 0,
                "win_rate": float("nan"),
                "profit_factor": float("nan"),
                "expectancy": float("nan"),
                "trade_sharpe": float("nan"),
                "average_holding_days": float("nan"),
                "maximum_holding_days": 0,
                "forced_exit_count": 0,
                "positive_year_ratio": float("nan"),
            }
        mean = self.sum_return / n
        variance = max(self.sum_sq_return / n - mean**2, 0.0)
        std = math.sqrt(variance)
        yearly_means = {
            year: self.yearly_sum[year] / max(self.yearly_count.get(year, 1), 1)
            for year in self.yearly_sum
        }
        return {
            "trade_count": n,
            "win_rate": self.win_count / n,
            "profit_factor": self.gross_profit / self.gross_loss
            if self.gross_loss > 0
            else (float("inf") if self.gross_profit > 0 else float("nan")),
            "expectancy": mean,
            "trade_sharpe": mean / std * math.sqrt(n) if std > 0 else float("nan"),
            "average_holding_days": self.duration_sum / n,
            "maximum_holding_days": self.max_duration,
            "forced_exit_count": self.forced_count,
            "forced_exit_ratio": self.forced_count / n,
            "positive_year_ratio": float(
                np.mean(np.asarray(list(yearly_means.values())) > 0)
            )
            if yearly_means
            else float("nan"),
            "yearly_mean_trade_return": {
                str(k): float(v) for k, v in sorted(yearly_means.items())
            },
        }


@dataclass
class VariantState:
    strategy: str
    family: str
    horizon: str
    params: dict[str, Any]
    variant_id: str
    periods: dict[str, MetricAccumulator] = field(
        default_factory=lambda: {name: MetricAccumulator() for name in PERIOD_NAMES}
    )


def make_variant_id(strategy: str, params: Mapping[str, Any]) -> str:
    return f"{strategy}__{sha256_text(stable_json(params))[:12]}"


def period_for_date(
    value: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> str:
    if value <= train_end:
        return "train"
    if value <= validation_end:
        return "validation"
    return "test"


def update_variant_from_trades(
    state: VariantState,
    trades: TradeBatch,
    round_trip_cost_decimal: float,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> None:
    if len(trades) == 0:
        return
    net_returns = trades.gross_returns - round_trip_cost_decimal
    entry_dates = pd.to_datetime(trades.entry_dates)
    period_labels = np.asarray(
        [period_for_date(date, train_end, validation_end) for date in entry_dates],
        dtype=object,
    )
    for period in PERIOD_NAMES:
        mask = period_labels == period
        if np.any(mask):
            state.periods[period].update(
                trades.entry_dates[mask],
                net_returns[mask],
                trades.durations[mask],
                trades.forced[mask],
            )


def variant_selection_score(state: VariantState, min_validation_trades: int) -> float:
    train = state.periods["train"].metrics()
    validation = state.periods["validation"].metrics()
    if validation["trade_count"] < min_validation_trades:
        return -1e12 + validation["trade_count"]
    train_pf = safe_float(train["profit_factor"], 0.0)
    val_pf = safe_float(validation["profit_factor"], 0.0)
    train_exp = safe_float(train["expectancy"], -1.0)
    val_exp = safe_float(validation["expectancy"], -1.0)
    forced = safe_float(validation.get("forced_exit_ratio"), 1.0)
    if train_pf <= 0 or val_pf <= 0:
        return -1e12
    # Validation dominates, train consistency prevents selecting a pure validation accident.
    score = (
        0.65 * math.log(max(val_pf, 1e-9))
        + 0.25 * math.log(max(train_pf, 1e-9))
        + 30.0 * val_exp
        + 10.0 * train_exp
        + 0.05 * math.log1p(validation["trade_count"])
        - 2.0 * forced
    )
    if train_exp <= 0:
        score -= 1.0
    if val_exp <= 0:
        score -= 2.0
    return score


# ---------------------------------------------------------------------------
# Candidate trade store and detailed strategy streams
# ---------------------------------------------------------------------------


def create_trade_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS candidate_trades")
    con.execute("DROP TABLE IF EXISTS accepted_trades")
    con.execute(
        """
        CREATE TABLE candidate_trades (
            strategy VARCHAR,
            variant_id VARCHAR,
            security_id VARCHAR,
            symbol VARCHAR,
            entry_date DATE,
            exit_date DATE,
            entry_price DOUBLE,
            exit_price DOUBLE,
            gross_return DOUBLE,
            score DOUBLE,
            duration INTEGER,
            forced BOOLEAN
        )
        """
    )
    con.execute(
        """
        CREATE TABLE accepted_trades (
            trade_id BIGINT,
            strategy VARCHAR,
            variant_id VARCHAR,
            security_id VARCHAR,
            symbol VARCHAR,
            entry_date DATE,
            exit_date DATE,
            entry_price DOUBLE,
            exit_price DOUBLE,
            gross_return DOUBLE,
            score DOUBLE,
            duration INTEGER,
            forced BOOLEAN,
            slot_weight DOUBLE
        )
        """
    )


def append_candidate_records(
    con: duckdb.DuckDBPyConnection,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return
    frame = pd.DataFrame.from_records(records)
    con.register("_candidate_append", frame)
    try:
        con.execute("INSERT INTO candidate_trades SELECT * FROM _candidate_append")
    finally:
        con.unregister("_candidate_append")


def accept_capacity_bounded_trades(
    candidates: pd.DataFrame,
    max_positions: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.assign(
            trade_id=pd.Series(dtype="int64"), slot_weight=pd.Series(dtype=float)
        )
    frame = candidates.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame = frame.sort_values(
        ["entry_date", "score", "security_id", "exit_date"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    heap: list[tuple[pd.Timestamp, str]] = []
    open_ids: set[str] = set()
    accepted_indices: list[int] = []
    for entry_date, group in frame.groupby("entry_date", sort=True):
        while heap and heap[0][0] <= entry_date:
            _, security_id = heapq.heappop(heap)
            open_ids.discard(security_id)
        for idx, row in group.iterrows():
            security_id = str(row["security_id"])
            if security_id in open_ids:
                continue
            if len(open_ids) >= max_positions:
                continue
            accepted_indices.append(idx)
            open_ids.add(security_id)
            heapq.heappush(heap, (pd.Timestamp(row["exit_date"]), security_id))
    accepted = frame.loc[accepted_indices].copy().reset_index(drop=True)
    accepted.insert(0, "trade_id", np.arange(1, len(accepted) + 1, dtype=np.int64))
    accepted["slot_weight"] = 1.0 / max_positions
    return accepted


def build_strategy_raw_daily_stream(
    con: duckdb.DuckDBPyConnection,
    strategy: str,
) -> pd.DataFrame:
    # One row per accepted trade/day; trade return is open->close on entry day,
    # close->open on exit day and close->close in between.
    query = """
        WITH marks AS (
            SELECT
                t.trade_id,
                t.strategy,
                t.entry_date,
                t.exit_date,
                t.entry_price,
                t.exit_price,
                t.slot_weight,
                b.date,
                b.close,
                LAG(b.close) OVER (PARTITION BY t.trade_id ORDER BY b.date) AS previous_close
            FROM accepted_trades t
            INNER JOIN market_bars b
              ON b.security_id = t.security_id
             AND b.date BETWEEN t.entry_date AND t.exit_date
            WHERE t.strategy = ?
        ), daily_trade_returns AS (
            SELECT
                date,
                trade_id,
                slot_weight,
                CASE
                    WHEN date = entry_date AND date = exit_date
                        THEN exit_price / entry_price - 1.0
                    WHEN date = entry_date
                        THEN close / entry_price - 1.0
                    WHEN date = exit_date
                        THEN exit_price / NULLIF(previous_close, 0) - 1.0
                    ELSE close / NULLIF(previous_close, 0) - 1.0
                END AS asset_return,
                CASE WHEN date = entry_date THEN slot_weight ELSE 0.0 END AS entry_turnover,
                CASE WHEN date = exit_date THEN slot_weight ELSE 0.0 END AS exit_turnover,
                CASE WHEN date = entry_date THEN 1 ELSE 0 END AS entry_orders,
                CASE WHEN date = exit_date THEN 1 ELSE 0 END AS exit_orders,
                slot_weight AS active_weight
            FROM marks
        )
        SELECT
            date,
            SUM(slot_weight * COALESCE(asset_return, 0.0)) AS gross_return,
            SUM(entry_turnover + exit_turnover) AS turnover_weight,
            SUM(entry_orders + exit_orders) AS order_count,
            SUM(active_weight) AS exposure
        FROM daily_trade_returns
        GROUP BY date
        ORDER BY date
    """
    stream = con.execute(query, [strategy]).fetchdf()
    if stream.empty:
        return pd.DataFrame(
            columns=["gross_return", "turnover_weight", "order_count", "exposure"],
            index=pd.DatetimeIndex([], name="date"),
        )
    stream["date"] = pd.to_datetime(stream["date"])
    return stream.set_index("date").sort_index()


def apply_costs_to_stream(
    raw: pd.DataFrame,
    side_cost_decimal: float,
    fixed_fee_eur: float,
    total_capital_eur: float,
    portfolio_weight: float = 1.0,
) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float)
    # Gross and proportional costs scale with sleeve weight. A fixed order fee is
    # a direct portfolio-level cash cost, independent of sleeve weight.
    contribution = portfolio_weight * (
        raw["gross_return"] - side_cost_decimal * raw["turnover_weight"]
    )
    fixed = raw["order_count"] * fixed_fee_eur / total_capital_eur
    return (contribution - fixed).rename("return")


# ---------------------------------------------------------------------------
# Reporting and orchestration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    command: str
    data: str
    output: str
    preset: str
    policy: str
    start: str
    train_end: str
    validation_end: str
    end: str
    initial_capital: float
    max_positions: int
    cost_bps_per_side: float
    slippage_bps_per_side: float
    fixed_fee_eur: float
    min_bars: int
    min_validation_trades: int
    max_symbols: int | None
    corporate_action_gate: bool
    overnight_ratio_min: float
    overnight_ratio_max: float
    batch_size: int
    checkpoint_every: int
    full_cartesian: bool
    max_variants_per_strategy: int
    combo_sizes: tuple[int, ...]
    weight_modes: tuple[str, ...]
    bootstrap_runs: int
    bootstrap_block_size: int
    top_equity_curves: int
    seed: int
    include_strategies: tuple[str, ...]
    exclude_strategies: tuple[str, ...]
    resume: bool


def config_hash(config: RunConfig) -> str:
    payload = dataclasses.asdict(config)
    payload.pop("resume", None)
    payload["_program_version"] = PROGRAM_VERSION
    try:
        payload["_program_sha256"] = (
            hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest().upper()
        )
    except OSError:
        payload["_program_sha256"] = "UNAVAILABLE"
    return sha256_text(stable_json(payload))[:16]


def flatten_variant_row(state: VariantState, selection_score: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "strategy": state.strategy,
        "family": state.family,
        "horizon": state.horizon,
        "variant_id": state.variant_id,
        "params": stable_json(state.params),
        "selection_score_train_validation": selection_score,
    }
    for period in PERIOD_NAMES:
        for key, value in state.periods[period].metrics().items():
            if key == "yearly_mean_trade_return":
                continue
            row[f"{period}_{key}"] = value
    return row


def choose_weight_vector(
    names: Sequence[str],
    raw_streams: Mapping[str, pd.DataFrame],
    mode: str,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> np.ndarray:
    n = len(names)
    if mode == "equal" or n == 1:
        return np.full(n, 1.0 / n)
    volatilities: list[float] = []
    sharpes: list[float] = []
    for name in names:
        series = raw_streams[name]["gross_return"].loc[validation_start:validation_end]
        vol = float(series.std(ddof=1))
        mean = float(series.mean())
        volatilities.append(vol if vol > 1e-12 else float("nan"))
        sharpes.append(mean / vol if vol > 1e-12 else float("nan"))
    if mode == "inverse_volatility":
        raw_weights = np.asarray(
            [1.0 / v if math.isfinite(v) and v > 0 else 0.0 for v in volatilities]
        )
    elif mode == "positive_sharpe":
        raw_weights = np.asarray(
            [max(s, 0.0) if math.isfinite(s) else 0.0 for s in sharpes]
        )
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    if raw_weights.sum() <= 0:
        return np.full(n, 1.0 / n)
    # Prevent one validation winner from receiving the whole portfolio.
    raw_weights = np.clip(raw_weights / raw_weights.sum(), 0.10 / n, 0.70)
    return raw_weights / raw_weights.sum()


def align_raw_streams(
    names: Sequence[str], streams: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    all_dates = pd.DatetimeIndex([])
    for name in names:
        all_dates = all_dates.union(streams[name].index)
    all_dates = all_dates.sort_values()
    aligned: dict[str, pd.DataFrame] = {}
    for name in names:
        aligned[name] = streams[name].reindex(all_dates).fillna(0.0)
    return aligned


def combine_streams(
    names: Sequence[str],
    weights: np.ndarray,
    aligned_raw: Mapping[str, pd.DataFrame],
    side_cost_decimal: float,
    fixed_fee_eur: float,
    total_capital_eur: float,
) -> tuple[pd.Series, pd.Series]:
    index = aligned_raw[names[0]].index
    combined = pd.Series(0.0, index=index, dtype=float)
    exposure = pd.Series(0.0, index=index, dtype=float)
    for name, weight in zip(names, weights):
        raw = aligned_raw[name]
        combined += weight * (
            raw["gross_return"] - side_cost_decimal * raw["turnover_weight"]
        )
        # Fixed fee affects the total portfolio once per order.
        combined -= raw["order_count"] * fixed_fee_eur / total_capital_eur
        exposure += weight * raw["exposure"]
    return combined, exposure


def split_daily_metrics(
    returns: pd.Series,
    config: RunConfig,
    trial_count: int,
    sharpe_std: float,
) -> dict[str, Any]:
    train_end = parse_date(config.train_end)
    validation_end = parse_date(config.validation_end)
    return {
        "train": daily_metrics(
            returns.loc[:train_end], config.initial_capital, trial_count, sharpe_std
        ),
        "validation": daily_metrics(
            returns.loc[train_end + pd.Timedelta(days=1) : validation_end],
            config.initial_capital,
            trial_count,
            sharpe_std,
        ),
        "test": daily_metrics(
            returns.loc[validation_end + pd.Timedelta(days=1) :],
            config.initial_capital,
            trial_count,
            sharpe_std,
        ),
        "full": daily_metrics(returns, config.initial_capital, trial_count, sharpe_std),
    }


def validation_combo_score(metrics: Mapping[str, Any]) -> float:
    validation = metrics["validation"]
    cagr = safe_float(validation.get("CAGR"), -1.0)
    sharpe = safe_float(validation.get("Sharpe"), -10.0)
    mdd = safe_float(validation.get("maximum_drawdown"), -1.0)
    pf = safe_float(validation.get("daily_profit_factor"), 0.0)
    if not math.isfinite(cagr) or not math.isfinite(sharpe):
        return -1e12
    return 2.0 * cagr + 0.5 * sharpe + 0.5 * math.log(max(pf, 1e-9)) - abs(mdd)


def dataframe_as_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def write_markdown_report(
    output_dir: Path,
    config: RunConfig,
    source_meta: Mapping[str, Any],
    variant_df: pd.DataFrame,
    winners_df: pd.DataFrame,
    individual_df: pd.DataFrame,
    combo_frames: Mapping[int, pd.DataFrame],
    blocked: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> None:
    lines = [
        "# Strategy Combo Research Lab",
        "",
        f"Generated: `{utc_now_iso()}`  ",
        f"Program version: `{PROGRAM_VERSION}`  ",
        f"Elapsed: `{elapsed_seconds / 3600:.2f} hours`",
        "",
        "## Safety and interpretation",
        "",
        "This is an offline research run. It performed no broker, account, market-data API or order calls. "
        "Parameter winners were selected using train/validation only; the test period was not used for selection. "
        "A positive backtest is not paper/live authority.",
        "",
        "## Data",
        "",
        "```json",
        json.dumps(source_meta, indent=2, default=str),
        "```",
        "",
        "## Run configuration",
        "",
        "```json",
        json.dumps(dataclasses.asdict(config), indent=2, default=str),
        "```",
        "",
        "## Parameter search",
        "",
        f"Variants evaluated: **{len(variant_df):,}**  ",
        f"Strategy winners: **{len(winners_df):,}**",
        "",
    ]
    if not individual_df.empty:
        columns = [
            "strategy",
            "family",
            "test_CAGR",
            "test_Sharpe",
            "test_maximum_drawdown",
            "test_daily_profit_factor",
            "test_terminal_equity",
            "accepted_trade_count",
        ]
        available = [c for c in columns if c in individual_df.columns]
        lines.extend(
            [
                "## Individual validation-selected strategies",
                "",
                dataframe_as_markdown(individual_df[available].head(25)),
                "",
            ]
        )
    for size, frame in combo_frames.items():
        lines.extend([f"## Top {size}-strategy combinations", ""])
        if frame.empty:
            lines.extend(["No eligible combinations.", ""])
        else:
            columns = [
                "strategies",
                "weight_mode",
                "validation_score",
                "test_CAGR",
                "test_Sharpe",
                "test_maximum_drawdown",
                "test_daily_profit_factor",
                "test_terminal_equity",
                "average_validation_correlation",
            ]
            available = [c for c in columns if c in frame.columns]
            lines.extend([dataframe_as_markdown(frame[available].head(25)), ""])
    lines.extend(["## Explicitly blocked or deferred transcript items", ""])
    lines.append(dataframe_as_markdown(pd.DataFrame(blocked)))
    lines.extend(
        [
            "",
            "## Methodological limitations",
            "",
            "- The search tests many hypotheses. Use the DSR probability, bootstrap results and forward shadow before trusting any winner.",
            "- Strategy combinations are equal-capital or validation-weighted sleeves, not simultaneous-signal confirmation rules.",
            "- The portfolio model uses fixed strategy slots and whole-share constraints are not simulated in this one-file research lab.\n- Extreme split-adjusted open/close discontinuities are blocked at identity level and written to `data_quality_exclusions.csv`.",
            "- Fixed fees are charged per accepted buy and sell order against total portfolio capital.",
            "- Cross-sectional Top-N rotational momentum is implemented with monthly ranking and next-open holding episodes; inverse-volatility multi-asset allocation remains data/product-review dependent.",
            "- Historical Shariah eligibility must be applied upstream or supplied in the dataset; this file does not reconstruct missing Shariah history.",
            "",
        ]
    )
    atomic_write_text(output_dir / "report.md", "\n".join(lines))


def run_lab(config: RunConfig) -> Path:
    started = time.time()
    start = parse_date(config.start)
    train_end = parse_date(config.train_end)
    validation_end = parse_date(config.validation_end)
    end = parse_date(config.end)
    if not (start < train_end < validation_end < end):
        raise ValueError("Dates must satisfy start < train_end < validation_end < end")

    run_hash = config_hash(config)
    output_root = Path(config.output)
    output_dir = output_root / f"run_{run_hash}"
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "research_lab.duckdb"
    checkpoint_path = output_dir / "screening_checkpoint.pkl"
    progress_path = output_dir / "progress.json"

    data_path = Path(config.data) if config.data else discover_default_data()
    source = MarketDataSource(
        data_path,
        start,
        end,
        config.min_bars,
        config.max_symbols,
        config.seed,
        database_path,
        corporate_action_gate=config.corporate_action_gate,
        overnight_ratio_min=config.overnight_ratio_min,
        overnight_ratio_max=config.overnight_ratio_max,
    )
    quality_exclusions = source.quality_exclusions()
    quality_exclusions.to_csv(
        output_dir / "data_quality_exclusions.csv",
        index=False,
    )
    if len(quality_exclusions):
        print(
            f"[data-quality] excluded identities={len(quality_exclusions):,} "
            f"ratio_bounds=[{config.overnight_ratio_min}, {config.overnight_ratio_max}]"
        )

    all_specs = strategy_registry()
    include = set(config.include_strategies)
    exclude = set(config.exclude_strategies)
    specs = []
    blocked = list(BLOCKED_TRANSCRIPT_ITEMS)
    for spec in all_specs:
        if include and spec.name not in include:
            continue
        if spec.name in exclude:
            continue
        if config.policy == "shariah" and spec.policy_status != "LONG_ONLY_GO":
            blocked.append(
                {
                    "name": spec.name,
                    "status": spec.policy_status,
                    "reason": spec.notes or "Policy blocked.",
                }
            )
            continue
        specs.append(spec)
    if not specs:
        raise ValueError("No strategies remain after include/exclude/policy filters")

    variants: dict[str, VariantState] = {}
    spec_by_name = {spec.name: spec for spec in specs}
    for index, spec in enumerate(specs):
        grid = one_factor_and_sampled_grid(
            spec.default_params,
            spec.choices,
            config.preset,
            config.full_cartesian,
            config.max_variants_per_strategy,
            config.seed + index * 997,
        )
        grid = [params for params in grid if params_valid(spec.name, params)]
        if not grid:
            raise RuntimeError(f"No valid parameter variants generated for {spec.name}")
        for params in grid:
            variant_id = make_variant_id(spec.name, params)
            variants[variant_id] = VariantState(
                strategy=spec.name,
                family=spec.family,
                horizon=spec.horizon,
                params=dict(params),
                variant_id=variant_id,
            )

    processed_ids: set[str] = set()
    if config.resume and checkpoint_path.exists():
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if checkpoint.get("config_hash") != run_hash:
            raise RuntimeError("Checkpoint config hash does not match this run")
        restored: dict[str, VariantState] = checkpoint["variants"]
        if set(restored) != set(variants):
            raise RuntimeError("Checkpoint variant registry does not match this run")
        variants = restored
        processed_ids = set(checkpoint.get("processed_ids", []))
        print(f"[resume] restored {len(processed_ids):,} processed identities")

    side_cost = (config.cost_bps_per_side + config.slippage_bps_per_side) / 10_000.0
    screening_round_trip_cost = (
        2.0 * side_cost
        + 2.0 * config.fixed_fee_eur * config.max_positions / config.initial_capital
    )
    total_identities = len(source.identities)
    print(
        f"[screen] identities={total_identities:,} strategies={len(specs)} "
        f"variants={len(variants):,} preset={config.preset}"
    )
    batch_counter = 0
    for bars in source.iter_batches(config.batch_size):
        batch_ids = bars["security_id"].astype(str).unique().tolist()
        if processed_ids and all(identity in processed_ids for identity in batch_ids):
            continue
        batch_counter += 1
        for security_id, frame in bars.groupby("security_id", sort=False):
            security_id = str(security_id)
            if security_id in processed_ids:
                continue
            frame = (
                frame.sort_values("date")
                .drop_duplicates("date", keep="last")
                .reset_index(drop=True)
            )
            if len(frame) < config.min_bars:
                processed_ids.add(security_id)
                continue
            cache = FeatureCache(frame)
            for state in variants.values():
                spec = spec_by_name[state.strategy]
                try:
                    trades = spec.builder(frame, cache, state.params)
                except Exception as exc:
                    raise RuntimeError(
                        f"Strategy {spec.name} failed for security_id={security_id} params={state.params}: {exc}"
                    ) from exc
                update_variant_from_trades(
                    state, trades, screening_round_trip_cost, train_end, validation_end
                )
            processed_ids.add(security_id)
        progress = {
            "status": "SCREENING_RUNNING",
            "processed_identities": len(processed_ids),
            "total_identities": total_identities,
            "percent": 100.0 * len(processed_ids) / max(total_identities, 1),
            "updated_at": utc_now_iso(),
        }
        atomic_write_json(progress_path, progress)
        print(
            f"[screen] {len(processed_ids):,}/{total_identities:,} "
            f"({progress['percent']:.1f}%)"
        )
        if batch_counter % config.checkpoint_every == 0:
            tmp = checkpoint_path.with_suffix(".tmp")
            with tmp.open("wb") as handle:
                pickle.dump(
                    {
                        "config_hash": run_hash,
                        "variants": variants,
                        "processed_ids": sorted(processed_ids),
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp.replace(checkpoint_path)

    print("[monthly] building cross-sectional monthly feature table")
    monthly_feature_count = rebuild_monthly_feature_table(source)
    rotational_states, rotational_winner, rotational_selected = (
        evaluate_rotational_variants(
            source, config, side_cost, train_end, validation_end
        )
    )

    variant_rows: list[dict[str, Any]] = []
    winners: dict[str, VariantState] = {}
    for strategy in sorted(spec_by_name):
        states = [state for state in variants.values() if state.strategy == strategy]
        scored = [
            (variant_selection_score(state, config.min_validation_trades), state)
            for state in states
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        winner_score, winner = scored[0]
        winners[strategy] = winner
        for score, state in scored:
            row = flatten_variant_row(state, score)
            row["selected_winner"] = state.variant_id == winner.variant_id
            variant_rows.append(row)

    if rotational_states:
        for state in rotational_states:
            score = variant_selection_score(
                state, max(12, config.min_validation_trades)
            )
            row = flatten_variant_row(state, score)
            row["selected_winner"] = bool(
                rotational_winner and state.variant_id == rotational_winner.variant_id
            )
            variant_rows.append(row)

    variant_df = pd.DataFrame(variant_rows).sort_values(
        ["strategy", "selection_score_train_validation"], ascending=[True, False]
    )
    variant_df.to_csv(output_dir / "individual_variants.csv", index=False)
    winner_rows = [row for row in variant_rows if row["selected_winner"]]
    winners_df = pd.DataFrame(winner_rows).sort_values(
        "selection_score_train_validation", ascending=False
    )
    winners_df.to_csv(output_dir / "individual_winners.csv", index=False)
    atomic_write_json(
        output_dir / "winner_parameters.json",
        {
            name: {
                "variant_id": state.variant_id,
                "params": state.params,
                "selection_score_train_validation": variant_selection_score(
                    state, config.min_validation_trades
                ),
            }
            for name, state in (
                {
                    **winners,
                    **(
                        {"rotational_momentum": rotational_winner}
                        if rotational_winner
                        else {}
                    ),
                }
            ).items()
        },
    )

    # Detailed second pass for winners only.
    print(
        f"[detail] rebuilding candidate trades for {len(winners)} symbol-level winners"
        + (" plus rotational momentum" if rotational_winner else "")
    )
    create_trade_tables(source.con)
    detail_records: list[dict[str, Any]] = []
    for bars in source.iter_batches(config.batch_size):
        for security_id, frame in bars.groupby("security_id", sort=False):
            frame = (
                frame.sort_values("date")
                .drop_duplicates("date", keep="last")
                .reset_index(drop=True)
            )
            cache = FeatureCache(frame)
            symbol = str(frame["symbol"].iloc[-1])
            for strategy, state in winners.items():
                spec = spec_by_name[strategy]
                trades = spec.builder(frame, cache, state.params)
                for i in range(len(trades)):
                    detail_records.append(
                        {
                            "strategy": strategy,
                            "variant_id": state.variant_id,
                            "security_id": str(security_id),
                            "symbol": symbol,
                            "entry_date": pd.Timestamp(trades.entry_dates[i]).date(),
                            "exit_date": pd.Timestamp(trades.exit_dates[i]).date(),
                            "entry_price": float(trades.entry_prices[i]),
                            "exit_price": float(trades.exit_prices[i]),
                            "gross_return": float(trades.gross_returns[i]),
                            "score": float(trades.scores[i]),
                            "duration": int(trades.durations[i]),
                            "forced": bool(trades.forced[i]),
                        }
                    )
                if len(detail_records) >= 100_000:
                    append_candidate_records(source.con, detail_records)
                    detail_records.clear()
        print("[detail] batch candidate trades appended")
    append_candidate_records(source.con, detail_records)

    if rotational_winner is not None and not rotational_selected.empty:
        rotational_accepted = merge_rotational_holding_episodes(
            rotational_selected,
            int(rotational_winner.params["top_n"]),
            rotational_winner.variant_id,
        )
        if not rotational_accepted.empty:
            source.con.register("_rotational_accepted", rotational_accepted)
            try:
                source.con.execute(
                    "INSERT INTO accepted_trades SELECT * FROM _rotational_accepted"
                )
            finally:
                source.con.unregister("_rotational_accepted")
    else:
        rotational_accepted = pd.DataFrame()

    strategy_streams: dict[str, pd.DataFrame] = {}
    individual_rows: list[dict[str, Any]] = []
    accepted_trade_counts: dict[str, int] = {}
    for strategy, state in winners.items():
        candidates = source.con.execute(
            "SELECT * FROM candidate_trades WHERE strategy = ? ORDER BY entry_date, score DESC",
            [strategy],
        ).fetchdf()
        accepted = accept_capacity_bounded_trades(candidates, config.max_positions)
        accepted_trade_counts[strategy] = int(len(accepted))
        if not accepted.empty:
            accepted["strategy"] = strategy
            accepted["variant_id"] = state.variant_id
            source.con.register(
                "_accepted_append",
                accepted[
                    [
                        "trade_id",
                        "strategy",
                        "variant_id",
                        "security_id",
                        "symbol",
                        "entry_date",
                        "exit_date",
                        "entry_price",
                        "exit_price",
                        "gross_return",
                        "score",
                        "duration",
                        "forced",
                        "slot_weight",
                    ]
                ],
            )
            try:
                source.con.execute(
                    "INSERT INTO accepted_trades SELECT * FROM _accepted_append"
                )
            finally:
                source.con.unregister("_accepted_append")
        raw = build_strategy_raw_daily_stream(source.con, strategy)
        strategy_streams[strategy] = raw
        net = apply_costs_to_stream(
            raw, side_cost, config.fixed_fee_eur, config.initial_capital, 1.0
        )
        split = split_daily_metrics(net, config, len(variant_df), 0.0)
        individual_row: dict[str, Any] = {
            "strategy": strategy,
            "family": state.family,
            "horizon": state.horizon,
            "variant_id": state.variant_id,
            "params": stable_json(state.params),
            "accepted_trade_count": int(len(accepted)),
            "candidate_trade_count": int(len(candidates)),
            "capacity_acceptance_ratio": float(len(accepted) / len(candidates))
            if len(candidates)
            else float("nan"),
            "average_exposure": float(raw["exposure"].mean()) if not raw.empty else 0.0,
            "turnover_weight": float(raw["turnover_weight"].sum())
            if not raw.empty
            else 0.0,
            "order_count": int(raw["order_count"].sum()) if not raw.empty else 0,
        }
        for period, metrics in split.items():
            for key, value in metrics.items():
                if key == "yearly_returns":
                    continue
                individual_row[f"{period}_{key}"] = value
        individual_rows.append(individual_row)
        # Save winner stream for audit/reuse.
        raw.assign(net_return=net).reset_index().to_parquet(
            output_dir / f"stream_{strategy}.parquet", index=False
        )

    if rotational_winner is not None:
        strategy = "rotational_momentum"
        raw = build_strategy_raw_daily_stream(source.con, strategy)
        strategy_streams[strategy] = raw
        net = apply_costs_to_stream(
            raw, side_cost, config.fixed_fee_eur, config.initial_capital, 1.0
        )
        split = split_daily_metrics(net, config, len(variant_df), 0.0)
        row = {
            "strategy": strategy,
            "family": rotational_winner.family,
            "horizon": rotational_winner.horizon,
            "variant_id": rotational_winner.variant_id,
            "params": stable_json(rotational_winner.params),
            "accepted_trade_count": int(len(rotational_accepted)),
            "candidate_trade_count": int(len(rotational_selected)),
            "capacity_acceptance_ratio": 1.0,
            "average_exposure": float(raw["exposure"].mean()) if not raw.empty else 0.0,
            "turnover_weight": float(raw["turnover_weight"].sum())
            if not raw.empty
            else 0.0,
            "order_count": int(raw["order_count"].sum()) if not raw.empty else 0,
        }
        for period, metrics in split.items():
            for key, value in metrics.items():
                if key != "yearly_returns":
                    row[f"{period}_{key}"] = value
        individual_rows.append(row)
        raw.assign(net_return=net).reset_index().to_parquet(
            output_dir / "stream_rotational_momentum.parquet", index=False
        )

    individual_df = pd.DataFrame(individual_rows)
    if not individual_df.empty:
        individual_df = individual_df.sort_values("validation_Sharpe", ascending=False)
    individual_df.to_csv(output_dir / "individual_portfolios.csv", index=False)

    # Eligible combination universe: a strategy needs at least one validation trade
    # and a non-empty daily stream. We still keep weak/negative strategies; all
    # combinations are evaluated as requested.
    all_winner_states = {
        **winners,
        **({"rotational_momentum": rotational_winner} if rotational_winner else {}),
    }
    eligible = [
        name
        for name, state in all_winner_states.items()
        if name in strategy_streams
        and not strategy_streams[name].empty
        and state.periods["validation"].trade_count
        >= max(1, config.min_validation_trades // 4)
    ]
    print(f"[combo] eligible strategies={len(eligible)}")
    aligned = align_raw_streams(eligible, strategy_streams) if eligible else {}
    validation_start = train_end + pd.Timedelta(days=1)
    combo_frames: dict[int, pd.DataFrame] = {}
    all_combo_sharpes: list[float] = []
    combination_trial_count = 0

    # First pass computes all combinations and their raw metrics.
    for size in config.combo_sizes:
        rows: list[dict[str, Any]] = []
        if size > len(eligible):
            combo_frames[size] = pd.DataFrame()
            continue
        total_for_size = math.comb(len(eligible), size) * len(config.weight_modes)
        print(f"[combo] size={size} evaluations={total_for_size:,}")
        done = 0
        for names_tuple in itertools.combinations(sorted(eligible), size):
            names = list(names_tuple)
            for mode in config.weight_modes:
                weights = choose_weight_vector(
                    names, aligned, mode, validation_start, validation_end
                )
                returns, exposure = combine_streams(
                    names,
                    weights,
                    aligned,
                    side_cost,
                    config.fixed_fee_eur,
                    config.initial_capital,
                )
                split = split_daily_metrics(returns, config, 1, 0.0)
                validation_score = validation_combo_score(split)
                corr_frame = pd.DataFrame(
                    {
                        name: aligned[name]["gross_return"].loc[
                            validation_start:validation_end
                        ]
                        for name in names
                    }
                ).fillna(0.0)
                corr = corr_frame.corr().to_numpy()
                upper = (
                    corr[np.triu_indices_from(corr, k=1)]
                    if len(names) > 1
                    else np.asarray([0.0])
                )
                combo_row: dict[str, Any] = {
                    "strategies": "+".join(names),
                    "strategy_count": size,
                    "weight_mode": mode,
                    "weights": stable_json(
                        {name: float(weight) for name, weight in zip(names, weights)}
                    ),
                    "validation_score": validation_score,
                    "average_validation_correlation": float(np.nanmean(upper))
                    if upper.size
                    else 0.0,
                    "maximum_theoretical_positions": size * config.max_positions,
                    "average_exposure": float(exposure.mean()),
                }
                for period, metrics in split.items():
                    for key, value in metrics.items():
                        if key == "yearly_returns":
                            continue
                        combo_row[f"{period}_{key}"] = value
                rows.append(combo_row)
                all_combo_sharpes.append(
                    safe_float(split["validation"].get("Sharpe"), 0.0)
                )
                combination_trial_count += 1
                done += 1
                if done % 1000 == 0:
                    print(f"[combo] size={size} {done:,}/{total_for_size:,}")
        frame = pd.DataFrame(rows).sort_values("validation_score", ascending=False)
        frame.to_csv(output_dir / f"combinations_{size}.csv", index=False)
        combo_frames[size] = frame

    # Recompute DSR probabilities using the observed dispersion of validation Sharpes.
    sharpe_std = (
        float(np.nanstd(np.asarray(all_combo_sharpes, dtype=float), ddof=1))
        if len(all_combo_sharpes) > 1
        else 0.0
    )
    for size, frame in combo_frames.items():
        if frame.empty:
            continue
        dsr_values = []
        for _, row in frame.iterrows():
            observed = safe_float(row.get("test_Sharpe"), float("nan"))
            observations = int(safe_float(row.get("test_observations"), 0.0))
            # Daily combo rows do not retain skew/kurtosis; use normal-return approximation.
            dsr_values.append(
                deflated_sharpe_probability(
                    observed,
                    observations,
                    combination_trial_count,
                    sharpe_std,
                    0.0,
                    3.0,
                )
            )
        frame["test_DSR_probability_all_combo_trials"] = dsr_values
        frame.to_csv(output_dir / f"combinations_{size}.csv", index=False)

    # Save top equity curves and bootstraps selected by validation score only.
    top_candidates: list[tuple[str, str, float, pd.Series]] = []
    for _, row in individual_df.iterrows():
        name = str(row["strategy"])
        net = apply_costs_to_stream(
            strategy_streams[name],
            side_cost,
            config.fixed_fee_eur,
            config.initial_capital,
            1.0,
        )
        score = safe_float(row.get("validation_Sharpe"), -1e12)
        top_candidates.append(("individual", name, score, net))
    for size, frame in combo_frames.items():
        for _, row in frame.head(max(config.top_equity_curves, 10)).iterrows():
            names = str(row["strategies"]).split("+")
            weights_map = json.loads(str(row["weights"]))
            weights = np.asarray([weights_map[name] for name in names], dtype=float)
            returns, _ = combine_streams(
                names,
                weights,
                aligned,
                side_cost,
                config.fixed_fee_eur,
                config.initial_capital,
            )
            top_candidates.append(
                (
                    f"combo_{size}",
                    str(row["strategies"]) + "__" + str(row["weight_mode"]),
                    float(row["validation_score"]),
                    returns,
                )
            )
    top_candidates.sort(key=lambda item: item[2], reverse=True)
    bootstrap_payload: dict[str, Any] = {}
    for rank, (kind, label, score, returns) in enumerate(
        top_candidates[: config.top_equity_curves], start=1
    ):
        safe_label = hashlib.sha256(label.encode()).hexdigest()[:12]
        equity = (1.0 + returns.fillna(0.0)).cumprod() * config.initial_capital
        pd.DataFrame(
            {
                "date": returns.index,
                "daily_return": returns.values,
                "equity": equity.values,
            }
        ).to_csv(
            output_dir / f"top_equity_{rank:03d}_{kind}_{safe_label}.csv", index=False
        )
        bootstrap_payload[f"{rank:03d}_{kind}_{label}"] = {
            "validation_selection_score": score,
            "bootstrap": block_bootstrap(
                returns.loc[validation_end + pd.Timedelta(days=1) :],
                config.bootstrap_runs,
                config.bootstrap_block_size,
                config.seed + rank,
            ),
        }
    atomic_write_json(output_dir / "top_bootstrap.json", bootstrap_payload)

    manifest = {
        "schema": PROGRAM_SCHEMA,
        "program_version": PROGRAM_VERSION,
        "status": "STRATEGY_COMBO_RESEARCH_COMPLETE",
        "generated_at": utc_now_iso(),
        "config_hash": run_hash,
        "data": source.metadata(),
        "config": dataclasses.asdict(config),
        "strategy_count": len(specs) + (1 if rotational_winner else 0),
        "variant_count": len(variant_df),
        "winner_count": len(winners) + (1 if rotational_winner else 0),
        "monthly_feature_count": monthly_feature_count,
        "eligible_combo_strategy_count": len(eligible),
        "combination_trial_count": combination_trial_count,
        "blocked_or_deferred": blocked,
        "financial_authority": "NONE",
        "strategy_authority": "NONE",
        "PAPER_STRATEGY_AUTHORITY": "NONE",
        "LIVE_STRATEGY_AUTHORITY": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "warnings": [
            "Multiple-testing risk is material.",
            "Test results were not used for parameter or combination selection, but this run reveals them for research.",
            "Whole-share constraints and cross-sleeve total-position caps are not simulated.",
            "Extreme open/close discontinuities are blocked at identity level by default.",
            "Historical Shariah eligibility is not reconstructed by this file.",
        ],
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    write_markdown_report(
        output_dir,
        config,
        source.metadata(),
        variant_df,
        winners_df,
        individual_df,
        combo_frames,
        blocked,
        time.time() - started,
    )
    atomic_write_json(
        progress_path,
        {
            "status": "STRATEGY_COMBO_RESEARCH_COMPLETE",
            "completed_at": utc_now_iso(),
            "elapsed_seconds": time.time() - started,
            "output_dir": str(output_dir.resolve()),
        },
    )
    print(f"[done] output: {output_dir.resolve()}")
    return output_dir


# ---------------------------------------------------------------------------
# V2 causal global portfolio ledger and orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V2Config:
    command: str
    data: str
    output: str
    preset: str
    policy: str
    start: str
    train_end: str
    validation_end: str
    end: str
    initial_capital: float
    global_max_positions: int
    max_security_weight: float
    max_sector_weight: float
    max_gross_exposure: float
    minimum_order_eur: float
    whole_shares: bool
    max_order_adv_fraction: float
    min_price: float
    min_median_dollar_volume: float
    liquidity_lookback: int
    allowed_exchanges: tuple[str, ...]
    cost_bps_per_side: float
    slippage_bps_per_side: float
    fx_cost_bps_per_side: float
    fixed_fee_eur: float
    min_bars: int
    min_validation_trades: int
    validation_max_drawdown: float
    max_symbols: int | None
    corporate_action_gate: bool
    overnight_ratio_min: float
    overnight_ratio_max: float
    batch_size: int
    checkpoint_every: int
    workers: int
    memory_budget_gb: float
    full_cartesian: bool
    max_variants_per_strategy: int
    combo_sizes: tuple[int, ...]
    weight_modes: tuple[str, ...]
    allow_invalid_strategies_in_combos: bool
    bootstrap_runs: int
    bootstrap_block_size: int
    top_equity_curves: int
    equity_extreme_return_threshold: float
    equity_hard_fail_return_threshold: float
    seed: int
    include_strategies: tuple[str, ...]
    exclude_strategies: tuple[str, ...]
    resume: bool
    parameter_grid_override: Mapping[str, Sequence[Mapping[str, Any]]] = field(
        default_factory=dict
    )


@dataclass
class V2Position:
    security_id: str
    ticker_for_display: str
    shares: int
    average_cost_local: float
    average_cost_eur: float
    last_mark_local: float
    last_mark_eur: float
    realized_pnl_eur: float
    strategy_attribution_weights: dict[str, float]
    entry_date: pd.Timestamp
    last_trade_date: pd.Timestamp
    sector: str
    currency: str
    median_dollar_volume: float


@dataclass
class V2LedgerResult:
    ledger: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    positions: pd.DataFrame
    attribution: pd.DataFrame
    contributors: pd.DataFrame
    overlap: pd.DataFrame
    metrics: dict[str, Any]
    accounting_failures: int
    unexplained_outliers: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def v2_registry(
    config: V2Config,
) -> tuple[list[StrategySpec], dict[str, VariantState], dict[str, Any]]:
    include = set(config.include_strategies)
    exclude = set(config.exclude_strategies)
    specs: list[StrategySpec] = []
    variants: dict[str, VariantState] = {}
    grids: dict[str, list[dict[str, Any]]] = {}
    for index, spec in enumerate(strategy_registry()):
        if include and spec.name not in include:
            continue
        if spec.name in exclude:
            continue
        if config.policy == "shariah" and spec.policy_status != "LONG_ONLY_GO":
            continue
        override = config.parameter_grid_override.get(spec.name)
        grid = (
            [dict(params) for params in override]
            if override is not None
            else one_factor_and_sampled_grid(
                spec.default_params,
                spec.choices,
                config.preset,
                config.full_cartesian,
                config.max_variants_per_strategy,
                config.seed + index * 997,
            )
        )
        grid = [dict(params) for params in grid if params_valid(spec.name, params)]
        if not grid:
            continue
        specs.append(spec)
        grids[spec.name] = grid
        for params in grid:
            variant_id = make_variant_id(spec.name, params)
            variants[variant_id] = VariantState(
                strategy=spec.name,
                family=spec.family,
                horizon=spec.horizon,
                params=params,
                variant_id=variant_id,
            )
    if (
        not include or "rotational_momentum" in include
    ) and "rotational_momentum" not in exclude:
        rotational_override = config.parameter_grid_override.get(
            "rotational_momentum"
        )
        grid = (
            [dict(params) for params in rotational_override]
            if rotational_override is not None
            else rotational_grid(config)
        )
        grids["rotational_momentum"] = grid
        for params in grid:
            variant_id = make_variant_id("rotational_momentum", params)
            variants[variant_id] = VariantState(
                strategy="rotational_momentum",
                family="cross_sectional_momentum",
                horizon="long",
                params=params,
                variant_id=variant_id,
            )
    return specs, variants, grids


def v2_run_fingerprint(config: V2Config, data_path: Path) -> dict[str, str]:
    specs, _, grids = v2_registry(config)
    config_payload = dataclasses.asdict(config)
    config_payload.pop("resume", None)
    config_payload.pop("output", None)
    program_hash = sha256_file(Path(__file__).resolve())
    data_hash = (
        sha256_file(data_path)
        if data_path.is_file()
        else sha256_text(
            stable_json(
                [
                    (str(path.relative_to(data_path)), sha256_file(path))
                    for path in sorted(data_path.rglob("*.parquet"))
                ]
            )
        )
    )
    registry_hash = sha256_text(
        stable_json(
            [
                (spec.name, spec.family, spec.horizon, spec.policy_status)
                for spec in specs
            ]
        )
    )
    grid_hash = sha256_text(stable_json(grids))
    investability = {
        key: config_payload[key]
        for key in (
            "min_price",
            "min_median_dollar_volume",
            "liquidity_lookback",
            "max_order_adv_fraction",
            "allowed_exchanges",
        )
    }
    portfolio = {
        key: config_payload[key]
        for key in (
            "initial_capital",
            "global_max_positions",
            "max_security_weight",
            "max_sector_weight",
            "max_gross_exposure",
            "minimum_order_eur",
            "whole_shares",
            "cost_bps_per_side",
            "slippage_bps_per_side",
            "fx_cost_bps_per_side",
            "fixed_fee_eur",
        )
    }
    fingerprints = {
        "program_sha256": program_hash,
        "data_sha256": data_hash,
        "config_sha256": sha256_text(stable_json(config_payload)),
        "strategy_registry_sha256": registry_hash,
        "parameter_grid_sha256": grid_hash,
        "investability_policy_sha256": sha256_text(stable_json(investability)),
        "portfolio_policy_sha256": sha256_text(stable_json(portfolio)),
    }
    fingerprints["run_id"] = sha256_text(stable_json(fingerprints))[:16]
    return fingerprints


def load_v2_fx(project_root: Path, calendar: pd.DatetimeIndex) -> pd.Series:
    path = (
        project_root / "data" / "research" / "phase11_4" / "private" / "eurusd.parquet"
    )
    frame = pd.read_parquet(path, columns=["date", "usd_per_eur"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    source = (
        frame.drop_duplicates("date", keep="last")
        .set_index("date")["usd_per_eur"]
        .sort_index()
    )
    aligned = (
        source.reindex(source.index.union(calendar))
        .sort_index()
        .ffill()
        .reindex(calendar)
    )
    aligned = 1.0 / aligned
    aligned.name = "eur_per_usd"
    return aligned


def load_v2_security_metadata(
    project_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = (
        project_root
        / "data"
        / "research"
        / "phase11_4"
        / "private"
        / "security-master.parquet"
    )
    if not path.is_file():
        return {}, {"status": "INVESTABILITY_METADATA_PARTIAL", "coverage_ratio": 0.0}
    frame = pd.read_parquet(path)
    records = {str(row["security_id"]): row for row in frame.to_dict("records")}
    complete = frame["exchange"].notna() & frame["category"].notna()
    return records, {
        "status": "INVESTABILITY_METADATA_PARTIAL",
        "coverage_ratio": float(complete.mean()) if len(frame) else 0.0,
        "reason": "exchange/category are identity metadata, not historical daily classifications",
    }


def _is_common_stock(metadata: Mapping[str, Any]) -> bool:
    category = str(metadata.get("category") or "").upper()
    blocked = ("WARRANT", "UNIT", "PREFERRED", "FUND", "ETF")
    return "COMMON STOCK" in category and not any(
        value in category for value in blocked
    )


def prepare_v2_frame(frame: pd.DataFrame, liquidity_lookback: int) -> pd.DataFrame:
    result = frame.sort_values("date").drop_duplicates("date", keep="last").copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["previous_close"] = result["close"].shift(1)
    dollar_volume = result["close"] * result["volume"]
    result["median_dollar_volume"] = (
        dollar_volume.rolling(liquidity_lookback, min_periods=liquidity_lookback)
        .median()
        .shift(1)
    )
    result["causal_volume_observations"] = (
        result["volume"].rolling(liquidity_lookback, min_periods=1).count().shift(1)
    )
    return result.reset_index(drop=True)


def v2_candidate_records(
    frame: pd.DataFrame,
    security_id: str,
    metadata: Mapping[str, Any],
    specs: Mapping[str, StrategySpec],
    variants: Iterable[VariantState],
    config: V2Config,
) -> list[dict[str, Any]]:
    prepared = prepare_v2_frame(frame, config.liquidity_lookback)
    cache = FeatureCache(prepared)
    by_date = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(prepared["date"])
    }
    records: list[dict[str, Any]] = []
    exchange = str(metadata.get("exchange") or "UNKNOWN").upper()
    metadata_common = _is_common_stock(metadata)
    allowed_exchange = exchange in set(config.allowed_exchanges)
    for state in variants:
        trades = specs[state.strategy].builder(prepared, cache, state.params)
        for index in range(len(trades)):
            entry_date = pd.Timestamp(trades.entry_dates[index]).normalize()
            location = by_date.get(entry_date)
            if location is None:
                continue
            row = prepared.iloc[location]
            previous_close = safe_float(row.get("previous_close"))
            median_dollar_volume = safe_float(row.get("median_dollar_volume"))
            observations = int(safe_float(row.get("causal_volume_observations"), 0.0))
            reasons: list[str] = []
            if not math.isfinite(previous_close) or previous_close < config.min_price:
                reasons.append("PRICE_BELOW_MINIMUM")
            if observations < config.liquidity_lookback:
                reasons.append("LIQUIDITY_WARMUP_INCOMPLETE")
            if (
                not math.isfinite(median_dollar_volume)
                or median_dollar_volume < config.min_median_dollar_volume
            ):
                reasons.append("ADV_BELOW_MINIMUM")
            if not metadata_common:
                reasons.append("NOT_PROVEN_COMMON_STOCK")
            if not allowed_exchange:
                reasons.append("EXCHANGE_NOT_ALLOWED")
            records.append(
                {
                    "strategy": state.strategy,
                    "family": state.family,
                    "variant_id": state.variant_id,
                    "params_json": stable_json(state.params),
                    "security_id": security_id,
                    "symbol": str(prepared["symbol"].iloc[-1]),
                    "sector": str(prepared["sector"].iloc[-1] or "UNKNOWN"),
                    "currency": str(prepared["currency"].iloc[-1] or "USD"),
                    "entry_date": entry_date,
                    "exit_date": pd.Timestamp(trades.exit_dates[index]).normalize(),
                    "entry_price": float(trades.entry_prices[index]),
                    "exit_price": float(trades.exit_prices[index]),
                    "score": float(trades.scores[index]),
                    "gross_return": float(trades.gross_returns[index]),
                    "duration": int(trades.durations[index]),
                    "forced": bool(trades.forced[index]),
                    "previous_close": previous_close,
                    "median_dollar_volume": median_dollar_volume,
                    "causal_volume_observations": observations,
                    "exchange": exchange,
                    "security_category": str(metadata.get("category") or "UNKNOWN"),
                    "investability_status": "INVESTABLE_GO"
                    if not reasons
                    else "INVESTABILITY_BLOCKED",
                    "investability_reasons": "|".join(reasons),
                }
            )
    return records


def v2_rotational_candidate_records(
    source: MarketDataSource,
    states: Sequence[VariantState],
    metadata: Mapping[str, Mapping[str, Any]],
    config: V2Config,
) -> list[dict[str, Any]]:
    allowed_ids = [
        security_id
        for security_id, values in metadata.items()
        if _is_common_stock(values)
        and str(values.get("exchange") or "").upper() in set(config.allowed_exchanges)
    ]
    if not allowed_ids or not states:
        return []
    source.con.register(
        "_v2_rotational_allowed_ids",
        pd.DataFrame({"security_id": sorted(allowed_ids)}),
    )
    records: list[dict[str, Any]] = []
    try:
        for state in states:
            params = state.params
            momentum_col = f"mom_{int(params['lookback'])}_{int(params['skip_days'])}"
            sma_col = f"sma_{int(params['trend_ma'])}"
            positive_clause = (
                f"AND {momentum_col} > 0"
                if bool(params["require_positive_momentum"])
                else ""
            )
            selected = source.con.execute(
                f"""
                WITH eligible AS (
                    SELECT m.*, {momentum_col} AS momentum
                    FROM monthly_features m
                    INNER JOIN _v2_rotational_allowed_ids a USING(security_id)
                    WHERE {momentum_col} IS NOT NULL
                      AND {sma_col} IS NOT NULL
                      AND signal_close > {sma_col}
                      AND signal_close >= ?
                      AND dollar_volume_20 >= ?
                      AND entry_open > 0 AND exit_open > 0
                      {positive_clause}
                ), ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY signal_date
                        ORDER BY momentum DESC, dollar_volume_20 DESC, security_id
                    ) AS rank_number
                    FROM eligible
                )
                SELECT * FROM ranked WHERE rank_number <= ?
                ORDER BY signal_date,rank_number
                """,
                [
                    config.min_price,
                    config.min_median_dollar_volume,
                    int(params["top_n"]),
                ],
            ).fetchdf()
            if not selected.empty:
                selected["entry_date"] = pd.to_datetime(selected["entry_date"])
                selected["exit_date"] = pd.to_datetime(selected["exit_date"])
            episodes = merge_rotational_holding_episodes(
                selected, int(params["top_n"]), state.variant_id
            )
            if episodes.empty:
                continue
            lookup = selected.set_index(["security_id", "entry_date"])
            for row in episodes.to_dict("records"):
                security_id = str(row["security_id"])
                entry_date = pd.Timestamp(row["entry_date"])
                source_row = lookup.loc[(security_id, entry_date)]
                if isinstance(source_row, pd.DataFrame):
                    source_row = source_row.iloc[0]
                identity = metadata[security_id]
                records.append(
                    {
                        "strategy": "rotational_momentum",
                        "family": "cross_sectional_momentum",
                        "variant_id": state.variant_id,
                        "params_json": stable_json(state.params),
                        "security_id": security_id,
                        "symbol": str(row["symbol"]),
                        "sector": str(
                            identity.get("sector")
                            or source_row.get("sector")
                            or "UNKNOWN"
                        ),
                        "currency": str(identity.get("currency") or "USD"),
                        "entry_date": entry_date,
                        "exit_date": pd.Timestamp(row["exit_date"]),
                        "entry_price": float(row["entry_price"]),
                        "exit_price": float(row["exit_price"]),
                        "score": float(row["score"]),
                        "gross_return": float(row["gross_return"]),
                        "duration": int(row["duration"]),
                        "forced": bool(row["forced"]),
                        "previous_close": float(source_row["signal_close"]),
                        "median_dollar_volume": float(source_row["dollar_volume_20"]),
                        "causal_volume_observations": config.liquidity_lookback,
                        "exchange": str(identity.get("exchange") or "UNKNOWN").upper(),
                        "security_category": str(identity.get("category") or "UNKNOWN"),
                        "investability_status": "INVESTABLE_GO",
                        "investability_reasons": "",
                    }
                )
    finally:
        source.con.unregister("_v2_rotational_allowed_ids")
    return records


def _fee_eur(notional: float, config: V2Config) -> float:
    variable = (
        config.cost_bps_per_side
        + config.slippage_bps_per_side
        + config.fx_cost_bps_per_side
    ) / 10_000.0
    return config.fixed_fee_eur + notional * variable


def _local_to_eur(value: float, currency: str, eur_per_usd: float) -> float:
    normalized = currency.upper()
    if normalized == "EUR":
        return value
    if normalized == "USD":
        return value * eur_per_usd
    raise RuntimeError(f"UNSUPPORTED_CURRENCY_BLOCKED:{normalized}")


def _target_weights(
    active: Mapping[tuple[str, str, str], Mapping[str, Any]],
    strategy_weights: Mapping[str, float],
    config: V2Config,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, Mapping[str, Any]],
    list[dict[str, Any]],
]:
    by_strategy: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in active.values():
        strategy = str(row["strategy"])
        security_id = str(row["security_id"])
        previous = by_strategy[strategy].get(security_id)
        if previous is None or float(row["score"]) > float(previous["score"]):
            by_strategy[strategy][security_id] = row
    raw: dict[str, float] = defaultdict(float)
    attribution: dict[str, dict[str, float]] = defaultdict(dict)
    representatives: dict[str, Mapping[str, Any]] = {}
    for strategy, securities in by_strategy.items():
        ranked = sorted(
            securities.values(),
            key=lambda row: (-float(row["score"]), str(row["security_id"])),
        )
        chosen = ranked[: config.global_max_positions]
        if not chosen:
            continue
        per_security = float(strategy_weights.get(strategy, 0.0)) / len(chosen)
        for row in chosen:
            security_id = str(row["security_id"])
            raw[security_id] += per_security
            attribution[security_id][strategy] = per_security
            representatives[security_id] = row
    audit: list[dict[str, Any]] = []
    final: dict[str, float] = {}
    sector_used: dict[str, float] = defaultdict(float)
    ranked_targets = sorted(
        raw, key=lambda sid: (-raw[sid], -float(representatives[sid]["score"]), sid)
    )
    for security_id in ranked_targets:
        row = representatives[security_id]
        sector = str(row.get("sector") or "UNKNOWN")
        reason = ""
        if len(final) >= config.global_max_positions:
            reason = "GLOBAL_POSITION_CAP"
            weight = 0.0
        else:
            weight = min(raw[security_id], config.max_security_weight)
            sector_room = max(config.max_sector_weight - sector_used[sector], 0.0)
            weight = min(weight, sector_room)
            if weight <= 0:
                reason = "SECTOR_CAP"
        if weight > 0:
            final[security_id] = weight
            sector_used[sector] += weight
            scale = weight / raw[security_id] if raw[security_id] else 0.0
            attribution[security_id] = {
                name: value * scale for name, value in attribution[security_id].items()
            }
        audit.append(
            {
                "security_id": security_id,
                "strategy_source_count": len(attribution[security_id]),
                "strategy_sources": "+".join(sorted(attribution[security_id])),
                "gross_requested_weight": raw[security_id],
                "net_target_weight": weight,
                "constraint_reason": reason,
            }
        )
    gross = sum(final.values())
    if gross > config.max_gross_exposure:
        scale = config.max_gross_exposure / gross
        final = {security_id: value * scale for security_id, value in final.items()}
        attribution = {
            security_id: {name: value * scale for name, value in values.items()}
            for security_id, values in attribution.items()
        }
    return final, attribution, representatives, audit


def _ledger_metrics(
    ledger: pd.DataFrame, fills: pd.DataFrame, initial_capital: float
) -> dict[str, Any]:
    if ledger.empty:
        return {"status": "INSUFFICIENT_SAMPLE", "observations": 0}
    returns = ledger.set_index("date")["daily_return"].astype(float)
    metrics = daily_metrics(returns, initial_capital)
    metrics.update(
        {
            "status": "GO",
            "calendar_observation_count": len(ledger),
            "cash_day_count": int((ledger["open_position_count"] == 0).sum()),
            "invested_day_count": int((ledger["open_position_count"] > 0).sum()),
            "missing_calendar_days": 0,
            "trade_count": int(
                (fills.get("side", pd.Series(dtype=str)) == "BUY").sum()
            ),
            "turnover_eur": float(ledger["turnover_eur"].sum()),
            "transaction_costs_eur": float(ledger["transaction_costs_eur"].sum()),
            "average_exposure": float(ledger["gross_exposure"].mean()),
            "maximum_security_weight": float(ledger["maximum_security_weight"].max()),
            "maximum_sector_weight": float(ledger["maximum_sector_weight"].max()),
            "accounting_failure_count": int(
                (ledger["reconciliation_error_eur"].abs() > 1e-7).sum()
            ),
        }
    )
    return metrics


def run_global_ledger(
    candidates: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    fx: pd.Series,
    config: V2Config,
    strategy_weights: Mapping[str, float],
    *,
    portfolio_name: str,
) -> V2LedgerResult:
    candidates = candidates.copy()
    if not candidates.empty:
        candidates["entry_date"] = pd.to_datetime(
            candidates["entry_date"]
        ).dt.normalize()
        candidates["exit_date"] = pd.to_datetime(candidates["exit_date"]).dt.normalize()
        candidates = candidates.loc[
            candidates["investability_status"].eq("INVESTABLE_GO")
        ]
    entries: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
    exits: dict[pd.Timestamp, list[tuple[str, str, str]]] = defaultdict(list)
    for row in candidates.to_dict("records"):
        key = (str(row["strategy"]), str(row["security_id"]), str(row["entry_date"]))
        row["_active_key"] = key
        entries[pd.Timestamp(row["entry_date"])].append(row)
        exits[pd.Timestamp(row["exit_date"])].append(key)
    indexed_frames = {
        security_id: frame.assign(date=pd.to_datetime(frame["date"]).dt.normalize())
        .drop_duplicates("date", keep="last")
        .set_index("date")
        for security_id, frame in frames.items()
    }
    positions: dict[str, V2Position] = {}
    active: dict[tuple[str, str, str], dict[str, Any]] = {}
    cash = float(config.initial_capital)
    previous_nav = cash
    realized_total = 0.0
    rows: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    contributor_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    accounting_failures = 0
    unexplained_outliers = 0
    for day in calendar:
        day = pd.Timestamp(day).normalize()
        fx_rate = safe_float(fx.get(day))
        if not math.isfinite(fx_rate) or fx_rate <= 0:
            raise RuntimeError(f"FX_RATE_UNAVAILABLE_BLOCKED:{day.date()}")
        before = {
            security_id: (
                position.shares,
                position.last_mark_local,
                position.last_mark_eur,
                position.currency,
                dict(position.strategy_attribution_weights),
                position.ticker_for_display,
                position.median_dollar_volume,
            )
            for security_id, position in positions.items()
        }
        open_prices: dict[str, tuple[float, float]] = {}
        close_prices: dict[str, tuple[float, float]] = {}
        stale_mark_count = 0
        for security_id in set(positions) | {
            str(row["security_id"]) for row in entries.get(day, [])
        }:
            frame = indexed_frames.get(security_id)
            if frame is None or day not in frame.index:
                stale_mark_count += int(security_id in positions)
                continue
            bar = frame.loc[day]
            open_local = safe_float(bar["open"])
            close_local = safe_float(bar["close"])
            currency = str(
                bar.get(
                    "currency",
                    positions[security_id].currency
                    if security_id in positions
                    else "USD",
                )
                or "USD"
            )
            if open_local > 0 and close_local > 0:
                open_prices[security_id] = (
                    open_local,
                    _local_to_eur(open_local, currency, fx_rate),
                )
                close_prices[security_id] = (
                    close_local,
                    _local_to_eur(close_local, currency, fx_rate),
                )
        for key in exits.get(day, []):
            active.pop(key, None)
        for row in entries.get(day, []):
            active[row["_active_key"]] = row
        target_weights, target_attribution, representatives, target_audit = (
            _target_weights(active, strategy_weights, config)
        )
        target_sector_weights: dict[str, float] = defaultdict(float)
        for security_id, weight in target_weights.items():
            target_sector_weights[
                str(representatives[security_id].get("sector") or "UNKNOWN")
            ] += weight
        for audit in target_audit:
            overlap_rows.append({"date": day, "portfolio": portfolio_name, **audit})
        open_market_value = sum(
            position.shares
            * open_prices.get(
                security_id, (position.last_mark_local, position.last_mark_eur)
            )[1]
            for security_id, position in positions.items()
        )
        nav_at_open = cash + open_market_value
        rebalance_required = bool(entries.get(day) or exits.get(day))
        desired_shares: dict[str, int] = {
            security_id: position.shares for security_id, position in positions.items()
        }
        if rebalance_required:
            desired_shares = {}
            for security_id, weight in target_weights.items():
                if security_id not in open_prices:
                    continue
                price_eur = open_prices[security_id][1]
                desired_shares[security_id] = math.floor(
                    nav_at_open * weight / price_eur
                )
        daily_fees = 0.0
        daily_turnover = 0.0
        rejected = 0

        def record_order(
            security_id: str,
            side: str,
            quantity: int,
            status: str,
            reason: str,
            notional: float,
            fee: float,
        ) -> None:
            orders.append(
                {
                    "date": day,
                    "portfolio": portfolio_name,
                    "order_id": sha256_text(
                        f"{portfolio_name}|{day}|{security_id}|{side}"
                    )[:20],
                    "security_id": security_id,
                    "side": side,
                    "quantity": quantity,
                    "order_notional_eur": notional,
                    "fee_eur": fee,
                    "status": status,
                    "rejection_reason": reason,
                }
            )

        for security_id in sorted(set(positions) - set(desired_shares)):
            desired_shares[security_id] = 0
        for security_id in sorted(desired_shares):
            current = positions.get(security_id)
            if current is None:
                continue
            current_shares = current.shares if current else 0
            target_shares = desired_shares[security_id]
            if target_shares >= current_shares or security_id not in open_prices:
                continue
            quantity = current_shares - target_shares
            open_local, open_eur = open_prices[security_id]
            notional = quantity * open_eur
            fee = _fee_eur(notional, config)
            proceeds = notional - fee
            cash += proceeds
            realized = quantity * (open_eur - current.average_cost_eur) - fee
            current.realized_pnl_eur += realized
            realized_total += realized
            current.shares -= quantity
            current.last_trade_date = day
            daily_fees += fee
            daily_turnover += notional
            record_order(security_id, "SELL", quantity, "FILLED", "", notional, fee)
            fills.append(
                {
                    "date": day,
                    "portfolio": portfolio_name,
                    "security_id": security_id,
                    "side": "SELL",
                    "shares": quantity,
                    "price_local": open_local,
                    "price_eur": open_eur,
                    "notional_eur": notional,
                    "fee_eur": fee,
                }
            )
            if current.shares == 0:
                positions.pop(security_id)
        for security_id in sorted(desired_shares):
            current_shares = (
                positions[security_id].shares if security_id in positions else 0
            )
            target_shares = desired_shares[security_id]
            if (
                target_shares == 0
                and current_shares == 0
                and security_id in target_weights
                and security_id in open_prices
            ):
                rejected += 1
                record_order(
                    security_id,
                    "BUY",
                    0,
                    "REJECTED",
                    "ORDER_REJECTED_UNAFFORDABLE_WHOLE_SHARE",
                    0.0,
                    0.0,
                )
                continue
            if target_shares <= current_shares or security_id not in open_prices:
                continue
            requested = target_shares - current_shares
            open_local, open_eur = open_prices[security_id]
            representative = representatives[security_id]
            adv_eur = _local_to_eur(
                safe_float(representative.get("median_dollar_volume"), 0.0),
                str(representative.get("currency") or "USD"),
                fx_rate,
            )
            max_adv_shares = math.floor(
                adv_eur * config.max_order_adv_fraction / open_eur
            )
            affordable = math.floor(
                max((cash - config.fixed_fee_eur), 0.0)
                / (
                    open_eur
                    * (
                        1
                        + (
                            config.cost_bps_per_side
                            + config.slippage_bps_per_side
                            + config.fx_cost_bps_per_side
                        )
                        / 10_000.0
                    )
                )
            )
            quantity = min(requested, max_adv_shares, affordable)
            reason = ""
            if quantity <= 0:
                reason = (
                    "ORDER_REJECTED_UNAFFORDABLE_WHOLE_SHARE"
                    if affordable <= 0
                    else "ORDER_REJECTED_ADV_CAP"
                )
            notional = quantity * open_eur
            fee = _fee_eur(notional, config) if quantity > 0 else 0.0
            if quantity > 0 and notional < config.minimum_order_eur:
                reason = "ORDER_REJECTED_MINIMUM_NOTIONAL"
                quantity = 0
                notional = 0.0
                fee = 0.0
            if quantity <= 0:
                rejected += 1
                record_order(security_id, "BUY", 0, "REJECTED", reason, 0.0, 0.0)
                continue
            cash -= notional + fee
            if cash < -1e-8:
                raise RuntimeError("NEGATIVE_CASH_INVARIANT")
            position = positions.get(security_id)
            if position:
                total_cost = (
                    position.shares * position.average_cost_eur + notional + fee
                )
                position.shares += quantity
                position.average_cost_eur = total_cost / position.shares
                position.average_cost_local = (
                    position.average_cost_local * (position.shares - quantity)
                    + open_local * quantity
                ) / position.shares
                position.last_trade_date = day
                position.strategy_attribution_weights = target_attribution.get(
                    security_id, {}
                )
            else:
                positions[security_id] = V2Position(
                    security_id=security_id,
                    ticker_for_display=str(representative["symbol"]),
                    shares=quantity,
                    average_cost_local=open_local,
                    average_cost_eur=(notional + fee) / quantity,
                    last_mark_local=open_local,
                    last_mark_eur=open_eur,
                    realized_pnl_eur=0.0,
                    strategy_attribution_weights=target_attribution.get(
                        security_id, {}
                    ),
                    entry_date=day,
                    last_trade_date=day,
                    sector=str(representative.get("sector") or "UNKNOWN"),
                    currency=str(representative.get("currency") or "USD"),
                    median_dollar_volume=safe_float(
                        representative.get("median_dollar_volume")
                    ),
                )
            daily_fees += fee
            daily_turnover += notional
            record_order(security_id, "BUY", quantity, "FILLED", "", notional, fee)
            fills.append(
                {
                    "date": day,
                    "portfolio": portfolio_name,
                    "security_id": security_id,
                    "side": "BUY",
                    "shares": quantity,
                    "price_local": open_local,
                    "price_eur": open_eur,
                    "notional_eur": notional,
                    "fee_eur": fee,
                }
            )
        if rebalance_required:
            for security_id, position in positions.items():
                if security_id in target_attribution:
                    position.strategy_attribution_weights = target_attribution[
                        security_id
                    ]
        local_price_pnl_total = 0.0
        fx_impact_total = 0.0
        for security_id in sorted(set(before) | set(positions)):
            (
                shares_before,
                previous_mark_local,
                previous_mark,
                currency_before,
                attribution_before,
                ticker_before,
                liquidity_before,
            ) = before.get(
                security_id,
                (
                    0,
                    0.0,
                    0.0,
                    positions[security_id].currency
                    if security_id in positions
                    else "USD",
                    {},
                    positions[security_id].ticker_for_display
                    if security_id in positions
                    else "",
                    positions[security_id].median_dollar_volume
                    if security_id in positions
                    else float("nan"),
                ),
            )
            position = positions.get(security_id)
            shares_after = position.shares if position else 0
            open_local, open_eur = open_prices.get(
                security_id, (previous_mark_local, previous_mark)
            )
            close_local, close_eur = close_prices.get(
                security_id,
                (
                    position.last_mark_local if position else open_local,
                    position.last_mark_eur if position else open_eur,
                ),
            )
            previous_conversion = (
                previous_mark / previous_mark_local
                if previous_mark_local > 0
                else _local_to_eur(1.0, currency_before, fx_rate)
            )
            current_conversion = (
                open_eur / open_local if open_local > 0 else previous_conversion
            )
            local_price_pnl = (
                shares_before * (open_local - previous_mark_local) * previous_conversion
                + shares_after * (close_local - open_local) * current_conversion
            )
            fx_impact = (
                shares_before * open_local * (current_conversion - previous_conversion)
            )
            security_pnl = local_price_pnl + fx_impact
            local_price_pnl_total += local_price_pnl
            fx_impact_total += fx_impact
            sources = (
                position.strategy_attribution_weights
                if position
                else attribution_before
            )
            source_total = sum(sources.values())
            market_value_for_security = position.shares * close_eur if position else 0.0
            for strategy, source_weight in sources.items():
                allocation = source_weight / source_total if source_total > 0 else 0.0
                attribution_rows.append(
                    {
                        "date": day,
                        "portfolio": portfolio_name,
                        "security_id": security_id,
                        "strategy": strategy,
                        "target_weight": source_weight,
                        "market_value_eur": market_value_for_security * allocation,
                        "strategy_pnl_eur": security_pnl * allocation,
                        "strategy_contribution": security_pnl
                        * allocation
                        / previous_nav
                        if previous_nav > 0
                        else 0.0,
                    }
                )
            contributor_rows.append(
                {
                    "date": day,
                    "portfolio": portfolio_name,
                    "security_id": security_id,
                    "symbol": position.ticker_for_display
                    if position
                    else ticker_before,
                    "shares": shares_after,
                    "previous_mark": previous_mark,
                    "current_mark": close_eur,
                    "security_return": close_eur / previous_mark - 1
                    if previous_mark > 0
                    else 0.0,
                    "portfolio_weight_before": shares_before
                    * previous_mark
                    / previous_nav
                    if previous_nav > 0
                    else 0.0,
                    "portfolio_contribution": security_pnl / previous_nav
                    if previous_nav > 0
                    else 0.0,
                    "local_price_pnl_eur": local_price_pnl,
                    "fx_impact_eur": fx_impact,
                    "total_security_pnl_eur": security_pnl,
                    "strategy_sources": "+".join(sorted(sources)),
                    "median_dollar_volume_20d": position.median_dollar_volume
                    if position
                    else liquidity_before,
                    "price": close_local,
                    "data_quality_status": "CORPORATE_ACTION_GATE_GO",
                    "investability_status": "INVESTABLE_GO",
                    "corporate_action_status": "GO",
                }
            )
            if position:
                position.last_mark_local = close_local
                position.last_mark_eur = close_eur
        market_value = sum(
            position.shares * position.last_mark_eur for position in positions.values()
        )
        nav = cash + market_value
        reconciliation_error = (nav - previous_nav) - (
            local_price_pnl_total + fx_impact_total - daily_fees
        )
        accounting_failures += int(abs(reconciliation_error) > 1e-7)
        daily_return = nav / previous_nav - 1.0 if previous_nav > 0 else 0.0
        if (
            abs(daily_return) > config.equity_hard_fail_return_threshold
            and abs(reconciliation_error) > 1e-7
        ):
            unexplained_outliers += 1
        sector_values: dict[str, float] = defaultdict(float)
        for position in positions.values():
            sector_values[position.sector] += position.shares * position.last_mark_eur
        unrealized = sum(
            position.shares * (position.last_mark_eur - position.average_cost_eur)
            for position in positions.values()
        )
        rows.append(
            {
                "date": day,
                "cash_eur": cash,
                "gross_market_value_eur": market_value,
                "net_market_value_eur": market_value,
                "portfolio_nav_eur": nav,
                "daily_return": daily_return,
                "realized_pnl_eur": realized_total,
                "unrealized_pnl_eur": unrealized,
                "transaction_costs_eur": daily_fees,
                "turnover_eur": daily_turnover,
                "open_position_count": len(positions),
                "unique_security_count": len(positions),
                "gross_exposure": market_value / nav if nav > 0 else 0.0,
                "net_exposure": market_value / nav if nav > 0 else 0.0,
                "maximum_security_weight": max(
                    (
                        position.shares * position.last_mark_eur / nav
                        for position in positions.values()
                    ),
                    default=0.0,
                ),
                "maximum_sector_weight": max(
                    (value / nav for value in sector_values.values()), default=0.0
                ),
                "target_gross_exposure": sum(target_weights.values()),
                "maximum_target_security_weight": max(
                    target_weights.values(), default=0.0
                ),
                "maximum_target_sector_weight": max(
                    target_sector_weights.values(), default=0.0
                ),
                "stale_mark_count": stale_mark_count,
                "rejected_order_count": rejected,
                "security_pnl_eur": local_price_pnl_total,
                "fee_contribution_eur": -daily_fees,
                "fx_impact_eur": fx_impact_total,
                "cash_return_eur": 0.0,
                "reconciliation_error_eur": reconciliation_error,
            }
        )
        for position in positions.values():
            value = position.shares * position.last_mark_eur
            position_rows.append(
                {
                    "date": day,
                    "portfolio": portfolio_name,
                    "security_id": position.security_id,
                    "ticker_for_display": position.ticker_for_display,
                    "shares": position.shares,
                    "average_cost_local": position.average_cost_local,
                    "average_cost_eur": position.average_cost_eur,
                    "last_mark_local": position.last_mark_local,
                    "last_mark_eur": position.last_mark_eur,
                    "market_value_eur": value,
                    "unrealized_pnl_eur": position.shares
                    * (position.last_mark_eur - position.average_cost_eur),
                    "realized_pnl_eur": position.realized_pnl_eur,
                    "strategy_attribution_weights": stable_json(
                        position.strategy_attribution_weights
                    ),
                    "entry_date": position.entry_date,
                    "last_trade_date": position.last_trade_date,
                    "sector": position.sector,
                    "currency": position.currency,
                }
            )
        previous_nav = nav
    ledger = pd.DataFrame(rows)
    orders_frame = pd.DataFrame(orders)
    fills_frame = pd.DataFrame(fills)
    result = V2LedgerResult(
        ledger=ledger,
        orders=orders_frame,
        fills=fills_frame,
        positions=pd.DataFrame(position_rows),
        attribution=pd.DataFrame(attribution_rows),
        contributors=pd.DataFrame(contributor_rows),
        overlap=pd.DataFrame(overlap_rows),
        metrics=_ledger_metrics(ledger, fills_frame, config.initial_capital),
        accounting_failures=accounting_failures,
        unexplained_outliers=unexplained_outliers,
    )
    return result


def v2_validation_score(
    metrics: Mapping[str, Any], config: V2Config
) -> tuple[float, list[str]]:
    trades = int(metrics.get("trade_count") or 0)
    cagr = safe_float(metrics.get("CAGR"), -1.0)
    sharpe = safe_float(metrics.get("Sharpe"), -10.0)
    calmar = safe_float(metrics.get("Calmar"), -10.0)
    pf = safe_float(metrics.get("daily_profit_factor"), 0.0)
    drawdown = safe_float(metrics.get("maximum_drawdown"), -1.0)
    turnover = safe_float(metrics.get("turnover_eur"), 0.0) / max(
        config.initial_capital, 1.0
    )
    concentration = max(
        safe_float(metrics.get("maximum_security_weight"), 1.0),
        safe_float(metrics.get("maximum_sector_weight"), 1.0),
    )
    failures: list[str] = []
    if trades < config.min_validation_trades:
        failures.append("MINIMUM_VALIDATION_TRADES")
    if cagr <= 0:
        failures.append("VALIDATION_CAGR_NOT_POSITIVE")
    if pf <= 1:
        failures.append("VALIDATION_PF_NOT_ABOVE_ONE")
    if drawdown < config.validation_max_drawdown:
        failures.append("VALIDATION_DRAWDOWN_GATE")
    if int(metrics.get("accounting_failure_count") or 0):
        failures.append("ACCOUNTING_FAILURE")
    score = (
        float(np.clip(sharpe, -3, 3))
        + float(np.clip(calmar, -3, 3))
        + math.log(max(pf, 1e-9))
        + 0.25 * safe_float(metrics.get("positive_year_ratio"), 0.0)
        - 2.0 * max(abs(drawdown) - 0.20, 0.0)
        - 0.01 * turnover
        - 0.5 * concentration
        - max(config.min_validation_trades - trades, 0)
        / max(config.min_validation_trades, 1)
    )
    return (score if not failures else -1e12 + score), failures


def _frames_and_calendar(
    source: MarketDataSource, config: V2Config
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    frames: dict[str, pd.DataFrame] = {}
    dates: set[pd.Timestamp] = set()
    for batch in source.iter_batches(config.batch_size):
        for security_id, frame in batch.groupby("security_id", sort=False):
            prepared = prepare_v2_frame(frame, config.liquidity_lookback)
            frames[str(security_id)] = prepared
            dates.update(pd.to_datetime(prepared["date"]).dt.normalize())
    calendar = pd.DatetimeIndex(
        sorted(
            date
            for date in dates
            if parse_date(config.start) <= date <= parse_date(config.end)
        )
    )
    return frames, calendar


def _slice_ledger_inputs(
    candidates: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sliced = candidates.loc[
        (pd.to_datetime(candidates["entry_date"]) >= start)
        & (pd.to_datetime(candidates["entry_date"]) <= end)
    ].copy()
    return sliced, calendar[(calendar >= start) & (calendar <= end)]


def _write_frame(path: Path, frame: pd.DataFrame, columns: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = frame.copy()
    if value.empty and columns:
        value = pd.DataFrame(columns=list(columns))
    if path.suffix == ".parquet":
        value.to_parquet(path, index=False)
    else:
        value.to_csv(path, index=False)


def _flatten_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    }


def _extreme_audits(
    result: V2LedgerResult, config: V2Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    extreme = result.ledger.loc[
        result.ledger["daily_return"].abs() >= config.equity_extreme_return_threshold
    ].copy()
    if extreme.empty:
        return extreme, pd.DataFrame(columns=result.contributors.columns)
    extreme["nav_before"] = extreme["portfolio_nav_eur"] / (1 + extreme["daily_return"])
    extreme["nav_after"] = extreme["portfolio_nav_eur"]
    contributors = result.contributors.loc[
        result.contributors["date"].isin(extreme["date"])
    ].copy()
    return extreme, contributors


def _create_v2_candidate_table(con: duckdb.DuckDBPyConnection, reset: bool) -> None:
    if reset:
        con.execute("DROP TABLE IF EXISTS v2_candidates")
        con.execute("DROP TABLE IF EXISTS v2_processed_identities")
    con.execute(
        """CREATE TABLE IF NOT EXISTS v2_candidates(
        strategy VARCHAR, "family" VARCHAR, variant_id VARCHAR, params_json VARCHAR,
        security_id VARCHAR, symbol VARCHAR, sector VARCHAR, currency VARCHAR,
        entry_date TIMESTAMP, exit_date TIMESTAMP, entry_price DOUBLE, exit_price DOUBLE,
        score DOUBLE, gross_return DOUBLE, duration INTEGER, forced BOOLEAN,
        previous_close DOUBLE, median_dollar_volume DOUBLE, causal_volume_observations INTEGER,
        exchange VARCHAR, security_category VARCHAR, investability_status VARCHAR,
        investability_reasons VARCHAR)"""
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS v2_processed_identities(security_id VARCHAR PRIMARY KEY)"
    )


def _v2_database_fingerprint(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    candidate_row = con.execute(
        """SELECT COUNT(*),COALESCE(BIT_XOR(HASH(
        strategy,variant_id,security_id,entry_date,exit_date,investability_status
        )),0) FROM v2_candidates"""
    ).fetchone()
    identity_row = con.execute(
        "SELECT COUNT(*),COALESCE(BIT_XOR(HASH(security_id)),0) FROM v2_processed_identities"
    ).fetchone()
    if candidate_row is None or identity_row is None:
        raise RuntimeError("V2 database fingerprint query returned no row")
    candidate_count, candidate_xor = candidate_row
    identity_count, identity_xor = identity_row
    return {
        "candidate_count": int(candidate_count),
        "candidate_hash_xor": int(candidate_xor),
        "processed_identity_count": int(identity_count),
        "processed_identity_hash_xor": int(identity_xor),
    }


def run_lab_v2(config: V2Config) -> Path:
    started = time.time()
    project_root = Path(__file__).resolve().parent
    data_path = Path(config.data) if config.data else discover_default_data()
    fingerprints = v2_run_fingerprint(config, data_path)
    output_dir = Path(config.output) / f"run_{fingerprints['run_id']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    database_path = output_dir / "research_lab.duckdb"
    source = MarketDataSource(
        data_path,
        parse_date(config.start),
        parse_date(config.end),
        config.min_bars,
        config.max_symbols,
        config.seed,
        database_path,
        corporate_action_gate=config.corporate_action_gate,
        overnight_ratio_min=config.overnight_ratio_min,
        overnight_ratio_max=config.overnight_ratio_max,
    )
    specs, variants, grids = v2_registry(config)
    spec_map = {spec.name: spec for spec in specs}
    ordinary_variants = [
        state for state in variants.values() if state.strategy in spec_map
    ]
    rotational_variants = [
        state for state in variants.values() if state.strategy == "rotational_momentum"
    ]
    metadata, metadata_status = load_v2_security_metadata(project_root)
    source.quality_exclusions().to_csv(
        output_dir / "data_quality_exclusions.csv", index=False
    )
    _create_v2_candidate_table(source.con, reset=not config.resume)
    processed = {
        str(row[0])
        for row in source.con.execute(
            "SELECT security_id FROM v2_processed_identities"
        ).fetchall()
    }
    for batch_number, batch in enumerate(
        source.iter_batches(config.batch_size), start=1
    ):
        for security_id, frame in batch.groupby("security_id", sort=False):
            security_id = str(security_id)
            if security_id in processed:
                continue
            records = v2_candidate_records(
                frame,
                security_id,
                metadata.get(security_id, {}),
                spec_map,
                ordinary_variants,
                config,
            )
            source.con.execute(
                "DELETE FROM v2_candidates WHERE security_id=?", [security_id]
            )
            if records:
                append = pd.DataFrame(records)
                source.con.register("_v2_append", append)
                source.con.execute("INSERT INTO v2_candidates SELECT * FROM _v2_append")
                source.con.unregister("_v2_append")
            source.con.execute(
                "INSERT OR IGNORE INTO v2_processed_identities VALUES (?)",
                [security_id],
            )
            processed.add(security_id)
        atomic_write_json(
            progress_path,
            {
                "status": "CANDIDATE_GENERATION_RUNNING",
                "run_id": fingerprints["run_id"],
                "processed_identities": len(processed),
                "total_identities": len(source.identities),
                "batch": batch_number,
                "updated_at": utc_now_iso(),
            },
        )
    if rotational_variants:
        rebuild_monthly_feature_table(source)
        rotational_records = v2_rotational_candidate_records(
            source,
            rotational_variants,
            metadata,
            config,
        )
        source.con.execute(
            "DELETE FROM v2_candidates WHERE strategy='rotational_momentum'"
        )
        if rotational_records:
            rotational_append = pd.DataFrame(rotational_records)
            source.con.register("_v2_rotational_append", rotational_append)
            source.con.execute(
                "INSERT INTO v2_candidates SELECT * FROM _v2_rotational_append"
            )
            source.con.unregister("_v2_rotational_append")
        atomic_write_json(
            progress_path,
            {
                "status": "ROTATIONAL_CANDIDATE_GENERATION_COMPLETE",
                "run_id": fingerprints["run_id"],
                "rotational_candidate_count": len(rotational_records),
                "updated_at": utc_now_iso(),
            },
        )
    candidates = source.con.execute(
        "SELECT * FROM v2_candidates ORDER BY variant_id,entry_date,security_id"
    ).fetchdf()
    investability_exclusions = candidates.loc[
        candidates["investability_status"].ne("INVESTABLE_GO")
    ].copy()
    investability_exclusions.to_csv(
        output_dir / "investability_exclusions.csv", index=False
    )
    coverage = {
        "schema": "strategy_combo_lab_v2_investability_coverage_v1",
        "status": metadata_status["status"],
        "candidate_count": len(candidates),
        "investable_candidate_count": int(
            candidates["investability_status"].eq("INVESTABLE_GO").sum()
        ),
        "coverage_ratio": float(
            candidates["investability_status"].eq("INVESTABLE_GO").mean()
        )
        if len(candidates)
        else 0.0,
        "historical_metadata_coverage_ratio": metadata_status["coverage_ratio"],
        "candidate_validity": False,
        "reason": metadata_status.get("reason"),
    }
    atomic_write_json(output_dir / "investability_coverage.json", coverage)
    frames, calendar = _frames_and_calendar(source, config)
    fx = load_v2_fx(project_root, calendar)
    if fx.isna().any():
        raise RuntimeError("FX_RATE_UNAVAILABLE_BLOCKED")
    train_start = parse_date(config.start)
    train_end = parse_date(config.train_end)
    validation_start = train_end + pd.Timedelta(days=1)
    validation_end = parse_date(config.validation_end)
    test_start = validation_end + pd.Timedelta(days=1)
    test_end = parse_date(config.end)
    variant_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    winner_states: dict[str, VariantState] = {}
    winner_validation_results: dict[str, V2LedgerResult] = {}
    winner_candidates: dict[str, pd.DataFrame] = {}
    for strategy in sorted({state.strategy for state in variants.values()}):
        ranked: list[
            tuple[float, VariantState, V2LedgerResult, dict[str, Any], list[str]]
        ] = []
        for state in [
            value for value in variants.values() if value.strategy == strategy
        ]:
            subset = candidates.loc[candidates["variant_id"].eq(state.variant_id)]
            selection_candidates, selection_calendar = _slice_ledger_inputs(
                subset, calendar, train_start, validation_end
            )
            selection_result = run_global_ledger(
                selection_candidates,
                frames,
                selection_calendar,
                fx,
                config,
                {strategy: 1.0},
                portfolio_name=f"variant:{state.variant_id}:selection",
            )
            validation_ledger = selection_result.ledger.loc[
                selection_result.ledger["date"].between(
                    validation_start, validation_end
                )
            ]
            if selection_result.fills.empty:
                validation_fills = selection_result.fills
            else:
                validation_fill_dates = pd.to_datetime(selection_result.fills["date"])
                validation_fills = selection_result.fills.loc[
                    validation_fill_dates.between(validation_start, validation_end)
                ]
            validation_metrics = _ledger_metrics(
                validation_ledger, validation_fills, config.initial_capital
            )
            score, failures = v2_validation_score(validation_metrics, config)
            row = {
                "strategy": strategy,
                "family": state.family,
                "variant_id": state.variant_id,
                "params": stable_json(state.params),
                "validation_score": score,
                "hard_gate_status": "GO"
                if not failures
                else "NO_VALID_PARAMETER_VARIANT",
                "hard_gate_failures": "|".join(failures),
                **_flatten_metrics("validation", validation_metrics),
            }
            variant_rows.append(row)
            ranked.append(
                (score, state, selection_result, validation_metrics, failures)
            )
        ranked.sort(key=lambda item: (-item[0], item[1].variant_id))
        valid = [item for item in ranked if not item[4]]
        if not valid:
            invalid_rows.extend(
                row for row in variant_rows if row["strategy"] == strategy
            )
            continue
        score, winner, result, _, _ = valid[0]
        winner_states[strategy] = winner
        winner_validation_results[strategy] = result
        winner_candidates[strategy] = candidates.loc[
            candidates["variant_id"].eq(winner.variant_id)
        ].copy()
    variant_df = pd.DataFrame(variant_rows)
    invalid_df = pd.DataFrame(invalid_rows)
    _write_frame(output_dir / "individual_variants.csv", variant_df)
    _write_frame(output_dir / "invalid_strategy_variants.csv", invalid_df)
    winner_payload = {
        strategy: {
            "variant_id": state.variant_id,
            "params": state.params,
            "winner_hash": sha256_text(
                stable_json(
                    {
                        "strategy": strategy,
                        "variant_id": state.variant_id,
                        "params": state.params,
                    }
                )
            ),
            "selection_closed_before_test": True,
        }
        for strategy, state in sorted(winner_states.items())
    }
    atomic_write_json(output_dir / "winner_parameters.json", winner_payload)
    winner_rows: list[dict[str, Any]] = []
    individual_portfolios: list[dict[str, Any]] = []
    winner_test_results: dict[str, V2LedgerResult] = {}
    for strategy, state in sorted(winner_states.items()):
        evaluation_candidates, evaluation_calendar = _slice_ledger_inputs(
            winner_candidates[strategy], calendar, train_start, test_end
        )
        test_result = run_global_ledger(
            evaluation_candidates,
            frames,
            evaluation_calendar,
            fx,
            config,
            {strategy: 1.0},
            portfolio_name=f"individual:{strategy}:test",
        )
        winner_test_results[strategy] = test_result
        validation_metrics = _ledger_metrics(
            winner_validation_results[strategy].ledger.loc[
                winner_validation_results[strategy]
                .ledger["date"]
                .between(validation_start, validation_end)
            ],
            winner_validation_results[strategy].fills.loc[
                pd.to_datetime(
                    winner_validation_results[strategy].fills["date"]
                ).between(validation_start, validation_end)
            ]
            if not winner_validation_results[strategy].fills.empty
            else winner_validation_results[strategy].fills,
            config.initial_capital,
        )
        test_ledger = test_result.ledger.loc[
            test_result.ledger["date"].between(test_start, test_end)
        ]
        if test_result.fills.empty:
            test_fills = test_result.fills
        else:
            test_fill_dates = pd.to_datetime(test_result.fills["date"])
            test_fills = test_result.fills.loc[
                test_fill_dates.between(test_start, test_end)
            ]
        test_metrics = _ledger_metrics(test_ledger, test_fills, config.initial_capital)
        row = {
            "strategy": strategy,
            "family": state.family,
            "variant_id": state.variant_id,
            "params": stable_json(state.params),
            "winner_hash": winner_payload[strategy]["winner_hash"],
            **_flatten_metrics("validation", validation_metrics),
            **_flatten_metrics("test", test_metrics),
        }
        winner_rows.append(row)
        individual_portfolios.append(row)
    winners_df = pd.DataFrame(winner_rows)
    _write_frame(output_dir / "individual_winners.csv", winners_df)
    _write_frame(
        output_dir / "individual_portfolios.csv", pd.DataFrame(individual_portfolios)
    )
    eligible = sorted(winner_states)
    invalid_combos: list[dict[str, Any]] = []
    combo_frames: dict[int, pd.DataFrame] = {}
    combo_results: dict[str, tuple[V2LedgerResult, pd.DataFrame, dict[str, float]]] = {}
    combo_trial_count = 0
    for size in config.combo_sizes:
        combo_rows: list[dict[str, Any]] = []
        for names_tuple in itertools.combinations(eligible, size):
            names = list(names_tuple)
            for mode in config.weight_modes:
                if mode == "equal":
                    weights = np.full(size, 1.0 / size)
                else:
                    vols = []
                    for name in names:
                        returns = (
                            winner_validation_results[name]
                            .ledger.set_index("date")["daily_return"]
                            .loc[validation_start:validation_end]
                        )
                        vol = float(returns.std(ddof=0))
                        vols.append(vol if vol > 1e-12 else float("nan"))
                    raw = np.asarray(
                        [1 / value if math.isfinite(value) else 0.0 for value in vols]
                    )
                    weights = (
                        raw / raw.sum() if raw.sum() > 0 else np.full(size, 1.0 / size)
                    )
                    weights = np.minimum(weights, config.max_security_weight * 2)
                    weights = weights / weights.sum()
                weight_map = {
                    name: float(value)
                    for name, value in zip(names, weights, strict=True)
                }
                combined_candidates = pd.concat(
                    [winner_candidates[name] for name in names], ignore_index=True
                )
                selection_candidates, selection_calendar = _slice_ledger_inputs(
                    combined_candidates, calendar, train_start, validation_end
                )
                result = run_global_ledger(
                    selection_candidates,
                    frames,
                    selection_calendar,
                    fx,
                    config,
                    weight_map,
                    portfolio_name=f"combo:{'+'.join(names)}:{mode}:validation",
                )
                validation_ledger = result.ledger.loc[
                    result.ledger["date"].between(validation_start, validation_end)
                ]
                if result.fills.empty:
                    validation_fills = result.fills
                else:
                    fill_dates = pd.to_datetime(result.fills["date"])
                    validation_fills = result.fills.loc[
                        fill_dates.between(validation_start, validation_end)
                    ]
                validation_metrics = _ledger_metrics(
                    validation_ledger, validation_fills, config.initial_capital
                )
                score, failures = v2_validation_score(validation_metrics, config)
                families = [winner_states[name].family for name in names]
                redundancy = 0.25 * (len(families) - len(set(families)))
                score -= redundancy
                return_frame = pd.concat(
                    {
                        name: winner_validation_results[name]
                        .ledger.set_index("date")["daily_return"]
                        .loc[validation_start:validation_end]
                        for name in names
                    },
                    axis=1,
                ).fillna(0.0)
                correlation = return_frame.corr().to_numpy(dtype=float)
                pairwise_correlation = (
                    correlation[np.triu_indices_from(correlation, k=1)]
                    if len(names) > 1
                    else np.asarray([0.0])
                )
                pairwise_correlation = np.nan_to_num(pairwise_correlation, nan=0.0)
                security_sets = {
                    name: set(
                        winner_candidates[name]
                        .loc[
                            pd.to_datetime(
                                winner_candidates[name]["entry_date"]
                            ).between(validation_start, validation_end),
                            "security_id",
                        ]
                        .astype(str)
                    )
                    for name in names
                }
                security_overlaps = []
                for left, right in itertools.combinations(names, 2):
                    union = security_sets[left] | security_sets[right]
                    security_overlaps.append(
                        len(security_sets[left] & security_sets[right]) / len(union)
                        if union
                        else 0.0
                    )
                key = f"{'+'.join(names)}__{mode}"
                row = {
                    "strategies": "+".join(names),
                    "strategy_count": size,
                    "weight_mode": mode,
                    "weights": stable_json(weight_map),
                    "validation_score": score,
                    "redundancy_penalty": redundancy,
                    "mean_pairwise_validation_correlation": float(
                        np.nanmean(pairwise_correlation)
                    ),
                    "maximum_pairwise_validation_correlation": float(
                        np.nanmax(pairwise_correlation)
                    ),
                    "mean_security_overlap_jaccard": float(
                        np.mean(security_overlaps) if security_overlaps else 0.0
                    ),
                    "overlap_trade_date_count": int(
                        result.overlap["date"].nunique()
                        if not result.overlap.empty
                        else 0
                    ),
                    "maximum_strategies_per_security": int(
                        result.overlap["strategy_source_count"].max()
                        if not result.overlap.empty
                        else 0
                    ),
                    "strategy_family_count": len(set(families)),
                    "hard_gate_status": "GO" if not failures else "INVALID_COMBINATION",
                    "hard_gate_failures": "|".join(failures),
                    **_flatten_metrics("validation", validation_metrics),
                }
                combo_trial_count += 1
                if failures:
                    invalid_combos.append(row)
                else:
                    combo_rows.append(row)
                    combo_results[key] = (result, combined_candidates, weight_map)
        frame = pd.DataFrame(combo_rows)
        if not frame.empty:
            frame = frame.sort_values(
                ["validation_score", "strategies", "weight_mode"],
                ascending=[False, True, True],
            )
        combo_frames[size] = frame
        _write_frame(output_dir / f"combinations_{size}.csv", frame)
    for missing_size in {2, 3, 4} - set(config.combo_sizes):
        _write_frame(output_dir / f"combinations_{missing_size}.csv", pd.DataFrame())
    _write_frame(output_dir / "invalid_combinations.csv", pd.DataFrame(invalid_combos))
    top_key: str | None = None
    top_validation_score = -math.inf
    for key, (result, _, _) in combo_results.items():
        validation_ledger = result.ledger.loc[
            result.ledger["date"].between(validation_start, validation_end)
        ]
        if result.fills.empty:
            validation_fills = result.fills
        else:
            fill_dates = pd.to_datetime(result.fills["date"])
            validation_fills = result.fills.loc[
                fill_dates.between(validation_start, validation_end)
            ]
        validation_metrics = _ledger_metrics(
            validation_ledger, validation_fills, config.initial_capital
        )
        score, failures = v2_validation_score(validation_metrics, config)
        if not failures and score > top_validation_score:
            top_key, top_validation_score = key, score
    if top_key is not None:
        _, final_candidates, final_weights = combo_results[top_key]
        portfolio_label = f"combo:{top_key}"
    elif winner_states:
        strategy = max(
            winner_states,
            key=lambda name: v2_validation_score(
                _ledger_metrics(
                    winner_validation_results[name].ledger.loc[
                        winner_validation_results[name]
                        .ledger["date"]
                        .between(validation_start, validation_end)
                    ],
                    winner_validation_results[name].fills,
                    config.initial_capital,
                ),
                config,
            )[0],
        )
        final_candidates = winner_candidates[strategy]
        final_weights = {strategy: 1.0}
        portfolio_label = f"individual:{strategy}"
    else:
        final_candidates = candidates.iloc[0:0]
        final_weights = {}
        portfolio_label = "cash:NO_VALID_PARAMETER_VARIANT"
    winner_freeze = {
        "schema": "strategy_combo_lab_v2_winner_freeze_v1",
        "created_before_test": True,
        "portfolio": portfolio_label,
        "weights": final_weights,
        "winner_parameters_hash": sha256_file(output_dir / "winner_parameters.json"),
    }
    atomic_write_json(output_dir / "selection-freeze.json", winner_freeze)
    final_result = run_global_ledger(
        final_candidates,
        frames,
        calendar,
        fx,
        config,
        final_weights,
        portfolio_name=portfolio_label,
    )
    final_test_ledger = final_result.ledger.loc[
        final_result.ledger["date"].between(test_start, test_end)
    ]
    if final_result.fills.empty:
        final_test_fills = final_result.fills
    else:
        final_fill_dates = pd.to_datetime(final_result.fills["date"])
        final_test_fills = final_result.fills.loc[
            final_fill_dates.between(test_start, test_end)
        ]
    final_test_metrics = _ledger_metrics(
        final_test_ledger, final_test_fills, config.initial_capital
    )
    _write_frame(output_dir / "portfolio_daily_ledger.parquet", final_result.ledger)
    _write_frame(
        output_dir / "portfolio_orders.parquet",
        final_result.orders,
        ("date", "order_id", "security_id", "side", "status"),
    )
    _write_frame(
        output_dir / "portfolio_fills.parquet",
        final_result.fills,
        ("date", "security_id", "side", "shares", "fee_eur"),
    )
    _write_frame(
        output_dir / "portfolio_positions.parquet",
        final_result.positions,
        ("date", "security_id", "shares"),
    )
    _write_frame(
        output_dir / "strategy_attribution.parquet",
        final_result.attribution,
        ("date", "security_id", "strategy"),
    )
    _write_frame(
        output_dir / "portfolio_contributors.parquet",
        final_result.contributors,
        ("date", "security_id", "portfolio_contribution"),
    )
    extreme, extreme_contributors = _extreme_audits(final_result, config)
    _write_frame(output_dir / "equity_extreme_days.csv", extreme)
    _write_frame(
        output_dir / "equity_extreme_day_contributors.csv", extreme_contributors
    )
    if final_result.unexplained_outliers:
        raise RuntimeError("UNEXPLAINED_EQUITY_OUTLIER")
    concentration = (
        final_result.positions.groupby("security_id", as_index=False).agg(
            maximum_market_value_eur=("market_value_eur", "max"),
            maximum_shares=("shares", "max"),
        )
        if not final_result.positions.empty
        else pd.DataFrame()
    )
    _write_frame(output_dir / "security_concentration_audit.csv", concentration)
    _write_frame(output_dir / "overlap_and_netting_audit.csv", final_result.overlap)
    top_equity = final_result.ledger[
        ["date", "daily_return", "portfolio_nav_eur"]
    ].rename(columns={"portfolio_nav_eur": "equity"})
    _write_frame(output_dir / "top_equity_001_selected.csv", top_equity)
    test_returns = final_result.ledger.set_index("date")["daily_return"].loc[
        test_start:test_end
    ]
    atomic_write_json(
        output_dir / "top_bootstrap.json",
        {
            portfolio_label: block_bootstrap(
                test_returns,
                config.bootstrap_runs,
                config.bootstrap_block_size,
                config.seed,
            )
        },
    )
    validation_eligible = variant_df.loc[
        variant_df.get("hard_gate_status", pd.Series(dtype=str)).eq("GO")
    ]
    validation_scores = pd.to_numeric(
        validation_eligible.get("validation_score", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    validation_sharpes = pd.to_numeric(
        variant_df.get("validation_Sharpe", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    top_decile = (
        validation_scores.loc[validation_scores >= validation_scores.quantile(0.90)]
        if not validation_scores.empty
        else validation_scores
    )
    family_trial_counts = {
        family: int(count)
        for family, count in pd.Series([state.family for state in variants.values()])
        .value_counts()
        .items()
    }
    all_trial_count = len(variants) + combo_trial_count
    adjusted_test_metrics = daily_metrics(
        test_returns,
        config.initial_capital,
        trial_count=max(all_trial_count, 1),
        sharpe_std_across_trials=float(validation_sharpes.std(ddof=1))
        if len(validation_sharpes) > 1
        else 0.0,
    )
    multiple_testing = {
        "schema": "strategy_combo_lab_v2_multiple_testing_v1",
        "status": "APPROXIMATION",
        "family_level_trial_count": len(variants),
        "family_trial_counts": family_trial_counts,
        "combo_level_trial_count": combo_trial_count,
        "all_trials_count": all_trial_count,
        "raw_test_sharpe": adjusted_test_metrics.get("Sharpe"),
        "family_adjusted_metric": "NOT_IDENTIFIABLE_WITH_ONE_VALIDATION_SPLIT",
        "all_trials_adjusted_DSR_probability": adjusted_test_metrics.get(
            "DSR_probability"
        ),
        "PBO": {
            "status": "NOT_IDENTIFIABLE_WITH_ONE_VALIDATION_SPLIT",
            "value": None,
        },
        "White_SPA": {
            "status": "APPROXIMATION",
            "value": None,
            "reason": "block bootstrap is published; no benchmark-residual SPA claim is made",
        },
        "parameter_stability": "NOT_PROVEN"
        if config.max_variants_per_strategy <= 1
        else "AVAILABLE_IN_VARIANT_TABLE",
        "top_decile_dispersion": float(top_decile.std(ddof=0))
        if len(top_decile)
        else None,
    }
    atomic_write_json(output_dir / "multiple_testing.json", multiple_testing)
    source.con.register("_ledger_export", final_result.ledger)
    source.con.execute(
        "CREATE OR REPLACE TABLE portfolio_daily_ledger AS SELECT * FROM _ledger_export"
    )
    source.con.unregister("_ledger_export")
    technical_go = (
        final_result.accounting_failures == 0
        and final_result.unexplained_outliers == 0
        and not final_result.ledger.empty
        and float(final_result.ledger["cash_eur"].min()) >= -1e-8
        and float(final_result.ledger["gross_exposure"].max())
        <= config.max_gross_exposure + 1e-8
        and float(final_result.ledger["target_gross_exposure"].max())
        <= config.max_gross_exposure + 1e-8
        and float(final_result.ledger["maximum_target_security_weight"].max())
        <= config.max_security_weight + 1e-8
        and float(final_result.ledger["maximum_target_sector_weight"].max())
        <= config.max_sector_weight + 1e-8
        and int(final_result.ledger["unique_security_count"].max())
        <= config.global_max_positions
    )
    run_status = {
        "schema": "strategy_combo_research_lab_v2_run_status_v1",
        "status": "PHASE11_5_STRATEGY_COMBO_LAB_V2_TECHNICAL_GO"
        if technical_go
        else "PHASE11_5_TECHNICAL_NO_GO",
        "portfolio_realism_status": "PHASE11_5_PORTFOLIO_REALISM_GO"
        if technical_go
        else "NO_GO",
        "causal_selection_status": "PHASE11_5_CAUSAL_SELECTION_GO",
        "extreme_day_attribution_status": "PHASE11_5_EXTREME_DAY_ATTRIBUTION_GO"
        if final_result.unexplained_outliers == 0
        else "NO_GO",
        "ledger_reconciliation_status": "PORTFOLIO_LEDGER_RECONCILIATION_GO"
        if final_result.accounting_failures == 0
        else "NO_GO",
        "accounting_failures": final_result.accounting_failures,
        "unexplained_outliers": final_result.unexplained_outliers,
        "financial_candidate_status": "NO_VALID_PARAMETER_VARIANT"
        if not winner_states
        else "RESEARCH_ONLY_NOT_FINANCIAL_FINALIST",
        "FINANCIAL_FINALIST_GO": False,
        "FORWARD_SHADOW_GO": False,
        "strategy_authority": "NONE",
        "PAPER_STRATEGY_AUTHORITY": "NONE",
        "LIVE_STRATEGY_AUTHORITY": "NONE",
        "execution_authority": "NONE",
        "financial_authority": "NONE",
        "automatic_submission": False,
        "broker_calls": 0,
        "realized_security_weight_drift_breach_days": int(
            (
                final_result.ledger["maximum_security_weight"]
                > config.max_security_weight + 1e-8
            ).sum()
        ),
        "realized_sector_weight_drift_breach_days": int(
            (
                final_result.ledger["maximum_sector_weight"]
                > config.max_sector_weight + 1e-8
            ).sum()
        ),
        "historical_shariah_coverage_ratio": 0.0,
        "historical_shariah_status": "SHARIAH_HISTORY_UNAVAILABLE",
        "shariah_candidate_valid": False,
    }
    atomic_write_json(output_dir / "run-status.json", run_status)
    manifest_config = dataclasses.asdict(config)
    manifest_config.pop("resume", None)
    manifest = {
        "schema": SCHEMA,
        "program_version": PROGRAM_VERSION,
        "status": run_status["status"],
        "fingerprints": fingerprints,
        "config": manifest_config,
        "data": source.metadata(),
        "investability": coverage,
        "strategy_count": len({state.strategy for state in variants.values()}),
        "variant_count": len(variants),
        "winner_count": len(winner_states),
        "combo_trial_count": combo_trial_count,
        "selected_portfolio": portfolio_label,
        "portfolio_metrics_full_calendar": final_result.metrics,
        "portfolio_metrics_untouched_test": final_test_metrics,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "financial_authority": "NONE",
        "automatic_submission": False,
        "broker_calls": 0,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    required_files = [
        "manifest.json",
        "progress.json",
        "report.md",
        "run-status.json",
        "program-freeze.json",
        "data_quality_exclusions.csv",
        "investability_exclusions.csv",
        "investability_coverage.json",
        "individual_variants.csv",
        "individual_winners.csv",
        "individual_portfolios.csv",
        "invalid_strategy_variants.csv",
        "combinations_2.csv",
        "combinations_3.csv",
        "combinations_4.csv",
        "invalid_combinations.csv",
        "winner_parameters.json",
        "top_bootstrap.json",
        "multiple_testing.json",
        "portfolio_daily_ledger.parquet",
        "portfolio_orders.parquet",
        "portfolio_fills.parquet",
        "portfolio_positions.parquet",
        "strategy_attribution.parquet",
        "portfolio_contributors.parquet",
        "equity_extreme_days.csv",
        "equity_extreme_day_contributors.csv",
        "security_concentration_audit.csv",
        "overlap_and_netting_audit.csv",
        "research_lab.duckdb",
    ]
    technical_markers = (
        [
            "STRATEGY_COMBO_LAB_V2_PIPELINE_GO",
            "PORTFOLIO_LEDGER_RECONCILIATION_GO",
            "DYNAMIC_TRANSACTION_COSTS_GO",
            "WHOLE_SHARE_ACCOUNTING_GO",
            "GLOBAL_POSITION_CAP_GO",
            "SECURITY_NETTING_GO",
            "FULL_CALENDAR_RETURNS_GO",
            "INVESTABILITY_GATE_GO",
            "CAUSAL_PARAMETER_SELECTION_GO",
            "EXTREME_DAY_ATTRIBUTION_GO",
            "BROKER_ISOLATION_GO",
        ]
        if technical_go
        else ["STRATEGY_COMBO_LAB_V2_PIPELINE_NO_GO"]
    )
    report = [
        "# Strategy Combo Research Lab V2",
        "",
        f"Run: `{fingerprints['run_id']}`",
        "",
        "## Validity",
        "",
        f"- Pipeline validity: `{run_status['status']}`",
        f"- Data validity: `{source.metadata()['quality_excluded_identity_count']} identities excluded by corporate-action QA`",
        f"- Investability validity: `{coverage['status']}`; candidate validity `{coverage['candidate_validity']}`",
        f"- Portfolio accounting validity: `{run_status['ledger_reconciliation_status']}`",
        f"- Parameter selection validity: `{run_status['causal_selection_status']}`",
        f"- Test performance: `{stable_json(final_test_metrics)}`",
        f"- Multiple testing: `{multiple_testing['status']}`",
        f"- Shariah validity: `{run_status['historical_shariah_status']}`",
        f"- Financial candidate status: `{run_status['financial_candidate_status']}`",
        "",
        "## Markers",
        "",
        *technical_markers,
        "",
        "FINANCIAL_FINALIST_GO=false",
        "FORWARD_SHADOW_GO=false",
        "STRATEGY_AUTHORITY=NONE",
        "PAPER_STRATEGY_AUTHORITY=NONE",
        "LIVE_STRATEGY_AUTHORITY=NONE",
        "EXECUTION_AUTHORITY=NONE",
        "",
    ]
    atomic_write_text(output_dir / "report.md", "\n".join(report))
    atomic_write_json(
        progress_path,
        {
            "status": "COMPLETE",
            "run_id": fingerprints["run_id"],
            "completed_at": utc_now_iso(),
            "elapsed_seconds": time.time() - started,
        },
    )
    database_logical_fingerprint = _v2_database_fingerprint(source.con)
    source.con.close()
    non_content_hashed_artifacts = {
        "program-freeze.json",
        "progress.json",
        "research_lab.duckdb",
    }
    freeze_payload = {
        "schema": "strategy_combo_research_lab_v2_program_freeze_v1",
        "status": "FROZEN_GO" if technical_go else "NO_GO",
        "program_sha256": fingerprints["program_sha256"],
        "run_id": fingerprints["run_id"],
        "database_logical_fingerprint": database_logical_fingerprint,
        "operational_artifacts_not_content_hashed": sorted(
            non_content_hashed_artifacts - {"program-freeze.json"}
        ),
        "artifact_hashes": {
            name: sha256_file(output_dir / name)
            for name in required_files
            if name not in non_content_hashed_artifacts
            and (output_dir / name).is_file()
        },
    }
    atomic_write_json(output_dir / "program-freeze.json", freeze_payload)
    return output_dir


def audit_v2_run(run_dir: Path) -> dict[str, Any]:
    required = (
        "manifest.json",
        "progress.json",
        "report.md",
        "run-status.json",
        "program-freeze.json",
        "portfolio_daily_ledger.parquet",
        "portfolio_orders.parquet",
        "portfolio_fills.parquet",
        "portfolio_positions.parquet",
        "strategy_attribution.parquet",
        "portfolio_contributors.parquet",
        "research_lab.duckdb",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    ledger = (
        pd.read_parquet(run_dir / "portfolio_daily_ledger.parquet")
        if not missing
        else pd.DataFrame()
    )
    fills = (
        pd.read_parquet(run_dir / "portfolio_fills.parquet")
        if not missing
        else pd.DataFrame()
    )
    manifest = (
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if not missing
        else {}
    )
    freeze = (
        json.loads((run_dir / "program-freeze.json").read_text(encoding="utf-8"))
        if not missing
        else {}
    )
    status = (
        json.loads((run_dir / "run-status.json").read_text(encoding="utf-8"))
        if not missing
        else {}
    )
    config = manifest.get("config", {})
    frozen_hashes = freeze.get("artifact_hashes", {})
    hash_mismatches = [
        name
        for name, expected in frozen_hashes.items()
        if not (run_dir / name).is_file()
        or sha256_file(run_dir / name) != str(expected)
    ]
    source_text = Path(__file__).read_text(encoding="utf-8")
    if not missing:
        with duckdb.connect(
            str(run_dir / "research_lab.duckdb"), read_only=True
        ) as con:
            database_logical_fingerprint = _v2_database_fingerprint(con)
    else:
        database_logical_fingerprint = {}
    forbidden_calls = [
        name
        for name in (
            "place" + "Order",
            "cancel" + "Order",
            "reqGlobal" + "Cancel",
            "req" + "Ids",
            "reqAutoOpen" + "Orders",
            "exercise" + "Options",
            "reqMkt" + "Data",
            "reqHistorical" + "Data",
        )
        if name in source_text
    ]
    checks: dict[str, Any] = {
        "missing_artifacts": missing,
        "artifact_hash_mismatches": hash_mismatches,
        "program_hash_match": freeze.get("program_sha256")
        == sha256_file(Path(__file__).resolve()),
        "database_logical_fingerprint_match": database_logical_fingerprint
        == freeze.get("database_logical_fingerprint"),
        "negative_cash_rows": int(
            (ledger.get("cash_eur", pd.Series(dtype=float)) < -1e-8).sum()
        ),
        "reconciliation_failure_rows": int(
            (
                ledger.get("reconciliation_error_eur", pd.Series(dtype=float)).abs()
                > 1e-7
            ).sum()
        ),
        "duplicate_calendar_rows": int(
            ledger.get("date", pd.Series(dtype=object)).duplicated().sum()
        ),
        "fractional_fill_rows": int(
            (
                pd.to_numeric(
                    fills.get("shares", pd.Series(dtype=float)), errors="coerce"
                )
                % 1
                != 0
            ).sum()
        ),
        "fee_reconciliation_error_eur": abs(
            float(ledger.get("transaction_costs_eur", pd.Series(dtype=float)).sum())
            - float(fills.get("fee_eur", pd.Series(dtype=float)).sum())
        ),
        "global_position_cap_breach_rows": int(
            (
                ledger.get("unique_security_count", pd.Series(dtype=float))
                > int(config.get("global_max_positions", 0))
            ).sum()
        ),
        "gross_exposure_cap_breach_rows": int(
            (
                ledger.get("gross_exposure", pd.Series(dtype=float))
                > float(config.get("max_gross_exposure", 0.0)) + 1e-8
            ).sum()
        ),
        "target_security_cap_breach_rows": int(
            (
                ledger.get("maximum_target_security_weight", pd.Series(dtype=float))
                > float(config.get("max_security_weight", 0.0)) + 1e-8
            ).sum()
        ),
        "target_sector_cap_breach_rows": int(
            (
                ledger.get("maximum_target_sector_weight", pd.Series(dtype=float))
                > float(config.get("max_sector_weight", 0.0)) + 1e-8
            ).sum()
        ),
        "target_gross_cap_breach_rows": int(
            (
                ledger.get("target_gross_exposure", pd.Series(dtype=float))
                > float(config.get("max_gross_exposure", 0.0)) + 1e-8
            ).sum()
        ),
        "authority_failures": int(
            status.get("strategy_authority") != "NONE"
            or status.get("PAPER_STRATEGY_AUTHORITY") != "NONE"
            or status.get("LIVE_STRATEGY_AUTHORITY") != "NONE"
            or status.get("execution_authority") != "NONE"
            or status.get("financial_authority") != "NONE"
            or bool(status.get("automatic_submission"))
        ),
        "forbidden_broker_call_names": forbidden_calls,
        "broker_calls": 0,
    }
    blocking = bool(
        missing
        or hash_mismatches
        or not checks["program_hash_match"]
        or not checks["database_logical_fingerprint_match"]
        or forbidden_calls
        or checks["authority_failures"]
        or checks["fee_reconciliation_error_eur"] > 1e-7
        or any(value for key, value in checks.items() if key.endswith("_rows"))
    )
    checks["status"] = "AUDIT_NO_GO" if blocking else "AUDIT_GO"
    atomic_write_json(run_dir / "audit-run.json", checks)
    return checks


def explain_v2_day(run_dir: Path, day: str) -> dict[str, Any]:
    date_value = parse_date(day)
    contributor_path = run_dir / "portfolio_contributors.parquet"
    if not contributor_path.is_file():
        contributor_path = run_dir / "equity_extreme_day_contributors.csv"
    ledger_path = run_dir / "portfolio_daily_ledger.parquet"
    if ledger_path.is_file():
        ledger = pd.read_parquet(ledger_path)
        contributors = (
            pd.read_parquet(contributor_path)
            if contributor_path.suffix == ".parquet" and contributor_path.is_file()
            else pd.read_csv(contributor_path)
            if contributor_path.is_file()
            else pd.DataFrame()
        )
        attribution_path = run_dir / "strategy_attribution.parquet"
        attribution = (
            pd.read_parquet(attribution_path)
            if attribution_path.is_file()
            else pd.DataFrame()
        )
        payload = {
            "schema": "strategy_combo_lab_v2_day_explanation_v1",
            "date": str(date_value.date()),
            "ledger": ledger.loc[
                pd.to_datetime(ledger["date"]).dt.normalize().eq(date_value)
            ].to_dict("records"),
            "contributors": contributors.loc[
                pd.to_datetime(
                    contributors.get("date", pd.Series(dtype=object)), errors="coerce"
                )
                .dt.normalize()
                .eq(date_value)
            ].to_dict("records")
            if not contributors.empty
            else [],
            "strategy_attribution": attribution.loc[
                pd.to_datetime(
                    attribution.get("date", pd.Series(dtype=object)), errors="coerce"
                )
                .dt.normalize()
                .eq(date_value)
            ].to_dict("records")
            if not attribution.empty
            else [],
        }
        return payload
    database = run_dir / "research_lab.duckdb"
    if not database.is_file():
        raise FileNotFoundError(f"No V2 or legacy evidence found in {run_dir}")
    with duckdb.connect(str(database), read_only=True) as con:
        active = con.execute(
            "SELECT * FROM accepted_trades WHERE entry_date<=? AND exit_date>=? ORDER BY strategy,security_id",
            [date_value.date(), date_value.date()],
        ).fetchdf()
        table_names = {
            str(row[0])
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        excluded_ids = (
            {
                str(row[0])
                for row in con.execute(
                    "SELECT security_id FROM data_quality_exclusions"
                ).fetchall()
            }
            if "data_quality_exclusions" in table_names
            else set()
        )
    data = discover_default_data()
    ids = active["security_id"].astype(str).unique().tolist()
    with duckdb.connect() as con:
        con.register("ids", pd.DataFrame({"security_id": ids}))
        bars = con.execute(
            """WITH x AS (SELECT b.*,
            LAG(close) OVER(PARTITION BY security_id ORDER BY date) previous_close,
            MEDIAN(close*volume) OVER(PARTITION BY security_id ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) median_dollar_volume_20d
            FROM read_parquet(?) b INNER JOIN ids USING(security_id))
            SELECT * FROM x WHERE CAST(date AS DATE)=?""",
            [str(data.resolve()), date_value.date()],
        ).fetchdf()
    merged = active.merge(
        bars, on="security_id", how="left", suffixes=("_trade", "_bar")
    )
    merged["security_return"] = merged["close"] / merged["previous_close"] - 1
    merged["portfolio_weight"] = merged["slot_weight"]
    merged["portfolio_contribution"] = merged["security_return"] * merged["slot_weight"]
    merged["ticker"] = merged.get("symbol", merged.get("ticker", ""))
    merged["price"] = merged["close"]
    merged["liquidity"] = merged["median_dollar_volume_20d"]
    merged["data_quality_status"] = merged["security_id"].map(
        lambda value: (
            "LEGACY_CORPORATE_ACTION_BLOCKED"
            if str(value) in excluded_ids
            else "LEGACY_CORPORATE_ACTION_GATE_ALLOWED"
        )
    )
    merged["v2_investability_status"] = np.where(
        (merged["previous_close"] >= 5.0)
        & (merged["median_dollar_volume_20d"] >= 5_000_000.0),
        "V2_PRICE_LIQUIDITY_GATE_GO",
        "V2_PRICE_OR_LIQUIDITY_BLOCKED",
    )
    merged = merged.sort_values(
        ["portfolio_contribution", "strategy", "security_id"],
        ascending=[False, True, True],
    )
    payload = {
        "schema": "strategy_combo_lab_v1_1_legacy_day_reconstruction_v1",
        "date": str(date_value.date()),
        "status": "LEGACY_SCHEMA_PARTIAL",
        "contributors": merged[
            [
                "security_id",
                "ticker",
                "security_return",
                "portfolio_weight",
                "portfolio_contribution",
                "strategy",
                "price",
                "liquidity",
                "data_quality_status",
                "v2_investability_status",
            ]
        ]
        .sort_values("portfolio_contribution", ascending=False)
        .to_dict("records"),
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-file parameter-search and exhaustive strategy-combination research lab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python strategy_combo_research_lab.py list
              python strategy_combo_research_lab.py validate
              python strategy_combo_research_lab.py run --preset smoke --max-symbols 50
              python strategy_combo_research_lab.py run --preset long --fixed-fee-eur 3
              python strategy_combo_research_lab.py run --preset extreme --full-cartesian --max-variants-per-strategy 5000
            """
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    from stocks.research.parameter_research_v2 import add_parameter_research_parser

    add_parameter_research_parser(sub)
    sub.add_parser(
        "list", help="List implemented and explicitly blocked transcript strategies."
    )

    validate = sub.add_parser(
        "validate", help="Validate and summarize the input dataset."
    )
    validate.add_argument(
        "--data",
        default="",
        help="Parquet/CSV file or directory; auto-detected when omitted.",
    )
    validate.add_argument("--start", default="2000-01-01")
    validate.add_argument("--end", default=str(pd.Timestamp.today().date()))
    validate.add_argument("--min-bars", type=int, default=260)
    validate.add_argument("--max-symbols", type=int, default=None)
    validate.add_argument(
        "--disable-corporate-action-gate",
        action="store_true",
    )
    validate.add_argument("--overnight-ratio-min", type=float, default=0.25)
    validate.add_argument("--overnight-ratio-max", type=float, default=4.0)
    validate.add_argument("--seed", type=int, default=20260722)

    run = sub.add_parser(
        "run",
        help="Run parameter search, detailed winners and all pair/triple/quad combinations.",
    )
    run.add_argument(
        "--data",
        default="",
        help="Parquet/CSV file or directory; auto-detected when omitted.",
    )
    run.add_argument("--output", default="output/research/strategy_combo_lab_v2")
    run.add_argument(
        "--preset", choices=["smoke", "standard", "long", "extreme"], default="long"
    )
    run.add_argument("--policy", choices=["shariah", "all"], default="shariah")
    run.add_argument("--start", default="2000-01-01")
    run.add_argument("--train-end", default="2011-12-31")
    run.add_argument("--validation-end", default="2018-12-31")
    run.add_argument("--end", default=str(pd.Timestamp.today().date()))
    run.add_argument("--initial-capital", type=float, default=2000.0)
    run.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Deprecated: interpreted as the global portfolio cap in v2.",
    )
    run.add_argument("--global-max-positions", type=int, default=4)
    run.add_argument("--max-security-weight", type=float, default=0.25)
    run.add_argument("--max-sector-weight", type=float, default=0.50)
    run.add_argument("--max-gross-exposure", type=float, default=1.00)
    run.add_argument("--minimum-order-eur", type=float, default=25.0)
    run.add_argument(
        "--whole-shares", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--max-order-adv-fraction", type=float, default=0.01)
    run.add_argument("--min-price", type=float, default=5.0)
    run.add_argument("--min-median-dollar-volume", type=float, default=5_000_000.0)
    run.add_argument("--liquidity-lookback", type=int, default=20)
    run.add_argument("--allowed-exchanges", default="NYSE,NASDAQ,NYSEMKT")
    run.add_argument("--cost-bps-per-side", type=float, default=10.0)
    run.add_argument("--slippage-bps-per-side", type=float, default=0.0)
    run.add_argument("--fixed-fee-eur", type=float, default=3.0)
    run.add_argument("--fx-cost-bps-per-side", type=float, default=0.0)
    run.add_argument("--min-bars", type=int, default=260)
    run.add_argument("--min-validation-trades", type=int, default=30)
    run.add_argument("--validation-max-drawdown", type=float, default=-0.35)
    run.add_argument("--max-symbols", type=int, default=None)
    run.add_argument(
        "--disable-corporate-action-gate",
        action="store_true",
        help=(
            "Disable the fail-closed identity filter for extreme open/close jumps. "
            "Not recommended for Phase 11.4 PIT data."
        ),
    )
    run.add_argument("--overnight-ratio-min", type=float, default=0.25)
    run.add_argument("--overnight-ratio-max", type=float, default=4.0)
    run.add_argument("--batch-size", type=int, default=100)
    run.add_argument("--checkpoint-every", type=int, default=5)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--memory-budget-gb", type=float, default=4.0)
    run.add_argument("--full-cartesian", action="store_true")
    run.add_argument(
        "--max-variants-per-strategy",
        type=int,
        default=0,
        help="0 uses preset defaults: smoke=1, standard=10, long=20, extreme=80.",
    )
    run.add_argument("--combo-sizes", default="2,3,4")
    run.add_argument("--weight-modes", default="equal,inverse_volatility")
    run.add_argument("--allow-invalid-strategies-in-combos", action="store_true")
    run.add_argument("--bootstrap-runs", type=int, default=500)
    run.add_argument("--bootstrap-block-size", type=int, default=20)
    run.add_argument("--top-equity-curves", type=int, default=30)
    run.add_argument("--equity-extreme-return-threshold", type=float, default=0.10)
    run.add_argument("--equity-hard-fail-return-threshold", type=float, default=0.50)
    run.add_argument("--seed", type=int, default=20260722)
    run.add_argument("--include-strategies", default="")
    run.add_argument("--exclude-strategies", default="")
    run.add_argument("--no-resume", action="store_true")
    audit = sub.add_parser(
        "audit-run", help="Audit a completed v2 run and publish audit-run.json."
    )
    audit.add_argument("--run-dir", required=True)
    explain = sub.add_parser(
        "explain-day", help="Explain a v2 or legacy portfolio day at security level."
    )
    explain.add_argument("--run-dir", required=True)
    explain.add_argument("--date", required=True)
    inspect = sub.add_parser(
        "inspect-ledger", help="Print bounded ledger/accounting diagnostics."
    )
    inspect.add_argument("--run-dir", required=True)
    inspect.add_argument("--rows", type=int, default=20)
    return parser


def command_list() -> int:
    rows = []
    for spec in strategy_registry():
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "horizon": spec.horizon,
                "policy_status": spec.policy_status,
                "description": spec.description,
            }
        )
    rows.append(
        {
            "name": "rotational_momentum",
            "family": "cross_sectional_momentum",
            "horizon": "long",
            "policy_status": "LONG_ONLY_GO",
            "description": "Monthly Top-N ranking on 6/9/12-month momentum with an absolute moving-average filter.",
        }
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nBlocked/deferred:")
    print(pd.DataFrame(BLOCKED_TRANSCRIPT_ITEMS).to_string(index=False))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    data_path = Path(args.data) if args.data else discover_default_data()
    temp_db = Path("output/research/strategy_combo_lab/validate.duckdb")
    source = MarketDataSource(
        data_path,
        parse_date(args.start),
        parse_date(args.end),
        args.min_bars,
        args.max_symbols,
        args.seed,
        temp_db,
        corporate_action_gate=not args.disable_corporate_action_gate,
        overnight_ratio_min=args.overnight_ratio_min,
        overnight_ratio_max=args.overnight_ratio_max,
    )
    print(json.dumps(source.metadata(), indent=2, default=str))
    exclusions = source.quality_exclusions()
    if not exclusions.empty:
        print("\nCorporate-action/data-quality exclusions:")
        print(exclusions.head(20).to_string(index=False))
    print(source.identities.head(20).to_string(index=False))
    return 0


def namespace_to_config(args: argparse.Namespace) -> V2Config:
    if not args.whole_shares:
        raise ValueError("FRACTIONAL_SHARE_ACCOUNTING_NOT_IMPLEMENTED_BLOCKED")
    if args.initial_capital <= 0:
        raise ValueError("--initial-capital must be positive")
    if args.global_max_positions <= 0 and args.max_positions is None:
        raise ValueError("--global-max-positions must be positive")
    if not 0 < args.max_security_weight <= 1:
        raise ValueError("--max-security-weight must be in (0, 1]")
    if not 0 < args.max_sector_weight <= 1:
        raise ValueError("--max-sector-weight must be in (0, 1]")
    if not 0 < args.max_gross_exposure <= 1:
        raise ValueError("--max-gross-exposure must be in (0, 1]")
    if not 0 < args.max_order_adv_fraction <= 1:
        raise ValueError("--max-order-adv-fraction must be in (0, 1]")
    combo_sizes = tuple(sorted({int(x) for x in parse_csv_list(args.combo_sizes)}))
    if any(size not in {2, 3, 4} for size in combo_sizes):
        raise ValueError("--combo-sizes may contain only 2,3,4")
    weight_modes = tuple(parse_csv_list(args.weight_modes))
    allowed_weight_modes = {"equal", "inverse_volatility"}
    if not set(weight_modes).issubset(allowed_weight_modes):
        raise ValueError(
            f"Unknown weight modes: {set(weight_modes) - allowed_weight_modes}"
        )
    max_variants = args.max_variants_per_strategy
    if max_variants <= 0:
        max_variants = {"smoke": 1, "standard": 10, "long": 20, "extreme": 80}[
            args.preset
        ]
    global_max_positions = args.max_positions or args.global_max_positions
    if args.max_positions is not None:
        print(
            "DEPRECATION: --max-positions is interpreted as global portfolio cap in v2",
            file=sys.stderr,
        )
    return V2Config(
        command="run",
        data=args.data,
        output=args.output,
        preset=args.preset,
        policy=args.policy,
        start=args.start,
        train_end=args.train_end,
        validation_end=args.validation_end,
        end=args.end,
        initial_capital=args.initial_capital,
        global_max_positions=global_max_positions,
        max_security_weight=args.max_security_weight,
        max_sector_weight=args.max_sector_weight,
        max_gross_exposure=args.max_gross_exposure,
        minimum_order_eur=args.minimum_order_eur,
        whole_shares=args.whole_shares,
        max_order_adv_fraction=args.max_order_adv_fraction,
        min_price=args.min_price,
        min_median_dollar_volume=args.min_median_dollar_volume,
        liquidity_lookback=args.liquidity_lookback,
        allowed_exchanges=tuple(parse_csv_list(args.allowed_exchanges)),
        cost_bps_per_side=args.cost_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        fixed_fee_eur=args.fixed_fee_eur,
        fx_cost_bps_per_side=args.fx_cost_bps_per_side,
        min_bars=args.min_bars,
        min_validation_trades=args.min_validation_trades,
        validation_max_drawdown=args.validation_max_drawdown,
        max_symbols=args.max_symbols,
        corporate_action_gate=not args.disable_corporate_action_gate,
        overnight_ratio_min=args.overnight_ratio_min,
        overnight_ratio_max=args.overnight_ratio_max,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        workers=args.workers,
        memory_budget_gb=args.memory_budget_gb,
        full_cartesian=args.full_cartesian,
        max_variants_per_strategy=max_variants,
        combo_sizes=combo_sizes,
        weight_modes=weight_modes,
        allow_invalid_strategies_in_combos=args.allow_invalid_strategies_in_combos,
        bootstrap_runs=args.bootstrap_runs,
        bootstrap_block_size=args.bootstrap_block_size,
        top_equity_curves=args.top_equity_curves,
        equity_extreme_return_threshold=args.equity_extreme_return_threshold,
        equity_hard_fail_return_threshold=args.equity_hard_fail_return_threshold,
        seed=args.seed,
        include_strategies=tuple(parse_csv_list(args.include_strategies)),
        exclude_strategies=tuple(parse_csv_list(args.exclude_strategies)),
        resume=not args.no_resume,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            return command_list()
        if args.command == "validate":
            return command_validate(args)
        if args.command == "run":
            config = namespace_to_config(args)
            run_lab_v2(config)
            return 0
        if args.command == "parameter-research":
            from stocks.research.parameter_research_v2 import run_parameter_research

            run_parameter_research(args, sys.modules[__name__])
            return 0
        if args.command == "audit-run":
            result = audit_v2_run(Path(args.run_dir))
            print(json.dumps(result, indent=2, default=str))
            return 0 if result["status"] == "AUDIT_GO" else 2
        if args.command == "explain-day":
            result = explain_v2_day(Path(args.run_dir), args.date)
            print(json.dumps(result, indent=2, default=str))
            return 0
        if args.command == "inspect-ledger":
            run_dir = Path(args.run_dir)
            ledger = pd.read_parquet(run_dir / "portfolio_daily_ledger.parquet")
            print(ledger.tail(max(1, args.rows)).to_string(index=False))
            print(json.dumps(audit_v2_run(run_dir), indent=2, default=str))
            return 0
        parser.error(f"Unknown command: {args.command}")
        return 2
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Screening checkpoints can be resumed by running the same command again.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
