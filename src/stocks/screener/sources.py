from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from stocks.screener.config import ScreenerConfig
from stocks.screener.models import (
    AssetMetadata,
    AssetSnapshot,
    FundamentalSnapshot,
    ShariahSnapshot,
)
from stocks.universe import broad_asset_metadata

UTC = timezone.utc
RELEVANT_FACTS = {
    "Assets",
    "CashAndCashEquivalentsAtCarryingValue",
    "EntityCommonStockSharesOutstanding",
    "LongTermDebtCurrent",
    "LongTermDebtNoncurrent",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetIncomeLoss",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsOfDividends",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
}
COMMODITY_ETFS = {
    "BNO",
    "CANE",
    "COPX",
    "CORN",
    "CPER",
    "DBA",
    "DBB",
    "DBC",
    "EART",
    "GLD",
    "IAU",
    "JO",
    "LIT",
    "NIB",
    "PALL",
    "PDBC",
    "PHO",
    "PICK",
    "PPLT",
    "REMX",
    "SCOP",
    "SLV",
    "SLX",
    "SOYB",
    "SPPP",
    "UGA",
    "UNG",
    "URA",
    "URNM",
    "USO",
    "WEAT",
    "WOOD",
    "XME",
}


def _as_utc(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def decision_time_for_session(session_date: date) -> datetime:
    return datetime.combine(session_date, time(23, 59, 59), tzinfo=UTC)


def latest_completed_session(now: datetime | None = None) -> date:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    calendar = xcals.get_calendar("XNYS")
    start = pd.Timestamp(current.date() - pd.Timedelta(days=14))
    end = pd.Timestamp(current.date())
    sessions = calendar.sessions_in_range(start, end)
    completed: list[pd.Timestamp] = []
    for session in sessions:
        close = calendar.session_close(session)
        if close <= pd.Timestamp(current):
            completed.append(session)
    if not completed:
        raise RuntimeError("no completed XNYS session found in bounded lookback")
    return completed[-1].date()


def trading_session_distance(older: date, newer: date) -> int:
    if older >= newer:
        return 0
    calendar = xcals.get_calendar("XNYS")
    if older < calendar.first_session.date() or newer > calendar.last_session.date():
        return int(len(pd.bdate_range(older, newer)) - 1)
    sessions = calendar.sessions_in_range(pd.Timestamp(older), pd.Timestamp(newer))
    return max(len(sessions) - 1, 0)


def select_pit_records(records: list[dict[str, Any]], decision_time: datetime) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        available = _as_utc(
            record.get("accepted_at")
            or record.get("decision_available_at")
            or record.get("available_at")
            or record.get("screened_at")
        )
        if available is not None and available <= decision_time:
            selected.append(record)
    return selected


class LocalScreenerSources:
    def __init__(self, project_root: Path, config: ScreenerConfig) -> None:
        self.project_root = project_root
        self.config = config
        self.causal_db = (
            project_root
            / "data"
            / "research"
            / "phase11_3"
            / "private"
            / "causal_research.sqlite3"
        )
        self.yfinance_dir = (
            project_root / "data" / "research" / "critical_trading" / "yfinance"
        )
        self.multitimeframe_yfinance_dir = (
            project_root
            / "data"
            / "research"
            / "multitimeframe"
            / "private"
            / "provider=YFINANCE"
        )
        self.security_master_path = (
            project_root / "data" / "research" / "phase11_4" / "private" / "security-master.parquet"
        )
        self.contract_path = project_root / "output" / "ibkr" / "contracts" / "stocks.parquet"
        self._connection: sqlite3.Connection | None = None
        self._security_master = self._load_security_master()
        self._contracts = self._load_contracts()
        self._broad_metadata = broad_asset_metadata(project_root)
        self._attestations = self._load_attestations()
        self._movers: dict[tuple[str, str], dict[str, Any]] = {}
        self._yfinance_sources: dict[str, str] = {}
        self.source_inventory: dict[str, Any] = {}

    def __enter__(self) -> LocalScreenerSources:
        if self.causal_db.exists():
            self._connection = sqlite3.connect(
                f"file:{self.causal_db}?mode=ro",
                uri=True,
            )
        return self

    def __exit__(self, *_: Any) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def load(
        self,
        screening_date: date,
        *,
        known_at: datetime | None = None,
    ) -> list[AssetSnapshot]:
        decision_time = (
            known_at.astimezone(UTC)
            if known_at is not None
            else decision_time_for_session(screening_date)
        )
        yfinance = self._load_yfinance()
        eodhd_symbols = self._eodhd_symbols()
        eodhd = {
            symbol: frame
            for symbol in eodhd_symbols
            if not (frame := self._load_eodhd(symbol)).empty
        }
        self._movers = self._load_movers(decision_time)
        symbols = sorted(
            (set(yfinance) | set(eodhd) | {symbol for symbol, _ in self._movers})
            - {"INDEX_VIX", "^VIX"}
        )
        bars_by_symbol: dict[str, pd.DataFrame] = {}
        source_by_symbol: dict[str, str] = {}
        conflict_by_symbol: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            yahoo_frame = yfinance.get(symbol, pd.DataFrame())
            eodhd_frame = eodhd.get(symbol, pd.DataFrame())
            conflict_by_symbol[symbol] = self._provider_conflict(yahoo_frame, eodhd_frame)
            yahoo_last = _last_date(yahoo_frame)
            eodhd_last = _last_date(eodhd_frame)
            if yahoo_last is not None and (eodhd_last is None or yahoo_last >= eodhd_last):
                bars_by_symbol[symbol] = yahoo_frame
                source_by_symbol[symbol] = self._yfinance_sources.get(
                    symbol,
                    "YFINANCE_LEGACY_AUTO_ADJUSTED",
                )
            elif eodhd_last is not None:
                bars_by_symbol[symbol] = eodhd_frame
                source_by_symbol[symbol] = "EODHD_ADJUSTED_CLOSE"

        snapshots: list[AssetSnapshot] = []
        for symbol in sorted(bars_by_symbol):
            bars = bars_by_symbol[symbol]
            bars = bars[bars.index.date <= screening_date].copy()
            if bars.empty:
                continue
            metadata = self._metadata(symbol)
            benchmark_symbol = self.config.benchmarks[metadata.asset_type]
            benchmark = bars_by_symbol.get(benchmark_symbol, pd.DataFrame())
            if not benchmark.empty:
                benchmark = benchmark[benchmark.index.date <= screening_date].copy()
            mover = self._movers.get((symbol, screening_date.isoformat()))
            snapshots.append(
                AssetSnapshot(
                    metadata=metadata,
                    bars=bars,
                    price_source=source_by_symbol[symbol],
                    price_source_timestamp=_as_utc(bars.index[-1]),
                    fundamental=(
                        None
                        if metadata.asset_type != "STOCK"
                        else self._fundamentals(symbol, decision_time)
                    ),
                    shariah=self._shariah(symbol, decision_time),
                    benchmark_symbol=benchmark_symbol,
                    benchmark_bars=benchmark,
                    mover_type=None if mover is None else str(mover.get("direction")),
                    mover_return=None if mover is None else float(mover.get("return", 0.0)),
                    provider_conflict=bool(conflict_by_symbol[symbol].get("conflict")),
                    provider_conflict_detail=conflict_by_symbol[symbol],
                )
            )
        self.source_inventory = {
            "knowledge_cutoff": decision_time.isoformat(),
            "market_data_cutoff": screening_date.isoformat(),
            "yfinance": {
                "status": "AVAILABLE" if yfinance else "UNAVAILABLE",
                "instrument_count": len(yfinance),
                "latest_session": _max_last_date(yfinance),
                "role": "qualified_ohlcv_primary_or_crosscheck",
            },
            "eodhd_causal_store": {
                "status": "AVAILABLE" if eodhd else "UNAVAILABLE",
                "instrument_count": len(eodhd),
                "latest_session": _max_last_date(eodhd),
                "role": "qualified_ohlcv_primary_or_crosscheck",
            },
            "sec_companyfacts": {
                "status": "AVAILABLE" if self._connection is not None else "UNAVAILABLE",
                "role": "point_in_time_fundamentals",
            },
            "phase11_3_movers": {
                "status": "AVAILABLE" if self._movers else "NO_CURRENT_EVENTS",
                "event_count_at_or_before_decision": len(self._movers),
                "role": "top_winner_top_loser_context",
            },
            "phase11_3_shariah": {
                "status": "AVAILABLE" if self._connection is not None else "UNAVAILABLE",
                "role": "point_in_time_hard_filter",
            },
            "security_master": {
                "status": "AVAILABLE" if self._security_master else "UNAVAILABLE",
                "instrument_count": len(self._security_master),
                "role": "identity_and_listing_metadata",
            },
            "ibkr_contract_cache": {
                "status": "AVAILABLE" if self._contracts else "UNAVAILABLE",
                "instrument_count": len(self._contracts),
                "role": "read_only_identity_only",
            },
        }
        return snapshots

    def _load_yfinance(self) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        self._yfinance_sources = {}
        for path in sorted(self.yfinance_dir.glob("*.parquet")):
            frame = pd.read_parquet(path)
            if frame.empty or "session_date" not in frame:
                continue
            frame.index = pd.to_datetime(frame["session_date"], utc=True)
            frame = frame.rename(columns={str(column): str(column).lower() for column in frame})
            frames[path.stem.upper()] = frame[
                [column for column in ("open", "high", "low", "close", "volume") if column in frame]
            ].sort_index()
            self._yfinance_sources[path.stem.upper()] = (
                "YFINANCE_LEGACY_AUTO_ADJUSTED"
            )
        pattern = "symbol=*/interval=1d/source_interval=1d/bars.parquet"
        for path in sorted(self.multitimeframe_yfinance_dir.glob(pattern)):
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            if "quality_status" in frame:
                frame = frame[
                    frame["quality_status"].astype(str).eq("VALIDATED_OHLC")
                ]
            if "is_partial" in frame:
                frame = frame[~frame["is_partial"].fillna(True).astype(bool)]
            timestamp_column = (
                "session_date"
                if "session_date" in frame
                else "timestamp_utc"
                if "timestamp_utc" in frame
                else None
            )
            required = {"open", "high", "low", "close", "volume"}
            if (
                frame.empty
                or timestamp_column is None
                or not required.issubset(frame.columns)
            ):
                continue
            frame = frame.copy()
            raw_close = pd.to_numeric(frame["close"], errors="coerce")
            adjusted_close = pd.to_numeric(
                frame.get("adjusted_close", raw_close), errors="coerce"
            )
            adjustment = (
                adjusted_close.div(raw_close)
                .replace([float("inf"), float("-inf")], pd.NA)
                .fillna(1.0)
            )
            for column in ("open", "high", "low", "close"):
                frame[column] = (
                    pd.to_numeric(frame[column], errors="coerce")
                    * adjustment
                )
            frame.index = pd.to_datetime(
                frame[timestamp_column], utc=True
            ).dt.normalize()
            frame = frame[list(required)].dropna(
                subset=["open", "high", "low", "close"]
            )
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
            if frame.empty:
                continue
            symbol = path.parts[-4].removeprefix("symbol=").upper()
            current = frames.get(symbol, pd.DataFrame())
            frame_last_date = _last_date(frame)
            current_last_date = _last_date(current)
            if frame_last_date is not None and (
                current_last_date is None
                or frame_last_date >= current_last_date
            ):
                frames[symbol] = frame
                self._yfinance_sources[symbol] = (
                    "YFINANCE_MULTITIMEFRAME_VALIDATED_ADJUSTED"
                )
        return frames

    def _eodhd_symbols(self) -> list[str]:
        if self._connection is None:
            return []
        rows = self._connection.execute(
            """
            SELECT DISTINCT substr(economic_key, 1, instr(economic_key, ':') - 1)
            FROM records
            WHERE dataset = 'prices'
            """
        ).fetchall()
        return sorted({str(row[0]).removesuffix(".US").upper() for row in rows if row[0]})

    def _load_eodhd(self, symbol: str) -> pd.DataFrame:
        if self._connection is None:
            return pd.DataFrame()
        provider_symbol = f"{symbol}.US"
        prefix = f"{provider_symbol}:"
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM records
            WHERE dataset = 'prices' AND economic_key >= ? AND economic_key < ?
            ORDER BY economic_key DESC
            LIMIT 420
            """,
            (prefix, prefix + "\uffff"),
        ).fetchall()
        payloads = [json.loads(row[0]) for row in reversed(rows)]
        valid = [
            {
                "session_date": item.get("timestamp"),
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("adjusted_close") or item.get("close"),
                "raw_close": item.get("close"),
                "volume": item.get("volume"),
            }
            for item in payloads
            if item.get("quality_status") == "OK"
        ]
        if not valid:
            return pd.DataFrame()
        frame = pd.DataFrame(valid)
        frame.index = pd.to_datetime(frame.pop("session_date"), utc=True)
        return frame.sort_index()

    def _load_security_master(self) -> dict[str, dict[str, Any]]:
        if not self.security_master_path.exists():
            return {}
        frame = pd.read_parquet(self.security_master_path)
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame = frame.sort_values(["ticker", "is_delisted", "last_price_date"])
        return {
            ticker: group.iloc[-1].to_dict()
            for ticker, group in frame.groupby("ticker", sort=False)
        }

    def _load_contracts(self) -> dict[str, dict[str, Any]]:
        if not self.contract_path.exists():
            return {}
        frame = pd.read_parquet(self.contract_path)
        return {
            str(row["symbol"]).upper(): row.to_dict()
            for _, row in frame.iterrows()
        }

    def _load_attestations(self) -> dict[str, dict[str, Any]]:
        if not self.config.shariah_attestations_path.exists():
            return {}
        payload = json.loads(self.config.shariah_attestations_path.read_text(encoding="utf-8"))
        return {
            str(item["symbol"]).upper(): item
            for item in payload.get("attestations", [])
        }

    def _metadata(self, symbol: str) -> AssetMetadata:
        master = self._security_master.get(symbol, {})
        contract = self._contracts.get(symbol, {})
        broad = self._broad_metadata.get(symbol, {})
        if symbol in COMMODITY_ETFS:
            asset_type = "COMMODITY_ETF"
        elif symbol in self.config.etf_symbols:
            asset_type = "ETF"
        else:
            asset_type = "STOCK"
        return AssetMetadata(
            asset_key=str(master.get("security_id") or f"SYMBOL:{symbol}"),
            symbol=symbol,
            name=_optional_text(master.get("name") or contract.get("long_name")),
            con_id=_optional_int(contract.get("con_id")),
            asset_type=asset_type,
            exchange=_optional_text(
                master.get("exchange")
                or contract.get("primary_exchange")
                or broad.get("primary_exchange")
            ),
            currency=_optional_text(
                master.get("currency")
                or contract.get("currency")
                or broad.get("currency")
            ),
            sector=_optional_text(master.get("sector") or broad.get("sector")),
            industry=_optional_text(master.get("industry")),
            category=_optional_text(
                master.get("category")
                or contract.get("category")
                or broad.get("product_structure")
            ),
            inactive=bool(master.get("is_delisted", False)),
        )

    def _fundamentals(
        self,
        symbol: str,
        decision_time: datetime,
    ) -> FundamentalSnapshot | None:
        if self._connection is None:
            return None
        prefix = f"{symbol}:"
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM records
            WHERE dataset = 'filings' AND economic_key >= ? AND economic_key < ?
            """,
            (prefix, prefix + "\uffff"),
        ).fetchall()
        parsed = []
        for (raw,) in rows:
            item = json.loads(raw)
            if item.get("record_type") != "COMPANYFACT":
                continue
            if item.get("concept") not in RELEVANT_FACTS:
                continue
            accepted = _as_utc(item.get("accepted_at"))
            if accepted is not None and accepted <= decision_time:
                parsed.append(item)
        if not parsed:
            return None

        annual = [
            item
            for item in parsed
            if item.get("form") == "10-K"
            and item.get("fiscal_period") == "FY"
            and item.get("period_start")
        ]
        net_income = _latest_duration_value(annual, {"NetIncomeLoss"})
        revenue = _latest_duration_value(
            annual,
            {"RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"},
        )
        operating_cash_flow = _latest_duration_value(
            annual,
            {"NetCashProvidedByUsedInOperatingActivities"},
        )
        capex = _latest_duration_value(annual, {"PaymentsToAcquirePropertyPlantAndEquipment"})
        dividends = _latest_duration_value(annual, {"PaymentsOfDividends"})
        free_cash_flow = (
            None
            if operating_cash_flow is None or capex is None
            else operating_cash_flow - abs(capex)
        )
        assets = _latest_instant_value(parsed, {"Assets"})
        cash = _latest_instant_value(parsed, {"CashAndCashEquivalentsAtCarryingValue"})
        current_debt = _latest_instant_value(parsed, {"LongTermDebtCurrent"}) or 0.0
        long_debt = _latest_instant_value(parsed, {"LongTermDebtNoncurrent"}) or 0.0
        debt = current_debt + long_debt if current_debt or long_debt else None
        share_facts = _instant_facts(parsed, {"EntityCommonStockSharesOutstanding"})
        shares = share_facts[-1][1] if share_facts else None
        previous_shares = None
        if len(share_facts) >= 2:
            latest_date = share_facts[-1][0]
            eligible = [
                value
                for period_end, value, _ in share_facts[:-1]
                if (latest_date - period_end).days >= 180
            ]
            previous_shares = eligible[-1] if eligible else share_facts[-2][1]
        selected_times = [
            _as_utc(item.get("accepted_at"))
            for item in parsed
            if item.get("concept") in RELEVANT_FACTS
        ]
        return FundamentalSnapshot(
            available_at=max(item for item in selected_times if item is not None),
            net_income=net_income,
            free_cash_flow=free_cash_flow,
            revenue=revenue,
            assets=assets,
            debt=debt,
            cash=cash,
            shares=shares,
            previous_shares=previous_shares,
            operating_cash_flow=operating_cash_flow,
            dividends=dividends,
        )

    def _shariah(self, symbol: str, decision_time: datetime) -> ShariahSnapshot:
        manual = self._attestations.get(symbol)
        if manual is not None:
            snapshot = ShariahSnapshot(
                status=str(manual["status"]),
                screened_at=_as_utc(manual.get("screened_at")),
                expires_at=_as_utc(manual.get("expires_at")),
                methodology=_optional_text(manual.get("methodology")),
                source=_optional_text(manual.get("source")),
            )
            if snapshot.screened_at is not None and snapshot.screened_at <= decision_time:
                return snapshot
        if self._connection is None:
            return ShariahSnapshot("SHARIAH_DATA_UNAVAILABLE", None, None, None, None)
        prefix = f"{symbol}:"
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM research_events
            WHERE event_type = 'SHARIAH_SCREEN'
              AND economic_key >= ? AND economic_key < ?
            """,
            (prefix, prefix + "\uffff"),
        ).fetchall()
        candidates = select_pit_records([json.loads(row[0]) for row in rows], decision_time)
        if not candidates:
            return ShariahSnapshot("SHARIAH_DATA_UNAVAILABLE", None, None, None, None)
        latest = max(candidates, key=lambda item: _as_utc(item.get("screened_at")) or datetime.min.replace(tzinfo=UTC))
        return ShariahSnapshot(
            status=str(latest.get("final_status") or "SHARIAH_DATA_INCOMPLETE"),
            screened_at=_as_utc(latest.get("screened_at")),
            expires_at=_as_utc(latest.get("expiry")),
            methodology=_optional_text(latest.get("methodology_id")),
            source="PHASE11_3_PIT_RECONSTRUCTION",
        )

    def _load_movers(self, decision_time: datetime) -> dict[tuple[str, str], dict[str, Any]]:
        if self._connection is None:
            return {}
        rows = self._connection.execute(
            "SELECT payload_json FROM research_events WHERE event_type = 'MOVER'"
        ).fetchall()
        selected = select_pit_records([json.loads(row[0]) for row in rows], decision_time)
        return {
            (str(item["symbol"]).upper(), str(item["session_date"])): item
            for item in selected
        }

    def _provider_conflict(
        self,
        yahoo: pd.DataFrame,
        eodhd: pd.DataFrame,
    ) -> dict[str, Any]:
        if yahoo.empty or eodhd.empty:
            return {"status": "ONE_PROVIDER_ONLY", "conflict": False}
        common = yahoo.index.intersection(eodhd.index)
        if common.empty:
            return {"status": "NO_OVERLAP", "conflict": False}
        common = common[-10:]
        differences = (
            (yahoo.loc[common, "close"] / eodhd.loc[common, "close"] - 1.0)
            .abs()
            .replace([float("inf")], pd.NA)
            .dropna()
        )
        maximum = float(differences.max()) if not differences.empty else 0.0
        conflict = maximum > self.config.thresholds["maximum_provider_close_difference"]
        return {
            "status": "PROVIDER_CONFLICT" if conflict else "PROVIDERS_ALIGNED",
            "conflict": conflict,
            "maximum_close_difference": maximum,
            "overlap_rows": len(common),
            "sources": ["YFINANCE_AUTO_ADJUSTED", "EODHD_ADJUSTED_CLOSE"],
        }


def _last_date(frame: pd.DataFrame) -> date | None:
    return None if frame.empty else pd.Timestamp(frame.index[-1]).date()


def _max_last_date(frames: dict[str, pd.DataFrame]) -> str | None:
    values = [_last_date(frame) for frame in frames.values() if not frame.empty]
    return None if not values else max(item for item in values if item is not None).isoformat()


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _latest_duration_value(
    facts: list[dict[str, Any]],
    concepts: set[str],
) -> float | None:
    candidates = [item for item in facts if item.get("concept") in concepts]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda item: (
            str(item.get("period_end") or ""),
            str(item.get("accepted_at") or ""),
        ),
    )
    return float(latest["value"])


def _instant_facts(
    facts: list[dict[str, Any]],
    concepts: set[str],
) -> list[tuple[date, float, datetime]]:
    deduplicated: dict[date, tuple[float, datetime]] = {}
    for item in facts:
        if item.get("concept") not in concepts or not item.get("period_end"):
            continue
        period_end = date.fromisoformat(str(item["period_end"])[:10])
        accepted = _as_utc(item.get("accepted_at"))
        if accepted is None:
            continue
        current = deduplicated.get(period_end)
        if current is None or accepted > current[1]:
            deduplicated[period_end] = (float(item["value"]), accepted)
    return [
        (period_end, value, accepted)
        for period_end, (value, accepted) in sorted(deduplicated.items())
    ]


def _latest_instant_value(
    facts: list[dict[str, Any]],
    concepts: set[str],
) -> float | None:
    candidates = _instant_facts(facts, concepts)
    return None if not candidates else candidates[-1][1]
