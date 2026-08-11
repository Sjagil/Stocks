from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocks.screener.config import ScreenerConfig
from stocks.screener.models import (
    AssetMetadata,
    AssetSnapshot,
    FundamentalSnapshot,
    ShariahSnapshot,
)
from stocks.screener.scoring import classify_scores, score_asset
from stocks.screener.service import screener_preview
from stocks.screener.sources import LocalScreenerSources, select_pit_records
from stocks.screener.storage import ScreenerLayout, ScreenerStore

UTC = timezone.utc
SCREEN_DATE = date(2026, 7, 22)


def _config() -> ScreenerConfig:
    return ScreenerConfig.load(Path(__file__).resolve().parents[1])


def test_repository_shariah_attestations_are_bounded_and_evidenced() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "config/screener/shariah_attestations_v1.json").read_text(
            encoding="utf-8"
        )
    )
    rows = payload["attestations"]
    symbols = [str(row["symbol"]).upper() for row in rows]

    assert len(symbols) == len(set(symbols))
    assert {
        "AAPL",
        "NVDA",
        "ON",
        "CSCO",
        "TXN",
        "CSX",
        "APH",
        "ANET",
        "AMAT",
        "LRCX",
        "SPUS",
        "SPSK",
        "HLAL",
        "UMMA",
    }.issubset(set(symbols))
    for row in rows:
        screened_at = datetime.fromisoformat(
            str(row["screened_at"]).replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            str(row["expires_at"]).replace("Z", "+00:00")
        )
        evidence = {str(value) for value in row["evidence"]}
        assert row["status"] == "SHARIAH_ELIGIBLE_PIT"
        assert timedelta(0) < expires_at - screened_at <= timedelta(days=32)
        if row["methodology"] == "AAOIFI_EXTERNAL_DUAL_SOURCE_CORROBORATION":
            assert any("musaffa.com/" in value for value in evidence)
            assert any("muslimxchange.com/" in value for value in evidence)
        else:
            assert row["methodology"] == "OFFICIAL_FUND_SHARIAH_CERTIFICATION"
            assert row["source"] == "OFFICIAL_FUND_PROVIDER"
            assert any(
                "wahed.com/" in value or "sp-funds.com/" in value
                for value in evidence
            )


def _bars(*, end: date = SCREEN_DATE, periods: int = 300, slope: float = 0.18) -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=periods, tz="UTC")
    close = 50.0 + np.arange(periods) * slope
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, 5_000_000.0),
        },
        index=index,
    )


def _fundamental() -> FundamentalSnapshot:
    return FundamentalSnapshot(
        available_at=datetime(2026, 5, 1, tzinfo=UTC),
        net_income=1_200_000_000.0,
        free_cash_flow=1_100_000_000.0,
        revenue=10_000_000_000.0,
        assets=6_000_000_000.0,
        debt=900_000_000.0,
        cash=1_000_000_000.0,
        shares=100_000_000.0,
        previous_shares=102_000_000.0,
        operating_cash_flow=1_400_000_000.0,
        dividends=200_000_000.0,
    )


def _shariah(status: str = "SHARIAH_ELIGIBLE_PIT") -> ShariahSnapshot:
    return ShariahSnapshot(
        status=status,
        screened_at=datetime(2026, 7, 1, tzinfo=UTC),
        expires_at=datetime(2026, 9, 30, tzinfo=UTC),
        methodology="TEST_PIT",
        source="TEST",
    )


def _snapshot(
    *,
    bars: pd.DataFrame | None = None,
    fundamental: FundamentalSnapshot | None = None,
    shariah: ShariahSnapshot | None = None,
) -> AssetSnapshot:
    return AssetSnapshot(
        metadata=AssetMetadata(
            asset_key="TEST:1",
            symbol="TEST",
            name="Test Corporation",
            con_id=123,
            asset_type="STOCK",
            exchange="NASDAQ",
            currency="USD",
            sector="Technology",
            industry="Software - Application",
            category="Domestic Common Stock",
            inactive=False,
        ),
        bars=_bars() if bars is None else bars,
        price_source="TEST_PIT",
        price_source_timestamp=datetime(2026, 7, 22, tzinfo=UTC),
        fundamental=_fundamental() if fundamental is None else fundamental,
        shariah=_shariah() if shariah is None else shariah,
        benchmark_symbol="SPY",
        benchmark_bars=_bars(slope=0.08),
    )


def test_scoring_is_deterministic_and_has_no_authority() -> None:
    first = score_asset(_snapshot(), screening_date=SCREEN_DATE, config=_config())
    second = score_asset(_snapshot(), screening_date=SCREEN_DATE, config=_config())
    assert first == second
    assert first.public["classification"] == "HIGH_POTENTIAL"
    assert first.public["execution_authority"] == "NONE"
    assert first.public["broker_calls"] == 0
    assert first.public["order_calls"] == 0
    assert first.public["industry"] == "Software - Application"
    assert first.public["category"] == "Domestic Common Stock"
    assert first.public["market_cap"] == pytest.approx(
        first.public["price"] * 100_000_000.0
    )
    assert first.public["fundamental_coverage"] == pytest.approx(1.0)
    assert first.public["median_dollar_volume_20d"] > 0


def test_point_in_time_selector_and_future_bar_block_lookahead() -> None:
    decision = datetime(2026, 7, 22, 23, 59, 59, tzinfo=UTC)
    records = [
        {"accepted_at": "2026-07-22T20:00:00Z", "value": 1},
        {"accepted_at": "2026-07-23T00:00:00Z", "value": 2},
    ]
    assert [item["value"] for item in select_pit_records(records, decision)] == [1]
    future = pd.concat([_bars(), _bars(end=date(2026, 7, 23), periods=1)])
    result = score_asset(_snapshot(bars=future), screening_date=SCREEN_DATE, config=_config())
    assert result.public["classification"] == "REJECTED"
    assert "LOOKAHEAD_PRICE_DATA_BLOCKED" in result.public["rejection_reasons"]


def test_stale_data_is_excluded() -> None:
    stale = _bars(end=date(2026, 7, 15))
    result = score_asset(_snapshot(bars=stale), screening_date=SCREEN_DATE, config=_config())
    assert result.public["classification"] == "REJECTED"
    assert "STALE_PRICE_DATA" in result.public["rejection_reasons"]


def test_missing_fundamentals_are_excluded() -> None:
    snapshot = replace(_snapshot(), fundamental=None)
    result = score_asset(snapshot, screening_date=SCREEN_DATE, config=_config())
    assert result.public["classification"] == "REJECTED"
    assert "MISSING_FUNDAMENTAL_DATA" in result.public["rejection_reasons"]


def test_etf_missing_holdings_fundamentals_is_explicit_and_not_fabricated() -> None:
    stock = _snapshot()
    snapshot = replace(
        stock,
        metadata=replace(
            stock.metadata,
            asset_key="ETF:SPUS",
            symbol="SPUS",
            name="SP Funds S&P 500 Sharia Industry Exclusions ETF",
            asset_type="ETF",
            exchange="ARCA",
            sector="SHARIAH_US_EQUITY",
            industry=None,
            category="Shariah Equity ETF",
        ),
        fundamental=None,
    )

    result = score_asset(
        snapshot,
        screening_date=SCREEN_DATE,
        config=_config(),
    )

    assert "MISSING_FUNDAMENTAL_DATA" not in result.public["rejection_reasons"]
    assert "ETF_HOLDINGS_FUNDAMENTALS_UNAVAILABLE" in result.public["warnings"]
    assert result.public["fundamental_applicability"] == "ETF_HOLDINGS_LEVEL"
    assert result.public["fundamental_coverage"] == 0.0
    assert "fundamental" not in result.public["effective_score_weights"]
    assert result.public["execution_authority"] == "NONE"
    assert result.public["broker_calls"] == 0


def test_shariah_filter_is_fail_closed() -> None:
    result = score_asset(
        _snapshot(shariah=_shariah("SHARIAH_DATA_INCOMPLETE")),
        screening_date=SCREEN_DATE,
        config=_config(),
    )
    assert result.public["classification"] == "REJECTED"
    assert "SHARIAH_DATA_INCOMPLETE" in result.public["rejection_reasons"]


@pytest.mark.parametrize(
    ("total", "fundamental", "technical", "expected"),
    [
        (75.0, 60.0, 65.0, "HIGH_POTENTIAL"),
        (74.999, 80.0, 80.0, "WATCHLIST"),
        (60.0, 20.0, 20.0, "WATCHLIST"),
        (45.0, 20.0, 20.0, "NEUTRAL"),
        (44.999, 80.0, 80.0, "REJECTED"),
    ],
)
def test_classification_boundaries(
    total: float,
    fundamental: float,
    technical: float,
    expected: str,
) -> None:
    assert (
        classify_scores(
            total_score=total,
            fundamental_score=fundamental,
            technical_score=technical,
            hard_filter_pass=True,
            long_term_trend_positive=True,
            momentum_positive=True,
            config=_config(),
        )
        == expected
    )
    assert (
        classify_scores(
            total_score=total,
            fundamental_score=fundamental,
            technical_score=technical,
            hard_filter_pass=False,
            long_term_trend_positive=True,
            momentum_positive=True,
            config=_config(),
        )
        == "REJECTED"
    )


def _record(screening_date: date, score: float, classification: str = "WATCHLIST") -> dict:
    return {
        "screening_date": screening_date.isoformat(),
        "asset_key": "TEST:1",
        "symbol": "TEST",
        "classification": classification,
        "total_score": score,
        "fundamental_score": score,
        "technical_score": score,
        "liquidity_score": score,
        "risk_score": score,
        "rejection_reasons": [],
    }


def _summary() -> dict:
    return {
        "data_quality_status": "GO",
        "screened_count": 1,
        "classification_counts": {"WATCHLIST": 1},
        "top_winners": [],
        "top_losers": [],
        "benchmarks": ["SPY"],
        "rejection_reason_counts": {},
    }


def test_append_only_duplicate_prevention_history_and_daily_change(tmp_path: Path) -> None:
    layout = ScreenerLayout(
        private_db=tmp_path / "private" / "screener.sqlite3",
        output_dir=tmp_path / "output",
    )
    store = ScreenerStore(layout)
    first_date = date(2026, 7, 21)
    second_date = first_date + timedelta(days=1)
    try:
        store.register(
            screening_date=first_date,
            decision_time="2026-07-21T23:59:59+00:00",
            config_hash="A" * 64,
            screener_version="TEST",
            records=[_record(first_date, 60.0)],
            private_records={"TEST:1": {}},
            summary_base=_summary(),
        )
        with pytest.raises(ValueError, match="DUPLICATE_SCREENING_DATE"):
            store.register(
                screening_date=first_date,
                decision_time="2026-07-21T23:59:59+00:00",
                config_hash="A" * 64,
                screener_version="TEST",
                records=[_record(first_date, 60.0)],
                private_records={"TEST:1": {}},
                summary_base=_summary(),
            )
        second = store.register(
            screening_date=second_date,
            decision_time="2026-07-22T23:59:59+00:00",
            config_hash="A" * 64,
            screener_version="TEST",
            records=[_record(second_date, 66.0)],
            private_records={"TEST:1": {}},
            summary_base=_summary(),
        )
        assert second["records"][0]["change"]["change_type"] == "SCORE_RISEN"
        history = store.history("test")
        assert len(history) == 2
        assert history[1]["change"]["score_change"] == 6.0
    finally:
        store.close()


def test_public_records_never_contain_broker_write_counters() -> None:
    result = score_asset(_snapshot(), screening_date=SCREEN_DATE, config=_config())
    serialized = json.dumps(result.public)
    assert "placeOrder" not in serialized
    assert "cancelOrder" not in serialized
    assert "reqIds" not in serialized


def test_screener_prefers_current_validated_multitimeframe_daily_bars(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "data/research/critical_trading/yfinance"
    legacy.mkdir(parents=True)
    legacy_frame = _bars(end=date(2026, 8, 4), periods=300)
    legacy_frame.reset_index(names="session_date").to_parquet(
        legacy / "AAPL.parquet", index=False
    )
    current = (
        tmp_path
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=AAPL/interval=1d/source_interval=1d"
    )
    current.mkdir(parents=True)
    current_frame = _bars(end=date(2026, 8, 7), periods=300)
    current_frame.reset_index(names="timestamp_utc").assign(
        session_date=lambda frame: frame["timestamp_utc"].dt.date,
        adjusted_close=lambda frame: frame["close"],
        quality_status="VALIDATED_OHLC",
        is_partial=False,
    ).to_parquet(current / "bars.parquet", index=False)

    sources = LocalScreenerSources(tmp_path, _config())
    frames = sources._load_yfinance()

    assert frames["AAPL"].index[-1].date() == date(2026, 8, 7)
    assert sources._yfinance_sources["AAPL"] == (
        "YFINANCE_MULTITIMEFRAME_VALIDATED_ADJUSTED"
    )


def test_screener_metadata_uses_broad_universe_identity_fallback(
    tmp_path: Path,
) -> None:
    universe_dir = tmp_path / "config" / "universes"
    universe_dir.mkdir(parents=True)
    (universe_dir / "broad_multi_asset_v1.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "group": "shariah_core",
                        "asset_type": "ETF",
                        "sleeve": "etf_core",
                        "instruments": {
                            "SPUS": {
                                "sector": "SHARIAH_US_EQUITY",
                                "product_structure": (
                                    "SHARIAH_EQUITY_INDEX_ETF"
                                ),
                                "primary_exchange": "ARCA",
                                "currency": "USD",
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    sources = LocalScreenerSources(tmp_path, _config())
    metadata = sources._metadata("SPUS")

    assert metadata.asset_type == "ETF"
    assert metadata.exchange == "ARCA"
    assert metadata.currency == "USD"
    assert metadata.sector == "SHARIAH_US_EQUITY"
    assert metadata.category == "SHARIAH_EQUITY_INDEX_ETF"
    assert metadata.con_id is None


def test_screener_preview_is_noncanonical_and_does_not_create_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_root = Path(__file__).resolve().parents[1] / "config"
    shutil.copytree(config_root, tmp_path / "config")
    fake_snapshot = _snapshot()
    monkeypatch.setattr(
        LocalScreenerSources,
        "load",
        lambda self, screening_date, known_at=None: [fake_snapshot],
    )
    monkeypatch.setattr(
        "stocks.screener.service.macro_context_at",
        lambda project_root, as_of: {},
    )

    known_at = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    result = screener_preview(
        tmp_path,
        as_of=SCREEN_DATE.isoformat(),
        known_at=known_at,
    )
    artifact = json.loads(
        (tmp_path / "output/screener/latest-preview.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "GO"
    assert artifact["canonical_research_evidence"] is False
    assert artifact["append_only_history_mutated"] is False
    assert artifact["knowledge_cutoff"] == known_at.isoformat()
    assert artifact["market_data_cutoff"] == SCREEN_DATE.isoformat()
    layout = ScreenerLayout.from_project_root(tmp_path)
    store = ScreenerStore(layout)
    try:
        assert store.run_count() == 0
    finally:
        store.close()


def test_runtime_score_uses_explicit_knowledge_cutoff_for_attestation() -> None:
    screened_at = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
    runtime_snapshot = replace(
        _snapshot(),
        shariah=ShariahSnapshot(
            status="SHARIAH_ELIGIBLE_PIT",
            screened_at=screened_at,
            expires_at=datetime(2026, 9, 1, tzinfo=UTC),
            methodology="TEST_CURRENT",
            source="TEST",
        ),
    )

    historical = score_asset(
        runtime_snapshot,
        screening_date=SCREEN_DATE,
        config=_config(),
    )
    current = score_asset(
        runtime_snapshot,
        screening_date=SCREEN_DATE,
        config=_config(),
        known_at=datetime(2026, 7, 24, 8, 0, tzinfo=UTC),
    )

    assert historical.public["classification"] == "REJECTED"
    assert historical.public["rejection_reasons"] == ["SHARIAH_ELIGIBLE_PIT"]
    assert current.public["classification"] != "REJECTED"
    assert current.public["rejection_reasons"] == []
