from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocks.ibkr.news import (
    NewsObservation,
    collect_ibkr_news,
    ibkr_news_schema,
    probe_ibkr_news,
)


def _env(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "APP_ENV=paper",
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7497",
                "IBKR_CLIENT_ID=17",
                "IBKR_READ_ONLY=true",
                "IBKR_ORDER_AUTHORITY=NONE",
                "IBKR_LIVE_TRADING_ENABLED=false",
                "IBKR_ALLOW_ORDER_TRANSMISSION=false",
                "IBKR_MARKET_DATA_TYPE=3",
            ]
        ),
        encoding="utf-8",
    )


def test_schema_exposes_only_read_only_news_calls() -> None:
    report = ibkr_news_schema()

    assert report["allowed_calls"] == [
        "reqNewsProviders",
        "reqHistoricalNews",
    ]
    assert "placeOrder" in report["forbidden_calls"]
    assert report["execution_authority"] == "NONE"


def test_capability_reports_missing_provider_subscription(tmp_path: Path) -> None:
    _env(tmp_path / ".env.ibkr")

    report = probe_ibkr_news(
        tmp_path,
        observer=lambda settings, contracts, hours, results: NewsObservation(
            connected=True,
            api_ready=True,
            provider_request_count=1,
        ),
    )

    assert report["status"] == "UNAVAILABLE_NO_PROVIDER_SUBSCRIPTION"
    assert report["subscription_purchase_automatic"] is False
    assert report["broker_write_calls"] == 0


def test_normal_connection_close_does_not_erase_connection_evidence(
    tmp_path: Path,
) -> None:
    _env(tmp_path / ".env.ibkr")
    report = probe_ibkr_news(
        tmp_path,
        observer=lambda settings, contracts, hours, results: NewsObservation(
            connected=True,
            api_ready=True,
            providers=[{"code": "TEST", "name": "Test"}],
            connection_closed_count=1,
            provider_request_count=1,
        ),
    )

    assert report["status"] == "AVAILABLE_NO_CURRENT_HEADLINES"
    assert report["tws_connected"] is True
    assert report["connection_closed_count"] == 1


def test_live_news_connection_uses_dedicated_read_only_settings(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env.ibkr.live"
    env.write_text(
        "\n".join(
            [
                "IBKR_ENVIRONMENT=LIVE",
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7496",
                "IBKR_QUOTE_CLIENT_ID=73",
                "IBKR_LIVE_CALLBACK_TIMEOUT_SECONDS=4",
            ]
        ),
        encoding="utf-8",
    )

    def observer(settings, contracts, hours, results):
        assert settings.port == 7496
        assert settings.client_id == 73
        assert settings.environment == "LIVE_READ_ONLY_OBSERVATION"
        return NewsObservation(
            connected=True,
            api_ready=True,
            provider_request_count=1,
        )

    report = probe_ibkr_news(
        tmp_path,
        env_file=env,
        observer=observer,
    )

    assert report["status"] == "UNAVAILABLE_NO_PROVIDER_SUBSCRIPTION"
    assert report["connection_environment"] == (
        "LIVE_READ_ONLY_OBSERVATION"
    )
    assert report["broker_write_calls"] == 0


def test_collection_is_bounded_private_and_idempotent(tmp_path: Path) -> None:
    _env(tmp_path / ".env.ibkr")
    contract_path = tmp_path / "output/ibkr/contracts/stocks.parquet"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "con_id": 123, "security_type": "STK"},
        ]
    ).to_parquet(contract_path, index=False)
    headline = {
        "published_at": "2999-01-01T00:00:00+00:00",
        "title": "AAA reports results",
        "source": "IBKR_TWS:TEST",
        "provider_code": "TEST",
        "symbols": ["AAA"],
        "article_reference_hash": "HASHED",
        "sentiment_polarity": 0.0,
    }

    def observer(settings, contracts, hours, results):
        assert hours == 720
        assert results == 100
        assert contracts == [{"symbol": "AAA", "con_id": 123}]
        return NewsObservation(
            connected=True,
            api_ready=True,
            providers=[{"code": "TEST", "name": "Test News"}],
            headlines=[headline],
            provider_request_count=1,
            historical_news_request_count=1,
            requested_symbol_count=1,
            completed_symbol_count=1,
        )

    first = collect_ibkr_news(
        tmp_path,
        symbols=["AAA"],
        lookback_hours=10_000,
        max_results_per_symbol=1_000,
        observer=observer,
    )
    second = collect_ibkr_news(
        tmp_path,
        symbols=["AAA"],
        lookback_hours=10_000,
        max_results_per_symbol=1_000,
        observer=observer,
    )
    private = (
        tmp_path / "data/news/ibkr/private/headlines.jsonl"
    ).read_text(encoding="utf-8")
    public = (
        tmp_path / "output/ibkr/news/collection.json"
    ).read_text(encoding="utf-8")

    assert first["new_private_headline_count"] == 1
    assert second["new_private_headline_count"] == 0
    assert second["duplicate_private_headline_count"] == 1
    assert len(private.splitlines()) == 1
    assert "raw-article-id" not in public
    assert second["raw_article_ids_published"] is False
    assert second["execution_authority"] == "NONE"
