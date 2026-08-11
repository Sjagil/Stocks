# Integrated Quant Platform

## Safety boundary

`stocks.quant_platform` is research-only. It does not import the IBKR live
adapter, cannot submit orders and reports `execution_authority=NONE` and
`broker_writes=0` at orchestration boundaries. The full manager produces
paper-validation proposals, never broker orders. Existing P0/P0.2 live safety
and manual approval remain separate and unchanged.

## Progressive architecture

| Level | Capabilities | Primary modules |
|---:|---|---|
| 1 | Multi-asset ingestion, canonical PIT data, Parquet/SQLite, performance and risk | `data`, `providers`, `storage`, `explorer`, `analytics` |
| 2 | Markowitz, efficient frontier, equal weight, minimum variance, maximum Sharpe, risk parity | `portfolio` |
| 3 | Factors, cross-sectional ranking, technical indicators and combination strategies | `factors`, `technical` |
| 4 | Professional fills/costs/corporate actions/delistings, walk-forward, regimes, strategy allocation | `backtest`, `validation`, `regime` |
| 5 | Cointegration/stat-arb, cross-sectional reversion, volatility, VaR/ES and Monte Carlo | `stat_arb`, `risk` |
| 6 | Temporal ML, meta-labeling, calibrated signals, news, SEC and event studies | `ml`, `intelligence` |
| 7 | Dynamic allocation, HRP, Black-Litterman, portfolio risk, costs and optimal execution | `allocation`, `execution` |
| 8 | Factor risk, alpha combination, mixture of experts, RL environment and portfolio manager | `professional`, `manager` |

The machine-readable source of truth is `capabilities.capability_registry()`;
it maps all 33 requested projects to concrete modules.

## Canonical data path

```text
EODHD / FRED / OpenExchangeRates / Bitvavo / CoinMarketCap
                              |
                              v
symbol, timestamp, OHLCV, asset_class, currency, source,
available_at, market_cap
                              |
                 +------------+-------------+
                 |                          |
              Parquet                    SQLite
                 |                          |
                 +------------+-------------+
                              v
                  as-of research and analytics
```

Provider fetches are HTTPS GET requests only. Credentials are read from the
existing environment by the CLI and are never included in stored manifests.
The routes follow the official documentation for
[EODHD historical EOD](https://eodhd.com/financial-apis/api-for-historical-data-and-volumes),
[FRED series observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html),
[OpenExchangeRates historical data](https://docs.openexchangerates.org/reference/historical-json),
[Bitvavo candles](https://docs.bitvavo.com/docs/rest-api/get-candlestick-data/), and
[CoinMarketCap latest quotes](https://coinmarketcap.com/api/documentation/guides/get-latest-crypto-prices).

## CLI

With `PYTHONPATH=src`:

```text
python -m stocks.quant_platform --root data/quant-platform fetch eodhd SPY.US --start 2020-01-01 --end 2026-01-01 --asset-class etf
python -m stocks.quant_platform --root data/quant-platform fetch fred VIXCLS --start 2020-01-01
python -m stocks.quant_platform --root data/quant-platform fetch openexchangerates EUR --start 2026-01-01
python -m stocks.quant_platform --root data/quant-platform fetch bitvavo BTC-EUR --interval 1d
python -m stocks.quant_platform --root data/quant-platform fetch coinmarketcap BTC --currency USD
python -m stocks.quant_platform --root data/quant-platform inventory
python -m stocks.quant_platform --root data/quant-platform analyze GLD --benchmark SPY
```

## Validation contract

- All timestamps are UTC and `available_at >= timestamp`.
- Macro revisions are stored as distinct vintages and selected with as-of time.
- Backtest orders have at least one bar of execution delay.
- Partial fills respect volume participation; missing bars cannot fill.
- Split/dividend actions and terminal delisting liquidation are explicit.
- Model validation uses ordered time splits; ML targets must clear transaction costs.
- Portfolio proposals must clear costs, limits, compliance and whole-quantity feasibility.
- Model feedback never updates production models automatically.
