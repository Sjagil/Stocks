from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import load_dotenv

from stocks.quant_platform.data import AssetClass
from stocks.quant_platform.explorer import MultiAssetMarketDataExplorer
from stocks.quant_platform.providers import (
    BitvavoAdapter,
    CoinMarketCapAdapter,
    EodhdAdapter,
    FredAdapter,
    OpenExchangeRatesAdapter,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m stocks.quant_platform")
    parser.add_argument("--root", type=Path, default=Path("data/quant-platform"))
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="ingest a canonical CSV, JSON or Parquet file")
    ingest.add_argument("path", type=Path)

    fetch = commands.add_parser("fetch", help="fetch read-only provider data and ingest it")
    fetch.add_argument("provider", choices=["eodhd", "fred", "openexchangerates", "bitvavo", "coinmarketcap"])
    fetch.add_argument("symbol")
    fetch.add_argument("--start")
    fetch.add_argument("--end")
    fetch.add_argument("--interval", default="1d")
    fetch.add_argument("--asset-class", choices=[item.value for item in AssetClass], default="equity")
    fetch.add_argument("--currency", default="USD")
    fetch.add_argument("--env-file", type=Path, default=Path(".env"))

    inventory = commands.add_parser("inventory", help="show stored multi-asset coverage")
    inventory.add_argument("--as-of")

    analyze = commands.add_parser("analyze", help="calculate performance and risk metrics")
    analyze.add_argument("symbol")
    analyze.add_argument("--benchmark")
    analyze.add_argument("--as-of")
    analyze.add_argument("--risk-free-rate", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    explorer = MultiAssetMarketDataExplorer(args.root)
    if args.command == "ingest":
        payload = _read_frame(args.path)
        result = explorer.ingest(payload)
    elif args.command == "fetch":
        load_dotenv(args.env_file, override=False)
        payload = _fetch(args)
        result = explorer.ingest(payload)
    elif args.command == "inventory":
        frame = explorer.observations(as_of=args.as_of).frame
        result = {
            "schema": "multi_asset_inventory_v1",
            "rows": len(frame),
            "symbols": sorted(frame["symbol"].unique().tolist()),
            "asset_classes": frame["asset_class"].value_counts().sort_index().to_dict(),
            "sources": frame["source"].value_counts().sort_index().to_dict(),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }
    else:
        result = explorer.analyze(
            args.symbol,
            benchmark_symbol=args.benchmark,
            as_of=args.as_of,
            annual_risk_free_rate=args.risk_free_rate,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError("ingest supports .csv, .json, .jsonl and .parquet")


def _fetch(args: argparse.Namespace) -> pd.DataFrame:
    if args.provider == "eodhd":
        adapter = EodhdAdapter()
        payload = adapter.fetch(
            api_key=_environment_secret("EODHD_API_KEY", "EOD_API_KEY", "EODHISTORICALDATA_API_KEY"),
            ticker=args.symbol,
            start=_required(args.start, "--start"),
            end=_required(args.end, "--end"),
        )
        return adapter.normalize(
            payload,
            symbol=args.symbol.split(".")[0],
            asset_class=AssetClass(args.asset_class),
            currency=args.currency,
        )
    if args.provider == "fred":
        adapter = FredAdapter()
        payload = adapter.fetch(
            api_key=_environment_secret("FRED_API_KEY"),
            series_id=args.symbol,
            observation_start=args.start,
            observation_end=args.end,
        )
        return adapter.normalize(payload, series_id=args.symbol)
    if args.provider == "openexchangerates":
        adapter = OpenExchangeRatesAdapter()
        payload = adapter.fetch_historical(
            app_id=_environment_secret("OPENEXCHANGERATES_APP_ID", "OPENEXCHANGE_API_KEY"),
            date=_required(args.start, "--start"),
            symbols=[args.symbol],
        )
        return adapter.normalize(payload, quote_currency=args.symbol)
    if args.provider == "bitvavo":
        adapter = BitvavoAdapter()
        payload = adapter.fetch(market=args.symbol, interval=args.interval)
        return adapter.normalize(payload, market=args.symbol)
    adapter = CoinMarketCapAdapter()
    payload = adapter.fetch_latest(
        api_key=_environment_secret("COINMARKETCAP_API_KEY", "CMC_API_KEY"),
        symbols=[args.symbol],
        quote_currency=args.currency,
    )
    return adapter.normalize(payload, quote_currency=args.currency)


def _environment_secret(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise ValueError(f"missing environment credential: {' or '.join(names)}")


def _required(value: str | None, option: str) -> str:
    if not value:
        raise ValueError(f"{option} is required for this provider")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
