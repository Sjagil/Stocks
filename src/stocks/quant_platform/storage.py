from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.quant_platform.data import CANONICAL_MARKET_DATA_COLUMNS, clean_market_data


class MultiAssetStore:
    """Canonical Parquet lake plus queryable SQLite catalog."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.parquet_root = self.root / "parquet"
        self.sqlite_path = self.root / "market_data.sqlite3"
        self.manifest_path = self.root / "manifest.json"

    def write(self, frame: pd.DataFrame) -> dict[str, Any]:
        data = clean_market_data(frame)
        self.root.mkdir(parents=True, exist_ok=True)
        parquet_files: list[str] = []
        for (asset_class, source, symbol), group in data.groupby(["asset_class", "source", "symbol"], sort=True):
            directory = self.parquet_root / f"asset_class={_safe(asset_class)}" / f"source={_safe(source)}" / f"symbol={_safe(symbol)}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "observations.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                merged = clean_market_data(pd.concat([existing, group], ignore_index=True))
            else:
                merged = clean_market_data(group)
            _atomic_parquet(path, merged)
            parquet_files.append(str(path))
        self._write_sqlite(data)
        manifest = {
            "schema": "multi_asset_store_manifest_v1",
            "canonical_columns": list(CANONICAL_MARKET_DATA_COLUMNS),
            "row_count_written": len(data),
            "symbols": sorted(data["symbol"].unique().tolist()),
            "asset_classes": sorted(data["asset_class"].unique().tolist()),
            "sources": sorted(data["source"].unique().tolist()),
            "parquet_files": parquet_files,
            "sqlite_path": str(self.sqlite_path),
            "broker_calls": 0,
            "broker_writes": 0,
        }
        _atomic_text(self.manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest

    def read(
        self,
        *,
        symbol: str | None = None,
        asset_class: str | None = None,
        as_of: Any | None = None,
    ) -> pd.DataFrame:
        if not self.sqlite_path.exists():
            return clean_market_data([])
        clauses: list[str] = []
        params: list[str] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.strip().upper())
        if asset_class:
            clauses.append("asset_class = ?")
            params.append(asset_class.strip().lower())
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            clauses.append("available_at <= ?")
            params.append(cutoff.isoformat())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT {', '.join(CANONICAL_MARKET_DATA_COLUMNS)} FROM observations{where} ORDER BY symbol, timestamp, available_at"
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = pd.read_sql_query(query, connection, params=params)
        return clean_market_data(rows)

    def _write_sqlite(self, frame: pd.DataFrame) -> None:
        records = frame.copy()
        records["timestamp"] = records["timestamp"].map(lambda value: value.isoformat())
        records["available_at"] = records["available_at"].map(lambda value: value.isoformat())
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL,
                    asset_class TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    source TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    market_cap REAL,
                    PRIMARY KEY (symbol, timestamp, source, available_at)
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timestamp, source, available_at) DO UPDATE SET
                    open=excluded.open, high=excluded.high, low=excluded.low,
                    close=excluded.close, volume=excluded.volume,
                    asset_class=excluded.asset_class, currency=excluded.currency,
                    market_cap=excluded.market_cap
                """,
                records.loc[:, list(CANONICAL_MARKET_DATA_COLUMNS)].itertuples(index=False, name=None),
            )
            connection.commit()


def _safe(value: Any) -> str:
    text = str(value).strip().upper()
    if not text or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in text):
        raise ValueError(f"unsafe partition value: {value}")
    return text


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
