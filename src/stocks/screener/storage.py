from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

UTC = timezone.utc
CANDIDATE_CLASSES = {"HIGH_POTENTIAL", "WATCHLIST"}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest().upper()


@dataclass(frozen=True)
class ScreenerLayout:
    private_db: Path
    output_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> ScreenerLayout:
        return cls(
            private_db=project_root / "data" / "screener" / "private" / "daily_screener.sqlite3",
            output_dir=project_root / "output" / "screener",
        )


class ScreenerStore:
    def __init__(self, layout: ScreenerLayout) -> None:
        self.layout = layout
        self.layout.private_db.parent.mkdir(parents=True, exist_ok=True)
        self.layout.output_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.layout.private_db)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS screener_runs (
                run_id TEXT PRIMARY KEY,
                screening_date TEXT NOT NULL UNIQUE,
                decision_time TEXT NOT NULL,
                status TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                screener_version TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS screener_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                screening_date TEXT NOT NULL,
                asset_key TEXT NOT NULL,
                symbol TEXT NOT NULL,
                classification TEXT NOT NULL,
                total_score REAL NOT NULL,
                public_json TEXT NOT NULL,
                private_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(screening_date, asset_key),
                FOREIGN KEY(run_id) REFERENCES screener_runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS screener_changes (
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                screening_date TEXT NOT NULL,
                asset_key TEXT NOT NULL,
                symbol TEXT NOT NULL,
                change_type TEXT NOT NULL,
                previous_classification TEXT,
                current_classification TEXT,
                previous_score REAL,
                current_score REAL,
                score_change REAL,
                payload_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(screening_date, asset_key, change_type),
                FOREIGN KEY(run_id) REFERENCES screener_runs(run_id)
            );
            """
        )
        self.connection.commit()

    def has_date(self, screening_date: date) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM screener_runs WHERE screening_date = ?",
            (screening_date.isoformat(),),
        ).fetchone()
        return row is not None

    def register(
        self,
        *,
        screening_date: date,
        decision_time: str,
        config_hash: str,
        screener_version: str,
        records: list[dict[str, Any]],
        private_records: dict[str, dict[str, Any]],
        summary_base: dict[str, Any],
    ) -> dict[str, Any]:
        if self.has_date(screening_date):
            raise ValueError("DUPLICATE_SCREENING_DATE")
        previous = self._previous_observations(screening_date)
        now = datetime.now(UTC).isoformat()
        run_id = f"SCR-{screening_date.isoformat()}-{config_hash[:12]}"
        changes: list[dict[str, Any]] = []
        current_by_key = {str(record["asset_key"]): record for record in records}
        for record in records:
            key = str(record["asset_key"])
            prior = previous.get(key)
            change = _classify_change(prior, record)
            record["change"] = change
            if change["change_type"] != "NO_CHANGE":
                changes.append(change)
        for key, prior in previous.items():
            if key in current_by_key or prior["classification"] not in CANDIDATE_CLASSES:
                continue
            changes.append(
                {
                    "asset_key": key,
                    "symbol": prior["symbol"],
                    "change_type": "CANDIDATE_REMOVED",
                    "previous_classification": prior["classification"],
                    "current_classification": None,
                    "previous_score": prior["total_score"],
                    "current_score": None,
                    "score_change": None,
                }
            )
        summary = {**summary_base, "changes": _change_summary(changes)}
        run_hash = _hash({"summary": summary, "records": records})
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO screener_runs
                (run_id, screening_date, decision_time, status, config_hash,
                 screener_version, summary_json, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    screening_date.isoformat(),
                    decision_time,
                    "GO",
                    config_hash,
                    screener_version,
                    _canonical_json(summary),
                    run_hash,
                    now,
                ),
            )
            for record in records:
                private = private_records.get(str(record["asset_key"]), {})
                self.connection.execute(
                    """
                    INSERT INTO screener_observations
                    (run_id, screening_date, asset_key, symbol, classification,
                     total_score, public_json, private_json, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        screening_date.isoformat(),
                        record["asset_key"],
                        record["symbol"],
                        record["classification"],
                        record["total_score"],
                        _canonical_json(record),
                        _canonical_json(private),
                        _hash(record),
                        now,
                    ),
                )
            for change in changes:
                payload_hash = _hash(change)
                self.connection.execute(
                    """
                    INSERT INTO screener_changes
                    (run_id, screening_date, asset_key, symbol, change_type,
                     previous_classification, current_classification,
                     previous_score, current_score, score_change, payload_json,
                     content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        screening_date.isoformat(),
                        change["asset_key"],
                        change["symbol"],
                        change["change_type"],
                        change.get("previous_classification"),
                        change.get("current_classification"),
                        change.get("previous_score"),
                        change.get("current_score"),
                        change.get("score_change"),
                        _canonical_json(change),
                        payload_hash,
                        now,
                    ),
                )
        return {
            "run_id": run_id,
            "content_hash": run_hash,
            "summary": summary,
            "records": records,
            "changes": changes,
        }

    def latest(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM screener_runs ORDER BY screening_date DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._run_payload(row)

    def by_date(self, screening_date: date) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM screener_runs WHERE screening_date = ?",
            (screening_date.isoformat(),),
        ).fetchone()
        return None if row is None else self._run_payload(row)

    def history(self, symbol: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT public_json FROM screener_observations
            WHERE upper(symbol) = upper(?)
            ORDER BY screening_date
            """,
            (symbol,),
        ).fetchall()
        return [json.loads(row["public_json"]) for row in rows]

    def all_public_records(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT public_json FROM screener_observations ORDER BY screening_date, symbol"
        ).fetchall()
        return [json.loads(row["public_json"]) for row in rows]

    def run_count(self) -> int:
        return int(self.connection.execute("SELECT count(*) FROM screener_runs").fetchone()[0])

    def observation_count(self) -> int:
        return int(
            self.connection.execute("SELECT count(*) FROM screener_observations").fetchone()[0]
        )

    def _run_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        observations = self.connection.execute(
            """
            SELECT public_json FROM screener_observations
            WHERE run_id = ? ORDER BY total_score DESC, symbol
            """,
            (row["run_id"],),
        ).fetchall()
        changes = self.connection.execute(
            "SELECT payload_json FROM screener_changes WHERE run_id = ? ORDER BY symbol",
            (row["run_id"],),
        ).fetchall()
        return {
            "run_id": row["run_id"],
            "screening_date": row["screening_date"],
            "decision_time": row["decision_time"],
            "status": row["status"],
            "config_hash": row["config_hash"],
            "screener_version": row["screener_version"],
            "summary": json.loads(row["summary_json"]),
            "content_hash": row["content_hash"],
            "records": [json.loads(item["public_json"]) for item in observations],
            "changes": [json.loads(item["payload_json"]) for item in changes],
        }

    def _previous_observations(self, screening_date: date) -> dict[str, dict[str, Any]]:
        previous_date = self.connection.execute(
            """
            SELECT max(screening_date) FROM screener_runs
            WHERE screening_date < ?
            """,
            (screening_date.isoformat(),),
        ).fetchone()[0]
        if previous_date is None:
            return {}
        rows = self.connection.execute(
            """
            SELECT asset_key, symbol, classification, total_score
            FROM screener_observations
            WHERE screening_date = ?
            """,
            (previous_date,),
        ).fetchall()
        return {
            str(row["asset_key"]): {
                "asset_key": row["asset_key"],
                "symbol": row["symbol"],
                "classification": row["classification"],
                "total_score": float(row["total_score"]),
            }
            for row in rows
        }


def write_public_artifacts(
    layout: ScreenerLayout,
    run: dict[str, Any],
    all_history: list[dict[str, Any]],
) -> dict[str, str]:
    screening_date = str(run["screening_date"])
    daily_dir = layout.output_dir / screening_date
    daily_dir.mkdir(parents=True, exist_ok=True)
    results_path = daily_dir / "screening-results.json"
    parquet_path = daily_dir / "screening-results.parquet"
    summary_path = daily_dir / "daily-summary.json"
    report_path = daily_dir / "daily-report.md"
    results_payload = {
        "schema": "daily_asset_screener_results_v1",
        "run_id": run["run_id"],
        "screening_date": screening_date,
        "content_hash": run["content_hash"],
        "records": run["records"],
    }
    results_path.write_text(
        json.dumps(results_payload, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(_parquet_records(run["records"])).to_parquet(parquet_path, index=False)
    summary_path.write_text(
        json.dumps(run["summary"], indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(run), encoding="utf-8")
    latest_path = layout.output_dir / "latest-summary.json"
    latest_path.write_text(
        json.dumps(
            {
                "schema": "daily_asset_screener_latest_v1",
                "run_id": run["run_id"],
                "screening_date": screening_date,
                "summary": run["summary"],
                "content_hash": run["content_hash"],
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    history_path = layout.output_dir / "candidate-history.parquet"
    pd.DataFrame(_parquet_records(all_history)).to_parquet(history_path, index=False)
    return {
        "results_json": str(results_path),
        "results_parquet": str(parquet_path),
        "daily_summary": str(summary_path),
        "daily_report": str(report_path),
        "latest_summary": str(latest_path),
        "candidate_history": str(history_path),
    }


def _classify_change(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    current_score = float(current["total_score"])
    current_class = str(current["classification"])
    if previous is None:
        change_type = (
            "NEW_CANDIDATE" if current_class in CANDIDATE_CLASSES else "NEW_OBSERVATION"
        )
        previous_score = None
        previous_class = None
        score_change = None
    else:
        previous_score = float(previous["total_score"])
        previous_class = str(previous["classification"])
        score_change = round(current_score - previous_score, 4)
        if previous_class != current_class:
            change_type = "CLASSIFICATION_CHANGED"
        elif score_change > 1e-9:
            change_type = "SCORE_RISEN"
        elif score_change < -1e-9:
            change_type = "SCORE_FALLEN"
        else:
            change_type = "NO_CHANGE"
    return {
        "asset_key": current["asset_key"],
        "symbol": current["symbol"],
        "change_type": change_type,
        "previous_classification": previous_class,
        "current_classification": current_class,
        "previous_score": previous_score,
        "current_score": current_score,
        "score_change": score_change,
    }


def _change_summary(changes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    def matching(*types: str) -> list[dict[str, Any]]:
        return [item for item in changes if item["change_type"] in types]

    risers = sorted(
        matching("SCORE_RISEN"),
        key=lambda item: item.get("score_change") or 0.0,
        reverse=True,
    )
    fallers = sorted(
        matching("SCORE_FALLEN"),
        key=lambda item: item.get("score_change") or 0.0,
    )
    return {
        "new_candidates": matching("NEW_CANDIDATE"),
        "removed_candidates": matching("CANDIDATE_REMOVED"),
        "classification_changes": matching("CLASSIFICATION_CHANGED"),
        "biggest_score_risers": risers[:20],
        "biggest_score_fallers": fallers[:20],
    }


def _parquet_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for record in records:
        converted.append(
            {
                key: _canonical_json(value) if isinstance(value, (dict, list)) else value
                for key, value in record.items()
            }
        )
    return converted


def _markdown_report(run: dict[str, Any]) -> str:
    summary = run["summary"]
    top = [
        item
        for item in run["records"]
        if item["classification"] in CANDIDATE_CLASSES
    ][:20]
    lines = [
        "# Daily Asset Screener",
        "",
        f"- Screening date: `{run['screening_date']}`",
        f"- Status: `{run['status']}`",
        f"- Data quality: `{summary['data_quality_status']}`",
        "- Execution authority: `NONE`",
        f"- Assets screened: `{summary['screened_count']}`",
        f"- High potential: `{summary['classification_counts'].get('HIGH_POTENTIAL', 0)}`",
        f"- Watchlist: `{summary['classification_counts'].get('WATCHLIST', 0)}`",
        "",
        "## Top Candidates",
        "",
        "| Symbol | Classification | Total | Fundamental | Technical |",
        "|---|---:|---:|---:|---:|",
    ]
    if top:
        lines.extend(
            f"| {item['symbol']} | {item['classification']} | {item['total_score']:.2f} | "
            f"{item['fundamental_score']:.2f} | {item['technical_score']:.2f} |"
            for item in top
        )
    else:
        lines.append("| None | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Daily Movers",
            "",
            f"- Top winners: `{', '.join(item['symbol'] for item in summary['top_winners'][:10]) or 'none'}`",
            f"- Top losers: `{', '.join(item['symbol'] for item in summary['top_losers'][:10]) or 'none'}`",
            "",
            "## Data Exclusions",
            "",
        ]
    )
    for reason, count in sorted(summary["rejection_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{reason}`: {count}")
    lines.extend(
        [
            "",
            "## Benchmarks",
            "",
            f"`{', '.join(summary['benchmarks'])}`",
            "",
            "This is a research-only report. It creates no order intent and grants no execution authority.",
            "",
        ]
    )
    return "\n".join(lines)
