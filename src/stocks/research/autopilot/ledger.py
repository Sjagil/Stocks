from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from stocks.research.autopilot.contracts import ComponentSpec, StrategySpec, stable_hash


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


@dataclass(frozen=True)
class AutopilotLayout:
    project_root: Path

    @property
    def private_root(self) -> Path:
        return self.project_root / "data" / "research" / "autopilot" / "private"

    @property
    def database(self) -> Path:
        return self.private_root / "research_autopilot.sqlite3"

    @property
    def output_root(self) -> Path:
        return self.project_root / "output" / "research" / "autopilot"

    @property
    def forward_root(self) -> Path:
        return self.project_root / "output" / "research" / "forward"


class ResearchLedger:
    def __init__(self, layout: AutopilotLayout) -> None:
        self.layout = layout
        layout.private_root.mkdir(parents=True, exist_ok=True)
        layout.output_root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(layout.database)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS components (
                component_name TEXT NOT NULL,
                version TEXT NOT NULL,
                component_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                PRIMARY KEY(component_name, version)
            );
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                strategy_hash TEXT NOT NULL UNIQUE,
                family TEXT NOT NULL,
                parent_strategy_id TEXT,
                seed INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                registered_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS parameter_sets (
                parameter_set_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                parameter_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS ensembles (
                ensemble_id TEXT PRIMARY KEY,
                ensemble_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                registered_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id TEXT PRIMARY KEY,
                cadence TEXT NOT NULL,
                family TEXT,
                config_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                stage INTEGER NOT NULL,
                cost_profile TEXT NOT NULL,
                status TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                trial_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id),
                FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                research_level TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS forward_registrations (
                registration_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL UNIQUE,
                strategy_hash TEXT NOT NULL,
                frozen_payload_json TEXT NOT NULL,
                frozen_hash TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                authority TEXT NOT NULL,
                FOREIGN KEY(strategy_id) REFERENCES strategies(strategy_id)
            );
            CREATE TABLE IF NOT EXISTS forward_observations (
                observation_id TEXT PRIMARY KEY,
                registration_id TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                session_date TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                FOREIGN KEY(registration_id) REFERENCES forward_registrations(registration_id),
                UNIQUE(registration_id, session_date)
            );
            CREATE TABLE IF NOT EXISTS dynamic_forward_registrations (
                strategy_id TEXT PRIMARY KEY,
                frozen_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                authority TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dynamic_forward_observations (
                observation_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                session_date TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                FOREIGN KEY(strategy_id)
                    REFERENCES dynamic_forward_registrations(strategy_id),
                UNIQUE(strategy_id, session_date, content_hash)
            );
            CREATE TABLE IF NOT EXISTS bulk_strategy_dna (
                strategy_id TEXT PRIMARY KEY,
                strategy_hash TEXT NOT NULL UNIQUE,
                family TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                registered_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bulk_trials (
                trial_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                cost_bps REAL NOT NULL,
                status TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                trial_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY(strategy_id) REFERENCES bulk_strategy_dna(strategy_id)
            );
            """
        )
        self.connection.commit()

    def register_components(self, components: Iterable[ComponentSpec]) -> int:
        inserted = 0
        now = datetime.now(UTC).isoformat()
        with self.connection:
            for component in components:
                payload = asdict(component)
                digest = stable_hash(payload)
                cursor = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO components
                    (component_name, version, component_hash, payload_json, registered_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        component.name,
                        component.version,
                        digest,
                        _json(payload),
                        now,
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def register_strategies(self, strategies: Iterable[StrategySpec]) -> dict[str, int]:
        inserted = 0
        existing = 0
        now = datetime.now(UTC).isoformat()
        with self.connection:
            for strategy in strategies:
                payload = asdict(strategy)
                row = self.connection.execute(
                    "SELECT strategy_hash FROM strategies WHERE strategy_id=?",
                    (strategy.strategy_id,),
                ).fetchone()
                if row is not None:
                    if row["strategy_hash"] != strategy.strategy_hash:
                        raise ValueError("STRATEGY_ID_HASH_COLLISION")
                    existing += 1
                    continue
                self.connection.execute(
                    """
                    INSERT INTO strategies
                    (strategy_id, strategy_hash, family, parent_strategy_id, seed,
                     payload_json, registered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        strategy.strategy_id,
                        strategy.strategy_hash,
                        strategy.family,
                        strategy.parent_strategy_id,
                        strategy.seed,
                        _json(payload),
                        now,
                    ),
                )
                inserted += 1
        return {"inserted": inserted, "existing": existing}

    def register_parameters(self, strategies: Iterable[StrategySpec]) -> int:
        inserted = 0
        now = datetime.now(UTC).isoformat()
        with self.connection:
            for strategy in strategies:
                payload = {
                    "strategy_id": strategy.strategy_id,
                    "parameters": strategy.parameters,
                    "entry_timeframe": strategy.entry_timeframe,
                    "confirmation_timeframe": strategy.confirmation_timeframe,
                    "regime_timeframe": strategy.regime_timeframe,
                }
                digest = stable_hash(payload)
                cursor = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO parameter_sets
                    (parameter_set_id, strategy_id, parameter_hash, payload_json,
                     registered_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        f"PARAM-{digest[:24]}",
                        strategy.strategy_id,
                        digest,
                        _json(payload),
                        now,
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def register_ensemble(self, payload: dict[str, Any]) -> dict[str, Any]:
        ensemble_id = str(payload["ensemble_id"])
        ensemble_hash = str(payload["ensemble_hash"])
        existing = self.connection.execute(
            "SELECT ensemble_hash FROM ensembles WHERE ensemble_id=?",
            (ensemble_id,),
        ).fetchone()
        if existing is not None and existing["ensemble_hash"] != ensemble_hash:
            raise ValueError("ENSEMBLE_ID_HASH_COLLISION")
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO ensembles
                (ensemble_id, ensemble_hash, payload_json, registered_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    ensemble_id,
                    ensemble_hash,
                    _json(payload),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return {
            "ensemble_id": ensemble_id,
            "inserted": cursor.rowcount == 1,
        }

    def ensembles(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT payload_json FROM ensembles ORDER BY ensemble_id"
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def register_campaign(self, payload: dict[str, Any]) -> str:
        immutable = {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "campaign_id"}
        }
        digest = stable_hash(immutable)
        campaign_id = str(payload.get("campaign_id") or f"CAMP-{digest[:20]}")
        created = str(payload.get("created_at") or datetime.now(UTC).isoformat())
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO campaigns
                (campaign_id, cadence, family, config_hash, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    payload["cadence"],
                    payload.get("family"),
                    digest,
                    _json({**payload, "campaign_id": campaign_id}),
                    created,
                ),
            )
        return campaign_id

    def append_trial(
        self,
        *,
        campaign_id: str,
        strategy_id: str,
        stage: int,
        cost_profile: str,
        status: str,
        metrics: dict[str, Any],
        provenance: dict[str, Any],
    ) -> str:
        payload = {
            "campaign_id": campaign_id,
            "strategy_id": strategy_id,
            "stage": stage,
            "cost_profile": cost_profile,
            "status": status,
            "metrics": metrics,
            "provenance": provenance,
        }
        digest = stable_hash(payload)
        trial_id = f"TRIAL-{digest[:24]}"
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO trials
                (trial_id, campaign_id, strategy_id, stage, cost_profile, status,
                 metrics_json, provenance_json, trial_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    campaign_id,
                    strategy_id,
                    stage,
                    cost_profile,
                    status,
                    _json(metrics),
                    _json(provenance),
                    digest,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return trial_id

    def append_decision(
        self,
        *,
        strategy_id: str,
        new_status: str,
        research_level: str,
        reasons: list[str],
        evidence: dict[str, Any],
    ) -> str:
        previous = self.latest_decision(strategy_id)
        payload = {
            "strategy_id": strategy_id,
            "previous_status": None if previous is None else previous["new_status"],
            "new_status": new_status,
            "research_level": research_level,
            "reasons": reasons,
            "evidence_hash": stable_hash(evidence),
        }
        digest = stable_hash(payload)
        decision_id = f"DEC-{digest[:24]}"
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO decisions
                (decision_id, strategy_id, previous_status, new_status,
                 research_level, reasons_json, evidence_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    strategy_id,
                    payload["previous_status"],
                    new_status,
                    research_level,
                    _json(reasons),
                    payload["evidence_hash"],
                    datetime.now(UTC).isoformat(),
                ),
            )
        return decision_id

    def latest_decision(self, strategy_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM decisions WHERE strategy_id=?
            ORDER BY created_at DESC, rowid DESC LIMIT 1
            """,
            (strategy_id,),
        ).fetchone()
        return None if row is None else dict(row)

    def strategy(self, strategy_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM strategies WHERE strategy_id=?",
            (strategy_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["latest_decision"] = self.latest_decision(strategy_id)
        payload["trials"] = self.trials(strategy_id)
        return payload

    def strategies(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT strategy_id, payload_json FROM strategies ORDER BY strategy_id"
        ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["latest_decision"] = self.latest_decision(row["strategy_id"])
            result.append(payload)
        return result

    def trials(self, strategy_id: str | None = None) -> list[dict[str, Any]]:
        if strategy_id is None:
            rows = self.connection.execute(
                "SELECT * FROM trials ORDER BY created_at, trial_id"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM trials WHERE strategy_id=? ORDER BY created_at, trial_id",
                (strategy_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "metrics": json.loads(row["metrics_json"]),
                "provenance": json.loads(row["provenance_json"]),
            }
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        tables = (
            "components",
            "strategies",
            "parameter_sets",
            "ensembles",
            "campaigns",
            "trials",
            "decisions",
            "forward_registrations",
            "forward_observations",
            "bulk_strategy_dna",
            "bulk_trials",
        )
        return {
            table: int(
                self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            )
            for table in tables
        }

    def bulk_strategies(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT strategy_id, strategy_hash, family, timeframe,
                   asset_class, payload_json, registered_at
            FROM bulk_strategy_dna
            ORDER BY strategy_id
            """
        ).fetchall()
        return [
            {
                **json.loads(row["payload_json"]),
                "strategy_id": row["strategy_id"],
                "strategy_hash": row["strategy_hash"],
                "family": row["family"],
                "timeframe": row["timeframe"],
                "asset_class": row["asset_class"],
                "registered_at": row["registered_at"],
            }
            for row in rows
        ]

    def bulk_trials(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT trial_id, strategy_id, cost_bps, status,
                   metrics_json, provenance_json, trial_hash, created_at
            FROM bulk_trials
            ORDER BY created_at, trial_id
            """
        ).fetchall()
        return [
            {
                "trial_id": row["trial_id"],
                "strategy_id": row["strategy_id"],
                "cost_bps": float(row["cost_bps"]),
                "status": row["status"],
                "metrics": json.loads(row["metrics_json"]),
                "provenance": json.loads(row["provenance_json"]),
                "trial_hash": row["trial_hash"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def register_bulk_strategies(
        self,
        strategies: Iterable[dict[str, Any]],
    ) -> dict[str, int]:
        inserted = 0
        existing = 0
        now = datetime.now(UTC).isoformat()
        with self.connection:
            for strategy in strategies:
                strategy_id = str(strategy["strategy_id"])
                strategy_hash = str(strategy["strategy_hash"])
                payload = {
                    key: value
                    for key, value in strategy.items()
                    if key not in {"strategy_id", "strategy_hash"}
                }
                if stable_hash(payload) != strategy_hash:
                    raise ValueError("BULK_STRATEGY_HASH_MISMATCH")
                row = self.connection.execute(
                    """
                    SELECT strategy_hash FROM bulk_strategy_dna
                    WHERE strategy_id=?
                    """,
                    (strategy_id,),
                ).fetchone()
                if row is not None:
                    if row["strategy_hash"] != strategy_hash:
                        raise ValueError("BULK_STRATEGY_ID_HASH_COLLISION")
                    existing += 1
                    continue
                self.connection.execute(
                    """
                    INSERT INTO bulk_strategy_dna(
                        strategy_id, strategy_hash, family, timeframe,
                        asset_class, payload_json, registered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        strategy_id,
                        strategy_hash,
                        str(strategy["family"]),
                        str(strategy["timeframe"]),
                        str(strategy["asset_class"]),
                        _json(strategy),
                        now,
                    ),
                )
                inserted += 1
        return {"inserted": inserted, "existing": existing}

    def append_bulk_trial(
        self,
        *,
        strategy_id: str,
        cost_bps: float,
        status: str,
        metrics: dict[str, Any],
        provenance: dict[str, Any],
    ) -> str:
        payload = {
            "strategy_id": strategy_id,
            "cost_bps": float(cost_bps),
            "status": status,
            "metrics": metrics,
            "provenance": provenance,
        }
        digest = stable_hash(payload)
        trial_id = f"BULK-TRIAL-{digest[:24]}"
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO bulk_trials(
                    trial_id, strategy_id, cost_bps, status, metrics_json,
                    provenance_json, trial_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    strategy_id,
                    float(cost_bps),
                    status,
                    _json(metrics),
                    _json(provenance),
                    digest,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return trial_id

    def register_forward(self, strategy_id: str) -> dict[str, Any]:
        strategy = self.strategy(strategy_id)
        if strategy is None:
            return {"status": "NOT_FOUND", "strategy_id": strategy_id}
        decision = strategy.get("latest_decision")
        if decision is None or decision["research_level"] != "FORWARD_OBSERVER_CANDIDATE":
            return {
                "status": "FORWARD_REGISTRATION_BLOCKED",
                "strategy_id": strategy_id,
                "reason": "FINANCIAL_ELIGIBILITY_NOT_GRANTED",
            }
        frozen = {
            key: value
            for key, value in strategy.items()
            if key not in {"latest_decision", "trials"}
        }
        frozen_hash = stable_hash(frozen)
        registration_id = f"FWD-{frozen_hash[:24]}"
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO forward_registrations
                (registration_id, strategy_id, strategy_hash, frozen_payload_json,
                 frozen_hash, registered_at, authority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    registration_id,
                    strategy_id,
                    frozen["strategy_hash"],
                    _json(frozen),
                    frozen_hash,
                    datetime.now(UTC).isoformat(),
                    "NONE",
                ),
            )
        return {
            "status": "GO",
            "registration_id": registration_id,
            "strategy_id": strategy_id,
            "strategy_hash": frozen["strategy_hash"],
            "frozen_hash": frozen_hash,
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
        }

    def forward_registrations(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM forward_registrations ORDER BY registered_at"
        ).fetchall()
        return [dict(row) for row in rows]

    def append_forward_observation(
        self,
        *,
        registration_id: str,
        session_date: str,
        payload: dict[str, Any],
    ) -> str:
        digest = stable_hash(
            {
                "registration_id": registration_id,
                "session_date": session_date,
                "payload": payload,
            }
        )
        observation_id = f"OBS-{digest[:24]}"
        existing = self.connection.execute(
            """
            SELECT observation_id, content_hash FROM forward_observations
            WHERE registration_id=? AND session_date=?
            """,
            (registration_id, session_date),
        ).fetchone()
        if existing is not None:
            if existing["content_hash"] != digest:
                raise ValueError("FORWARD_OBSERVATION_IMMUTABILITY_CONFLICT")
            return str(existing["observation_id"])
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO forward_observations
                (observation_id, registration_id, observed_at, session_date,
                 payload_json, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    registration_id,
                    datetime.now(UTC).isoformat(),
                    session_date,
                    _json(payload),
                    digest,
                ),
            )
        return observation_id

    def forward_observations(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM forward_observations ORDER BY session_date, observation_id"
        ).fetchall()
        return [
            {**dict(row), "payload": json.loads(row["payload_json"])}
            for row in rows
        ]

    def register_dynamic_forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        strategy_id = str(payload["strategy_id"])
        frozen = {
            key: value
            for key, value in payload.items()
            if key not in {"registered_at", "authority"}
        }
        frozen_hash = stable_hash(frozen)
        existing = self.connection.execute(
            """
            SELECT frozen_hash FROM dynamic_forward_registrations
            WHERE strategy_id=?
            """,
            (strategy_id,),
        ).fetchone()
        if existing is not None and existing["frozen_hash"] != frozen_hash:
            raise ValueError("DYNAMIC_FORWARD_FROZEN_HASH_MISMATCH")
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO dynamic_forward_registrations
                (strategy_id, frozen_hash, payload_json, registered_at, authority)
                VALUES (?, ?, ?, ?, 'NONE')
                """,
                (
                    strategy_id,
                    frozen_hash,
                    _json(frozen),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return {
            "strategy_id": strategy_id,
            "frozen_hash": frozen_hash,
            "inserted": cursor.rowcount == 1,
            "authority": "NONE",
        }

    def append_dynamic_forward_observation(
        self,
        *,
        strategy_id: str,
        session_date: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        immutable = {
            **payload,
            "strategy_id": strategy_id,
            "session_date": session_date,
        }
        content_hash = stable_hash(immutable)
        observation_id = f"DYNFWD-{content_hash[:24]}"
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO dynamic_forward_observations
                (observation_id, strategy_id, session_date, payload_json,
                 content_hash, observed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    strategy_id,
                    session_date,
                    _json(immutable),
                    content_hash,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return {
            "observation_id": observation_id,
            "strategy_id": strategy_id,
            "content_hash": content_hash,
            "inserted": cursor.rowcount == 1,
        }

    def dynamic_forward_status(self) -> dict[str, Any]:
        registrations = self.connection.execute(
            """
            SELECT strategy_id, frozen_hash, payload_json, registered_at, authority
            FROM dynamic_forward_registrations ORDER BY strategy_id
            """
        ).fetchall()
        observations = self.connection.execute(
            """
            SELECT observation_id, strategy_id, session_date, content_hash,
                   observed_at
            FROM dynamic_forward_observations
            ORDER BY session_date, strategy_id
            """
        ).fetchall()
        return {
            "registration_count": len(registrations),
            "observation_count": len(observations),
            "registrations": [dict(row) for row in registrations],
            "observations": [dict(row) for row in observations],
            "authority": "NONE",
        }
