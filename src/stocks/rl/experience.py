from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from stocks.p3.io import atomic_write_json
from stocks.rl.contracts import stable_hash


DEFAULT_DATABASE = Path("data/rl/private/experience.sqlite3")
DEFAULT_STATUS = Path("output/rl/experience-store-status.json")


@dataclass(frozen=True)
class ExperienceDecision:
    timestamp: str
    policy_version: str
    state_hash: str
    observation: list[float]
    available_actions: list[str]
    action_mask: list[int]
    chosen_action: str
    action_probability: float
    portfolio_state: dict[str, Any]
    market_regime: str
    signal_id: str
    strategy_id: str
    asset: str
    episode_id: str
    decision_type: str
    top_features: list[dict[str, Any]] = field(default_factory=list)
    entry: float | None = None
    exit: float | None = None
    execution_authority: str = "NONE"
    money_control: bool = False

    @property
    def decision_id(self) -> str:
        return stable_hash(
            {
                "timestamp": self.timestamp,
                "policy_version": self.policy_version,
                "state_hash": self.state_hash,
                "episode_id": self.episode_id,
                "decision_type": self.decision_type,
            }
        )

    def validate(self) -> None:
        if not self.timestamp or not self.policy_version or not self.state_hash:
            raise ValueError("experience decision identity is incomplete")
        if len(self.action_mask) != len(self.available_actions):
            raise ValueError("available actions and mask length differ")
        if not 0.0 <= float(self.action_probability) <= 1.0:
            raise ValueError("action probability must be between zero and one")
        if self.execution_authority != "NONE" or self.money_control:
            raise ValueError("RL experience decisions cannot carry money authority")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["decision_id"] = self.decision_id
        payload["schema"] = "rl_experience_decision_v1"
        return payload


@dataclass(frozen=True)
class ExperienceOutcome:
    decision_id: str
    episode_id: str
    timestamp: str
    reward: float
    reward_components: dict[str, float]
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    slippage: float
    mfe: float
    mae: float
    holding_duration: int
    outcome: dict[str, Any]

    @property
    def outcome_id(self) -> str:
        return stable_hash(
            {
                "decision_id": self.decision_id,
                "episode_id": self.episode_id,
                "timestamp": self.timestamp,
                "outcome": self.outcome,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.decision_id or not self.episode_id:
            raise ValueError("experience outcome identity is incomplete")
        payload = asdict(self)
        payload["outcome_id"] = self.outcome_id
        payload["schema"] = "rl_experience_outcome_v1"
        return payload


class ExperienceStore:
    """Restart-safe append-only SQLite store for decisions and later outcomes."""

    def __init__(self, project_root: Path, database_path: Path = DEFAULT_DATABASE) -> None:
        self.project_root = project_root.resolve()
        self.path = (self.project_root / database_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    chosen_action TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reward REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES decisions(decision_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_outcomes_decision
                    ON outcomes(decision_id);
                CREATE INDEX IF NOT EXISTS idx_decisions_episode
                    ON decisions(episode_id, timestamp);
                CREATE TABLE IF NOT EXISTS episode_events (
                    event_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_episode_events_episode
                    ON episode_events(episode_id, timestamp);
                CREATE TABLE IF NOT EXISTS training_job_events (
                    event_id TEXT PRIMARY KEY,
                    job_key TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_training_jobs_key
                    ON training_job_events(job_key, timestamp);
                """
            )

    def append_decision(self, decision: ExperienceDecision) -> bool:
        payload = decision.to_dict()
        return self._append_immutable(
            table="decisions",
            identity_column="decision_id",
            identity=decision.decision_id,
            columns={
                "timestamp": decision.timestamp,
                "policy_version": decision.policy_version,
                "state_hash": decision.state_hash,
                "episode_id": decision.episode_id,
                "decision_type": decision.decision_type,
                "chosen_action": decision.chosen_action,
                "asset": decision.asset,
                "strategy_id": decision.strategy_id,
            },
            payload=payload,
        )

    def append_outcome(self, outcome: ExperienceOutcome) -> bool:
        payload = outcome.to_dict()
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM decisions WHERE decision_id = ?",
                (outcome.decision_id,),
            ).fetchone()
        if not exists:
            raise ValueError("cannot append outcome for unknown RL decision")
        return self._append_immutable(
            table="outcomes",
            identity_column="outcome_id",
            identity=outcome.outcome_id,
            columns={
                "decision_id": outcome.decision_id,
                "episode_id": outcome.episode_id,
                "timestamp": outcome.timestamp,
                "reward": float(outcome.reward),
            },
            payload=payload,
        )

    def append_episode_event(
        self,
        episode_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        timestamp: str | None = None,
    ) -> bool:
        timestamp = timestamp or _now()
        event_id = stable_hash(
            {
                "episode_id": episode_id,
                "event_type": event_type,
                "timestamp": timestamp,
                "payload": payload,
            }
        )
        return self._append_immutable(
            table="episode_events",
            identity_column="event_id",
            identity=event_id,
            columns={
                "episode_id": episode_id,
                "timestamp": timestamp,
                "event_type": event_type,
            },
            payload=payload,
        )

    def claim_training_job(self, job_key: str, payload: dict[str, Any]) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT status FROM training_job_events
                WHERE job_key = ? ORDER BY timestamp DESC, rowid DESC LIMIT 1
                """,
                (job_key,),
            ).fetchone()
            if row and row[0] in {"CLAIMED", "RUNNING", "COMPLETED"}:
                connection.rollback()
                return False
            event = self._training_event(job_key, "CLAIMED", payload)
            self._insert_event(connection, "training_job_events", event)
            connection.commit()
            return True

    def finish_training_job(
        self,
        job_key: str,
        *,
        status: str,
        payload: dict[str, Any],
    ) -> bool:
        if status not in {"COMPLETED", "FAILED", "REJECTED"}:
            raise ValueError("unsupported terminal training job status")
        event = self._training_event(job_key, status, payload)
        with self._connection() as connection:
            latest = connection.execute(
                """
                SELECT status FROM training_job_events
                WHERE job_key = ? ORDER BY timestamp DESC, rowid DESC LIMIT 1
                """,
                (job_key,),
            ).fetchone()
            if not latest or latest[0] != "CLAIMED":
                return False
            self._insert_event(connection, "training_job_events", event)
            return True

    def recover_open_episodes(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                WITH latest AS (
                    SELECT episode_id, MAX(rowid) AS latest_row
                    FROM episode_events GROUP BY episode_id
                )
                SELECT e.payload_json FROM episode_events e
                JOIN latest l ON e.rowid = l.latest_row
                WHERE e.event_type NOT IN ('CLOSED', 'CANCELLED')
                ORDER BY e.timestamp
                """
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def unresolved_decisions(self, *, asset: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT d.payload_json FROM decisions d
            LEFT JOIN outcomes o ON o.decision_id = d.decision_id
            WHERE o.decision_id IS NULL
        """
        parameters: tuple[Any, ...] = ()
        if asset:
            query += " AND d.asset = ?"
            parameters = (asset.upper(),)
        query += " ORDER BY d.timestamp, d.rowid"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [json.loads(row[0]) for row in rows]

    def statistics(self) -> dict[str, Any]:
        with self._connection() as connection:
            decisions = int(connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])
            outcomes = int(connection.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])
            holds = int(
                connection.execute(
                    "SELECT COUNT(*) FROM decisions WHERE chosen_action IN ('HOLD','SKIP')"
                ).fetchone()[0]
            )
            closed_episodes = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT episode_id) FROM episode_events WHERE event_type='CLOSED'"
                ).fetchone()[0]
            )
            reward = connection.execute(
                "SELECT AVG(reward), MIN(reward), MAX(reward) FROM outcomes"
            ).fetchone()
            action_rows = connection.execute(
                "SELECT chosen_action, COUNT(*) FROM decisions GROUP BY chosen_action"
            ).fetchall()
            recent_rewards = [
                float(row[0])
                for row in connection.execute(
                    "SELECT reward FROM outcomes ORDER BY timestamp DESC, rowid DESC LIMIT 50"
                ).fetchall()
            ]
            joined_rows = connection.execute(
                """
                SELECT d.payload_json, o.payload_json
                FROM outcomes o JOIN decisions d ON d.decision_id = o.decision_id
                ORDER BY o.timestamp, o.rowid
                """
            ).fetchall()
            decision_payload_rows = connection.execute(
                "SELECT payload_json FROM decisions ORDER BY timestamp, rowid"
            ).fetchall()
        net_pnl = 0.0
        equity_curve = [0.0]
        reward_by_regime: dict[str, list[float]] = {}
        pnl_by_regime: dict[str, float] = {}
        actions_by_regime: dict[str, dict[str, int]] = {}
        proposed_turnovers: list[float] = []
        for (decision_json,) in decision_payload_rows:
            decision_payload = json.loads(decision_json)
            regime = str(decision_payload.get("market_regime") or "UNKNOWN")
            chosen_action = str(
                decision_payload.get("chosen_action") or "UNKNOWN"
            )
            regime_actions = actions_by_regime.setdefault(regime, {})
            regime_actions[chosen_action] = regime_actions.get(chosen_action, 0) + 1
            turnover = decision_payload.get("portfolio_state", {}).get(
                "proposed_turnover"
            )
            if turnover is not None:
                proposed_turnovers.append(float(turnover))
        for decision_json, outcome_json in joined_rows:
            decision_payload = json.loads(decision_json)
            outcome_payload = json.loads(outcome_json)
            regime = str(decision_payload.get("market_regime") or "UNKNOWN")
            pnl = float(outcome_payload.get("realized_pnl", 0.0) or 0.0)
            reward_value = float(outcome_payload.get("reward", 0.0) or 0.0)
            net_pnl += pnl
            equity_curve.append(net_pnl)
            reward_by_regime.setdefault(regime, []).append(reward_value)
            pnl_by_regime[regime] = pnl_by_regime.get(regime, 0.0) + pnl
        running_peak = equity_curve[0]
        maximum_drawdown = 0.0
        for value in equity_curve:
            running_peak = max(running_peak, value)
            maximum_drawdown = max(maximum_drawdown, running_peak - value)
        return {
            "schema": "rl_experience_store_status_v1",
            "status": "GO",
            "database": self.path.relative_to(self.project_root).as_posix(),
            "decision_count": decisions,
            "outcome_count": outcomes,
            "unresolved_decision_count": decisions - outcomes,
            "hold_or_skip_count": holds,
            "closed_episode_count": closed_episodes,
            "mean_reward": reward[0],
            "minimum_reward": reward[1],
            "maximum_reward": reward[2],
            "rolling_50_mean_reward": (
                sum(recent_rewards) / len(recent_rewards) if recent_rewards else None
            ),
            "action_distribution": {str(name): int(count) for name, count in action_rows},
            "net_pnl": net_pnl if outcomes else None,
            "maximum_drawdown": maximum_drawdown if outcomes else None,
            "reward_by_regime": {
                regime: sum(values) / len(values)
                for regime, values in reward_by_regime.items()
            },
            "pnl_by_regime": pnl_by_regime,
            "actions_by_regime": actions_by_regime,
            "average_proposed_turnover": (
                sum(proposed_turnovers) / len(proposed_turnovers)
                if proposed_turnovers
                else None
            ),
            "append_only_decisions": True,
            "append_only_outcomes": True,
            "restart_safe": True,
            "execution_authority": "NONE",
            "broker_writes": 0,
            "generated_at": _now(),
        }

    def publish_status(self, path: Path = DEFAULT_STATUS) -> dict[str, Any]:
        status = self.statistics()
        atomic_write_json(self.project_root / path, status)
        return status

    def _append_immutable(
        self,
        *,
        table: str,
        identity_column: str,
        identity: str,
        columns: dict[str, Any],
        payload: dict[str, Any],
    ) -> bool:
        payload_json = _canonical_json(payload)
        payload_hash = stable_hash(payload)
        with self._connection() as connection:
            existing = connection.execute(
                f"SELECT payload_hash FROM {table} WHERE {identity_column} = ?",
                (identity,),
            ).fetchone()
            if existing:
                if existing[0] != payload_hash:
                    raise ValueError(f"immutable RL {table} identity collision")
                return False
            values = {
                identity_column: identity,
                **columns,
                "payload_json": payload_json,
                "payload_hash": payload_hash,
                "created_at": _now(),
            }
            names = ",".join(values)
            placeholders = ",".join("?" for _ in values)
            connection.execute(
                f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            return True

    def _training_event(
        self, job_key: str, status: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        timestamp = _now()
        body = {"job_key": job_key, "status": status, "payload": payload}
        return {
            "event_id": stable_hash({**body, "timestamp": timestamp}),
            "job_key": job_key,
            "timestamp": timestamp,
            "status": status,
            "payload_json": _canonical_json(body),
            "payload_hash": stable_hash(body),
            "created_at": timestamp,
        }

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, table: str, values: dict[str, Any]
    ) -> None:
        names = ",".join(values)
        placeholders = ",".join("?" for _ in values)
        connection.execute(
            f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
            tuple(values.values()),
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DEFAULT_DATABASE",
    "DEFAULT_STATUS",
    "ExperienceDecision",
    "ExperienceOutcome",
    "ExperienceStore",
]
