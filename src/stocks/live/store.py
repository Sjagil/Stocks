from __future__ import annotations

import sqlite3
from pathlib import Path

from stocks.ibkr.paper_execution.storage import PaperExecutionStore


class LiveExecutionStore(PaperExecutionStore):
    """Separate append-only execution ledger for supervised live canaries."""

    def initialize(self) -> dict[str, object]:
        result = super().initialize()
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_submission_claims (
                  session_date TEXT PRIMARY KEY,
                  intent_id TEXT NOT NULL UNIQUE,
                  claimed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        return result

    def claim_daily_submission(
        self,
        *,
        session_date: str,
        intent_id: str,
    ) -> str:
        try:
            with self.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO daily_submission_claims(
                      session_date, intent_id
                    ) VALUES (?, ?)
                    """,
                    (session_date, intent_id),
                )
                conn.commit()
        except sqlite3.IntegrityError:
            return "LIVE_DAILY_SUBMISSION_ALREADY_CLAIMED"
        self.append_event(
            intent_id,
            "LIVE_DAILY_SUBMISSION_CLAIMED",
            {"session_date": session_date},
        )
        return "LIVE_DAILY_SUBMISSION_CLAIMED"

    @classmethod
    def from_project_root(cls, project_root: Path) -> "LiveExecutionStore":
        return cls(
            project_root
            / "data"
            / "execution"
            / "live"
            / "private"
            / "live_execution.sqlite3"
        )
