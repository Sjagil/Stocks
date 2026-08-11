from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def load_manifest(
    path: Path,
) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    manifest_path = Path(
        args.manifest
    )

    rows = load_manifest(
        manifest_path
    )

    conn = sqlite3.connect(
        db_path
    )

    try:
        conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        updated = 0
        cleared = 0

        for row in rows:
            accession = str(
                row.get("accession")
                or ""
            ).strip()

            if not accession:
                continue

            metadata = (
                row.get("metadata")
                or {}
            )

            amends_accession = (
                metadata.get(
                    "amends_accession"
                )
            )

            if amends_accession:
                cursor = conn.execute(
                    """
                    UPDATE sec_intel_filings
                    SET amends_accession = ?
                    WHERE accession = ?
                    """,
                    (
                        str(
                            amends_accession
                        ),
                        accession,
                    ),
                )

                updated += cursor.rowcount

            else:
                cursor = conn.execute(
                    """
                    UPDATE sec_intel_filings
                    SET amends_accession = NULL
                    WHERE accession = ?
                      AND amends_accession IS NOT NULL
                    """,
                    (
                        accession,
                    ),
                )

                cleared += cursor.rowcount

        conn.commit()

        audit_rows = conn.execute(
            """
            SELECT
                accession,
                report_period,
                form_type,
                amends_accession,
                json_extract(
                    metadata_json,
                    '$.amendment_type'
                ) AS amendment_type
            FROM sec_intel_filings
            WHERE form_type LIKE '13F-HR%'
              AND (
                  amends_accession IS NOT NULL
                  OR json_extract(
                      metadata_json,
                      '$.is_amendment'
                  ) = 1
              )
            ORDER BY
                report_period,
                accepted_at
            """
        ).fetchall()

    finally:
        conn.close()

    print(
        json.dumps(
            {
                "status": "PASS",
                "updated": updated,
                "cleared": cleared,
                "amendment_rows": len(
                    audit_rows
                ),
            },
            indent=2,
        )
    )

    for result in audit_rows:
        print(
            " | ".join(
                str(value)
                for value in result
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
