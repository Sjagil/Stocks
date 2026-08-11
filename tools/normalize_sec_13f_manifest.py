from __future__ import annotations

import argparse
import csv
import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def local_name(tag: str) -> str:
    return (
        tag.rsplit("}", 1)[-1]
        .split(":", 1)[-1]
        .lower()
    )


def to_float(value: Any) -> float | None:
    text = (
        str(value or "")
        .strip()
        .replace(",", "")
    )

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def to_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def parse_information_table(
    path: Path,
) -> list[dict[str, Any]]:
    root = ET.fromstring(
        path.read_bytes()
    )

    positions: list[dict[str, Any]] = []

    for node in root.iter():
        if local_name(node.tag) != "infotable":
            continue

        position: dict[str, Any] = {
            "shares": None,
            "value": None,
            "put_call": None,
        }

        for child in node.iter():
            name = local_name(child.tag)
            text = (child.text or "").strip()

            if not text:
                continue

            if name == "value":
                position["value"] = to_float(text)

            elif name in {
                "sshprnamt",
                "sharesorprincipalamount",
            }:
                position["shares"] = to_float(text)

            elif name == "putcall":
                position["put_call"] = text.upper()

        positions.append(position)

    return positions


def infer_value_multiplier(
    positions: list[dict[str, Any]],
) -> tuple[float, float, int]:
    ratios: list[float] = []

    for position in positions:
        # Alleen gewone aandelen voor de unit-test.
        if position.get("put_call"):
            continue

        shares = to_float(
            position.get("shares")
        )
        value = to_float(
            position.get("value")
        )

        if (
            shares is None
            or value is None
            or shares <= 0
            or value <= 0
        ):
            continue

        ratios.append(
            value / shares
        )

    if not ratios:
        raise RuntimeError(
            "Geen bruikbare equity value/share-ratio's."
        )

    median_raw = statistics.median(ratios)

    # Moderne dollarfiling:
    # value / shares geeft direct een normale koers.
    if 1.0 <= median_raw <= 20_000.0:
        return 1.0, median_raw, len(ratios)

    # Feitelijke thousand-dollarfiling:
    # raw ratio is centen/fracties, x1000 is plausibel.
    thousand_adjusted = median_raw * 1_000.0

    if (
        0.0001 <= median_raw < 1.0
        and 1.0 <= thousand_adjusted <= 20_000.0
    ):
        return (
            1_000.0,
            median_raw,
            len(ratios),
        )

    raise RuntimeError(
        "Ambigue 13F-value-eenheid: "
        f"median_raw={median_raw:.8f}, "
        f"median_x1000={thousand_adjusted:.8f}"
    )


def load_audit(
    path: Path,
) -> dict[str, dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return {
            str(row["accession"]).strip(): row
            for row in csv.DictReader(handle)
        }


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--content-root",
        required=True,
    )
    parser.add_argument(
        "--audit",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    content_root = Path(args.content_root)
    audit_path = Path(args.audit)

    audit_by_accession = load_audit(
        audit_path
    )

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    normalized: list[dict[str, Any]] = []
    failures: list[str] = []

    for row in rows:
        accession = str(
            row.get("accession") or ""
        ).strip()

        metadata = dict(
            row.get("metadata") or {}
        )

        audit = audit_by_accession.get(
            accession,
            {},
        )

        content_path = (
            content_root
            / str(row["content_path"])
        )

        try:
            positions = parse_information_table(
                content_path
            )

            (
                multiplier,
                median_raw,
                ratio_count,
            ) = infer_value_multiplier(
                positions
            )

        except Exception as exc:
            failures.append(
                f"{accession}: {exc!r}"
            )
            continue

        is_amendment = to_bool(
            audit.get("is_amendment")
        )

        amendment_type = (
            str(
                audit.get("amendment_type")
                or ""
            )
            .strip()
            .upper()
            or None
        )

        metadata.update(
            {
                "is_amendment": is_amendment,
                "amendment_type": amendment_type,
                "amendment_number": (
                    audit.get(
                        "amendment_number"
                    )
                    or None
                ),
                "report_type": (
                    audit.get("report_type")
                    or None
                ),
                "primary_document": (
                    audit.get(
                        "primary_document"
                    )
                    or None
                ),
                "thirteen_f_value_multiplier": (
                    multiplier
                ),
                "value_unit_inference": {
                    "method": (
                        "median_equity_value_per_share"
                    ),
                    "raw_median_value_per_share": (
                        median_raw
                    ),
                    "equity_observations": (
                        ratio_count
                    ),
                    "selected_multiplier": (
                        multiplier
                    ),
                },
            }
        )

        row["metadata"] = metadata
        normalized.append(row)

        manager = metadata.get(
            "manager_key",
            row.get("filer_cik"),
        )

        print(
            f"{str(manager):12} "
            f"{accession} "
            f"period={row.get('report_period')} "
            f"multiplier={multiplier:,.0f} "
            f"median_raw={median_raw:.6f} "
            f"amendment={is_amendment} "
            f"type={amendment_type}"
        )

    if failures:
        print("\nNORMALISATIE GESTOPT")

        for failure in failures:
            print(f"  {failure}")

        return 1

    # Bepaal voor amendments de oorspronkelijke filing
    # binnen hetzelfde manager/kwartaal.
    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in normalized:
        key = (
            str(
                row.get("filer_cik")
                or row.get("issuer_cik")
                or ""
            ),
            str(
                row.get("report_period")
                or ""
            ),
        )

        groups[key].append(row)

    for group_rows in groups.values():
        group_rows.sort(
            key=lambda item: (
                str(
                    item.get("accepted_at")
                    or ""
                ),
                str(
                    item.get("accession")
                    or ""
                ),
            )
        )

        effective_base: str | None = None

        for row in group_rows:
            metadata = row["metadata"]
            accession = str(
                row["accession"]
            )

            amendment_type = str(
                metadata.get("amendment_type")
                or ""
            ).upper()

            is_amendment = bool(
                metadata.get("is_amendment")
            )

            if not is_amendment:
                effective_base = accession
                continue

            metadata["amends_accession"] = (
                effective_base
            )

            if amendment_type == "RESTATEMENT":
                # Restatement wordt de nieuwe volledige basis.
                effective_base = accession

            elif amendment_type == "NEW HOLDINGS":
                # New Holdings is een aanvulling; basis blijft staan.
                pass

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    backup_path = manifest_path.with_name(
        f"{manifest_path.name}."
        f"backup-normalize-{timestamp}"
    )

    backup_path.write_bytes(
        manifest_path.read_bytes()
    )

    normalized.sort(
        key=lambda item: (
            str(
                (item.get("metadata") or {}).get(
                    "manager_key"
                )
            ),
            str(item.get("report_period")),
            str(item.get("accepted_at")),
        )
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for row in normalized:
            handle.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )

    multiplier_counts: dict[str, int] = (
        defaultdict(int)
    )

    for row in normalized:
        multiplier = float(
            row["metadata"][
                "thirteen_f_value_multiplier"
            ]
        )

        multiplier_counts[
            f"{multiplier:g}"
        ] += 1

    print()
    print(
        json.dumps(
            {
                "status": "PASS",
                "filings_normalized": len(
                    normalized
                ),
                "multiplier_counts": dict(
                    multiplier_counts
                ),
                "backup": str(
                    backup_path
                ),
                "manifest": str(
                    manifest_path
                ),
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
