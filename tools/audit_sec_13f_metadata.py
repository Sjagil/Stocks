from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


SEC_ARCHIVES_BASE = (
    "https://www.sec.gov/Archives/edgar/data"
)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(
        ":",
        1,
    )[-1].lower()


def first_text(
    root: ET.Element,
    *names: str,
) -> str | None:
    wanted = {
        name.lower()
        for name in names
    }

    for node in root.iter():
        if local_name(node.tag) not in wanted:
            continue

        text = (
            node.text or ""
        ).strip()

        if text:
            return text

    return None


def to_float(
    value: Any,
) -> float | None:
    text = str(
        value or ""
    ).strip().replace(
        ",",
        "",
    )

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def to_bool(
    value: Any,
) -> bool:
    return str(
        value or ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "x",
    }


def load_builder(
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        "sec13f_builder",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "13F-buildermodule kon niet worden geladen."
        )

    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(module)

    return module


def raw_information_table_total(
    path: Path,
) -> tuple[float, int]:
    root = ET.fromstring(
        path.read_bytes()
    )

    total = 0.0
    count = 0

    for info in root.iter():
        if local_name(info.tag) != "infotable":
            continue

        value = None

        for child in info.iter():
            if local_name(child.tag) == "value":
                value = to_float(child.text)
                break

        if value is not None:
            total += value

        count += 1

    return total, count


def primary_document_metadata(
    client,
    manager_cik: str,
    accession: str,
) -> dict[str, Any]:
    cik_path = str(
        int(manager_cik)
    )

    accession_path = accession.replace(
        "-",
        "",
    )

    base_url = (
        f"{SEC_ARCHIVES_BASE}/"
        f"{cik_path}/"
        f"{accession_path}"
    )

    index_payload = client.get_json(
        f"{base_url}/index.json"
    )

    items = (
        index_payload.get("directory", {})
        .get("item", [])
    )

    candidates = [
        str(item.get("name") or "")
        for item in items
        if isinstance(item, dict)
        and str(
            item.get("name") or ""
        ).lower().endswith(".xml")
    ]

    for filename in candidates:
        url = (
            f"{base_url}/"
            f"{urllib.parse.quote(filename)}"
        )

        content = client.get_bytes(url)

        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            continue

        summary_total = to_float(
            first_text(
                root,
                "tableValueTotal",
            )
        )

        if summary_total is None:
            continue

        amendment_type = first_text(
            root,
            "amendmentType",
        )

        amendment_number = first_text(
            root,
            "amendmentNumber",
        )

        report_type = first_text(
            root,
            "reportType",
        )

        is_amendment_text = first_text(
            root,
            "isAmendment",
        )

        return {
            "primary_document": filename,
            "primary_document_url": url,
            "summary_table_value_total": (
                summary_total
            ),
            "is_amendment": to_bool(
                is_amendment_text
            ),
            "amendment_type": (
                amendment_type
            ),
            "amendment_number": (
                amendment_number
            ),
            "report_type": report_type,
        }

    raise RuntimeError(
        "Geen primary document met "
        f"tableValueTotal gevonden voor {accession}"
    )


def infer_multiplier(
    raw_total: float,
    summary_total: float,
) -> tuple[float, float]:
    if raw_total <= 0 or summary_total <= 0:
        return 1.0, 1.0

    candidates = []

    for multiplier in (
        1.0,
        1_000.0,
    ):
        calculated = (
            raw_total * multiplier
        )

        relative_error = abs(
            calculated - summary_total
        ) / summary_total

        candidates.append(
            (
                relative_error,
                multiplier,
            )
        )

    relative_error, multiplier = min(
        candidates,
        key=lambda item: item[0],
    )

    return multiplier, relative_error


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
        "--builder",
        default=(
            "./tools/build_sec_13f_manager_input.py"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "./results/sec_intelligence/"
            "13f_metadata_audit.csv"
        ),
    )

    args = parser.parse_args()

    user_agent = str(
        os.getenv("SEC_USER_AGENT") or ""
    ).strip()

    if not user_agent:
        raise SystemExit(
            "SEC_USER_AGENT ontbreekt."
        )

    builder = load_builder(
        Path(args.builder)
    )

    client = builder.HttpClient(
        user_agent=user_agent,
        delay_seconds=0.18,
    )

    manifest_path = Path(
        args.manifest
    )
    content_root = Path(
        args.content_root
    )
    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for raw_line in manifest_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if not raw_line.strip():
            continue

        manifest_row = json.loads(
            raw_line
        )

        accession = str(
            manifest_row.get("accession")
            or ""
        )

        manager_cik = str(
            manifest_row.get("filer_cik")
            or manifest_row.get("issuer_cik")
            or ""
        )

        metadata = (
            manifest_row.get("metadata")
            or {}
        )

        manager_key = str(
            metadata.get("manager_key")
            or manager_cik
        )

        content_path = (
            content_root
            / str(
                manifest_row["content_path"]
            )
        )

        try:
            raw_total, position_count = (
                raw_information_table_total(
                    content_path
                )
            )

            primary = (
                primary_document_metadata(
                    client,
                    manager_cik,
                    accession,
                )
            )

            summary_total = float(
                primary[
                    "summary_table_value_total"
                ]
            )

            multiplier, relative_error = (
                infer_multiplier(
                    raw_total,
                    summary_total,
                )
            )

            current_multiplier = float(
                metadata.get(
                    "thirteen_f_value_multiplier",
                    1.0,
                )
            )

            row = {
                "manager_key": manager_key,
                "accession": accession,
                "form_type": manifest_row.get(
                    "form_type"
                ),
                "report_period": (
                    manifest_row.get(
                        "report_period"
                    )
                ),
                "position_count": (
                    position_count
                ),
                "raw_value_total": (
                    raw_total
                ),
                "summary_value_total": (
                    summary_total
                ),
                "inferred_multiplier": (
                    multiplier
                ),
                "current_manifest_multiplier": (
                    current_multiplier
                ),
                "relative_error": (
                    relative_error
                ),
                "multiplier_change_needed": (
                    multiplier
                    != current_multiplier
                ),
                "is_amendment": primary[
                    "is_amendment"
                ],
                "amendment_type": primary[
                    "amendment_type"
                ],
                "amendment_number": primary[
                    "amendment_number"
                ],
                "report_type": primary[
                    "report_type"
                ],
                "primary_document": primary[
                    "primary_document"
                ],
                "error": "",
            }

        except Exception as exc:
            row = {
                "manager_key": manager_key,
                "accession": accession,
                "form_type": manifest_row.get(
                    "form_type"
                ),
                "report_period": (
                    manifest_row.get(
                        "report_period"
                    )
                ),
                "position_count": "",
                "raw_value_total": "",
                "summary_value_total": "",
                "inferred_multiplier": "",
                "current_manifest_multiplier": "",
                "relative_error": "",
                "multiplier_change_needed": "",
                "is_amendment": "",
                "amendment_type": "",
                "amendment_number": "",
                "report_type": "",
                "primary_document": "",
                "error": repr(exc),
            }

        rows.append(row)

        print(
            f"{manager_key:12} "
            f"{accession} "
            f"period={row['report_period']} "
            f"multiplier={row['inferred_multiplier']} "
            f"amendment={row['is_amendment']} "
            f"type={row['amendment_type']} "
            f"error={row['error']}"
        )

    fieldnames = [
        "manager_key",
        "accession",
        "form_type",
        "report_period",
        "position_count",
        "raw_value_total",
        "summary_value_total",
        "inferred_multiplier",
        "current_manifest_multiplier",
        "relative_error",
        "multiplier_change_needed",
        "is_amendment",
        "amendment_type",
        "amendment_number",
        "report_type",
        "primary_document",
        "error",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    errors = sum(
        1
        for row in rows
        if row["error"]
    )

    multiplier_changes = sum(
        1
        for row in rows
        if row[
            "multiplier_change_needed"
        ] is True
    )

    amendments = sum(
        1
        for row in rows
        if row["is_amendment"] is True
    )

    print()
    print(
        json.dumps(
            {
                "status": (
                    "PASS"
                    if errors == 0
                    else "DEGRADED"
                ),
                "filings_checked": len(rows),
                "errors": errors,
                "multiplier_changes_needed": (
                    multiplier_changes
                ),
                "amendments_detected": (
                    amendments
                ),
                "output": str(
                    output_path
                ),
            },
            indent=2,
        )
    )

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
