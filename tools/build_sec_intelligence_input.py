#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SUPPORTED_FORMS = {
    "3",
    "3/A",
    "4",
    "4/A",
    "5",
    "5/A",
    "144",
    "144/A",
    "8-K",
    "8-K/A",
    "SC 13D",
    "SC 13D/A",
    "SC 13G",
    "SC 13G/A",
    "SCHEDULE 13D",
    "SCHEDULE 13D/A",
    "SCHEDULE 13G",
    "SCHEDULE 13G/A",
}

DATA_SEC_ROOT = "https://data.sec.gov/submissions"
ARCHIVE_ROOT = "https://www.sec.gov/Archives/edgar/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a real SEC filing JSONL manifest and download filing content "
            "for sec_ownership_and_event_intelligence_v1.py."
        )
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--cik")
    parser.add_argument("--ticker-map")
    parser.add_argument("--submissions-json")
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", default=date.today().isoformat())
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"))
    parser.add_argument("--request-delay", type=float, default=0.20)
    return parser.parse_args()


def http_get(url: str, user_agent: str, retries: int = 4) -> bytes:
    headers = {
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Host": re.sub(r"^https?://([^/]+).*$", r"\1", url),
    }

    for attempt in range(1, retries + 1):
        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == retries:
                raise

            wait = max(2.0, float(attempt * 3))
            print(
                f"HTTP {exc.code} voor {url}; opnieuw over {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
        except URLError:
            if attempt == retries:
                raise
            time.sleep(float(attempt * 3))

    raise RuntimeError(f"Download mislukt: {url}")


def load_json_url(url: str, user_agent: str) -> dict[str, Any]:
    payload = http_get(url, user_agent)
    result = json.loads(payload.decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Verwacht JSON-object van {url}")
    return result


def resolve_cik(
    symbol: str,
    cik_argument: str | None,
    ticker_map_path: str | None,
) -> str:
    if cik_argument:
        digits = re.sub(r"\D", "", cik_argument)
        if not digits:
            raise ValueError("--cik bevat geen cijfers")
        return digits.zfill(10)

    if not ticker_map_path:
        raise ValueError(
            "Gebruik --cik of --ticker-map om de CIK voor het symbool te bepalen"
        )

    path = Path(ticker_map_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    records: list[dict[str, Any]]
    if isinstance(data, dict):
        records = [
            value
            for value in data.values()
            if isinstance(value, dict)
        ]
    elif isinstance(data, list):
        records = [
            value
            for value in data
            if isinstance(value, dict)
        ]
    else:
        raise ValueError("Onverwacht ticker-mapformaat")

    wanted = symbol.upper()

    for record in records:
        ticker = str(
            record.get("ticker")
            or record.get("symbol")
            or ""
        ).upper()

        if ticker != wanted:
            continue

        cik = (
            record.get("cik_str")
            or record.get("cik")
            or record.get("issuer_cik")
        )

        if cik is None:
            continue

        digits = re.sub(r"\D", "", str(cik))
        if digits:
            return digits.zfill(10)

    raise ValueError(f"Geen CIK gevonden voor {symbol}")


def normalize_columnar_records(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(payload.get("filings"), dict):
        recent = payload["filings"].get("recent")
        if isinstance(recent, dict):
            payload = recent

    accession_numbers = payload.get("accessionNumber")

    if not isinstance(accession_numbers, list):
        return []

    records: list[dict[str, Any]] = []

    for index in range(len(accession_numbers)):
        record: dict[str, Any] = {}

        for key, values in payload.items():
            if isinstance(values, list):
                record[key] = values[index] if index < len(values) else None

        records.append(record)

    return records


def supplemental_files(
    submissions: dict[str, Any],
    since: str,
    until: str,
) -> list[str]:
    filings = submissions.get("filings")
    if not isinstance(filings, dict):
        return []

    files = filings.get("files")
    if not isinstance(files, list):
        return []

    selected: list[str] = []

    for item in files:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or "").strip()
        filing_from = str(item.get("filingFrom") or "")
        filing_to = str(item.get("filingTo") or "")

        if not name:
            continue

        if filing_to and filing_to < since:
            continue

        if filing_from and filing_from > until:
            continue

        selected.append(name)

    return selected


def split_items(value: Any) -> list[str]:
    if value is None:
        return []

    return sorted(
        set(
            re.findall(
                r"\b\d\.\d{2}\b",
                str(value),
            )
        )
    )


def main() -> int:
    args = parse_args()

    if not args.user_agent:
        raise SystemExit(
            "SEC_USER_AGENT ontbreekt. Gebruik een echte naam en e-mailadres."
        )

    symbol = args.symbol.upper()
    since = date.fromisoformat(args.since).isoformat()
    until = date.fromisoformat(args.until).isoformat()

    output_root = Path(args.output_root).resolve()
    raw_root = output_root / "raw" / symbol
    cache_root = output_root / "submissions-cache"
    manifest_path = output_root / f"{symbol}_filings.jsonl"
    error_path = output_root / f"{symbol}_download_errors.json"

    raw_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    cik = resolve_cik(
        symbol=symbol,
        cik_argument=args.cik,
        ticker_map_path=args.ticker_map,
    )

    if args.submissions_json:
        submissions_path = Path(args.submissions_json).resolve()
        submissions = json.loads(
            submissions_path.read_text(encoding="utf-8")
        )
    else:
        submissions_url = f"{DATA_SEC_ROOT}/CIK{cik}.json"
        print(f"Submissions downloaden: {submissions_url}")
        submissions = load_json_url(
            submissions_url,
            args.user_agent,
        )

        submissions_path = cache_root / f"CIK{cik}.json"
        submissions_path.write_text(
            json.dumps(submissions, indent=2),
            encoding="utf-8",
        )

        time.sleep(args.request_delay)

    filing_records = normalize_columnar_records(submissions)

    for filename in supplemental_files(submissions, since, until):
        url = f"{DATA_SEC_ROOT}/{filename}"
        print(f"Historische submissions downloaden: {url}")

        supplemental = load_json_url(url, args.user_agent)
        filing_records.extend(
            normalize_columnar_records(supplemental)
        )

        cache_path = cache_root / filename
        cache_path.write_text(
            json.dumps(supplemental, indent=2),
            encoding="utf-8",
        )

        time.sleep(args.request_delay)

    by_accession: dict[str, dict[str, Any]] = {}

    for record in filing_records:
        accession = str(record.get("accessionNumber") or "").strip()
        form = str(record.get("form") or "").strip().upper()
        filing_date = str(record.get("filingDate") or "").strip()

        if not accession:
            continue

        if form not in SUPPORTED_FORMS:
            continue

        if not filing_date or not since <= filing_date <= until:
            continue

        by_accession[accession] = record

    selected = sorted(
        by_accession.values(),
        key=lambda item: (
            str(item.get("filingDate") or ""),
            str(item.get("acceptanceDateTime") or ""),
            str(item.get("accessionNumber") or ""),
        ),
    )

    manifest_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for number, record in enumerate(selected, start=1):
        accession = str(record["accessionNumber"]).strip()
        form = str(record["form"]).strip().upper()
        accepted_at = str(
            record.get("acceptanceDateTime")
            or ""
        ).strip()

        if not accepted_at:
            errors.append(
                {
                    "accession": accession,
                    "form": form,
                    "error": "acceptanceDateTime ontbreekt",
                }
            )
            continue

        accession_compact = accession.replace("-", "")
        archive_cik = str(int(cik))

        url = (
            f"{ARCHIVE_ROOT}/{archive_cik}/"
            f"{accession_compact}/{accession}.txt"
        )

        target = raw_root / f"{accession}.txt"

        print(
            f"[{number}/{len(selected)}] "
            f"{symbol} {form} {accession}"
        )

        if not target.exists() or target.stat().st_size == 0:
            try:
                content = http_get(url, args.user_agent)
                target.write_bytes(content)
                time.sleep(args.request_delay)
            except Exception as exc:
                errors.append(
                    {
                        "accession": accession,
                        "form": form,
                        "url": url,
                        "error": str(exc),
                    }
                )
                continue

        relative_path = target.relative_to(output_root).as_posix()
        items = split_items(record.get("items"))

        manifest_records.append(
            {
                "accession": accession,
                "form_type": form,
                "accepted_at": accepted_at,
                "accepted_timezone": "America/New_York",
                "symbol": symbol,
                "issuer_cik": cik,
                "filer_cik": accession.split("-", 1)[0],
                "filing_date": record.get("filingDate"),
                "report_period": record.get("reportDate") or None,
                "source_path": relative_path,
                "content_path": relative_path,
                "metadata": {
                    "items": items,
                    "primary_document": record.get("primaryDocument"),
                    "primary_document_description": record.get(
                        "primaryDocDescription"
                    ),
                    "archive_url": url,
                    "source": "SEC_EDGAR",
                },
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in manifest_records:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    error_path.write_text(
        json.dumps(errors, indent=2),
        encoding="utf-8",
    )

    summary = {
        "symbol": symbol,
        "cik": cik,
        "since": since,
        "until": until,
        "candidate_filings": len(selected),
        "manifest_records": len(manifest_records),
        "download_failures": len(errors),
        "manifest": str(manifest_path),
        "raw_root": str(raw_root),
        "errors": str(error_path),
        "excluded_13f": True,
        "reason_13f_excluded": (
            "Issuer submissions metadata does not provide actual "
            "institutional position tables. 13F requires manager filings "
            "and information-table XML."
        ),
    }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
