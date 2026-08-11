from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEC_SUBMISSIONS_BASE = (
    "https://data.sec.gov/submissions"
)
SEC_ARCHIVES_BASE = (
    "https://www.sec.gov/Archives/edgar/data"
)
EODHD_MAPPING_URL = (
    "https://eodhd.com/api/id-mapping"
)

INFO_TABLE_PATTERN = re.compile(
    rb"<(?:[A-Za-z0-9_]+:)?infoTable\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat().replace(
        "+00:00",
        "Z",
    )


def clean_cik(value: Any) -> str:
    digits = re.sub(
        r"\D",
        "",
        str(value or ""),
    )
    return digits.zfill(10)


def cik_path(value: Any) -> str:
    return str(
        int(clean_cik(value))
    )


def normalize_datetime(value: Any) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    if re.fullmatch(r"\d{14}", text):
        parsed = datetime.strptime(
            text,
            "%Y%m%d%H%M%S",
        ).replace(
            tzinfo=timezone.utc
        )
        return parsed.isoformat().replace(
            "+00:00",
            "Z",
        )

    if text.endswith("Z"):
        return text

    if re.search(
        r"[+-]\d{2}:\d{2}$",
        text,
    ):
        return text

    return text + "Z"


class HttpClient:
    def __init__(
        self,
        user_agent: str,
        delay_seconds: float = 0.18,
    ) -> None:
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.last_request_at = 0.0

    def get_bytes(
        self,
        url: str,
        *,
        attempts: int = 5,
    ) -> bytes:
        last_error: Exception | None = None

        for attempt in range(attempts):
            elapsed = (
                time.monotonic()
                - self.last_request_at
            )

            if elapsed < self.delay_seconds:
                time.sleep(
                    self.delay_seconds - elapsed
                )

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                    "Accept": (
                        "application/json,"
                        "application/xml,"
                        "text/xml,"
                        "text/plain,*/*"
                    ),
                },
            )

            try:
                self.last_request_at = (
                    time.monotonic()
                )

                with urllib.request.urlopen(
                    request,
                    timeout=60,
                ) as response:
                    content = response.read()

                    content_encoding = str(
                        response.headers.get(
                            "Content-Encoding",
                            "",
                        )
                        or ""
                    ).lower()

                    # urllib.request decodeert gzip en
                    # deflate niet automatisch.
                    if content.startswith(b"\x1f\x8b"):
                        content = gzip.decompress(content)

                    elif "deflate" in content_encoding:
                        try:
                            content = zlib.decompress(content)
                        except zlib.error:
                            content = zlib.decompress(
                                content,
                                -zlib.MAX_WBITS,
                            )

                    return content

            except urllib.error.HTTPError as exc:
                last_error = exc

                if exc.code not in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    body = exc.read().decode(
                        "utf-8",
                        errors="replace",
                    )
                    raise RuntimeError(
                        f"HTTP {exc.code} voor {url}: "
                        f"{body[:500]}"
                    ) from exc

            except urllib.error.URLError as exc:
                last_error = exc

            time.sleep(
                min(
                    12.0,
                    1.5 * (2**attempt),
                )
            )

        raise RuntimeError(
            f"Download mislukt na {attempts} pogingen: "
            f"{url}; laatste fout={last_error}"
        )

    def get_json(
        self,
        url: str,
    ) -> dict[str, Any]:
        payload = self.get_bytes(url)

        parsed = json.loads(
            payload.decode("utf-8")
        )

        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"JSON-response is geen object: {url}"
            )

        return parsed


def columnar_rows(
    columns: dict[str, Any],
) -> list[dict[str, Any]]:
    lengths = [
        len(value)
        for value in columns.values()
        if isinstance(value, list)
    ]

    if not lengths:
        return []

    row_count = max(lengths)
    rows: list[dict[str, Any]] = []

    for index in range(row_count):
        row: dict[str, Any] = {}

        for key, values in columns.items():
            if isinstance(values, list):
                row[key] = (
                    values[index]
                    if index < len(values)
                    else None
                )

        rows.append(row)

    return rows


def load_manager_filings(
    client: HttpClient,
    cik: str,
    max_filings: int,
) -> list[dict[str, Any]]:
    submissions_url = (
        f"{SEC_SUBMISSIONS_BASE}/"
        f"CIK{clean_cik(cik)}.json"
    )

    submissions = client.get_json(
        submissions_url
    )

    filings = submissions.get("filings") or {}
    recent = filings.get("recent") or {}

    rows = columnar_rows(recent)

    selected = [
        row
        for row in rows
        if str(
            row.get("form") or ""
        ).upper()
        in {
            "13F-HR",
            "13F-HR/A",
        }
    ]

    selected.sort(
        key=lambda row: (
            str(
                row.get("reportDate") or ""
            ),
            str(
                row.get("acceptanceDateTime")
                or row.get("filingDate")
                or ""
            ),
        ),
        reverse=True,
    )

    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in selected:
        accession = str(
            row.get("accessionNumber") or ""
        ).strip()

        if not accession or accession in seen:
            continue

        seen.add(accession)
        result.append(row)

        if len(result) >= max_filings:
            break

    return result


def document_candidates(
    index_payload: dict[str, Any],
) -> list[str]:
    directory = (
        index_payload.get("directory") or {}
    )
    items = directory.get("item") or []

    names = [
        str(item.get("name") or "")
        for item in items
        if isinstance(item, dict)
    ]

    xml_names = [
        name
        for name in names
        if name.lower().endswith(".xml")
        and not name.lower().endswith(
            (
                "_cal.xml",
                "_def.xml",
                "_lab.xml",
                "_pre.xml",
            )
        )
        and name.lower()
        not in {
            "filingsummary.xml",
            "metalinks.json",
        }
    ]

    def priority(name: str) -> tuple[int, str]:
        lowered = name.lower()

        preferred = any(
            token in lowered
            for token in (
                "infotable",
                "informationtable",
                "form13f",
                "13f",
            )
        )

        return (
            0 if preferred else 1,
            lowered,
        )

    return sorted(
        xml_names,
        key=priority,
    )


def extract_xml_from_submission(
    content: bytes,
) -> bytes | None:
    blocks = re.findall(
        rb"<DOCUMENT>(.*?)</DOCUMENT>",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    for block in blocks:
        if not INFO_TABLE_PATTERN.search(block):
            continue

        text_match = re.search(
            rb"<TEXT>(.*?)</TEXT>",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )

        candidate = (
            text_match.group(1)
            if text_match
            else block
        )

        xml_start = candidate.find(b"<?xml")

        if xml_start >= 0:
            candidate = candidate[xml_start:]

        return candidate.strip()

    return None


def download_information_table(
    client: HttpClient,
    cik: str,
    accession: str,
) -> tuple[bytes, str]:
    accession_without_dashes = (
        accession.replace("-", "")
    )

    base_url = (
        f"{SEC_ARCHIVES_BASE}/"
        f"{cik_path(cik)}/"
        f"{accession_without_dashes}"
    )

    index_url = f"{base_url}/index.json"
    index_payload = client.get_json(index_url)

    for filename in document_candidates(
        index_payload
    ):
        document_url = (
            f"{base_url}/"
            f"{urllib.parse.quote(filename)}"
        )

        content = client.get_bytes(
            document_url
        )

        if INFO_TABLE_PATTERN.search(content):
            return content, document_url

    submission_url = (
        f"{base_url}/{accession}.txt"
    )
    submission = client.get_bytes(
        submission_url
    )

    extracted = extract_xml_from_submission(
        submission
    )

    if extracted is None:
        raise RuntimeError(
            "Geen information-table XML gevonden voor "
            f"{accession}"
        )

    return extracted, submission_url


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(
        ":",
        1,
    )[-1]


def extract_positions(
    content: bytes,
) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise RuntimeError(
            f"Information-table XML is ongeldig: {exc}"
        ) from exc

    rows: list[dict[str, str]] = []

    for node in root.iter():
        if local_name(node.tag).lower() != "infotable":
            continue

        row: dict[str, str] = {}

        for child in node.iter():
            name = local_name(
                child.tag
            ).lower()

            text = (
                child.text or ""
            ).strip()

            if not text:
                continue

            if name == "cusip":
                row["cusip"] = text.upper()
            elif name == "nameofissuer":
                row["issuer_name"] = text
            elif name == "titleofclass":
                row["title_of_class"] = text
            elif name == "figi":
                row["figi"] = text.upper()
            elif name == "putcall":
                row["put_call"] = text.upper()

        if row.get("cusip"):
            rows.append(row)

    return rows


def load_mapping_cache(
    path: Path,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        return {}

    return {
        str(key).upper(): value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def save_mapping_cache(
    path: Path,
    cache: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            cache,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def resolve_cusip(
    client: HttpClient,
    token: str,
    cusip: str,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "filter[cusip]": cusip,
            "page[limit]": 10,
            "page[offset]": 0,
            "api_token": token,
            "fmt": "json",
        }
    )

    payload = client.get_json(
        f"{EODHD_MAPPING_URL}?{query}"
    )

    data = payload.get("data") or []

    if not isinstance(data, list):
        data = []

    candidates = [
        item
        for item in data
        if isinstance(item, dict)
    ]

    if not candidates:
        return {
            "cusip": cusip,
            "resolved": False,
            "resolved_at": utc_now(),
        }

    candidates.sort(
        key=lambda item: (
            0
            if str(
                item.get("symbol") or ""
            ).upper().endswith(".US")
            else 1,
            str(
                item.get("symbol") or ""
            ),
        )
    )

    selected = candidates[0]
    eodhd_symbol = str(
        selected.get("symbol") or ""
    ).strip().upper()

    if eodhd_symbol.endswith(".US"):
        symbol = eodhd_symbol[:-3]
    else:
        symbol = eodhd_symbol

    return {
        "cusip": cusip,
        "resolved": bool(symbol),
        "symbol": symbol or None,
        "eodhd_symbol": eodhd_symbol or None,
        "cik": selected.get("cik"),
        "figi": selected.get("figi"),
        "isin": selected.get("isin"),
        "lei": selected.get("lei"),
        "resolved_at": utc_now(),
    }


def write_security_map(
    path: Path,
    cache: dict[str, dict[str, Any]],
) -> int:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved = [
        value
        for value in cache.values()
        if value.get("resolved")
        and value.get("symbol")
    ]

    resolved.sort(
        key=lambda item: (
            str(item.get("symbol") or ""),
            str(item.get("cusip") or ""),
        )
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "cusip",
                "cik",
                "figi",
                "isin",
                "eodhd_symbol",
                "source",
                "resolved_at",
            ],
        )

        writer.writeheader()

        for item in resolved:
            writer.writerow(
                {
                    "symbol": item.get("symbol"),
                    "cusip": item.get("cusip"),
                    "cik": item.get("cik"),
                    "figi": item.get("figi"),
                    "isin": item.get("isin"),
                    "eodhd_symbol": item.get(
                        "eodhd_symbol"
                    ),
                    "source": "EODHD_ID_MAPPING",
                    "resolved_at": item.get(
                        "resolved_at"
                    ),
                }
            )

    return len(resolved)


def write_unresolved(
    path: Path,
    issuer_by_cusip: dict[str, set[str]],
    cache: dict[str, dict[str, Any]],
) -> int:
    unresolved = [
        cusip
        for cusip in sorted(issuer_by_cusip)
        if not cache.get(
            cusip,
            {},
        ).get("resolved")
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cusip",
                "issuer_names",
            ],
        )

        writer.writeheader()

        for cusip in unresolved:
            writer.writerow(
                {
                    "cusip": cusip,
                    "issuer_names": " | ".join(
                        sorted(
                            issuer_by_cusip.get(
                                cusip,
                                set(),
                            )
                        )
                    ),
                }
            )

    return len(unresolved)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--registry",
        default=(
            "./config/sec_13f_managers.json"
        ),
    )
    parser.add_argument(
        "--root",
        default=(
            "./data/sec_intelligence/13f"
        ),
    )
    parser.add_argument(
        "--max-filings",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--refresh-mappings",
        action="store_true",
    )

    args = parser.parse_args()

    user_agent = str(
        os.getenv("SEC_USER_AGENT") or ""
    ).strip()

    eodhd_token = str(
        os.getenv("EODHD_API_KEY")
        or os.getenv("EOD_API_KEY")
        or os.getenv(
            "EODHISTORICALDATA_API_KEY"
        )
        or ""
    ).strip()

    if not user_agent:
        raise SystemExit(
            "SEC_USER_AGENT ontbreekt. "
            "Zet naam/project plus contactadres in .env."
        )

    if not eodhd_token:
        raise SystemExit(
            "Geen EODHD API-key in het proces."
        )

    registry_path = Path(args.registry)
    root = Path(args.root)

    raw_root = root / "raw"
    manifest_dir = root / "manifests"
    resolver_dir = root / "resolver"
    cache_dir = root / "cache"
    errors_dir = root / "errors"

    for directory in (
        raw_root,
        manifest_dir,
        resolver_dir,
        cache_dir,
        errors_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    registry = json.loads(
        registry_path.read_text(
            encoding="utf-8-sig"
        )
    )

    managers = [
        manager
        for manager in (
            registry.get("managers") or []
        )
        if manager.get("enabled", True)
    ]

    sec_client = HttpClient(
        user_agent=user_agent,
        delay_seconds=0.18,
    )
    eod_client = HttpClient(
        user_agent=user_agent,
        delay_seconds=0.10,
    )

    downloaded: list[dict[str, Any]] = []
    issuer_by_cusip: dict[
        str,
        set[str],
    ] = {}
    errors: list[dict[str, Any]] = []

    for manager in managers:
        manager_key = str(
            manager["manager_key"]
        ).strip().upper()

        manager_cik = clean_cik(
            manager["cik"]
        )

        print(
            f"\n[{manager_key}] filings ophalen..."
        )

        try:
            filings = load_manager_filings(
                sec_client,
                manager_cik,
                args.max_filings,
            )
        except Exception as exc:
            errors.append(
                {
                    "manager_key": manager_key,
                    "stage": "submissions",
                    "error": repr(exc),
                }
            )
            print(
                f"  ERROR submissions: {exc}"
            )
            continue

        print(
            f"  geselecteerde filings: "
            f"{len(filings)}"
        )

        manager_raw = (
            raw_root / manager_key
        )
        manager_raw.mkdir(
            parents=True,
            exist_ok=True,
        )

        for filing in filings:
            accession = str(
                filing.get(
                    "accessionNumber"
                )
                or ""
            ).strip()

            report_period = str(
                filing.get("reportDate")
                or ""
            ).strip()

            form_type = str(
                filing.get("form")
                or ""
            ).strip()

            accepted_at = normalize_datetime(
                filing.get(
                    "acceptanceDateTime"
                )
                or filing.get("filingDate")
            )

            safe_period = (
                report_period
                or "unknown-period"
            )

            filename = (
                f"{safe_period}_"
                f"{accession.replace('-', '')}_"
                "infotable.xml"
            )

            output_path = (
                manager_raw / filename
            )

            try:
                if output_path.exists():
                    content = output_path.read_bytes()
                    source_url = "LOCAL_CACHE"
                else:
                    content, source_url = (
                        download_information_table(
                            sec_client,
                            manager_cik,
                            accession,
                        )
                    )

                    output_path.write_bytes(
                        content
                    )

                positions = extract_positions(
                    content
                )

                if not positions:
                    raise RuntimeError(
                        "XML bevat 0 infoTable-posities"
                    )

                for position in positions:
                    cusip = position["cusip"]

                    issuer_by_cusip.setdefault(
                        cusip,
                        set(),
                    ).add(
                        position.get(
                            "issuer_name",
                            "",
                        )
                    )

                downloaded.append(
                    {
                        "manager": manager,
                        "accession": accession,
                        "form_type": form_type,
                        "accepted_at": accepted_at,
                        "report_period": report_period,
                        "content_path": (
                            output_path.relative_to(
                                root
                            ).as_posix()
                        ),
                        "source_url": source_url,
                        "position_count": len(
                            positions
                        ),
                        "cusips": sorted(
                            {
                                item["cusip"]
                                for item in positions
                            }
                        ),
                    }
                )

                print(
                    f"  {accession}: "
                    f"{len(positions)} posities"
                )

            except Exception as exc:
                errors.append(
                    {
                        "manager_key": manager_key,
                        "accession": accession,
                        "stage": "filing_download",
                        "error": repr(exc),
                    }
                )

                print(
                    f"  ERROR {accession}: {exc}"
                )

    cache_path = (
        cache_dir / "eodhd_cusip_cache.json"
    )
    mapping_cache = load_mapping_cache(
        cache_path
    )

    all_cusips = sorted(
        issuer_by_cusip
    )

    print(
        f"\nUnieke CUSIP's: {len(all_cusips)}"
    )

    for index, cusip in enumerate(
        all_cusips,
        start=1,
    ):
        existing = mapping_cache.get(
            cusip
        )

        if (
            existing is not None
            and not args.refresh_mappings
        ):
            continue

        try:
            mapping_cache[cusip] = (
                resolve_cusip(
                    eod_client,
                    eodhd_token,
                    cusip,
                )
            )

        except Exception as exc:
            mapping_cache[cusip] = {
                "cusip": cusip,
                "resolved": False,
                "error": repr(exc),
                "resolved_at": utc_now(),
            }

        if index % 25 == 0:
            save_mapping_cache(
                cache_path,
                mapping_cache,
            )
            print(
                f"  mappings verwerkt: "
                f"{index}/{len(all_cusips)}"
            )

    save_mapping_cache(
        cache_path,
        mapping_cache,
    )

    security_map_path = (
        resolver_dir / "security_map.csv"
    )
    resolved_count = write_security_map(
        security_map_path,
        mapping_cache,
    )

    unresolved_path = (
        resolver_dir / "unresolved_cusips.csv"
    )
    unresolved_count = write_unresolved(
        unresolved_path,
        issuer_by_cusip,
        mapping_cache,
    )

    manifest_path = (
        manifest_dir
        / "manager_13f_filings.jsonl"
    )

    downloaded.sort(
        key=lambda item: (
            str(
                item["manager"][
                    "manager_key"
                ]
            ),
            str(item["report_period"]),
            str(item["accepted_at"]),
        )
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for item in downloaded:
            manager = item["manager"]

            cusip_to_symbol = {
                cusip: mapping_cache[
                    cusip
                ]["symbol"]
                for cusip in item["cusips"]
                if mapping_cache.get(
                    cusip,
                    {},
                ).get("resolved")
                and mapping_cache[cusip].get(
                    "symbol"
                )
            }

            row = {
                "accepted_at": item[
                    "accepted_at"
                ],
                "accession": item[
                    "accession"
                ],
                "content_path": item[
                    "content_path"
                ],
                "filer_cik": clean_cik(
                    manager["cik"]
                ),
                "issuer_cik": clean_cik(
                    manager["cik"]
                ),
                "form_type": item[
                    "form_type"
                ],
                "report_period": item[
                    "report_period"
                ],
                "symbol": None,
                "metadata": {
                    "manager_key": manager[
                        "manager_key"
                    ],
                    "manager_name": manager[
                        "manager_name"
                    ],
                    "associated_person": manager[
                        "associated_person"
                    ],
                    "authority": manager[
                        "authority"
                    ],
                    "delayed_context_only": True,
                    "source_url": item[
                        "source_url"
                    ],
                    "position_count": item[
                        "position_count"
                    ],
                    "cusip_to_symbol": (
                        cusip_to_symbol
                    ),
                },
            }

            handle.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )

    errors_path = (
        errors_dir / "download_errors.json"
    )
    errors_path.write_text(
        json.dumps(
            errors,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "status": (
            "PASS"
            if downloaded
            else "FAIL"
        ),
        "managers_enabled": len(managers),
        "filings_downloaded": len(
            downloaded
        ),
        "unique_cusips": len(
            all_cusips
        ),
        "resolved_cusips": (
            resolved_count
        ),
        "unresolved_cusips": (
            unresolved_count
        ),
        "errors": len(errors),
        "manifest": str(
            manifest_path
        ),
        "security_map": str(
            security_map_path
        ),
    }

    print(
        "\n"
        + json.dumps(
            summary,
            indent=2,
        )
    )

    return (
        0
        if downloaded
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
