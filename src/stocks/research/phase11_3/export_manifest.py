from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KNOWN_SCHEMAS = {"stocks_phase11_3_export_v1"}
ALLOWED_PIT = {"PIT_ELIGIBLE_NEXT_SESSION", "PIT_ELIGIBLE_DATE", "PIT_ELIGIBLE_ACCEPTED_AT", "PIT_PARTIAL", "FORWARD_ONLY"}
BLOCKED_LICENSES = {"LICENSE_BLOCKED", "REDISTRIBUTION_PROHIBITED"}


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class ManifestValidation:
    status: str
    valid: bool
    manifest: dict[str, Any]
    path: Path


def validate_export_manifest(path: Path) -> ManifestValidation:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") not in KNOWN_SCHEMAS:
            return ManifestValidation("DATASCRAPER_EXPORT_SCHEMA_BLOCKED", False, payload, path)
        if payload.get("PIT_classification") not in ALLOWED_PIT:
            return ManifestValidation("DATASCRAPER_EXPORT_PIT_BLOCKED", False, payload, path)
        if payload.get("license_classification") in BLOCKED_LICENSES:
            return ManifestValidation("DATASCRAPER_EXPORT_LICENSE_BLOCKED", False, payload, path)
        expected = payload.get("manifest_hash")
        unhashed = dict(payload)
        unhashed.pop("manifest_hash", None)
        if not expected or stable_hash(unhashed) != expected:
            return ManifestValidation("DATASCRAPER_EXPORT_HASH_MISMATCH", False, payload, path)
        data_path = path.parent / str(payload.get("data_file"))
        if not data_path.is_file() or file_hash(data_path) != payload.get("content_hash"):
            return ManifestValidation("DATASCRAPER_EXPORT_HASH_MISMATCH", False, payload, path)
        return ManifestValidation("DATASCRAPER_EXPORT_ACCEPTED", True, payload, path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return ManifestValidation("DATASCRAPER_EXPORT_SCHEMA_BLOCKED", False, {}, path)
