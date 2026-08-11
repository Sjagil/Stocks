from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from stocks.execution.idempotency import stable_hash


MANIFEST_SCHEMA = "live_writer_integrity_manifest_v2"
VERIFY_SCHEMA = "live_writer_integrity_verification_v2"
HISTORY_SCHEMA = "live_writer_integrity_history_v1"
FREEZE_MARKER = "LIVE_CANARY_WRITER_OFFLINE_FROZEN_GO"


def normalized_file_hash(path: Path) -> str:
    """Hash source deterministically across Windows/Unix line endings."""
    if path.is_symlink():
        raise ValueError(f"SYMLINK_BLOCKED:{path}")
    data = path.read_bytes()
    if path.suffix.lower() in {
        ".cfg",
        ".ini",
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }:
        text = data.decode("utf-8-sig")
        data = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_manifest(
    project_root: Path,
    sources: Iterable[str],
    *,
    operator: str,
    reason: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    normalized_sources = tuple(sorted({_normalize_relative(path) for path in sources}))
    files: list[dict[str, Any]] = []
    blockers: list[str] = []
    for relative in normalized_sources:
        path = _inside_root(root, relative)
        if not path.exists():
            blockers.append(f"MISSING_CRITICAL_FILE:{relative}")
            continue
        if not path.is_file():
            blockers.append(f"NOT_A_FILE:{relative}")
            continue
        try:
            digest = normalized_file_hash(path)
        except (UnicodeDecodeError, ValueError) as exc:
            blockers.append(str(exc))
            continue
        files.append(
            {
                "path": relative,
                "sha256": digest,
                "category": _category(relative),
                "size_bytes": path.stat().st_size,
            }
        )
    unauthorized = _unexpected_live_modules(root, normalized_sources)
    blockers.extend(f"UNAUTHORIZED_LIVE_MODULE:{item}" for item in unauthorized)
    category_hashes = {
        category: stable_hash(
            {
                row["path"]: row["sha256"]
                for row in files
                if row["category"] == category
            }
        )
        for category in sorted({str(row["category"]) for row in files})
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at": _now(),
        "status": "GO" if not blockers else "NO_GO",
        "freeze_status": FREEZE_MARKER if not blockers else "NO_GO",
        "operator": _safe_operator(operator),
        "reason": str(reason).strip(),
        "path_format": "POSIX_RELATIVE",
        "line_ending_normalization": "CRLF_AND_CR_TO_LF_FOR_TEXT",
        "symlink_policy": "BLOCK",
        "files": files,
        "source_hashes": {row["path"]: row["sha256"] for row in files},
        "category_hashes": category_hashes,
        "configuration_hash": category_hashes.get("CONFIG"),
        "strategy_registry_hash": category_hashes.get("STRATEGY_REGISTRY"),
        "authority_registry_hash": category_hashes.get("AUTHORITY"),
        "broker_adapter_hash": category_hashes.get("BROKER_ADAPTER"),
        "risk_engine_hash": category_hashes.get("RISK_ENGINE"),
        "order_router_hash": category_hashes.get("ORDER_ROUTER"),
        "reconciliation_hash": category_hashes.get("RECONCILIATION"),
        "unauthorized_live_modules": unauthorized,
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "live_trading_allowed": False,
    }
    manifest["manifest_hash"] = stable_hash(
        {
            "source_hashes": manifest["source_hashes"],
            "category_hashes": category_hashes,
            "path_format": manifest["path_format"],
            "line_ending_normalization": manifest["line_ending_normalization"],
            "symlink_policy": manifest["symlink_policy"],
        }
    )
    return manifest


def verify_manifest(
    project_root: Path,
    sources: Iterable[str],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    expected_sources = tuple(sorted({_normalize_relative(path) for path in sources}))
    expected = {
        _normalize_relative(str(path)): str(digest)
        for path, digest in dict(frozen.get("source_hashes", {})).items()
    }
    current = build_manifest(
        project_root,
        expected_sources,
        operator="VERIFY_ONLY",
        reason="RUNTIME_VERIFICATION",
    )
    current_hashes = dict(current["source_hashes"])
    expected_set = set(expected)
    current_set = set(expected_sources)
    missing = sorted(expected_set - set(current_hashes))
    extra = sorted(current_set - expected_set)
    changed = sorted(
        path
        for path in expected_set & set(current_hashes)
        if expected[path] != current_hashes[path]
    )
    blockers = list(current.get("blockers", []))
    blockers.extend(f"FROZEN_FILE_MISSING:{path}" for path in missing)
    blockers.extend(f"UNAUTHORIZED_CRITICAL_FILE:{path}" for path in extra)
    blockers.extend(f"CRITICAL_FILE_CHANGED:{path}" for path in changed)
    schema_ok = frozen.get("schema") == MANIFEST_SCHEMA
    if not schema_ok:
        blockers.append("LEGACY_OR_INVALID_FREEZE_SCHEMA")
    if frozen.get("freeze_status") != FREEZE_MARKER:
        blockers.append("FREEZE_MARKER_INVALID")
    status = "GO" if not blockers else "NO_GO"
    return {
        "schema": VERIFY_SCHEMA,
        "verified_at": _now(),
        "status": status,
        "writer_hash_integrity": status == "GO",
        "freeze_schema_valid": schema_ok,
        "expected_manifest_hash": frozen.get("manifest_hash"),
        "current_manifest_hash": current.get("manifest_hash"),
        "missing_files": missing,
        "extra_critical_files": extra,
        "changed_files": changed,
        "unauthorized_live_modules": current.get("unauthorized_live_modules", []),
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "live_trading_allowed": False,
    }


def freeze_manifest(
    project_root: Path,
    sources: Iterable[str],
    *,
    operator: str,
    reason: str,
    re_freeze: bool,
    confirmed: bool,
) -> dict[str, Any]:
    root = project_root.resolve()
    output = root / "output" / "ibkr" / "live"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "freeze-status.json"
    previous = _read_json(path)
    if re_freeze and not confirmed:
        return _blocked("EXPLICIT_REFREEZE_CONFIRMATION_REQUIRED")
    if re_freeze and (not operator.strip() or not reason.strip()):
        return _blocked("OPERATOR_AND_REASON_REQUIRED")
    manifest = build_manifest(
        root,
        sources,
        operator=operator,
        reason=reason,
    )
    if manifest["status"] != "GO":
        return manifest
    if path.exists() and not re_freeze:
        return _blocked("EXISTING_FREEZE_REQUIRES_EXPLICIT_REFREEZE")
    previous_hash = previous.get("manifest_hash")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    event = {
        "schema": HISTORY_SCHEMA,
        "recorded_at": _now(),
        "event": "REFREEZE" if previous else "INITIAL_FREEZE",
        "operator": manifest["operator"],
        "reason": manifest["reason"],
        "old_manifest_hash": previous_hash,
        "new_manifest_hash": manifest["manifest_hash"],
        "changed_files": _changed_paths(previous, manifest),
        "execution_authority": "NONE",
    }
    with (output / "writer-integrity-history.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return manifest


def inspect_manifest(project_root: Path, sources: Iterable[str]) -> dict[str, Any]:
    root = project_root.resolve()
    path = root / "output" / "ibkr" / "live" / "freeze-status.json"
    frozen = _read_json(path)
    current = build_manifest(
        root,
        sources,
        operator="INSPECT_ONLY",
        reason="NON_MUTATING_INSPECTION",
    )
    freeze_exists = bool(frozen)
    verification = (
        verify_manifest(root, sources, frozen) if freeze_exists else {}
    )
    if not freeze_exists:
        inspection_status = "NO_FREEZE"
        blockers = ["FROZEN_MANIFEST_NOT_FOUND"]
    elif current.get("status") != "GO":
        inspection_status = "CURRENT_MANIFEST_INVALID"
        blockers = list(current.get("blockers", []))
    elif verification.get("status") != "GO":
        inspection_status = "MISMATCH"
        blockers = list(verification.get("blockers", []))
    else:
        inspection_status = "MATCH"
        blockers = []
    status = "GO" if inspection_status == "MATCH" else "NO_GO"
    return {
        "schema": "live_writer_integrity_inspection_v2",
        "status": status,
        "inspection_status": inspection_status,
        "writer_hash_integrity": status == "GO",
        "freeze_exists": freeze_exists,
        "frozen_schema": frozen.get("schema"),
        "frozen_manifest_hash": frozen.get("manifest_hash"),
        "current_manifest_hash": current.get("manifest_hash"),
        "current_status": current.get("status"),
        "source_count": len(current.get("files", [])),
        "missing_files": verification.get("missing_files", []),
        "extra_critical_files": verification.get(
            "extra_critical_files", []
        ),
        "changed_files": verification.get("changed_files", []),
        "unauthorized_live_modules": current.get(
            "unauthorized_live_modules", []
        ),
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "live_trading_allowed": False,
    }


def _unexpected_live_modules(root: Path, sources: Iterable[str]) -> list[str]:
    allowed = set(sources)
    live_root = root / "src" / "stocks" / "live"
    if not live_root.exists():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in live_root.glob("*.py")
        if path.name != "__init__.py"
        and path.relative_to(root).as_posix() not in allowed
    )


def _category(relative: str) -> str:
    lower = relative.lower()
    if lower.startswith("config/"):
        return "CONFIG"
    if "authority" in lower or "approval" in lower:
        return "AUTHORITY"
    if "adapter" in lower or "callbacks" in lower:
        return "BROKER_ADAPTER"
    if "risk" in lower or "capital" in lower:
        return "RISK_ENGINE"
    if "submission" in lower or "router" in lower or "automatic" in lower:
        return "ORDER_ROUTER"
    if "reconcil" in lower or "position" in lower:
        return "RECONCILIATION"
    if "strategy" in lower or "signals" in lower:
        return "STRATEGY_REGISTRY"
    if lower.startswith("tests/"):
        return "TEST_EVIDENCE"
    return "EXECUTION_CODE"


def _normalize_relative(value: str) -> str:
    normalized = PurePosixPath(str(value).replace("\\", "/")).as_posix()
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"UNSAFE_CRITICAL_PATH:{value}")
    return normalized


def _inside_root(root: Path, relative: str) -> Path:
    candidate = root / Path(relative)
    resolved_parent = candidate.parent.resolve()
    if root != resolved_parent and root not in resolved_parent.parents:
        raise ValueError(f"UNSAFE_CRITICAL_PATH:{relative}")
    return candidate


def _safe_operator(value: str) -> str:
    clean = "".join(character for character in value.strip() if character.isalnum() or character in "-_.@")
    return clean[:80] or "UNSPECIFIED_OPERATOR"


def _changed_paths(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    before = dict(old.get("source_hashes", {}))
    after = dict(new.get("source_hashes", {}))
    return sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _blocked(blocker: str) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "NO_GO",
        "freeze_status": "NO_GO",
        "blockers": [blocker],
        "execution_authority": "NONE",
        "live_trading_allowed": False,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "FREEZE_MARKER",
    "MANIFEST_SCHEMA",
    "build_manifest",
    "freeze_manifest",
    "inspect_manifest",
    "normalized_file_hash",
    "verify_manifest",
]
