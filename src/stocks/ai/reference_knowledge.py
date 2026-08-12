from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.p3.io import atomic_write_json, read_json


CONFIG_PATH = Path("config/ai/reference_patterns_v1.json")
REFERENCE_ROOT = Path("reference_repos")
OUTPUT_ROOT = Path("output/reference_repos")
REPORT_PATH = Path("reports/REFERENCE_REPOS_MAXIMUM_INTEGRATION.md")
EXPECTED_REPOSITORY_COUNT = 14


def publish_reference_knowledge(project_root: Path) -> dict[str, Any]:
    """Inventory local design oracles without importing or installing them."""

    root = project_root.resolve()
    config = read_json(root / CONFIG_PATH)
    configured = config.get("repositories", [])
    directories = sorted(
        path.name for path in (root / REFERENCE_ROOT).iterdir() if path.is_dir()
    )
    configured_names = sorted(str(item.get("repo")) for item in configured)
    if len(configured) != EXPECTED_REPOSITORY_COUNT:
        raise ValueError("reference pattern registry must contain exactly 14 repositories")
    if directories != configured_names:
        raise ValueError("reference directory and registry identities do not match")

    previous_heads = read_json(root / OUTPUT_ROOT / "reference-heads.json")
    previous_by_repo = {
        row["repo"]: row.get("head")
        for row in previous_heads.get("repositories", [])
    }
    inventory_rows: list[dict[str, Any]] = []
    license_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    integration_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    for spec in configured:
        name = str(spec["repo"])
        repo = root / REFERENCE_ROOT / name
        head = _git(repo, "rev-parse", "HEAD")
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_git(repo, "status", "--porcelain"))
        head_rows.append(
            {
                "repo": name,
                "head": head,
                "branch": branch,
                "dirty": dirty,
                "changed_since_previous_inventory": bool(
                    previous_by_repo.get(name)
                    and previous_by_repo.get(name) != head
                ),
            }
        )
        inspected: list[dict[str, Any]] = []
        for relative in spec.get("inspected_files", []):
            path = repo / str(relative)
            if not path.is_file():
                raise ValueError(f"configured reference source missing: {name}/{relative}")
            inspected.append(
                {
                    "path": str(relative),
                    "sha256": sha256_file(path).upper(),
                    "line_count": _line_count(path),
                    "source_inspected": True,
                }
            )
        license_path = _license_path(repo)
        license_rows.append(
            {
                "repo": name,
                "license_file": license_path.name if license_path else None,
                "license_sha256": (
                    sha256_file(license_path).upper() if license_path else None
                ),
                "license_family": _license_family(license_path),
                "runtime_dependency": False,
                "source_copied": False,
                "reference_only": True,
            }
        )
        inventory_rows.append(
            {
                "repo": name,
                "role": spec.get("role"),
                "head": head,
                "branch": branch,
                "inspected_files": inspected,
                "patterns": spec.get("patterns", []),
            }
        )
        provenance_rows.extend(
            {
                "repo": name,
                "head": head,
                "source_path": item["path"],
                "source_sha256": item["sha256"],
                "use": "DESIGN_PATTERN_ONLY",
                "copied_lines": 0,
                "runtime_import": False,
            }
            for item in inspected
        )
        native_artifacts = _native_artifacts(name)
        present = [
            relative for relative in native_artifacts if (root / relative).exists()
        ]
        matrix_rows.append(
            {
                "repo": name,
                "reference_role": spec.get("role"),
                "patterns": spec.get("patterns", []),
                "native_integration": spec.get("native_integration"),
                "native_artifacts": present,
                "preserved_exclusions": spec.get("explicit_exclusions", []),
                "second_engine_created": False,
                "runtime_dependency_added": False,
            }
        )
        complete = len(present) == len(native_artifacts) and bool(native_artifacts)
        integration_rows.append(
            {
                "repo": name,
                "status": "NATIVE_INTEGRATION_PRESENT" if complete else "PARTIAL_NATIVE_COVERAGE",
                "expected_native_artifacts": native_artifacts,
                "present_native_artifacts": present,
                "missing_native_artifacts": sorted(set(native_artifacts) - set(present)),
                "broker_authority": "NONE",
            }
        )
        attribution_rows.append(
            {
                "repo": name,
                "head": head,
                "derived_value": spec.get("native_integration"),
                "evidence_artifacts": present,
                "patterns_count": len(spec.get("patterns", [])),
                "source_files_inspected": len(inspected),
                "copied_source": False,
            }
        )

    generated_at = datetime.now(UTC).isoformat()
    common = {
        "generated_at": generated_at,
        "repository_count": len(configured),
        "reference_only": True,
        "runtime_dependencies_added": False,
        "source_code_copied": False,
        "execution_authority": "NONE",
        "broker_writes": 0,
    }
    artifacts = {
        "inventory.json": {
            "schema": "reference_repo_inventory_v1",
            **common,
            "repositories": inventory_rows,
        },
        "license-audit.json": {
            "schema": "reference_repo_license_audit_v1",
            **common,
            "repositories": license_rows,
        },
        "reference-native-matrix.json": {
            "schema": "reference_native_matrix_v1",
            **common,
            "repositories": matrix_rows,
        },
        "provenance.json": {
            "schema": "reference_repo_provenance_v1",
            **common,
            "source_files": provenance_rows,
        },
        "integration-status.json": {
            "schema": "reference_repo_integration_status_v1",
            **common,
            "status": (
                "GO"
                if all(row["status"] == "NATIVE_INTEGRATION_PRESENT" for row in integration_rows)
                else "PARTIAL"
            ),
            "repositories": integration_rows,
        },
        "reference-heads.json": {
            "schema": "reference_repo_heads_v1",
            **common,
            "repositories": head_rows,
        },
        "value-attribution.json": {
            "schema": "reference_repo_value_attribution_v1",
            **common,
            "repositories": attribution_rows,
        },
    }
    for name, payload in artifacts.items():
        payload["content_hash"] = stable_hash(payload)
        atomic_write_json(root / OUTPUT_ROOT / name, payload)
    _write_report(root / REPORT_PATH, artifacts)
    return artifacts["integration-status.json"]


def _native_artifacts(repo: str) -> list[str]:
    mapping = {
        "qlib": ["src/stocks/ai/panel.py", "src/stocks/ai/modeling.py", "src/stocks/ai/intelligence.py"],
        "pybroker": ["src/stocks/ai/modeling.py", "output/ai/decision-intelligence/oos-predictions.parquet"],
        "vectorbt": ["src/stocks/quant_platform/data.py", "src/stocks/ai/modeling.py"],
        "finrl_x": ["src/stocks/ai/intelligence.py", "src/stocks/portfolio/learning_integration.py"],
        "trademaster": ["src/stocks/rl", "config/rl_reward_v1.json"],
        "quantstats": ["src/stocks/ai/modeling.py", "output/ai/decision-intelligence/tournament.json"],
        "rd_agent": ["src/stocks/ai/contracts.py", "src/stocks/ai/governance.py"],
        "fingpt": ["src/stocks/ai/contracts.py", "src/stocks/ai/plane.py"],
        "finrobot": ["src/stocks/ai/contracts.py", "src/stocks/ai/plane.py"],
        "lean": ["src/stocks/portfolio/orchestrator.py", "src/stocks/portfolio/execution_bridge.py"],
        "ib_async": ["src/stocks/ibkr", "src/stocks/live"],
        "lean_ibkr": ["src/stocks/ibkr", "src/stocks/live"],
        "nautilus_trader": ["src/stocks/execution", "src/stocks/live"],
        "lumibot": ["src/stocks/portfolio/orchestrator.py", "src/stocks/operations/service.py"],
    }
    return mapping[repo]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"git metadata unavailable for {repo.name}: {' '.join(args)}")
    return result.stdout.strip()


def _license_path(repo: Path) -> Path | None:
    candidates = sorted(
        path
        for path in repo.iterdir()
        if path.is_file() and path.name.lower().startswith(("license", "copying"))
    )
    return candidates[0] if candidates else None


def _license_family(path: Path | None) -> str:
    if path is None:
        return "NOT_FOUND"
    text = path.read_text(encoding="utf-8", errors="replace").lower()[:12_000]
    if "commons clause" in text:
        return "COMMONS_CLAUSE"
    if "apache license" in text and "version 2.0" in text:
        return "APACHE-2.0"
    if "gnu affero general public license" in text:
        return "AGPL"
    if "gnu lesser general public license" in text:
        return "LGPL"
    if "gnu general public license" in text:
        return "GPL"
    if "mit license" in text or "permission is hereby granted" in text:
        return "MIT"
    if "bsd" in text and "redistribution" in text:
        return "BSD"
    return "OTHER_REVIEW_REQUIRED"


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _write_report(path: Path, artifacts: dict[str, dict[str, Any]]) -> None:
    status = artifacts["integration-status.json"]
    rows = status["repositories"]
    lines = [
        "# Reference Repositories Maximum Integration",
        "",
        f"Generated: {status['generated_at']}",
        "",
        "All 14 repositories are treated as local, file-level design oracles. No repository is imported, installed, or granted runtime, portfolio, money, or broker authority.",
        "",
        f"Overall status: **{status['status']}**",
        "",
        "| Repository | Native status | Missing artifacts |",
        "|---|---|---|",
    ]
    for row in rows:
        missing = ", ".join(row["missing_native_artifacts"]) or "None"
        lines.append(f"| {row['repo']} | {row['status']} | {missing} |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Source lines copied: 0",
            "- Runtime reference dependencies added: 0",
            "- Second trading or broker engine created: no",
            "- Execution authority: NONE",
            "- Broker writes: 0",
            "",
            "The machine-readable inventory, license audit, source hashes, repository heads, native mapping, integration status, and value attribution are under `output/reference_repos/`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


__all__ = ["publish_reference_knowledge"]
