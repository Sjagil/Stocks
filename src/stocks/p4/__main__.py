from __future__ import annotations

import argparse
import json
from pathlib import Path

from stocks.p4.data import (
    PITDataCatalog,
    SourceAttestation,
    ingest_point_in_time_bundle,
    ingest_point_in_time_snapshot,
)
from stocks.p4.forward import preregister_phase11_14_candidates
from stocks.p4.publisher import publish_p4_readiness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P4 data and forward evidence")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("publish")
    subparsers.add_parser("preregister")
    subparsers.add_parser("audit-data")
    bundle = subparsers.add_parser("ingest-bundle")
    bundle.add_argument("manifest", type=Path)
    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("dataset")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--provider", required=True)
    ingest.add_argument("--source-version", required=True)
    ingest.add_argument("--license-id", required=True)
    ingest.add_argument("--obtained-at", required=True)
    ingest.add_argument("--operator", required=True)
    ingest.add_argument("--licensed-for-research", action="store_true")
    ingest.add_argument("--complete-history-attested", action="store_true")
    ingest.add_argument("--point-in-time-semantics-attested", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "publish":
        result = publish_p4_readiness(root)
    elif args.command == "preregister":
        result = preregister_phase11_14_candidates(root)
    elif args.command == "audit-data":
        result = PITDataCatalog(root).audit()
    elif args.command == "ingest-bundle":
        result = ingest_point_in_time_bundle(root, args.manifest)
    else:
        result = ingest_point_in_time_snapshot(
            root,
            args.dataset,
            args.source,
            SourceAttestation(
                provider=args.provider,
                source_version=args.source_version,
                license_id=args.license_id,
                licensed_for_research=args.licensed_for_research,
                complete_history_attested=args.complete_history_attested,
                point_in_time_semantics_attested=args.point_in_time_semantics_attested,
                obtained_at=args.obtained_at,
                operator=args.operator,
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") not in {"BLOCKED", "NO_GO"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
