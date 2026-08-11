from __future__ import annotations

import json
import re
from pathlib import Path


PATTERNS = {
    "raw_account": re.compile(r"\b(?:DU[0-9A-Z_]{4,}|U[0-9]{4,})\b"),
    "account_fingerprint": re.compile(r'(?i)"?account_fingerprint"?\s*[:=]'),
    "broker_identifier": re.compile(r'(?i)"?(broker_order_id|exec_id|perm_id)"?\s*[:=]'),
    "approval_challenge": re.compile(r'(?i)"?approval_challenge"?\s*[:=]'),
    "secret": re.compile(r'(?i)"?(password|passwd|api[_-]?key|access[_-]?token|provider[_-]?secret)"?\s*[:=]'),
}


def scan_public_artifacts(output_dir: Path) -> dict[str, object]:
    matches = {name: 0 for name in PATTERNS}
    for path in output_dir.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "privacy" in payload:
            payload = dict(payload)
            payload.pop("privacy", None)
            text = json.dumps(payload, sort_keys=True, default=str)
        for name, pattern in PATTERNS.items():
            matches[name] += len(pattern.findall(text))
    return {
        "status": "PRIVACY_GO" if not any(matches.values()) else "PRIVACY_BLOCKED",
        "matches": matches,
        "public_artifact_count": len(list(output_dir.glob("*.json"))),
    }
