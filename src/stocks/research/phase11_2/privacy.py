from __future__ import annotations

import json
import re
from pathlib import Path


PATTERNS = {
    "provider_secret": re.compile(r'(?i)"?(api[_-]?key|api[_-]?token|app_id|provider[_-]?secret|access[_-]?token)"?\s*[:=]'),
    "raw_account": re.compile(r"\b(?:DU[0-9A-Z_]{4,}|U[0-9]{4,})\b"),
    "raw_accession": re.compile(r'(?i)"?(accession|accession_number)"?\s*[:=]'),
    "private_path": re.compile(r"(?i)data[/\\]research[/\\]phase11_2[/\\]private"),
    "raw_article_body": re.compile(r'(?i)"?(content|article_body|raw_text)"?\s*:'),
}


def scan_public_artifacts(output_dir: Path) -> dict[str, object]:
    matches = {name: 0 for name in PATTERNS}
    files = list(output_dir.glob("*.json"))
    for path in files:
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
    return {"status": "PRIVACY_GO" if not any(matches.values()) else "PRIVACY_BLOCKED", "matches": matches, "artifact_count": len(files)}
