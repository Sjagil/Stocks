from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .export_manifest import ManifestValidation, validate_export_manifest


DEFAULT_DATASCRAPER_ROOT = Path(r"C:\Users\alhar\Documents\datascraper")
EXPORT_RELATIVE = Path("output") / "stocks_research_exports" / "phase11_3"


class DatascraperAdapter:
    def __init__(self, root: Path = DEFAULT_DATASCRAPER_ROOT) -> None:
        self.root = root.resolve()
        self.export_root = self.root / EXPORT_RELATIVE

    def inventory(self) -> dict[str, Any]:
        path = self.export_root / "source-inventory.json"
        if not path.is_file():
            return {"status": "DATASCRAPER_INVENTORY_MISSING", "connectors": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def manifests(self) -> list[ManifestValidation]:
        return [validate_export_manifest(path) for path in sorted(self.export_root.glob("*/batch-*/manifest.json"))]

    def rows(self, validation: ManifestValidation) -> Iterator[dict[str, Any]]:
        if not validation.valid:
            return
        path = validation.path.parent / validation.manifest["data_file"]
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
