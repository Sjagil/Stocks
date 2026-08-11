from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stocks.universe import broad_etf_symbols


@dataclass(frozen=True)
class ScreenerConfig:
    version: str
    screener_version: str
    weights: dict[str, float]
    thresholds: dict[str, float]
    classifications: dict[str, float]
    benchmarks: dict[str, str]
    etf_symbols: frozenset[str]
    excluded_product_terms: tuple[str, ...]
    shariah_attestations_path: Path
    shariah_attestations_hash: str
    config_hash: str

    @classmethod
    def load(cls, project_root: Path) -> ScreenerConfig:
        path = project_root / "config" / "screener" / "daily_screener_v1.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        weights = {str(key): float(value) for key, value in raw["weights"].items()}
        if set(weights) != {
            "fundamental",
            "technical",
            "liquidity",
            "risk",
            "macro",
        }:
            raise ValueError(
                "screener weights must define the five canonical components"
            )
        if not 0 <= weights["macro"] <= 0.10:
            raise ValueError("screener macro weight must be in [0, 0.10]")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("screener weights must sum to one")
        attestations_path = project_root / str(raw["shariah_attestations_path"])
        attestations = json.loads(attestations_path.read_text(encoding="utf-8"))
        attestations_canonical = json.dumps(
            attestations,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        attestations_hash = hashlib.sha256(attestations_canonical).hexdigest().upper()
        canonical = json.dumps(
            {"config": raw, "shariah_attestations_hash": attestations_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return cls(
            version=str(raw["version"]),
            screener_version=str(raw["screener_version"]),
            weights=weights,
            thresholds={str(key): float(value) for key, value in raw["thresholds"].items()},
            classifications={
                str(key): float(value) for key, value in raw["classifications"].items()
            },
            benchmarks={str(key): str(value) for key, value in raw["benchmarks"].items()},
            etf_symbols=frozenset(
                {
                    *(str(item).upper() for item in raw["etf_symbols"]),
                    *broad_etf_symbols(project_root),
                }
            ),
            excluded_product_terms=tuple(
                str(item).upper() for item in raw["excluded_product_terms"]
            ),
            shariah_attestations_path=attestations_path,
            shariah_attestations_hash=attestations_hash,
            config_hash=hashlib.sha256(canonical).hexdigest().upper(),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "screener_version": self.screener_version,
            "weights": self.weights,
            "thresholds": self.thresholds,
            "classifications": self.classifications,
            "benchmarks": self.benchmarks,
            "shariah_attestations_hash": self.shariah_attestations_hash,
            "config_hash": self.config_hash,
        }
