from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stocks.macro.contracts import MacroSeriesSpec, stable_hash


@dataclass(frozen=True)
class MacroConfig:
    version: str
    config_hash: str
    minimum_score_coverage: float
    partial_score_coverage: float
    hysteresis: dict[str, float]
    portfolio: dict[str, float]
    screener: dict[str, float]
    strategy: dict[str, Any]
    score_weights: dict[str, dict[str, float]]
    sector_mappings: dict[str, dict[str, float]]
    regional_mappings: dict[str, dict[str, float]]
    events: tuple[dict[str, Any], ...]
    series: dict[str, MacroSeriesSpec]

    @classmethod
    def load(cls, project_root: Path) -> MacroConfig:
        path = project_root / "config" / "macro" / "macro_v1.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        specs: dict[str, MacroSeriesSpec] = {}
        for item in raw["series"]:
            spec = MacroSeriesSpec(
                canonical_id=str(item["id"]),
                name=str(item["name"]),
                category=str(item["category"]),
                region=str(item["region"]),
                frequency=str(item["frequency"]),
                unit=str(item["unit"]),
                transformation=str(item["transform"]),
                release_lag_days=int(item["release_lag_days"]),
                revision_sensitive=bool(item["revision_sensitive"]),
                direction=int(item["direction"]),
                minimum_history=int(item["minimum_history"]),
                stale_days=int(item["stale_days"]),
                primary_source=str(item["primary"]),
                fallback_source=(
                    None if item.get("fallback") is None else str(item["fallback"])
                ),
                provider_id=(
                    None
                    if item.get("provider_id") is None
                    else str(item["provider_id"])
                ),
                vintage_capable=bool(item["vintage_capable"]),
            )
            spec.validate()
            if spec.canonical_id in specs:
                raise ValueError(f"DUPLICATE_MACRO_SERIES:{spec.canonical_id}")
            specs[spec.canonical_id] = spec
        score_weights = {
            str(score): {
                str(series_id): float(weight)
                for series_id, weight in weights.items()
            }
            for score, weights in raw["score_weights"].items()
        }
        for score, weights in score_weights.items():
            unknown = set(weights) - set(specs)
            if unknown:
                raise ValueError(
                    f"UNKNOWN_SCORE_SERIES:{score}:{','.join(sorted(unknown))}"
                )
            if not weights or abs(sum(weights.values()) - 1.0) > 1e-9:
                raise ValueError(f"INVALID_MACRO_SCORE_WEIGHTS:{score}")
        minimum = float(raw["minimum_score_coverage"])
        partial = float(raw["partial_score_coverage"])
        if not 0 <= minimum <= partial <= 1:
            raise ValueError("INVALID_MACRO_COVERAGE_THRESHOLDS")
        return cls(
            version=str(raw["version"]),
            config_hash=stable_hash(raw),
            minimum_score_coverage=minimum,
            partial_score_coverage=partial,
            hysteresis={
                str(key): float(value)
                for key, value in raw["hysteresis"].items()
            },
            portfolio={
                str(key): float(value)
                for key, value in raw["portfolio"].items()
            },
            screener={
                str(key): float(value)
                for key, value in raw["screener"].items()
            },
            strategy=dict(raw["strategy"]),
            score_weights=score_weights,
            sector_mappings={
                str(key): {
                    str(score): float(weight)
                    for score, weight in values.items()
                }
                for key, values in raw["sector_mappings"].items()
            },
            regional_mappings={
                str(key): {
                    str(score): float(weight)
                    for score, weight in values.items()
                }
                for key, values in raw["regional_mappings"].items()
            },
            events=tuple(dict(item) for item in raw["events"]),
            series=specs,
        )

    def public_registry(self) -> dict[str, Any]:
        return {
            "schema": "macro_series_registry_v1",
            "version": self.version,
            "config_hash": self.config_hash,
            "series_count": len(self.series),
            "series": [
                spec.__dict__
                for spec in sorted(
                    self.series.values(),
                    key=lambda item: item.canonical_id,
                )
            ],
        }
