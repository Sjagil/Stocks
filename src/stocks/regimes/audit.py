from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.regimes.model import FrozenHMM


def audit_transition_stability(
    model: FrozenHMM,
    probabilities: pd.DataFrame,
    *,
    minimum_state_fraction: float,
    minimum_duration: float,
    maximum_chatter_ratio: float,
) -> dict[str, Any]:
    decoded = probabilities.idxmax(axis=1)
    occupancy = pd.Series(
        {
            model.raw_to_label[state]: model.training_state_occupancy[state]
            for state in range(model.n_regimes)
        }
    )
    changes = decoded.ne(decoded.shift(1))
    single_bar = (
        changes
        & decoded.shift(-1).eq(decoded.shift(1))
        & decoded.shift(1).notna()
    )
    chatter = float(single_bar.mean()) if len(single_bar) else 0.0
    durations = {
        model.raw_to_label[state]: float(model.expected_durations[state])
        for state in range(model.n_regimes)
    }
    checks = {
        "converged": model.converged,
        "minimum_state_fraction": bool(
            occupancy.ge(minimum_state_fraction).all()
        ),
        "minimum_expected_duration": bool(
            all(value >= minimum_duration for value in durations.values())
        ),
        "maximum_chatter_ratio": chatter <= maximum_chatter_ratio,
        "probabilities_sum_to_one": bool(
            np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
        ),
    }
    return {
        "status": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "state_occupancy": occupancy.to_dict(),
        "expected_durations": durations,
        "single_bar_chatter_ratio": chatter,
    }


class HMMStatePersistence:
    def __init__(self, private_root: Path):
        self.private_root = private_root
        self.private_root.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: FrozenHMM) -> dict[str, str]:
        payload = model.payload()
        digest = _hash(payload)
        path = self.private_root / f"model-{digest}.json"
        if not path.exists():
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        pointer = self.private_root / "current-model.json"
        pointer.write_text(
            json.dumps(
                {"model_hash": digest, "model_path": str(path)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"model_hash": digest, "model_path": str(path)}

    def append_state(self, payload: dict[str, Any]) -> None:
        path = self.private_root / "filtered-states.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n"
            )


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest().upper()
