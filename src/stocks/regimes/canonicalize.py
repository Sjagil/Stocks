from __future__ import annotations

from dataclasses import dataclass

import numpy as np


THREE_STATE_LABELS = (
    "RISK_ON_TREND",
    "NEUTRAL_CHOPPY",
    "STRESS_HIGH_VOL",
)


@dataclass(frozen=True)
class CanonicalStateMap:
    raw_to_label: dict[int, str]
    label_to_raw: dict[str, int]


def canonicalize_states(
    variances: np.ndarray,
    conditional_returns: np.ndarray,
    *,
    inflation_signature: np.ndarray | None = None,
) -> CanonicalStateMap:
    variance = np.asarray(variances, dtype=float)
    returns = np.asarray(conditional_returns, dtype=float)
    if len(variance) not in {3, 4} or variance.shape != returns.shape:
        raise ValueError("HMM_CANONICALIZATION_DIMENSION_MISMATCH")
    stress = int(np.argmax(variance))
    remaining = [state for state in range(len(variance)) if state != stress]
    mapping: dict[int, str] = {stress: "STRESS_HIGH_VOL"}
    if len(variance) == 4:
        if inflation_signature is None:
            raise ValueError("FOUR_STATE_INFLATION_SIGNATURE_REQUIRED")
        signature = np.asarray(inflation_signature, dtype=float)
        inflation = max(remaining, key=lambda state: signature[state])
        mapping[inflation] = "INFLATION_RATE_SHOCK"
        remaining.remove(inflation)
    risk_on = max(remaining, key=lambda state: returns[state])
    mapping[risk_on] = "RISK_ON_TREND"
    remaining.remove(risk_on)
    mapping[remaining[0]] = "NEUTRAL_CHOPPY"
    return CanonicalStateMap(
        raw_to_label=mapping,
        label_to_raw={label: raw for raw, label in mapping.items()},
    )
