from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocks.research.macro_pairs import (
    _macro_risk_on_gate,
    _portfolio_returns,
    _registry_pair_inventory,
)


def test_macro_gate_is_point_in_time_and_fail_closed_before_first_record(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output" / "macro"
    output.mkdir(parents=True)
    (output / "history.json").write_text(
        json.dumps(
            {
                "history": [
                    {
                        "as_of": "2020-01-31T14:00:00Z",
                        "regime": {"market_regime": "RISK_OFF"},
                    },
                    {
                        "as_of": "2020-02-29T14:00:00Z",
                        "regime": {"market_regime": "RISK_ON"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    gate, audit = _macro_risk_on_gate(tmp_path)
    assert gate.tolist() == [False, True]
    assert audit["point_in_time"] is True
    assert audit["future_returns_used"] is False


def test_macro_pair_changes_only_the_prespecified_gate() -> None:
    dates = pd.bdate_range("2020-01-01", periods=320)
    close = 100.0 + pd.Series(range(320), dtype=float)
    frame = pd.DataFrame(
        {
            "date": dates,
            "security_id": "SEC-1",
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
        }
    )
    parameters = {"fast": 20, "slow": 50}
    baseline = _portfolio_returns(
        frame,
        "ma_crossover",
        parameters,
        10.0,
        macro_gate=None,
    )
    gate = pd.Series(False, index=dates)
    macro = _portfolio_returns(
        frame,
        "ma_crossover",
        parameters,
        10.0,
        macro_gate=gate,
    )
    assert baseline.sum() > 0
    assert macro.sum() == 0


def test_every_generated_strategy_has_an_identical_parameter_macro_pair() -> None:
    inventory = _registry_pair_inventory()
    assert inventory["status"] == "GO"
    assert inventory["pair_count"] > 0
    assert all(
        row["identical_non_macro_parameters"]
        for row in inventory["pairs"]
    )
    assert all(
        row["macro_filter"] == "macro_risk_on"
        for row in inventory["pairs"]
    )
    assert inventory["execution_authority"] == "NONE"
