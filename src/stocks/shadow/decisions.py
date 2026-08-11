from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.shadow.clock import validate_decision_clock
from stocks.shadow.models import (
    ShadowDecision,
    ShadowDecisionRequest,
    ShadowSignal,
    ShadowTargetPortfolio,
    ShadowTargetPosition,
    model_to_jsonable,
)
from stocks.shadow.registry import StrategyContract
from stocks.shadow.validation import validate_target_portfolio


@dataclass(frozen=True)
class UniverseSnapshot:
    instrument_count: int
    con_ids: tuple[int, ...]
    symbols: tuple[str, ...]
    regions: tuple[str, ...]
    sleeves: tuple[str, ...]
    currencies: tuple[str, ...]
    eligibility_status: dict[int, str]
    exclusion_reasons: dict[int, str]
    contract_hashes: dict[int, str]
    session_hashes: dict[int, str]
    dataset_hashes: dict[int, str]
    universe_hash: str


def frozen_fixture_universe(project_root: Path) -> UniverseSnapshot:
    rows = [
        (756733, "SPY", "united_states", "equity", "USD"),
        (101484826, "GLD", "global", "commodity", "USD"),
        (39039301, "BIL", "united_states", "cash", "USD"),
    ]
    dataset_hashes = {}
    for con_id, *_ in rows:
        path = project_root / "data" / "total_returns" / "security_type=STK" / f"con_id={con_id}" / "interval=1d" / "total_returns.parquet"
        dataset_hashes[con_id] = sha256_file(path) or stable_hash({"missing_fixture_dataset": con_id})
    core: dict[str, Any] = {
        "instrument_count": len(rows),
        "con_ids": [row[0] for row in rows],
        "symbols": [row[1] for row in rows],
        "regions": [row[2] for row in rows],
        "sleeves": [row[3] for row in rows],
        "currencies": [row[4] for row in rows],
        "eligibility_status": {row[0]: "ELIGIBLE" for row in rows},
        "exclusion_reasons": {},
        "contract_hashes": {row[0]: stable_hash({"contract": row[0], "symbol": row[1]}) for row in rows},
        "session_hashes": {row[0]: stable_hash({"session": row[0], "clock": "frozen"}) for row in rows},
        "dataset_hashes": dataset_hashes,
    }
    return UniverseSnapshot(
        instrument_count=len(rows),
        con_ids=tuple(row[0] for row in rows),
        symbols=tuple(row[1] for row in rows),
        regions=tuple(row[2] for row in rows),
        sleeves=tuple(row[3] for row in rows),
        currencies=tuple(row[4] for row in rows),
        eligibility_status={row[0]: "ELIGIBLE" for row in rows},
        exclusion_reasons={},
        contract_hashes=core["contract_hashes"],
        session_hashes=core["session_hashes"],
        dataset_hashes=dataset_hashes,
        universe_hash=stable_hash(core),
    )


def fixture_decision_request(project_root: Path, contract: StrategyContract) -> ShadowDecisionRequest:
    universe = frozen_fixture_universe(project_root)
    return ShadowDecisionRequest(
        strategy_id=contract.strategy_id,
        strategy_version=contract.strategy_version,
        decision_timestamp="2026-07-21T21:05:00+00:00",
        information_cutoff_timestamp="2026-07-21T20:59:00+00:00",
        first_executable_timestamp="2026-07-22T13:30:00+00:00",
        dataset_manifest_hash=stable_hash(universe.dataset_hashes),
        dataset_content_hashes={str(key): value for key, value in universe.dataset_hashes.items()},
        universe_hash=universe.universe_hash,
        parameter_hash=stable_hash({"fixture": "phase8_2", "weights": "fixed"}),
    )


def build_fixture_signals(request: ShadowDecisionRequest, universe: UniverseSnapshot) -> list[ShadowSignal]:
    values = [Decimal("0.80"), Decimal("0.30"), Decimal("0.05")]
    return [
        ShadowSignal(
            decision_id=decision_id_for(request),
            con_id=con_id,
            feature_name="fixture_momentum",
            feature_value=value,
            feature_timestamp="2026-07-21T20:00:00+00:00",
            available_at="2026-07-21T20:01:00+00:00",
            source_dataset=f"total_returns:{con_id}",
            source_content_hash=universe.dataset_hashes[con_id],
            calculation_version="phase8_2_fixture_signal_v1",
            signal_value=value,
            signal_status="VALID",
        )
        for con_id, value in zip(universe.con_ids, values, strict=True)
    ]


def build_fixture_target(request: ShadowDecisionRequest) -> ShadowTargetPortfolio:
    decision_id = decision_id_for(request)
    positions = (
        ShadowTargetPosition(756733, "SPY", "united_states", "equity", "USD", Decimal("0.40")),
        ShadowTargetPosition(101484826, "GLD", "global", "commodity", "USD", Decimal("0.30")),
        ShadowTargetPosition(39039301, "BIL", "united_states", "cash", "USD", Decimal("0.20")),
    )
    target_hash = stable_hash([model_to_jsonable(item) for item in positions] + [{"cash_weight": "0.10"}])
    return ShadowTargetPortfolio(decision_id=decision_id, positions=positions, cash_weight=Decimal("0.10"), target_portfolio_hash=target_hash, status="TARGET_PORTFOLIO_VALID")


def build_decision(contract: StrategyContract, request: ShadowDecisionRequest, target: ShadowTargetPortfolio, universe: UniverseSnapshot) -> ShadowDecision:
    clock = validate_decision_clock(
        frequency="FIXTURE_MANUAL",
        information_cutoff_timestamp=request.information_cutoff_timestamp,
        decision_timestamp=request.decision_timestamp,
        first_executable_timestamp=request.first_executable_timestamp,
        dataset_content_hashes=request.dataset_content_hashes,
    )
    target_validation = validate_target_portfolio(target, eligible_con_ids=set(universe.con_ids))
    status = "SHADOW_FIXTURE_VALIDATED" if clock["status"] == "GO" and target_validation["status"] == "GO" else "SHADOW_FIXTURE_BLOCKED"
    return ShadowDecision(
        decision_id=decision_id_for(request),
        strategy_id=contract.strategy_id,
        strategy_version=contract.strategy_version,
        strategy_hash=contract.strategy_hash,
        decision_timestamp=request.decision_timestamp,
        information_cutoff_timestamp=request.information_cutoff_timestamp,
        first_executable_timestamp=request.first_executable_timestamp,
        dataset_manifest_hash=request.dataset_manifest_hash,
        dataset_content_hashes=request.dataset_content_hashes,
        universe_hash=universe.universe_hash,
        parameter_hash=request.parameter_hash,
        eligible_instruments=universe.con_ids,
        blocked_instruments=(),
        block_reasons={},
        signal_count=len(universe.con_ids),
        target_portfolio_hash=target.target_portfolio_hash,
        authority="NONE",
        status=status,
        created_at="2026-07-21T21:05:01+00:00",
    )


def decision_id_for(request: ShadowDecisionRequest) -> str:
    return f"SHADOW-DECISION-{stable_hash(model_to_jsonable(request))[:20]}"
