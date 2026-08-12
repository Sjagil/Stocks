from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from stocks.data.multitimeframe import canonical_interval
from stocks.portfolio.swing import StrategyTimeframeContract


REGISTRY_PATH = Path("config/research_contracts/stocks_strategy_timeframe_registry_v1.json")


def declared_research_signal_timeframe_contract(
    project_root: Path,
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return an explicit registry declaration, never a performance inference.

    Legacy native strategies only declared a signal timeframe.  Registry V3
    gives those strategies a truthful single-timeframe, research-only contract.
    It does not claim higher-timeframe evidence, stable setup identity, economic
    qualification, or execution authority.
    """
    explicit = candidate.get("strategy_timeframe_contract")
    if isinstance(explicit, Mapping):
        return _validated_contract(
            explicit,
            declaration_source="CANDIDATE_EXPLICIT_DECLARATION",
            architecture_id=str(explicit.get("architecture_id") or "CANDIDATE_EXPLICIT"),
            research_only=bool(
                explicit.get("research_only", False)
                or candidate.get("classification")
                not in {"PAPER_CANDIDATE", "LIVE_CANARY_CANDIDATE", "CONTROLLED_LIVE"}
            ),
        )

    registry = _read_registry(project_root)
    native = registry.get("research_only_native_contracts")
    if not isinstance(native, Mapping):
        return None
    try:
        timeframe = canonical_interval(str(candidate.get("timeframe") or ""))
    except ValueError:
        return None
    declaration = native.get(timeframe)
    if not isinstance(declaration, Mapping):
        return None
    return _validated_contract(
        declaration,
        declaration_source=str(registry.get("contract_id") or "UNAVAILABLE"),
        architecture_id=str(
            declaration.get("architecture_id") or f"NATIVE_{timeframe.upper()}_RESEARCH_ONLY"
        ),
        research_only=True,
    )


def _validated_contract(
    declaration: Mapping[str, Any],
    *,
    declaration_source: str,
    architecture_id: str,
    research_only: bool,
) -> dict[str, Any] | None:
    try:
        contract = StrategyTimeframeContract(
            entry_timeframe=str(declaration["entry_timeframe"]),
            setup_timeframe=str(declaration["setup_timeframe"]),
            context_timeframes=tuple(declaration.get("context_timeframes", ())),
            structural_timeframe=str(declaration["structural_timeframe"]),
            management_timeframe=str(declaration["management_timeframe"]),
            exit_timeframe=str(declaration["exit_timeframe"]),
            required_timeframes=tuple(declaration["required_timeframes"]),
            optional_timeframes=tuple(declaration.get("optional_timeframes", ())),
            session=str(declaration.get("session", "RTH")),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return {
        **contract.as_dict(),
        "architecture_id": architecture_id,
        "declaration_source": declaration_source,
        "declaration_status": "EXPLICIT_RESEARCH_ONLY" if research_only else "EXPLICIT",
        "research_only": research_only,
        "multi_timeframe_edge_claimed": bool(contract.context_timeframes),
        "strategy_authority": "NONE" if research_only else "CONFIG_CONTROLLED",
        "execution_authority": "NONE",
    }


def _read_registry(project_root: Path) -> dict[str, Any]:
    try:
        value = json.loads((project_root / REGISTRY_PATH).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = ["declared_research_signal_timeframe_contract"]
