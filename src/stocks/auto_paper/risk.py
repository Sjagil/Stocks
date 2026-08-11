from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from stocks.auto_paper.config import AutoPaperConfig
from stocks.auto_paper.contracts import AutoSignal, MarketQuote, PortfolioState


@dataclass(frozen=True)
class EntryRiskContext:
    decision_time: str
    market_session_open: bool
    signal_seen: bool
    shariah_status: str
    strategy_authority_go: bool
    financial_finalist_go: bool
    forward_shadow_go: bool
    quote: MarketQuote
    open_order_for_con_id: bool
    existing_position_for_con_id: bool
    entries_today: int
    sector: str
    event_cluster: str
    kill_switch_clear: bool


def evaluate_entry_risk(signal: AutoSignal, portfolio: PortfolioState, config: AutoPaperConfig, context: EntryRiskContext) -> dict[str, object]:
    decision_at = datetime.fromisoformat(context.decision_time)
    blockers = []
    if signal.side != "BUY" or signal.security_type != "STK" or signal.target_quantity <= 0 or signal.target_quantity % 1 != 0:
        blockers.append("ENTRY_INSTRUMENT_OR_SIDE_BLOCKED")
    if not context.strategy_authority_go:
        blockers.append("STRATEGY_AUTHORITY_BLOCKED")
    if not context.financial_finalist_go:
        blockers.append("FINANCIAL_FINALIST_REQUIRED")
    if not context.forward_shadow_go:
        blockers.append("FORWARD_SHADOW_GO_REQUIRED")
    if context.shariah_status != "SHARIAH_ELIGIBLE":
        blockers.append("SHARIAH_STATUS_LOST" if context.shariah_status == "SHARIAH_INELIGIBLE" else "SHARIAH_STATUS_STALE")
    if not context.market_session_open:
        blockers.append("MARKET_SESSION_CLOSED")
    signal_age = (decision_at - datetime.fromisoformat(signal.generated_at)).total_seconds()
    if (
        decision_at < datetime.fromisoformat(signal.available_at)
        or decision_at > datetime.fromisoformat(signal.expires_at)
        or signal_age < 0
        or signal_age > config.max_signal_age_seconds
    ):
        blockers.append("STALE_SIGNAL")
    quote_age = (decision_at - datetime.fromisoformat(context.quote.observed_at)).total_seconds()
    if quote_age < 0 or quote_age > config.max_quote_age_seconds:
        blockers.append("STALE_QUOTE")
    if context.quote.spread_bps > config.max_spread_bps:
        blockers.append("WIDE_SPREAD")
    if context.signal_seen:
        blockers.append("DUPLICATE_INTENT_DETECTED")
    if context.open_order_for_con_id:
        blockers.append("OPEN_ORDER_FOR_CON_ID")
    if context.existing_position_for_con_id:
        blockers.append("EXISTING_POSITION_BLOCKED")
    if context.entries_today >= config.max_new_positions_per_day:
        blockers.append("DAILY_ENTRY_LIMIT_REACHED")
    if -portfolio.daily_pnl_eur >= config.max_daily_loss_eur:
        blockers.append("DAILY_LOSS_LIMIT_REACHED")
    if len(portfolio.positions) >= config.max_open_positions:
        blockers.append("POSITION_LIMIT_REACHED")
    notional = signal.target_quantity * min(signal.maximum_limit_price, context.quote.ask)
    if notional > config.max_order_notional_eur:
        blockers.append("ORDER_NOTIONAL_LIMIT_REACHED")
    if portfolio.exposure_eur + notional > config.max_portfolio_exposure_eur:
        blockers.append("PORTFOLIO_EXPOSURE_REACHED")
    if portfolio.sector_exposure_pct.get(context.sector, Decimal("0")) > config.max_sector_exposure_pct:
        blockers.append("SECTOR_EXPOSURE_REACHED")
    if portfolio.event_cluster_exposure_pct.get(context.event_cluster, Decimal("0")) > config.max_event_cluster_exposure_pct:
        blockers.append("EVENT_CLUSTER_LIMIT_REACHED")
    if not portfolio.snapshot_complete or portfolio.reconciliation_status != "PAPER_RECONCILED_EMPTY":
        blockers.append("BROKER_RECONCILIATION_MISMATCH")
    if not context.kill_switch_clear:
        blockers.append("KILL_SWITCH_ACTIVE")
    return {
        "status": "ENTRY_RISK_GO" if not blockers else "ENTRY_RISK_BLOCKED",
        "blockers": blockers,
        "estimated_notional_eur": str(notional),
        "risk_reducing_exit_unaffected": True,
    }
