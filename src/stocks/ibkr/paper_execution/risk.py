from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from stocks.execution.idempotency import economic_order_key, stable_hash
from stocks.ibkr.paper_execution.errors import (
    ACCOUNT_FINGERPRINT_MISMATCH,
    APPROVAL_REQUIRED,
    CANARY_INSTRUMENT_NOT_SUITABLE,
    DUPLICATE_INTENT,
    INVALID_QUANTITY,
    KILL_SWITCH_ACTIVE,
    LIMIT_PRICE_INVALID,
    OPEN_ORDER_LIMIT_EXCEEDED,
    ORDER_NOTIONAL_EXCEEDED,
    ORDER_TYPE_BLOCKED,
    OUTSIDE_RTH_BLOCKED,
    PAPER_RISK_APPROVED_MANUAL_CANARY,
    POSITION_LIMIT_EXCEEDED,
    RECONCILIATION_BLOCKED,
    SESSION_BLOCKED,
    SHORT_POSITION_BLOCKED,
    STALE_FX_BLOCKED,
    STRATEGY_INTENT_BLOCKED,
    TIME_IN_FORCE_BLOCKED,
    UNRESOLVED_CONTRACT,
)
from stocks.ibkr.paper_execution.models import ManualPaperIntent, PaperWriterConfig
from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def prepare_intent(
    project_root: Path,
    config: PaperWriterConfig,
    *,
    con_id: int,
    side: str,
    quantity: Decimal,
    limit_price: Decimal,
    reason: str,
) -> tuple[ManualPaperIntent, dict[str, object]]:
    now = datetime.now(timezone.utc)
    symbol = _symbol_for_con_id(con_id)
    fx_rate = Decimal("1") if symbol["currency"] == "EUR" else Decimal("0.92")
    estimated_local = quantity * limit_price
    estimated_eur = estimated_local * fx_rate * Decimal("1.10")
    key = economic_order_key(
        strategy_id="MANUAL_OPERATOR",
        strategy_version="PHASE9",
        decision_id="MANUAL_OPERATOR_INTENT",
        con_id=con_id,
        side=side,
        target_position=quantity,
        session_date=now.date().isoformat(),
    )
    intent = ManualPaperIntent(
        intent_id=f"MANUAL-PAPER-{stable_hash({'key': key, 'created_at': now.isoformat()})[:20]}",
        economic_order_key=key,
        intent_source="MANUAL_OPERATOR",
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=10)).isoformat(),
        account_fingerprint=config.approved_account_fingerprint,
        con_id=con_id,
        symbol=symbol["symbol"],
        security_type=symbol["security_type"],
        currency=symbol["currency"],
        exchange=symbol["exchange"],
        side=side,
        quantity=quantity,
        order_type="LIMIT",
        limit_price=limit_price,
        estimated_notional_local=estimated_local,
        estimated_notional_eur=estimated_eur,
        fx_rate=fx_rate,
        fx_rate_timestamp=now.isoformat(),
        session_date=now.date().isoformat(),
        outside_rth=False,
        time_in_force="DAY",
        contract_hash=stable_hash({"con_id": con_id, **symbol}),
        operator_reason=reason,
    )
    store = PaperExecutionStore(project_root / "data" / "execution" / "phase9" / "private" / "paper_execution.sqlite3")
    return intent, evaluate_risk(intent, config=config, store=store, approval_valid=False)


def evaluate_risk(
    intent: ManualPaperIntent,
    *,
    config: PaperWriterConfig,
    store: PaperExecutionStore,
    approval_valid: bool,
    existing_long_quantity: Decimal = Decimal("0"),
    open_orders: int = 0,
    open_positions: int = 0,
    new_orders_today: int = 0,
    closing_orders_today: int = 0,
    reconciliation_go: bool = True,
    kill_switch_armed: bool = True,
    session_ready: bool = True,
    duplicate_intent: bool = False,
) -> dict[str, object]:
    if intent.account_fingerprint != config.approved_account_fingerprint:
        return _blocked(ACCOUNT_FINGERPRINT_MISMATCH)
    if intent.intent_source != "MANUAL_OPERATOR":
        return _blocked(STRATEGY_INTENT_BLOCKED)
    if intent.security_type != "STK" or not intent.contract_hash:
        return _blocked(UNRESOLVED_CONTRACT)
    if not session_ready:
        return _blocked(SESSION_BLOCKED)
    if intent.outside_rth:
        return _blocked(OUTSIDE_RTH_BLOCKED)
    if intent.order_type != "LIMIT":
        return _blocked(ORDER_TYPE_BLOCKED)
    if intent.time_in_force != "DAY":
        return _blocked(TIME_IN_FORCE_BLOCKED)
    if intent.quantity <= 0 or intent.quantity != Decimal("1"):
        return _blocked(INVALID_QUANTITY if intent.quantity <= 0 else CANARY_INSTRUMENT_NOT_SUITABLE)
    if intent.side == "SELL" and existing_long_quantity <= 0:
        return _blocked("SELL_WITHOUT_POSITION_BLOCKED")
    if intent.side == "SELL" and intent.quantity > existing_long_quantity:
        return _blocked(SHORT_POSITION_BLOCKED)
    if intent.limit_price <= 0:
        return _blocked(LIMIT_PRICE_INVALID)
    if intent.estimated_notional_eur > config.max_order_notional_eur:
        return _blocked(ORDER_NOTIONAL_EXCEEDED)
    if open_orders >= config.max_open_orders:
        return _blocked(OPEN_ORDER_LIMIT_EXCEEDED)
    if open_positions >= config.max_positions and intent.side == "BUY":
        return _blocked(POSITION_LIMIT_EXCEEDED)
    if intent.side == "BUY" and new_orders_today >= config.max_new_orders_per_day:
        return _blocked("OPENING_ORDER_DAILY_LIMIT_BLOCKED")
    if intent.side == "SELL" and closing_orders_today >= config.max_closing_orders_per_day:
        return _blocked("CLOSING_ORDER_DAILY_LIMIT_BLOCKED")
    if _fx_stale(intent.fx_rate_timestamp):
        return _blocked(STALE_FX_BLOCKED)
    duplicate_owner = store.economic_order_key_owner(intent.economic_order_key)
    if duplicate_intent or (duplicate_owner is not None and duplicate_owner != intent.intent_id):
        return _blocked(DUPLICATE_INTENT)
    if not reconciliation_go:
        return _blocked(RECONCILIATION_BLOCKED)
    if not kill_switch_armed:
        return _blocked(KILL_SWITCH_ACTIVE)
    if not approval_valid:
        return _blocked(APPROVAL_REQUIRED)
    return {"status": "GO", "risk_status": PAPER_RISK_APPROVED_MANUAL_CANARY}


def evaluate_closing_sell_risk(
    intent: ManualPaperIntent,
    *,
    config: PaperWriterConfig,
    local_long_quantity: Decimal,
    broker_long_quantity: Decimal,
    broker_position_snapshot_complete: bool,
    local_position_reconciled: bool,
    same_con_id: bool = True,
    same_account_fingerprint: bool = True,
    approval_valid: bool = True,
    closing_orders_today: int = 0,
) -> dict[str, object]:
    if intent.account_fingerprint != config.approved_account_fingerprint:
        return _blocked(ACCOUNT_FINGERPRINT_MISMATCH)
    if intent.intent_source != "MANUAL_OPERATOR":
        return _blocked(STRATEGY_INTENT_BLOCKED)
    if intent.security_type != "STK" or not intent.contract_hash:
        return _blocked(UNRESOLVED_CONTRACT)
    if intent.outside_rth:
        return _blocked(OUTSIDE_RTH_BLOCKED)
    if intent.order_type != "LIMIT":
        return _blocked(ORDER_TYPE_BLOCKED)
    if intent.time_in_force != "DAY":
        return _blocked(TIME_IN_FORCE_BLOCKED)
    if intent.quantity <= 0 or intent.quantity != Decimal("1"):
        return _blocked(INVALID_QUANTITY if intent.quantity <= 0 else CANARY_INSTRUMENT_NOT_SUITABLE)
    if intent.limit_price <= 0:
        return _blocked(LIMIT_PRICE_INVALID)
    if intent.estimated_notional_eur > config.max_order_notional_eur:
        return _blocked(ORDER_NOTIONAL_EXCEEDED)
    if _fx_stale(intent.fx_rate_timestamp):
        return _blocked(STALE_FX_BLOCKED)
    if not approval_valid:
        return _blocked(APPROVAL_REQUIRED)
    if intent.side != "SELL":
        return {"status": "NO_GO", "risk_status": "CLOSING_SELL_SIDE_REQUIRED"}
    if closing_orders_today >= config.max_closing_orders_per_day:
        return {"status": "NO_GO", "risk_status": "CLOSING_ORDER_DAILY_LIMIT_BLOCKED"}
    if not broker_position_snapshot_complete:
        return {"status": "NO_GO", "risk_status": "POSITION_SNAPSHOT_INCOMPLETE"}
    if not same_con_id:
        return {"status": "NO_GO", "risk_status": "LOCAL_BROKER_POSITION_MISMATCH"}
    if not same_account_fingerprint:
        return {"status": "NO_GO", "risk_status": "LOCAL_BROKER_POSITION_MISMATCH"}
    if not local_position_reconciled or local_long_quantity != broker_long_quantity:
        return {"status": "NO_GO", "risk_status": "LOCAL_BROKER_POSITION_MISMATCH"}
    if local_long_quantity <= 0:
        return {"status": "NO_GO", "risk_status": "SELL_WITHOUT_POSITION_BLOCKED"}
    if intent.quantity > local_long_quantity or intent.quantity > broker_long_quantity:
        return {"status": "NO_GO", "risk_status": "SELL_EXCEEDS_RECONCILED_POSITION"}
    return {"status": "GO", "risk_status": "CLOSING_SELL_ALLOWED"}


def _blocked(status: str) -> dict[str, object]:
    return {"status": "NO_GO", "risk_status": status}


def _fx_stale(timestamp: str) -> bool:
    return datetime.fromisoformat(timestamp) < datetime.now(timezone.utc) - timedelta(days=1)


def _symbol_for_con_id(con_id: int) -> dict[str, str]:
    known = {
        756733: {"symbol": "SPY", "security_type": "STK", "currency": "USD", "exchange": "SMART"},
        39039301: {"symbol": "BIL", "security_type": "STK", "currency": "USD", "exchange": "SMART"},
        101484826: {"symbol": "GLD", "security_type": "STK", "currency": "USD", "exchange": "SMART"},
        8677881: {"symbol": "ON", "security_type": "STK", "currency": "USD", "exchange": "SMART"},
    }
    return known.get(con_id, {"symbol": f"CON{con_id}", "security_type": "STK", "currency": "EUR", "exchange": "SMART"})
