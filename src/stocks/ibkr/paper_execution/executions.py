from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def record_execution(store: PaperExecutionStore, *, exec_identity: str, intent_id: str, quantity: Decimal, submitted_quantity: Decimal, already_filled: Decimal) -> dict[str, object]:
    if already_filled + quantity > submitted_quantity:
        return {"status": "NO_GO", "execution_status": "FILLED_QUANTITY_EXCEEDS_SUBMITTED"}
    status = store.append_execution_once(exec_identity, intent_id, {"quantity": str(quantity), "submitted_quantity": str(submitted_quantity)})
    return {"status": "GO", "execution_status": status}


@dataclass(frozen=True)
class FillExecution:
    exec_id: str
    intent_id: str
    account_fingerprint: str
    perm_id: str
    broker_order_id: str
    con_id: int
    symbol: str
    currency: str
    side: str
    quantity: Decimal
    price: Decimal
    execution_time: str
    submitted_quantity: Decimal
    fx_rate: Decimal = Decimal("1")

    def payload(self) -> dict[str, Any]:
        return {
            "exec_id": self.exec_id,
            "fill_fingerprint": fill_fingerprint(self),
            "intent_id": self.intent_id,
            "account_fingerprint_hash": stable_hash(self.account_fingerprint),
            "perm_id_hash": stable_hash(self.perm_id),
            "broker_order_id_hash": stable_hash(self.broker_order_id),
            "con_id": self.con_id,
            "symbol": self.symbol,
            "currency": self.currency,
            "side": self.side,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "execution_time": self.execution_time,
            "submitted_quantity": str(self.submitted_quantity),
            "fx_rate": str(self.fx_rate),
        }


@dataclass(frozen=True)
class PositionProjection:
    account_fingerprint: str
    con_id: int
    symbol: str
    currency: str
    long_quantity: Decimal
    average_cost_local: Decimal
    average_cost_eur: Decimal
    realized_pnl_local: Decimal
    realized_pnl_eur: Decimal
    commission_local: Decimal
    commission_eur: Decimal
    cash_impact_local: Decimal
    cash_impact_eur: Decimal
    last_execution_id: str | None
    last_updated_at: str | None
    position_status: str
    state_hash: str

    def public_payload(self) -> dict[str, Any]:
        return {
            "account_fingerprint_hash": stable_hash(self.account_fingerprint),
            "con_id": self.con_id,
            "symbol": self.symbol,
            "currency": self.currency,
            "long_quantity": str(self.long_quantity),
            "average_cost_local": str(self.average_cost_local),
            "average_cost_eur": str(self.average_cost_eur),
            "realized_pnl_local": str(self.realized_pnl_local),
            "realized_pnl_eur": str(self.realized_pnl_eur),
            "commission_local": str(self.commission_local),
            "commission_eur": str(self.commission_eur),
            "cash_impact_local": str(self.cash_impact_local),
            "cash_impact_eur": str(self.cash_impact_eur),
            "last_execution_hash": None if self.last_execution_id is None else stable_hash(self.last_execution_id),
            "last_updated_at": self.last_updated_at,
            "position_status": self.position_status,
            "state_hash": self.state_hash,
        }


def fill_fingerprint(fill: FillExecution) -> str:
    return stable_hash(
        {
            "account_fingerprint": fill.account_fingerprint,
            "perm_id": fill.perm_id,
            "broker_order_id": fill.broker_order_id,
            "con_id": fill.con_id,
            "side": fill.side,
            "quantity": str(fill.quantity),
            "price": str(fill.price),
            "execution_time": fill.execution_time,
        }
    )


def record_fill_execution(store: PaperExecutionStore, fill: FillExecution) -> dict[str, object]:
    if not fill.exec_id:
        return {"status": "NO_GO", "execution_status": "EXECUTION_ID_MISSING_BLOCKED"}
    if fill.quantity <= 0 or fill.quantity > fill.submitted_quantity:
        return {"status": "NO_GO", "execution_status": "FILLED_QUANTITY_EXCEEDS_SUBMITTED"}
    existing_rows = store.list_executions()
    existing_same_id = next(
        (
            row
            for row in existing_rows
            if str(row["exec_identity"]) == fill.exec_id
        ),
        None,
    )
    if existing_same_id is None:
        already_filled = sum(
            (
                Decimal(str(row["payload"].get("quantity", "0")))
                for row in existing_rows
                if row["payload"].get("intent_id") == fill.intent_id
                and str(row["payload"].get("side", "")).upper()
                == fill.side.upper()
            ),
            Decimal("0"),
        )
        if already_filled + fill.quantity > fill.submitted_quantity:
            return {
                "status": "NO_GO",
                "execution_status": "FILLED_QUANTITY_EXCEEDS_SUBMITTED",
            }
    status = store.append_execution_once(fill.exec_id, fill.intent_id, fill.payload())
    mapped = {
        "EXECUTION_RECORDED": "EXECUTION_ACCEPTED",
        "DUPLICATE_EXECUTION_IGNORED": "IDEMPOTENT_REPLAY",
        "EXECUTION_CONFLICT_BLOCKED": "EXECUTION_CONFLICT_BLOCKED",
    }.get(status, status)
    if mapped == "EXECUTION_ACCEPTED":
        store.append_event(fill.intent_id, "FILL_EXECUTION_ACCEPTED", {"execution_hash": stable_hash(fill.exec_id), "fill_fingerprint": fill_fingerprint(fill)})
        if fill.side.upper() == "BUY":
            total_filled = sum(
                (
                    Decimal(str(row["payload"].get("quantity", "0")))
                    for row in store.list_executions()
                    if row["payload"].get("intent_id") == fill.intent_id
                    and str(row["payload"].get("side", "")).upper()
                    == "BUY"
                ),
                Decimal("0"),
            )
            store.apply_fill_to_capital_reservation(
                fill.intent_id,
                filled_notional_eur=(
                    fill.quantity * fill.price * fill.fx_rate
                ),
                order_complete=total_filled >= fill.submitted_quantity,
            )
        elif fill.side.upper() == "SELL":
            projection = project_position_from_store(store)
            position = projection.get("position", {})
            if (
                projection.get("status") == "GO"
                and position.get("position_status") == "CLOSED"
            ):
                released = store.release_capital_for_con_id(
                    fill.con_id, reason="CANONICAL_POSITION_CLOSED"
                )
                store.append_event(
                    fill.intent_id,
                    "CANONICAL_POSITION_EPISODE_CLOSED",
                    {
                        "canonical_fill_evidence": True,
                        "position_state_hash": position.get("state_hash"),
                        "released_reservation_count": released,
                    },
                )
    return {"status": "GO" if mapped != "EXECUTION_CONFLICT_BLOCKED" else "NO_GO", "execution_status": mapped}


def project_position_from_store(store: PaperExecutionStore) -> dict[str, Any]:
    return project_position(store.list_executions(), store.list_commissions())


def project_position(execution_rows: list[dict[str, Any]], commission_rows: list[dict[str, Any]]) -> dict[str, Any]:
    commissions = _commissions_by_execution(commission_rows)
    pending_commissions = []
    orphan_commissions = [row for row in commission_rows if row["payload"].get("exec_identity") not in {item["exec_identity"] for item in execution_rows}]
    quantity = Decimal("0")
    average_cost = Decimal("0")
    realized = Decimal("0")
    commission_total = Decimal("0")
    cash = Decimal("0")
    meta = {
        "account_fingerprint": "PHASE9_OFFLINE_ACCOUNT",
        "con_id": 0,
        "symbol": "UNKNOWN",
        "currency": "USD",
        "fx_rate": Decimal("1"),
        "last_execution_id": None,
        "last_updated_at": None,
    }
    duplicate_economic_fills = 0
    negative_position_prevented = 0
    seen_fingerprints: set[str] = set()
    for row in execution_rows:
        payload = row["payload"]
        fingerprint = str(payload.get("fill_fingerprint", row["payload_hash"]))
        if fingerprint in seen_fingerprints:
            duplicate_economic_fills += 1
            continue
        seen_fingerprints.add(fingerprint)
        side = str(payload["side"]).upper()
        fill_quantity = Decimal(str(payload["quantity"]))
        price = Decimal(str(payload["price"]))
        fx_rate = Decimal(str(payload.get("fx_rate", "1")))
        execution_commission = commissions.get(str(payload["exec_id"]))
        if execution_commission is None:
            pending_commissions.append(str(payload["exec_id"]))
            execution_commission = Decimal("0")
        commission_total += execution_commission
        meta = {
            "account_fingerprint": "HASHED_ACCOUNT",
            "con_id": int(payload["con_id"]),
            "symbol": str(payload["symbol"]),
            "currency": str(payload["currency"]),
            "fx_rate": fx_rate,
            "last_execution_id": str(payload["exec_id"]),
            "last_updated_at": row["created_at"],
        }
        if side == "BUY":
            new_quantity = quantity + fill_quantity
            average_cost = ((quantity * average_cost) + (fill_quantity * price) + execution_commission) / new_quantity
            quantity = new_quantity
            cash -= (fill_quantity * price) + execution_commission
        elif side == "SELL":
            if fill_quantity > quantity:
                negative_position_prevented += 1
                return {
                    "status": "NO_GO",
                    "projection_status": "NEGATIVE_POSITION_BLOCKED",
                    "negative_position_prevented": negative_position_prevented,
                    "duplicate_economic_fills": duplicate_economic_fills,
                    "orphan_commissions": len(orphan_commissions),
                }
            cost_basis = average_cost * fill_quantity
            proceeds = fill_quantity * price
            realized += proceeds - cost_basis - execution_commission
            quantity -= fill_quantity
            cash += proceeds - execution_commission
            if quantity == 0:
                average_cost = Decimal("0")
        else:
            return {"status": "NO_GO", "projection_status": "UNKNOWN_SIDE_BLOCKED"}
    fx_rate = Decimal(str(meta["fx_rate"]))
    status = "GO"
    projection_status = "POSITION_PROJECTED"
    if pending_commissions:
        projection_status = "RECONCILIATION_PENDING_COMMISSION"
    position_status = "CLOSED" if quantity == 0 and execution_rows else "PARTIALLY_OPEN" if quantity > 0 else "EMPTY"
    if quantity < 0:
        status = "NO_GO"
        projection_status = "NEGATIVE_POSITION_BLOCKED"
    state_base = {
        "con_id": meta["con_id"],
        "symbol": meta["symbol"],
        "long_quantity": str(quantity),
        "average_cost_local": str(average_cost),
        "realized_pnl_local": str(realized),
        "commission_local": str(commission_total),
        "cash_impact_local": str(cash),
        "position_status": position_status,
    }
    projection = PositionProjection(
        account_fingerprint=str(meta["account_fingerprint"]),
        con_id=int(str(meta["con_id"])),
        symbol=str(meta["symbol"]),
        currency=str(meta["currency"]),
        long_quantity=quantity,
        average_cost_local=average_cost,
        average_cost_eur=average_cost * fx_rate,
        realized_pnl_local=realized,
        realized_pnl_eur=realized * fx_rate,
        commission_local=commission_total,
        commission_eur=commission_total * fx_rate,
        cash_impact_local=cash,
        cash_impact_eur=cash * fx_rate,
        last_execution_id=None if meta["last_execution_id"] is None else str(meta["last_execution_id"]),
        last_updated_at=None if meta["last_updated_at"] is None else str(meta["last_updated_at"]),
        position_status=position_status,
        state_hash=stable_hash(state_base),
    )
    return {
        "status": status,
        "projection_status": projection_status,
        "position": projection.public_payload(),
        "pending_commission_count": len(pending_commissions),
        "orphan_commissions": len(orphan_commissions),
        "duplicate_economic_fills": duplicate_economic_fills,
        "negative_position_prevented": negative_position_prevented,
        "order_level_cash_impact_status": "ORDER_LEVEL_CASH_IMPACT_RECONCILED" if not pending_commissions else "ACCOUNT_CASH_SCOPE_INCOMPLETE",
    }


def _commissions_by_execution(commission_rows: list[dict[str, Any]]) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for row in commission_rows:
        payload = row["payload"]
        exec_id = str(payload.get("exec_identity", ""))
        amount = Decimal(str(payload.get("amount", "0")))
        if exec_id:
            out[exec_id] = amount
    return out
