from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from filelock import FileLock, Timeout

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.reconciliation.adapter import BrokerObservationAdapter, zero_read_counters, zero_write_counters
from stocks.ibkr.reconciliation.callbacks import BrokerObserverState
from stocks.ibkr.reconciliation.errors import (
    BROKER_OBSERVATION_AUTHORITY,
    EXECUTION_AUTHORITY,
    Phase8Blocked,
)
from stocks.ibkr.reconciliation.models import (
    BrokerAccountSnapshot,
    BrokerExecutionSnapshot,
    BrokerObservationSnapshot,
    BrokerOpenOrderSnapshot,
    BrokerPositionSnapshot,
    ObservationScope,
    SnapshotComponentAudit,
    model_to_jsonable,
)
from stocks.ibkr.reconciliation.requests import Phase8Config


def snapshot_from_state(
    state: BrokerObserverState,
    config: Phase8Config,
    *,
    started_at: str,
    completed_at: str,
) -> BrokerObservationSnapshot:
    account_status = _required_status(state.account_summary_end.is_set())
    position_status = _required_status(state.position_end.is_set())
    same_status = _required_status(
        state.same_client_open_order_end.is_set()
    )
    all_status = _required_status(state.all_api_open_order_end.is_set())
    exec_status = _status_from_count(len(state.executions), state.exec_details_end.is_set())
    completeness = "NO_EXECUTIONS_RETURNED_WITHIN_REQUEST_SCOPE" if not state.executions else "CURRENT_SESSION_OBSERVED"
    execution = BrokerExecutionSnapshot(
        executions=tuple(state.executions),
        commissions=tuple(state.commissions),
        status=exec_status,
        execution_scope="CURRENT_AVAILABLE_TWS_SCOPE",
        request_filter="ExecutionFilter()",
        requested_from=None,
        requested_until=completed_at,
        tws_trade_log_scope_known=False,
        execution_history_complete=False,
        completeness_status=completeness,
        content_hash=stable_hash(model_to_jsonable(state.executions)),
    )
    account = BrokerAccountSnapshot(tuple(state.account_values), account_status, stable_hash(model_to_jsonable(state.account_values)))
    positions = BrokerPositionSnapshot(tuple(state.positions), position_status, stable_hash(model_to_jsonable(state.positions)))
    same = BrokerOpenOrderSnapshot(ObservationScope.SAME_CLIENT, tuple(state.same_client_open_orders), same_status, stable_hash(model_to_jsonable(state.same_client_open_orders)))
    all_api = BrokerOpenOrderSnapshot(ObservationScope.ALL_API_CLIENTS, tuple(state.all_api_open_orders), all_status, stable_hash(model_to_jsonable(state.all_api_open_orders)))
    component_audits = (
        _component("accountsummary", started_at, completed_at, len(state.account_values), account.status, account.content_hash, config),
        _component("positions", started_at, completed_at, len(state.positions), positions.status, positions.content_hash, config),
        _component("same_client_open_orders", started_at, completed_at, len(state.same_client_open_orders), same.status, same.content_hash, config),
        _component("all_api_open_orders", started_at, completed_at, len(state.all_api_open_orders), all_api.status, all_api.content_hash, config),
        _component("executions", started_at, completed_at, len(state.executions), execution.status, execution.content_hash, config),
    )
    span = Decimal(str(max(0.0, (datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds())))
    snapshot_id = stable_hash(
        {
            "started_at": started_at,
            "completed_at": completed_at,
            "account": account.content_hash,
            "positions": positions.content_hash,
            "same": same.content_hash,
            "all": all_api.content_hash,
            "executions": execution.content_hash,
        }
    )[:24]
    payload = {
        "snapshot_id": snapshot_id,
        "snapshot_started_at": started_at,
        "snapshot_completed_at": completed_at,
        "account": account.content_hash,
        "positions": positions.content_hash,
        "same": same.content_hash,
        "all": all_api.content_hash,
        "executions": execution.content_hash,
    }
    return BrokerObservationSnapshot(
        snapshot_id=snapshot_id,
        snapshot_started_at=started_at,
        snapshot_completed_at=completed_at,
        snapshot_span_seconds=span,
        component_timestamps={item.name: {"started_at": item.started_at, "completed_at": item.completed_at} for item in component_audits},
        snapshot_atomic=False,
        account=account,
        positions=positions,
        same_client_open_orders=same,
        all_api_open_orders=all_api,
        executions=execution,
        component_audits=component_audits,
        server_version=state.server_version,
        broker_observation_authority=BROKER_OBSERVATION_AUTHORITY,
        execution_authority=EXECUTION_AUTHORITY,
        content_hash=stable_hash(payload),
    )


def capture_snapshot(config: Phase8Config) -> tuple[BrokerObservationSnapshot, dict[str, int], dict[str, int]]:
    lock_path = (
        config.env_file.parent
        / "data"
        / "broker"
        / "phase8"
        / "private"
        / "observer.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock_path), timeout=0):
            started = _now()
            adapter = BrokerObservationAdapter(config)
            state = adapter.capture()
            completed = _now()
            return (
                snapshot_from_state(
                    state,
                    config,
                    started_at=started,
                    completed_at=completed,
                ),
                adapter.read_counters,
                adapter.write_counters,
            )
    except Timeout as exc:
        raise Phase8Blocked("OBSERVER_SINGLE_FLIGHT_BUSY") from exc


def stability_status(first: BrokerObservationSnapshot, second: BrokerObservationSnapshot) -> dict[str, object]:
    if not snapshot_components_complete(first) or not snapshot_components_complete(
        second
    ):
        return {
            "stability_status": "SNAPSHOT_INCOMPLETE_BLOCKED",
            "snapshot_atomic": False,
            "stable": False,
            "first_snapshot_hash": first.content_hash,
            "second_snapshot_hash": second.content_hash,
        }
    equal = (
        _position_key(first) == _position_key(second)
        and _order_key(first.same_client_open_orders.open_orders) == _order_key(second.same_client_open_orders.open_orders)
        and _order_key(first.all_api_open_orders.open_orders) == _order_key(second.all_api_open_orders.open_orders)
        and {item.execution_id for item in first.executions.executions} == {item.execution_id for item in second.executions.executions}
        and _account_fingerprints(first) == _account_fingerprints(second)
    )
    return {
        "stability_status": "BROKER_SNAPSHOT_STABLE_GO" if equal else "STATE_CHANGED_DURING_CAPTURE",
        "snapshot_atomic": False,
        "stable": equal,
        "first_snapshot_hash": first.content_hash,
        "second_snapshot_hash": second.content_hash,
    }


def snapshot_components_complete(snapshot: BrokerObservationSnapshot) -> bool:
    statuses = {
        audit.name: audit.request_status
        for audit in snapshot.component_audits
    }
    return (
        statuses.get("accountsummary") == "COMPLETE"
        and statuses.get("positions") == "COMPLETE"
        and statuses.get("same_client_open_orders") == "COMPLETE"
        and statuses.get("all_api_open_orders") == "COMPLETE"
        and statuses.get("executions") in {"COMPLETE", "EMPTY_COMPLETE"}
    )


def empty_counters() -> dict[str, int]:
    return {**zero_read_counters(), **zero_write_counters()}


def _status_from_count(count: int, completed: bool) -> str:
    if not completed:
        return "CALLBACK_TIMEOUT"
    return "COMPLETE" if count > 0 else "EMPTY_COMPLETE"


def _required_status(completed: bool) -> str:
    return "COMPLETE" if completed else "CALLBACK_TIMEOUT"


def _component(name: str, started: str, completed: str, count: int, status: str, content_hash: str, config: Phase8Config) -> SnapshotComponentAudit:
    return SnapshotComponentAudit(name, started, completed, count, Decimal(str(config.request_timeout_seconds)), status, content_hash)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _position_key(snapshot: BrokerObservationSnapshot) -> set[tuple[str, int, str]]:
    return {(item.account_fingerprint, item.con_id, str(item.position_quantity)) for item in snapshot.positions.positions}


def _order_key(orders: tuple[object, ...]) -> set[tuple[str, str]]:
    return {(getattr(item, "broker_order_id"), getattr(item, "order_status")) for item in orders}


def _account_fingerprints(snapshot: BrokerObservationSnapshot) -> set[str]:
    return {item.account_fingerprint for item in snapshot.account.values}
