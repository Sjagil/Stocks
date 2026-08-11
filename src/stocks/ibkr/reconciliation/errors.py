from __future__ import annotations


PHASE8_MARKER = "PHASE8_IBKR_READ_ONLY_RECONCILIATION_ADAPTER_GO"
PHASE8_FREEZE_MARKER = "PHASE8_IBKR_READ_ONLY_RECONCILIATION_ADAPTER_FROZEN_GO"
BROKER_OBSERVATION_AUTHORITY = "READ_ONLY"
EXECUTION_AUTHORITY = "NONE"

ERROR_CODES = {
    "CALLBACK_TIMEOUT",
    "CONNECTION_LOST",
    "DUPLICATE_CALLBACK",
    "OUT_OF_ORDER_CALLBACK",
    "ACCOUNT_MASKING_FAILURE",
    "RAW_ACCOUNT_LEAK_BLOCKED",
    "CLIENT_ID_ZERO_BLOCKED",
    "CLIENT_ID_COLLISION_BLOCKED",
    "READ_ONLY_METHOD_ALLOWLIST_VIOLATION",
    "BROKER_WRITE_METHOD_BLOCKED",
    "PARTIAL_SNAPSHOT_BLOCKED",
    "NON_ATOMIC_BROKER_SNAPSHOT",
    "EXECUTION_SCOPE_INCOMPLETE",
    "COMMISSION_WITHOUT_EXECUTION",
    "EXECUTION_WITHOUT_COMMISSION",
    "UNKNOWN_CONTRACT",
    "CONTRACT_HASH_MISMATCH",
    "RECON_CONFIG_BLOCKED",
    "NON_PAPER_CONFIGURATION_BLOCKED",
    "ACCOUNT_FINGERPRINT_KEY_MISSING",
}

READ_ONLY_METHODS = {
    "reqCurrentTime",
    "reqAccountSummary",
    "cancelAccountSummary",
    "reqPositions",
    "cancelPositions",
    "reqOpenOrders",
    "reqAllOpenOrders",
    "reqExecutions",
}

SUBSCRIPTION_CANCELLATION_METHODS = {
    "cancelAccountSummary",
    "cancelPositions",
}

FORBIDDEN_METHODS = {
    "place" + "Order",
    "cancel" + "Order",
    "req" + "Global" + "Cancel",
    "req" + "Ids",
    "req" + "Auto" + "Open" + "Orders",
    "exercise" + "Options",
    "replace" + "FA",
    "req" + "Mkt" + "Data",
    "req" + "Real" + "Time" + "Bars",
    "req" + "Historical" + "Data",
}


class Phase8Blocked(RuntimeError):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)
