from __future__ import annotations

from dataclasses import dataclass, field


REQUIRED_SUBSCRIPTIONS = frozenset(
    {"ACCOUNT", "POSITIONS", "OPEN_ORDERS", "EXECUTIONS"}
)


@dataclass
class ConnectionRecoveryState:
    connected: bool = False
    client_id: int | None = None
    generation: int = 0
    subscriptions: set[str] = field(default_factory=set)
    reconciled: bool = False

    def connect(self, client_id: int) -> str:
        if self.connected:
            if self.client_id == client_id:
                return "CLIENT_ID_REUSE_BLOCKED"
            return "CONCURRENT_CONNECTION_BLOCKED"
        self.connected = True
        self.client_id = client_id
        self.generation += 1
        self.subscriptions.clear()
        self.reconciled = False
        return "CONNECTED_RECONCILIATION_REQUIRED"

    def disconnect(self) -> str:
        self.connected = False
        self.subscriptions.clear()
        self.reconciled = False
        return "DISCONNECTED_EXECUTION_BLOCKED"

    def subscribe(self, component: str) -> str:
        normalized = component.strip().upper()
        if not self.connected:
            return "SUBSCRIPTION_BLOCKED_DISCONNECTED"
        if normalized not in REQUIRED_SUBSCRIPTIONS:
            return "UNKNOWN_SUBSCRIPTION_BLOCKED"
        before = len(self.subscriptions)
        self.subscriptions.add(normalized)
        return (
            "SUBSCRIPTION_IDEMPOTENT"
            if len(self.subscriptions) == before
            else "SUBSCRIPTION_ACTIVE"
        )

    def mark_reconciled(self) -> str:
        if not self.connected:
            return "RECONCILIATION_BLOCKED_DISCONNECTED"
        if self.subscriptions != set(REQUIRED_SUBSCRIPTIONS):
            return "RECONCILIATION_BLOCKED_SUBSCRIPTIONS_INCOMPLETE"
        self.reconciled = True
        return "CONNECTION_READY"

    @property
    def execution_ready(self) -> bool:
        return bool(
            self.connected
            and self.reconciled
            and self.subscriptions == set(REQUIRED_SUBSCRIPTIONS)
        )


__all__ = ["ConnectionRecoveryState", "REQUIRED_SUBSCRIPTIONS"]
