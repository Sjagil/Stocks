from __future__ import annotations


class ExecutionError(Exception):
    """Base error for offline execution-control-plane failures."""


class AuthorityNotGranted(ExecutionError):
    """Raised when non-NONE authority is requested in Phase 7."""


class IdempotencyConflict(ExecutionError):
    """Raised when an economic order key is reused with different payload."""


class InvalidStateTransition(ExecutionError):
    """Raised when an order event violates the state machine."""

