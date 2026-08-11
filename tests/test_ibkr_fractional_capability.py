from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from stocks.ibkr.fractional_capability import (
    FractionalProbeSettings,
    _parse_error_args,
    probe_fractional_contract_capability,
)


class FakeApp:
    def __init__(
        self,
        details: list[Any],
        *,
        connected: bool = True,
        ready: bool = True,
        complete: bool = True,
    ) -> None:
        self.details = details
        self.connected = connected
        self.ready = ready
        self.complete = complete
        self.errors: list[dict[str, Any]] = []
        self.requests = 0
        self.closed = False

    def connect(self, host: str, port: int, client_id: int) -> None:
        return None

    def is_connected(self) -> bool:
        return self.connected

    def start(self) -> None:
        return None

    def wait_ready(self, timeout_seconds: float) -> bool:
        return self.ready

    def request_contract_details(self, request_id: int, contract: Any) -> None:
        self.requests += 1

    def wait_complete(self, timeout_seconds: float) -> bool:
        return self.complete

    def server_version(self) -> int:
        return 200

    def close(self) -> None:
        self.closed = True


def _details(
    *,
    con_id: int = 265598,
    min_size: Any = Decimal("0.0001"),
    increment: Any = Decimal("0.0001"),
    suggested: Any = Decimal("0.01"),
) -> Any:
    return SimpleNamespace(
        contract=SimpleNamespace(conId=con_id),
        minSize=min_size,
        sizeIncrement=increment,
        suggestedSizeIncrement=suggested,
    )


def _run(tmp_path: Path, app: FakeApp, **settings: Any) -> dict[str, Any]:
    return probe_fractional_contract_capability(
        tmp_path,
        settings=FractionalProbeSettings(
            host="127.0.0.1",
            port=7496,
            client_id=settings.get("client_id", 3093),
            timeout_seconds=settings.get("timeout_seconds", 1.0),
        ),
        con_id=265598,
        symbol="AAPL",
        currency="USD",
        reserved_client_ids=settings.get("reserved_client_ids", {91, 92, 93}),
        app_factory=lambda: app,
    )


def test_fractional_increment_is_observed_without_granting_writer_authority(
    tmp_path: Path,
) -> None:
    app = FakeApp([_details()])
    report = _run(tmp_path, app)

    assert report["status"] == "GO_CONTRACT_METADATA_ONLY"
    assert report["classification"] == "CONTRACT_FRACTIONAL_INCREMENT_OBSERVED"
    assert report["contract_fractional_increment_observed"] is True
    assert report["account_fractional_permission_proven"] is False
    assert report["fractional_bracket_support_proven"] is False
    assert report["fractional_writer_activation_allowed"] is False
    assert report["execution_authority"] == "NONE"
    assert report["forbidden_write_counters"]["place_order_calls"] == 0
    assert app.requests == 1
    assert app.closed is True


def test_whole_share_metadata_remains_no_go(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        FakeApp([_details(min_size=1, increment=1, suggested=1)]),
    )

    assert report["classification"] == "CONTRACT_WHOLE_SHARE_ONLY_OBSERVED"
    assert report["status"] == "NO_GO"


def test_unset_ibapi_decimal_metadata_is_unproven(tmp_path: Path) -> None:
    unset = Decimal("170141183460469231731687303715884105727")
    report = _run(
        tmp_path,
        FakeApp([_details(min_size=unset, increment=unset, suggested=unset)]),
    )

    assert report["classification"] == "CONTRACT_FRACTIONAL_CAPABILITY_UNPROVEN"
    assert report["contract_size_metadata"]["size_increment"] is None


def test_callback_timeout_is_bounded_and_closes_connection(tmp_path: Path) -> None:
    app = FakeApp([], complete=False)
    report = _run(tmp_path, app)

    assert report["status"] == "PROBE_CALLBACK_TIMEOUT"
    assert report["read_only_request_counters"]["contract_details_requests"] == 1
    assert app.closed is True


def test_ambiguous_exact_contract_details_are_blocked(tmp_path: Path) -> None:
    report = _run(tmp_path, FakeApp([_details(), _details()]))

    assert report["status"] == "AMBIGUOUS_CONTRACT_BLOCKED"
    assert report["contract_match_count"] == 2


def test_reserved_client_id_blocks_before_connect(tmp_path: Path) -> None:
    app = FakeApp([_details()])
    report = _run(
        tmp_path,
        app,
        client_id=93,
        reserved_client_ids={91, 92, 93},
    )

    assert report["status"] == "PROBE_PREFLIGHT_BLOCKED"
    assert "PROBE_CLIENT_ID_COLLIDES_WITH_CONFIGURED_CLIENT" in report["blockers"]
    assert app.requests == 0
    assert report["account_ids_stored"] == 0
    assert report["credentials_stored"] == 0


def test_error_parser_supports_current_and_legacy_ibapi_signatures() -> None:
    assert _parse_error_args((1786132738292, 2104, "farm ready")) == (
        2104,
        "farm ready",
    )
    assert _parse_error_args((2104, "farm ready")) == (2104, "farm ready")
