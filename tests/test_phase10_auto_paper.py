from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from stocks.auto_paper.authority import AuthorityDependencies, entry_authority, foundation_authority
from stocks.auto_paper.audit import (
    COUNTERS,
    IMMUTABLE_PHASE9_ARTIFACTS,
    PRIVATE_DEPENDENCIES,
    _phase9_progression_valid,
    _selected_hashes_unchanged,
    frozen_dependency_audit,
)
from stocks.auto_paper.config import AutoPaperConfig, load_auto_paper_config
from stocks.auto_paper.contracts import AssetGroup, AutoSignal, MarketQuote, PortfolioState, Position, Regime, ShariahSnapshot
from stocks.auto_paper.entries import prepare_shadow_entry
from stocks.auto_paper.exits import evaluate_risk_reducing_exit
from stocks.auto_paper.kill_switch import evaluate_kill_switches
from stocks.auto_paper.evaluation import financial_evaluation_fixture
from stocks.auto_paper.movers_adapter import classify_candidate
from stocks.auto_paper.portfolio import BLOCKED_SLEEVES, regime_allocation
from stocks.auto_paper.privacy import scan_public_artifacts
from stocks.auto_paper.replay import replay_fixture
from stocks.auto_paper.risk import EntryRiskContext, evaluate_entry_risk
from stocks.auto_paper.scheduler import run_bounded_scheduler
from stocks.auto_paper.shariah_gate import evaluate_shariah
from stocks.auto_paper.storage import AutoPaperStore
from stocks.auto_paper.strategies import STRATEGY_IDS, evaluate_strategy
from stocks.execution.idempotency import stable_hash
from stocks.data.phase5_common import sha256_file


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def config(**changes: object) -> AutoPaperConfig:
    base = AutoPaperConfig(
        enabled=False,
        client_id=1017,
        strategy_allowlist=(STRATEGY_IDS[0],),
        product_allowlist=("ETF-1", "GOLD-1"),
        max_order_notional_eur=Decimal("50"),
        max_new_positions_per_day=1,
        max_closing_orders_per_day=4,
        max_open_positions=2,
        max_portfolio_exposure_eur=Decimal("100"),
        max_daily_loss_eur=Decimal("20"),
        max_sector_exposure_pct=Decimal("25"),
        max_event_cluster_exposure_pct=Decimal("15"),
        max_signal_age_seconds=300,
        max_quote_age_seconds=15,
        max_spread_bps=Decimal("40"),
        rth_only=True,
        limit_only=True,
        require_shariah_fresh=True,
        heartbeat_timeout_seconds=30,
    )
    return replace(base, **changes)


def signal(**changes: object) -> AutoSignal:
    base = AutoSignal(
        signal_id="SIG-1",
        strategy_id=STRATEGY_IDS[0],
        strategy_version="1",
        generated_at=NOW.isoformat(),
        available_at=NOW.isoformat(),
        expires_at=(NOW + timedelta(minutes=5)).isoformat(),
        session_date="2026-07-22",
        con_id=123,
        symbol="TEST",
        exchange="SMART",
        currency="EUR",
        security_type="STK",
        asset_group=AssetGroup.SHARIAH_STOCK,
        side="BUY",
        target_quantity=Decimal("1"),
        reference_price=Decimal("20"),
        maximum_limit_price=Decimal("21"),
        entry_reason="synthetic test",
        exit_reason=None,
        confidence=Decimal("0.8"),
        expected_holding_period="5d",
        source_provenance={"fixture": "offline"},
        source_provenance_hash=stable_hash({"fixture": "offline"}),
        feature_snapshot_hash="F" * 64,
        portfolio_snapshot_hash="P" * 64,
        shariah_snapshot_hash="S" * 64,
    )
    return replace(base, **changes)


def portfolio(**changes: object) -> PortfolioState:
    base = PortfolioState(
        nav_eur=Decimal("1000"),
        exposure_eur=Decimal("0"),
        daily_pnl_eur=Decimal("0"),
        positions=(),
        sector_exposure_pct={},
        event_cluster_exposure_pct={},
        reconciliation_status="PAPER_RECONCILED_EMPTY",
        snapshot_complete=True,
    )
    return replace(base, **changes)


def context(**changes: object) -> EntryRiskContext:
    base = EntryRiskContext(
        decision_time=NOW.isoformat(),
        market_session_open=True,
        signal_seen=False,
        shariah_status="SHARIAH_ELIGIBLE",
        strategy_authority_go=True,
        financial_finalist_go=True,
        forward_shadow_go=True,
        quote=MarketQuote(Decimal("20"), Decimal("20.02"), NOW.isoformat()),
        open_order_for_con_id=False,
        existing_position_for_con_id=False,
        entries_today=0,
        sector="TECH",
        event_cluster="EARNINGS",
        kill_switch_clear=True,
    )
    return replace(base, **changes)


def shariah(**changes: object) -> ShariahSnapshot:
    base = ShariahSnapshot(
        status="SHARIAH_ELIGIBLE",
        methodology="fixture",
        methodology_version="1",
        screened_at=NOW.isoformat(),
        financials_available_at=(NOW - timedelta(days=1)).isoformat(),
        expires_at=(NOW + timedelta(days=30)).isoformat(),
        business_activity_pass=True,
        financial_ratio_pass=True,
        non_permissible_income_pass=True,
        product_structure="COMMON_STOCK",
    )
    return replace(base, **changes)


def test_default_disabled_and_foundation_authority(tmp_path: Path) -> None:
    loaded, errors = load_auto_paper_config(tmp_path)
    assert errors == []
    assert loaded.enabled is False
    assert foundation_authority()["execution_authority"] == "NONE"
    assert foundation_authority()["automatic_submission"] is False


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ({"phase9_full_frozen_go": False}, "PHASE9_FULL_FREEZE_REQUIRED"),
        ({"financial_finalist_go": False}, "FINANCIAL_FINALIST_REQUIRED"),
        ({"forward_shadow_go": False}, "FORWARD_SHADOW_GO_REQUIRED"),
    ],
)
def test_authority_dependencies_fail_closed(change: dict[str, bool], blocker: str) -> None:
    deps = AuthorityDependencies(True, True, True, True, True, True, True, True, True, True)
    result = entry_authority(config(enabled=True), replace(deps, **change), STRATEGY_IDS[0])
    assert blocker in result["blockers"]
    assert result["execution_authority"] == "NONE"


def test_strategy_allowlist_blocks() -> None:
    deps = AuthorityDependencies(True, True, True, True, True, True, True, True, True, True)
    result = entry_authority(config(enabled=True, strategy_allowlist=()), deps, STRATEGY_IDS[0])
    assert "STRATEGY_NOT_ALLOWLISTED" in result["blockers"]


def test_complete_entry_authority_contract_is_explicit() -> None:
    deps = AuthorityDependencies(True, True, True, True, True, True, True, True, True, True)
    result = entry_authority(config(enabled=True), deps, STRATEGY_IDS[0])
    assert result["status"] == "GO"
    assert result["authority_type"] == "AUTOMATED_PAPER_ENTRY"
    assert result["automatic_submission"] is True


def test_shariah_stock_etf_and_blocked_products() -> None:
    stock = evaluate_shariah(
        security_type="STK", asset_group=AssetGroup.SHARIAH_STOCK, product_id="STOCK",
        snapshot=shariah(), product_allowlist=(), decision_time=NOW.isoformat(),
    )
    stale = evaluate_shariah(
        security_type="STK", asset_group=AssetGroup.SHARIAH_STOCK, product_id="STOCK",
        snapshot=shariah(expires_at=(NOW - timedelta(seconds=1)).isoformat()),
        product_allowlist=(), decision_time=NOW.isoformat(),
    )
    etf = evaluate_shariah(
        security_type="STK", asset_group=AssetGroup.APPROVED_SHARIAH_EQUITY_ETF, product_id="ETF-X",
        snapshot=shariah(product_structure="PHYSICAL_EQUITY_ETF", shariah_certificate=True),
        product_allowlist=(), decision_time=NOW.isoformat(),
    )
    assert stock["status"] == "SHARIAH_ELIGIBLE"
    assert stale["status"] == "SHARIAH_STATUS_STALE"
    assert etf["status"] == "SHARIAH_MANUAL_REVIEW_REQUIRED"
    for security_type in ("FUT", "OPT", "BOND", "CFD", "SWAP"):
        result = evaluate_shariah(
            security_type=security_type, asset_group=AssetGroup.SHARIAH_STOCK, product_id="X",
            snapshot=shariah(), product_allowlist=(), decision_time=NOW.isoformat(),
        )
        assert result["status"] == "SHARIAH_PRODUCT_STRUCTURE_BLOCKED"


def test_shariah_ineligible_and_allowlisted_product_require_complete_structure() -> None:
    ineligible = evaluate_shariah(
        security_type="STK", asset_group=AssetGroup.SHARIAH_STOCK, product_id="STOCK",
        snapshot=shariah(status="SHARIAH_INELIGIBLE"), product_allowlist=(), decision_time=NOW.isoformat(),
    )
    incomplete_etf = evaluate_shariah(
        security_type="STK", asset_group=AssetGroup.APPROVED_SHARIAH_EQUITY_ETF, product_id="ETF-1",
        snapshot=shariah(product_structure="PHYSICAL_EQUITY_ETF", shariah_certificate=True),
        product_allowlist=("ETF-1",), decision_time=NOW.isoformat(),
    )
    eligible_etf = evaluate_shariah(
        security_type="STK", asset_group=AssetGroup.APPROVED_SHARIAH_EQUITY_ETF, product_id="ETF-1",
        snapshot=shariah(product_structure="PHYSICAL_EQUITY_ETF", underlying_assets=("STOCK-BASKET",), shariah_certificate=True),
        product_allowlist=("ETF-1",), decision_time=NOW.isoformat(),
    )
    assert ineligible["status"] == "SHARIAH_INELIGIBLE"
    assert incomplete_etf["status"] == "SHARIAH_DATA_INCOMPLETE"
    assert eligible_etf["status"] == "SHARIAH_ELIGIBLE"


@pytest.mark.parametrize(
    ("ctx_changes", "portfolio_changes", "expected"),
    [
        ({"signal_seen": True}, {}, "DUPLICATE_INTENT_DETECTED"),
        ({"quote": MarketQuote(Decimal("20"), Decimal("20.02"), (NOW - timedelta(seconds=16)).isoformat())}, {}, "STALE_QUOTE"),
        ({"quote": MarketQuote(Decimal("20"), Decimal("21"), NOW.isoformat())}, {}, "WIDE_SPREAD"),
        ({"entries_today": 1}, {}, "DAILY_ENTRY_LIMIT_REACHED"),
        ({}, {"daily_pnl_eur": Decimal("-20")}, "DAILY_LOSS_LIMIT_REACHED"),
        ({}, {"positions": (Position(1, Decimal("1"), Decimal("10"), "A", "A"), Position(2, Decimal("1"), Decimal("10"), "B", "B"))}, "POSITION_LIMIT_REACHED"),
        ({}, {"exposure_eur": Decimal("90")}, "PORTFOLIO_EXPOSURE_REACHED"),
        ({}, {"sector_exposure_pct": {"TECH": Decimal("26")}}, "SECTOR_EXPOSURE_REACHED"),
        ({}, {"event_cluster_exposure_pct": {"EARNINGS": Decimal("16")}}, "EVENT_CLUSTER_LIMIT_REACHED"),
    ],
)
def test_entry_risk_blocks(ctx_changes: dict[str, object], portfolio_changes: dict[str, object], expected: str) -> None:
    result = evaluate_entry_risk(signal(), portfolio(**portfolio_changes), config(), context(**ctx_changes))
    assert expected in result["blockers"]


def test_expired_signal_blocks() -> None:
    result = evaluate_entry_risk(
        signal(expires_at=(NOW - timedelta(seconds=1)).isoformat()), portfolio(), config(), context()
    )
    assert "STALE_SIGNAL" in result["blockers"]


def test_signal_age_and_future_quote_fail_closed() -> None:
    old = signal(generated_at=(NOW - timedelta(seconds=301)).isoformat())
    future_quote = MarketQuote(Decimal("20"), Decimal("20.02"), (NOW + timedelta(seconds=1)).isoformat())
    assert "STALE_SIGNAL" in evaluate_entry_risk(old, portfolio(), config(), context())["blockers"]
    assert "STALE_QUOTE" in evaluate_entry_risk(signal(), portfolio(), config(), context(quote=future_quote))["blockers"]


def test_one_shadow_buy_is_idempotent(tmp_path: Path) -> None:
    store = AutoPaperStore(tmp_path / "auto.sqlite3")
    store.initialize()
    risk = evaluate_entry_risk(signal(), portfolio(), config(), context())
    first = prepare_shadow_entry(store, signal(), account_fingerprint="MASKED", risk=risk)
    second = prepare_shadow_entry(store, signal(), account_fingerprint="MASKED", risk=risk)
    assert first.status == "GO"
    assert first.authority == "NONE"
    assert first.hypothetical_order["broker_submission"] is False
    assert second.status == "BLOCKED"
    assert second.decision_code == "SIGNAL_DUPLICATE"
    assert store.count("hypothetical_orders") == 1


def test_provenance_hash_mismatch_is_not_registered(tmp_path: Path) -> None:
    store = AutoPaperStore(tmp_path / "auto.sqlite3")
    store.initialize()
    altered = signal(source_provenance_hash="0" * 64)
    decision = prepare_shadow_entry(
        store,
        altered,
        account_fingerprint="MASKED",
        risk=evaluate_entry_risk(altered, portfolio(), config(), context()),
    )
    assert decision.status == "BLOCKED"
    assert decision.decision_code == "SIGNAL_DATA_STALE"
    assert store.count("signals") == 0


def test_risk_reducing_sell_and_safety_rules() -> None:
    valid = evaluate_risk_reducing_exit(
        con_id=123, local_con_id=123, broker_con_id=123, sell_quantity=Decimal("1"),
        local_quantity=Decimal("1"), broker_quantity=Decimal("1"), account_match=True,
        snapshot_complete=True, reason="HARD_STOP_LOSS", limit_price=Decimal("19"), entries_today=99,
    )
    too_large = evaluate_risk_reducing_exit(
        con_id=123, local_con_id=123, broker_con_id=123, sell_quantity=Decimal("2"),
        local_quantity=Decimal("1"), broker_quantity=Decimal("1"), account_match=True,
        snapshot_complete=True, reason="HARD_STOP_LOSS", limit_price=Decimal("19"),
    )
    mismatch = evaluate_risk_reducing_exit(
        con_id=123, local_con_id=123, broker_con_id=456, sell_quantity=Decimal("1"),
        local_quantity=Decimal("1"), broker_quantity=Decimal("1"), account_match=True,
        snapshot_complete=True, reason="HARD_STOP_LOSS", limit_price=Decimal("19"),
    )
    assert valid["status"] == "EXIT_RISK_REDUCING_ALLOWED"
    assert valid["authority_type"] == "AUTOMATED_PAPER_RISK_REDUCING_EXIT"
    assert valid["execution_authority"] == "NONE"
    assert valid["automatic_submission"] is False
    assert valid["entry_count_ignored"] == 99
    assert too_large["status"] == "SELL_EXCEEDS_RECONCILED_POSITION"
    assert mismatch["status"] == "EXIT_BLOCKED_POSITION_MISMATCH"


def test_sell_without_position_cannot_create_short() -> None:
    result = evaluate_risk_reducing_exit(
        con_id=123, local_con_id=123, broker_con_id=123, sell_quantity=Decimal("1"),
        local_quantity=Decimal("0"), broker_quantity=Decimal("0"), account_match=True,
        snapshot_complete=True, reason="HARD_STOP_LOSS", limit_price=Decimal("19"),
    )
    assert result["status"] == "SELL_WITHOUT_POSITION_BLOCKED"


def test_replay_restart_duplicate_and_late_commission(tmp_path: Path) -> None:
    store = AutoPaperStore(tmp_path / "replay.sqlite3")
    store.initialize()
    result = replay_fixture(store)
    assert result["status"] == "REPLAY_GO"
    assert result["partial_fill"] is True
    assert result["duplicate_callback"] is True
    assert result["late_commission"] is True
    assert result["restart_recovery"] is True
    assert result["disconnect_recovery"] == "BOUNDED_RECONNECT_FIXTURE_GO"
    assert result["brokerwrite_calls"] == 0
    second_run = replay_fixture(store, run_id="SYNTHETIC-RUN-2")
    assert second_run["status"] == "REPLAY_GO"


def test_kill_switches_block_unknown_broker_state() -> None:
    result = evaluate_kill_switches({"UNKNOWN_BROKER_ORDER": True, "UNKNOWN_EXECUTION": True})
    assert result["status"] == "KILL_SWITCH_ACTIVE"
    assert result["new_entries_allowed"] is False
    assert result["automatic_state_corrections"] == 0


def test_scheduler_is_bounded_without_busy_loop() -> None:
    result = run_bounded_scheduler(start_time=NOW.isoformat(), interval_seconds=1, max_iterations=2_000)
    assert result["interval_seconds"] == 60
    assert result["iterations_completed"] == 1_440
    assert result["bounded"] is True
    assert result["busy_loop"] is False
    assert result["automatic_submission"] is False
    assert result["records"][0]["tasks"] == [
        "REFRESH_UNIVERSE",
        "REFRESH_SHARIAH_STATE",
        "REFRESH_FUNDAMENTALS",
        "REFRESH_NEWS",
        "CALCULATE_MOVERS",
    ]


def test_strategies_and_portfolio_remain_shadow_only() -> None:
    for strategy_id in STRATEGY_IDS:
        result = evaluate_strategy(strategy_id, {})
        assert result["status"] in {"REJECTED", "WATCHLIST", "SHADOW_CANDIDATE"}
        assert result["automatic_paper_eligibility"] is False
    for regime in Regime:
        assert sum(regime_allocation(regime).values()) == Decimal("1")
    assert set(BLOCKED_SLEEVES) == {"BONDS", "CONVENTIONAL_FIXED_INCOME", "FUTURES", "OPTIONS"}


def test_mover_pipeline_requires_shariah_and_rejects_impairment() -> None:
    candidate = {
        "mover_type": "TOP_GAINERS",
        "shariah_eligible": True,
        "liquid": True,
        "news_attributed": True,
        "fundamentals_available": True,
        "technical_acceptance": True,
        "event_cluster_known": True,
    }
    assert classify_candidate(candidate)["status"] == "MOVER_CANDIDATE_ACCEPTED"
    assert classify_candidate(candidate | {"shariah_eligible": False})["status"] == "MOVER_CANDIDATE_REJECTED"
    assert classify_candidate(candidate | {"permanent_impairment": True})["status"] == "MOVER_CANDIDATE_REJECTED"


def test_financial_evaluation_uses_no_synthetic_positive_evidence() -> None:
    result = financial_evaluation_fixture()
    assert result["provider_capability_status"] == "PROVIDER_CAPABILITY_BLOCKED"
    assert result["evidence_status"] == "PIT_DATA_INCOMPLETE"
    assert result["synthetic_positive_evidence_used"] is False
    assert result["FINANCIAL_FINALIST_GO"] is False
    assert all(row["event_study_observations"] == 0 for row in result["strategies"])


def test_privacy_and_secret_scan_clean(tmp_path: Path) -> None:
    (tmp_path / "public.json").write_text('{"status":"GO","count":1}', encoding="utf-8")
    result = scan_public_artifacts(tmp_path)
    assert result["status"] == "PRIVACY_GO"
    assert all(count == 0 for count in result["matches"].values())


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        ('{"account":"DU123456"}', "raw_account"),
        ('{"account_fingerprint":"forbidden"}', "account_fingerprint"),
        ('{"broker_order_id":123}', "broker_identifier"),
        ('{"approval_challenge":"forbidden"}', "approval_challenge"),
        ('{"provider_secret":"forbidden"}', "secret"),
    ],
)
def test_privacy_scan_detects_forbidden_public_fields(tmp_path: Path, payload: str, category: str) -> None:
    (tmp_path / "public.json").write_text(payload, encoding="utf-8")
    result = scan_public_artifacts(tmp_path)
    assert result["status"] == "PRIVACY_BLOCKED"
    assert result["matches"][category] > 0


def test_phase10_source_has_no_brokerwrite_invocation() -> None:
    source_root = Path(__file__).parents[1] / "src" / "stocks" / "auto_paper"
    joined = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
    for invocation in (".placeOrder(", ".cancelOrder(", ".reqGlobalCancel(", ".reqIds(", "from ibapi", "EClient("):
        assert invocation not in joined
    assert all(value == 0 for value in COUNTERS.values())


def test_frozen_dependency_audit_is_read_only() -> None:
    project_root = Path(__file__).parents[1]
    protected = PRIVATE_DEPENDENCIES | IMMUTABLE_PHASE9_ARTIFACTS
    before = {name: sha256_file(project_root / relative) for name, relative in protected.items()}
    assert all(before.values())
    assert frozen_dependency_audit(project_root)["status"] == "GO"
    after = {name: sha256_file(project_root / relative) for name, relative in protected.items()}
    assert after == before


@pytest.mark.parametrize(
    ("status", "fill", "close", "blockers", "expected"),
    [
        (
            "NO_GO",
            False,
            False,
            ["fill_canary", "closing_sell_canary"],
            True,
        ),
        (
            "NO_GO",
            True,
            False,
            ["closing_sell_canary"],
            True,
        ),
        (
            "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO",
            True,
            True,
            [],
            True,
        ),
        ("NO_GO", True, False, [], False),
        ("NO_GO", False, True, ["fill_canary"], False),
    ],
)
def test_phase10_accepts_only_monotonic_phase9_progress(
    status: str,
    fill: bool,
    close: bool,
    blockers: list[str],
    expected: bool,
) -> None:
    payload = {
        "status": status,
        "checks": {
            "submit_cancel_canary": True,
            "fill_canary": fill,
            "closing_sell_canary": close,
        },
        "open_blockers": blockers,
        "execution_authority": "NONE",
        "strategy_authority": "NONE",
        "shadow_authority": "NONE",
        "live_authority": "NONE",
    }

    assert _phase9_progression_valid(payload) is expected


def test_phase10_freeze_ignores_only_declared_append_only_hash_changes() -> None:
    before = {
        "immutable": "A",
        "append_only": "B",
    }
    after = {
        "immutable": "A",
        "append_only": "C",
    }

    assert _selected_hashes_unchanged(
        before,
        after,
        {"immutable"},
    )
    assert not _selected_hashes_unchanged(
        before,
        {**after, "immutable": "CHANGED"},
        {"immutable"},
    )
