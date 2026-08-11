from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from stocks.live.level_one_reauthorization import (
    AUTHORITATIVE_SOURCES,
    build_p02_freeze,
    build_prepare_artifact,
    policy_integrity,
    rank_manual_review_candidates,
    validate_prepare_artifact,
    verify_p02_freeze,
)
from stocks.live.integrity import normalized_file_hash


PROJECT_ROOT = Path(__file__).parents[1]


def _candidate(**overrides):
    values = {
        "candidate_id": "C-1",
        "strategy_id": "S-1",
        "symbol": "TEST",
        "con_id": 123,
        "asset_class": "STOCK",
        "entry_price": "70",
        "stop_price": "65",
        "target_price": "85",
        "share_price_eur": "70",
        "desired_qty": "2",
        "normal_qty": "2",
        "canary_qty": "1",
        "fractional_allowed": False,
        "strategy_authorized": True,
        "shariah_allowed": True,
        "contract_resolved": True,
        "economics_go": True,
        "risk_go": True,
        "liquidity_go": True,
        "expected_net_return": "0.08",
        "risk_adjusted_opportunity": "0.8",
        "validation_quality": "0.9",
        "diversification": "0.7",
        "liquidity": "0.9",
        "regime_fit": "0.8",
        "event_risk": "0.1",
        "account_snapshot_hash": "SNAPSHOT",
        "account_fingerprint": "ACCOUNT",
        "market_data_timestamp": "2026-08-10T16:00:00+00:00",
        "actual_notional_eur": "70",
        "actual_portfolio_weight": "0.0374",
        "planned_total_risk_eur": "5",
        "cash_before_eur": "1870",
        "cash_after_eur": "1800",
        "portfolio_heat_before": "0",
        "portfolio_heat_after": "0.0027",
    }
    values.update(overrides)
    return values


def _bindings():
    return {
        "p0_hash": "P0",
        "writer_freeze_hash": "WRITER",
        "policy_hash": "POLICY",
    }


def _seed_policy_and_sources(tmp_path) -> None:
    for relative in AUTHORITATIVE_SOURCES:
        source = PROJECT_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    policy_source = (
        PROJECT_ROOT / "output/ibkr/live/whole-share-canary-policy-v1.json"
    )
    policy_target = (
        tmp_path / "output/ibkr/live/whole-share-canary-policy-v1.json"
    )
    policy_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(policy_source, policy_target)
    prior_hashes = {
        relative: normalized_file_hash(tmp_path / relative)
        for relative in AUTHORITATIVE_SOURCES
        if relative != "src/stocks/live/level_one_reauthorization.py"
    }
    prior = {
        "status": "GO",
        "manifest_hash": "PRIOR",
        "source_hashes": prior_hashes,
    }
    prior_path = tmp_path / "output/ibkr/live/freeze-status.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")


def test_policy_and_p02_freeze_bind_all_authoritative_sources(tmp_path) -> None:
    _seed_policy_and_sources(tmp_path)

    assert policy_integrity(tmp_path)["status"] == "GO"
    frozen = build_p02_freeze(tmp_path)
    verified = verify_p02_freeze(tmp_path)

    assert frozen["status"] == "GO"
    assert verified["status"] == "GO"
    assert "src/stocks/portfolio/manager.py" in frozen["source_hashes"]
    assert frozen["manual_approval_required"] is True
    assert frozen["automatic_submission"] is False

    changed = tmp_path / "src/stocks/portfolio/manager.py"
    changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    mismatch = verify_p02_freeze(tmp_path)
    assert mismatch["status"] == "NO_GO"
    assert "WRITER_FREEZE_MISMATCH" in mismatch["blockers"]


def test_prepare_is_non_transmitting_expiring_whole_share_record() -> None:
    now = datetime(2026, 8, 10, 16, tzinfo=UTC)
    prepared = build_prepare_artifact(_candidate(), bindings=_bindings(), now=now)

    assert prepared["status"] == "GO"
    assert prepared["state"] == "PREPARED"
    assert prepared["parent_qty"] == prepared["stop_qty"] == prepared["target_qty"] == "1"
    assert prepared["approval_required"] is True
    assert prepared["transmission_allowed_by_this_artifact"] is False
    assert prepared["intent_created"] is False
    assert prepared["submitted"] is False
    json.dumps(prepared)


def test_fractional_or_unproven_candidate_never_prepares() -> None:
    fractional = build_prepare_artifact(
        _candidate(canary_qty="1.5"), bindings=_bindings()
    )
    unproven = build_prepare_artifact(
        _candidate(shariah_allowed=False), bindings=_bindings()
    )

    assert fractional["status"] == "NO_GO"
    assert "WHOLE_SHARE_CANARY_REQUIRED" in fractional["blockers"]
    assert unproven["status"] == "NO_GO"
    assert "SHARIAH_ALLOWED_REQUIRED" in unproven["blockers"]


def test_prepare_expiry_price_stop_and_account_drift_fail_closed() -> None:
    now = datetime(2026, 8, 10, 16, tzinfo=UTC)
    prepared = build_prepare_artifact(_candidate(), bindings=_bindings(), now=now)

    valid = validate_prepare_artifact(
        prepared,
        current_price=Decimal("70.50"),
        current_stop=Decimal("65"),
        current_account_snapshot_hash="SNAPSHOT",
        current_account_fingerprint="ACCOUNT",
        now=now + timedelta(minutes=1),
    )
    drifted = validate_prepare_artifact(
        prepared,
        current_price=Decimal("72"),
        current_stop=Decimal("64"),
        current_account_snapshot_hash="CHANGED",
        current_account_fingerprint="OTHER",
        now=now + timedelta(minutes=11),
    )

    assert valid["status"] == "GO"
    assert drifted["status"] == "NO_GO"
    assert set(drifted["blockers"]) == {
        "ACCOUNT_FINGERPRINT_MISMATCH",
        "PREPARE_EXPIRED",
        "REPREPARE_REQUIRED_PRICE_DRIFT",
        "REPREPARE_REQUIRED_STOP_DRIFT",
        "REVALIDATION_REQUIRED_ACCOUNT_DRIFT",
    }
    assert drifted["transmission_allowed"] is False


def test_cross_asset_ranking_has_no_share_price_bias_and_cash_competes() -> None:
    stock = _candidate(
        candidate_id="STOCK",
        asset_class="STOCK",
        share_price_eur="5",
        expected_net_return="0.03",
    )
    etf = _candidate(
        candidate_id="ETF",
        asset_class="ETF",
        share_price_eur="250",
        expected_net_return="0.06",
    )
    commodity = _candidate(
        candidate_id="COMMODITY",
        asset_class="COMMODITY_VEHICLE",
        share_price_eur="80",
        expected_net_return="0.09",
    )

    ranked = rank_manual_review_candidates([stock, etf, commodity])
    cash = rank_manual_review_candidates(
        [_candidate(expected_net_return="0", risk_adjusted_opportunity="0")]
    )

    assert ranked["selected_candidate"]["candidate_id"] == "COMMODITY"
    assert ranked["share_price_used_for_ranking"] is False
    assert cash["selected_action"] == "NO_TRADE"
    assert cash["cash_competes"] is True


def test_zero_and_multiple_candidates_are_deterministic() -> None:
    assert rank_manual_review_candidates([])["selected_action"] == "NO_TRADE"
    result = rank_manual_review_candidates(
        [_candidate(candidate_id="A"), _candidate(candidate_id="B")]
    )
    assert len(result["ranked_candidates"]) == 2
    assert result["selected_candidate"]["candidate_id"] == "B"
