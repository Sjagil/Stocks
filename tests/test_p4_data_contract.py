from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stocks.p4.completion import publish_requirement_audit
from stocks.p4.data import (
    DATASET_SCHEMAS,
    PITDataCatalog,
    SourceAttestation,
    ingest_point_in_time_bundle,
    ingest_point_in_time_snapshot,
    validate_point_in_time_frame,
)
from stocks.p4.forward import (
    freeze_forward_evaluation_protocol,
    preregister_phase11_14_candidates,
)


def _membership_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "universe_id": "SP500",
                "security_id": "SEC-1",
                "effective_from": "2020-01-01T00:00:00Z",
                "effective_to": "2021-01-01T00:00:00Z",
                "available_at": "2020-01-02T00:00:00Z",
                "is_member": True,
            },
            {
                "universe_id": "SP500",
                "security_id": "SEC-1",
                "effective_from": "2021-01-01T00:00:00Z",
                "effective_to": None,
                "available_at": "2021-01-02T00:00:00Z",
                "is_member": False,
            },
        ]
    )


def test_pit_contract_accepts_adjacent_intervals_and_sorts() -> None:
    result = validate_point_in_time_frame(
        _membership_frame().iloc[::-1], DATASET_SCHEMAS["universe_membership"]
    )
    assert result.iloc[0]["effective_from"].year == 2020
    assert str(result["available_at"].dtype) == "datetime64[ns, UTC]"


def test_pit_contract_rejects_missing_duplicate_and_future_rows() -> None:
    frame = _membership_frame()
    with pytest.raises(ValueError, match="columns missing"):
        validate_point_in_time_frame(
            frame.drop(columns=["available_at"]),
            DATASET_SCHEMAS["universe_membership"],
        )
    with pytest.raises(ValueError, match="duplicate immutable identities"):
        validate_point_in_time_frame(
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            DATASET_SCHEMAS["universe_membership"],
        )
    future = frame.copy()
    future.loc[0, "available_at"] = "2099-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="future availability"):
        validate_point_in_time_frame(
            future, DATASET_SCHEMAS["universe_membership"]
        )


def test_pit_contract_rejects_overlapping_and_invalid_intervals() -> None:
    overlap = _membership_frame()
    overlap.loc[1, "effective_from"] = "2020-12-01T00:00:00Z"
    with pytest.raises(ValueError, match="overlapping effective intervals"):
        validate_point_in_time_frame(
            overlap, DATASET_SCHEMAS["universe_membership"]
        )
    invalid = _membership_frame()
    invalid.loc[0, "effective_to"] = "2019-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="invalid effective intervals"):
        validate_point_in_time_frame(
            invalid, DATASET_SCHEMAS["universe_membership"]
        )


def test_missing_delisting_return_is_preserved_not_zero_imputed() -> None:
    frame = pd.DataFrame(
        [
            {
                "security_id": "DELISTED-1",
                "delisting_date": "2020-05-01T00:00:00Z",
                "available_at": "2020-05-02T00:00:00Z",
                "delisting_return": None,
                "reason": "UNKNOWN",
            }
        ]
    )
    result = validate_point_in_time_frame(frame, DATASET_SCHEMAS["delistings"])
    assert pd.isna(result.iloc[0]["delisting_return"])


def test_research_only_attestation_cannot_turn_production_gates_green(
    tmp_path: Path,
) -> None:
    source = tmp_path / "daily_prices.csv"
    pd.DataFrame(
        [
            {
                "security_id": "SEC-1",
                "session_date": "2020-01-02T00:00:00Z",
                "available_at": "2020-01-03T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "adjustment_version": "v1",
            }
        ]
    ).to_csv(source, index=False)
    attestation = SourceAttestation(
        provider="TEST",
        source_version="v1",
        license_id="TEST-ONLY",
        licensed_for_research=False,
        complete_history_attested=True,
        point_in_time_semantics_attested=True,
        obtained_at="2026-08-11T00:00:00Z",
        operator="pytest",
    )
    result = ingest_point_in_time_snapshot(
        tmp_path, "daily_prices", source, attestation
    )
    assert result["status"] == "RESEARCH_ONLY"
    audit = PITDataCatalog(tmp_path).audit()
    assert audit["gates"]["PIT_DATA_GO"] is False
    assert audit["current_membership_substitution_allowed"] is False
    assert audit["current_shariah_substitution_allowed"] is False
    assert audit["zero_delisting_return_imputation_allowed"] is False


def test_empty_catalog_does_not_report_vacuous_integrity_or_coverage(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "p4_data_policy_v1.json").write_text(
        json.dumps({"target_universe": "BOUNDED_US_V1"}), encoding="utf-8"
    )

    audit = PITDataCatalog(tmp_path).audit()

    assert audit["snapshot_integrity_verified"] is False
    assert audit["bundle_coherence"]["target_universe_present"] is False
    assert audit["bundle_coherence"]["member_coverage"] == {
        "fundamentals": False,
        "shariah_classification": False,
        "daily_prices": False,
    }


def test_attestation_timestamp_must_be_valid_and_not_future() -> None:
    base = {
        "provider": "TEST",
        "source_version": "v1",
        "license_id": "LICENSE",
        "licensed_for_research": True,
        "complete_history_attested": True,
        "point_in_time_semantics_attested": True,
        "operator": "pytest",
    }
    with pytest.raises(ValueError, match="obtained_at is invalid"):
        SourceAttestation(obtained_at="not-a-date", **base).validate()
    with pytest.raises(ValueError, match="obtained_at is in the future"):
        SourceAttestation(
            obtained_at="2099-01-01T00:00:00Z", **base
        ).validate()
    invalid_boolean = {**base, "licensed_for_research": "false"}
    with pytest.raises(ValueError, match="licensed_for_research must be boolean"):
        SourceAttestation(
            obtained_at="2026-08-11T00:00:00Z", **invalid_boolean
        ).validate()


def test_daily_price_availability_and_ohlc_are_causal_and_consistent() -> None:
    base = {
        "security_id": "SEC-1",
        "session_date": "2020-01-02T00:00:00Z",
        "available_at": "2020-01-03T00:00:00Z",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
        "adjustment_version": "v1",
    }
    early = pd.DataFrame([{**base, "available_at": "2020-01-01T00:00:00Z"}])
    with pytest.raises(ValueError, match="available_at precedes session_date"):
        validate_point_in_time_frame(early, DATASET_SCHEMAS["daily_prices"])
    invalid = pd.DataFrame([{**base, "high": 99.5}])
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        validate_point_in_time_frame(invalid, DATASET_SCHEMAS["daily_prices"])


def test_catalog_revokes_tampered_production_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "daily_prices.csv"
    pd.DataFrame(
        [
            {
                "security_id": "SEC-1",
                "session_date": "2020-01-02T00:00:00Z",
                "available_at": "2020-01-03T00:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "adjustment_version": "v1",
            }
        ]
    ).to_csv(source, index=False)
    attestation = SourceAttestation(
        provider="TEST",
        source_version="v1",
        license_id="LICENSE",
        licensed_for_research=True,
        complete_history_attested=True,
        point_in_time_semantics_attested=True,
        obtained_at="2026-08-11T00:00:00Z",
        operator="pytest",
    )
    result = ingest_point_in_time_snapshot(
        tmp_path, "daily_prices", source, attestation
    )
    snapshot = tmp_path / result["snapshot"]["normalized_path"]
    snapshot.write_bytes(b"tampered")
    audit = PITDataCatalog(tmp_path).audit()
    assert audit["snapshot_integrity_verified"] is False
    assert audit["latest_production_eligible_snapshots"]["daily_prices"] is None
    with pytest.raises(ValueError, match="immutable P4 normalized snapshot collision"):
        ingest_point_in_time_snapshot(
            tmp_path, "daily_prices", source, attestation
        )


def test_catalog_requires_coherent_complete_bounded_universe_bundle(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "p4_data_policy_v1.json").write_text(
        json.dumps({"target_universe": "BOUNDED_US_V1"}), encoding="utf-8"
    )
    frames = {
        "security_master": pd.DataFrame(
            [
                {
                    "security_id": "SEC-1",
                    "symbol": "AAA",
                    "exchange": "NYSE",
                    "currency": "USD",
                    "effective_from": "2019-01-01T00:00:00Z",
                    "effective_to": None,
                    "available_at": "2019-01-01T00:00:00Z",
                    "listing_status": "ACTIVE",
                },
                {
                    "security_id": "SEC-2",
                    "symbol": "OLD",
                    "exchange": "NASDAQ",
                    "currency": "USD",
                    "effective_from": "2019-01-01T00:00:00Z",
                    "effective_to": "2021-01-04T00:00:00Z",
                    "available_at": "2019-01-01T00:00:00Z",
                    "listing_status": "DELISTED",
                },
            ]
        ),
        "universe_membership": pd.DataFrame(
            [
                {
                    "universe_id": "BOUNDED_US_V1",
                    "security_id": "SEC-1",
                    "effective_from": "2020-01-01T00:00:00Z",
                    "effective_to": None,
                    "available_at": "2020-01-01T00:00:00Z",
                    "is_member": True,
                },
                {
                    "universe_id": "BOUNDED_US_V1",
                    "security_id": "SEC-2",
                    "effective_from": "2020-01-01T00:00:00Z",
                    "effective_to": "2021-01-04T00:00:00Z",
                    "available_at": "2020-01-01T00:00:00Z",
                    "is_member": True,
                },
            ]
        ),
        "delistings": pd.DataFrame(
            [
                {
                    "security_id": "SEC-2",
                    "delisting_date": "2021-01-04T00:00:00Z",
                    "available_at": "2021-01-04T00:00:00Z",
                    "delisting_return": -0.25,
                    "reason": "ACQUISITION",
                }
            ]
        ),
        "corporate_actions": pd.DataFrame(
            [
                {
                    "security_id": "SEC-1",
                    "action_date": "2020-02-01T00:00:00Z",
                    "available_at": "2020-02-01T00:00:00Z",
                    "action_type": "DIVIDEND",
                    "value": 0.25,
                }
            ]
        ),
        "fundamentals": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "period_end": "2020-03-31T00:00:00Z",
                    "available_at": "2020-05-01T00:00:00Z",
                    "metric": "assets",
                    "value": 100.0,
                    "revision_id": "R1",
                }
                for security_id in ("SEC-1", "SEC-2")
            ]
        ),
        "shariah_classification": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "effective_from": "2020-01-01T00:00:00Z",
                    "effective_to": None,
                    "available_at": "2020-01-01T00:00:00Z",
                    "status": "COMPLIANT",
                    "methodology_version": "AAOIFI-V1",
                }
                for security_id in ("SEC-1", "SEC-2")
            ]
        ),
        "daily_prices": pd.DataFrame(
            [
                {
                    "security_id": security_id,
                    "session_date": "2020-01-02T00:00:00Z",
                    "available_at": "2020-01-03T00:00:00Z",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                    "adjustment_version": "V1",
                }
                for security_id in ("SEC-1", "SEC-2")
            ]
        ),
    }
    manifest = {
        "schema": "p4_pit_bundle_manifest_v1",
        "target_universe": "BOUNDED_US_V1",
        "datasets": {},
    }
    for dataset, frame in frames.items():
        source = tmp_path / f"{dataset}.csv"
        frame.to_csv(source, index=False)
        manifest["datasets"][dataset] = {
            "source": source.name,
            "attestation": {
                "provider": "TEST",
                "source_version": "V1",
                "license_id": "LICENSE",
                "licensed_for_research": True,
                "complete_history_attested": True,
                "point_in_time_semantics_attested": True,
                "obtained_at": "2026-08-11T00:00:00Z",
                "operator": "pytest",
            },
        }
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bad_root = tmp_path / "bad-root"
    bad_config = bad_root / "config"
    bad_config.mkdir(parents=True)
    (bad_config / "p4_data_policy_v1.json").write_text(
        json.dumps({"target_universe": "BOUNDED_US_V1"}), encoding="utf-8"
    )
    incomplete_source = tmp_path / "daily_prices-incomplete.csv"
    frames["daily_prices"].iloc[[0]].to_csv(incomplete_source, index=False)
    bad_manifest = json.loads(json.dumps(manifest))
    bad_manifest["datasets"]["daily_prices"]["source"] = (
        incomplete_source.name
    )
    bad_manifest_path = tmp_path / "bundle-incomplete.json"
    bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    rejected = ingest_point_in_time_bundle(bad_root, bad_manifest_path)
    assert rejected["status"] == "NO_GO"
    assert not (bad_root / "data/p4/private/catalog.json").exists()

    result = ingest_point_in_time_bundle(tmp_path, manifest_path)
    assert result["status"] == "GO"
    assert set(result["snapshots"]) == set(DATASET_SCHEMAS)

    audit = PITDataCatalog(tmp_path).audit()
    assert audit["gates"] == {
        "PIT_DATA_GO": True,
        "SURVIVORSHIP_GO": True,
        "SHARIAH_PIT_GO": True,
    }
    assert audit["bundle_coherence"]["status"] == "GO"

    incomplete_prices = frames["daily_prices"].iloc[[0]]
    source = tmp_path / "daily_prices-v2.csv"
    incomplete_prices.to_csv(source, index=False)
    ingest_point_in_time_snapshot(
        tmp_path,
        "daily_prices",
        source,
        SourceAttestation(
            provider="TEST",
            source_version="V2",
            license_id="LICENSE",
            licensed_for_research=True,
            complete_history_attested=True,
            point_in_time_semantics_attested=True,
            obtained_at="2026-08-11T00:00:00Z",
            operator="pytest",
        ),
    )
    audit = PITDataCatalog(tmp_path).audit()
    assert audit["gates"]["PIT_DATA_GO"] is False
    assert audit["bundle_coherence"]["member_coverage"]["daily_prices"] is False
    assert (
        audit["bundle_coherence"]["coverage"]["daily_prices"]
        ["missing_member_security_count"]
        == 1
    )


def test_forward_protocol_binds_each_candidate_to_simultaneous_baselines(
    tmp_path: Path,
) -> None:
    research = tmp_path / "output/research/phase11_14"
    research.mkdir(parents=True)
    (research / "qualification-boundary.json").write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "robust_strategy_ids": ["S1"],
                "data_end_by_strategy": {"S1": "2026-01-01T00:00:00Z"},
                "frozen_at": "2026-01-02T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (research / "qualification.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "strategy_id": "S1",
                        "source_strategy_id": "SOURCE",
                        "formula": "frozen",
                        "timeframe": "4h",
                        "asset_class": "STOCK",
                        "frozen_profile": "balanced",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config"
    config.mkdir()
    policy = {
        "status": "FROZEN",
        "candidate_baselines": [
            "PASSIVE_UNDERLYING_BUY_AND_HOLD",
            "DO_NOTHING_CASH",
        ],
        "comparison_clock": "SAME_FORWARD_DECISION_TIMESTAMP",
        "threshold_relaxation_allowed": False,
        "historical_backfill_counts_as_forward": False,
    }
    (config / "p4_forward_evaluation_v1.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    registration = preregister_phase11_14_candidates(tmp_path)
    protocol = freeze_forward_evaluation_protocol(tmp_path)
    assert registration["status"] == "FROZEN"
    assert protocol["status"] == "FROZEN"
    assert protocol["registration_hash"] == registration["registration_hash"]
    assert protocol["candidate_protocols"][0]["baselines"] == (
        policy["candidate_baselines"]
    )
    policy["candidate_baselines"] = ["CHANGED_AFTER_FREEZE"]
    (config / "p4_forward_evaluation_v1.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    assert freeze_forward_evaluation_protocol(tmp_path)["status"] == "BLOCKED"


def test_requirement_audit_maps_all_43_and_never_claims_profitability(
    tmp_path: Path,
) -> None:
    external = {
        "PIT_DATA_GO": False,
        "SURVIVORSHIP_GO": False,
        "SHARIAH_PIT_GO": False,
        "FORWARD_EVIDENCE_GO": False,
        "RL_INCREMENTAL_EVIDENCE_GO": False,
        "RL_FORWARD_EVIDENCE_GO": False,
        "RL_POLICY_PROMOTION_GO": False,
    }
    audit = publish_requirement_audit(
        tmp_path,
        {"economic_and_external_gates": external},
        {"closed_episodes": 0},
    )
    assert audit["all_requirements_mapped"] is True
    assert [item["number"] for item in audit["requirements"]] == list(range(1, 44))
    assert audit["production_ready"] is False
    assert audit["profitability_proven"] is False
    assert audit["rl_live_enabled"] is False
    assert audit["broker_writes"] == 0
    assert "PIT_DATA_GO" in audit["requirements"][-1]["blockers"]
    assert audit["evidence_paths_verified"] is False
    assert "src/stocks/rl/environment.py" in audit["missing_evidence_paths"]
    assert (tmp_path / "reports/P4_RL_REQUIREMENT_AUDIT.md").is_file()
