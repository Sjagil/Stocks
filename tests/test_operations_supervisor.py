from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import stocks.operations.service as service
from stocks.application.phase_gates import PhaseGateStatus


def _phase1(root: Path) -> PhaseGateStatus:
    return PhaseGateStatus(
        name="phase1",
        status="PHASE1_FROZEN",
        frozen=True,
        report_path=root / "PHASE1_FREEZE_REPORT.md",
    )


def test_machine_run_is_bounded_and_append_only(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".venv-ibkr" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-ibkr" / "Scripts" / "python.exe").touch()
    monkeypatch.setattr(service, "phase1_freeze_status", _phase1)
    monkeypatch.setattr(
        service,
        "_command_step",
        lambda *_args, **_kwargs: {"status": "GO", "schema": "step"},
    )
    monkeypatch.setattr(service, "_research_due", lambda _root: False)

    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=1,
    )

    assert result["status"] == "GO"
    assert result["bounded"] is True
    assert result["cycles_completed"] == 1
    assert result["interval_seconds"] == 60
    assert result["execution_authority"] == "NONE"
    assert result["cycles"][0]["status"] == "GO"
    cycle_log = (
        tmp_path / "data" / "operations" / "private" / "cycles.jsonl"
    )
    rows = cycle_log.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["broker_writes"] == 0
    machine_status = json.loads(
        (tmp_path / "output" / "operations" / "machine-status.json").read_text(
            encoding="utf-8"
        )
    )
    assert machine_status["last_cycle_id"] == result["cycles"][0]["cycle_id"]
    assert machine_status["status"] == "GO"
    heartbeat = json.loads(
        (tmp_path / "runtime" / "heartbeat.json").read_text(
            encoding="utf-8"
        )
    )
    assert heartbeat["runtime_status"] == "GO"
    assert heartbeat["cycle_id"] == result["cycles"][0]["cycle_id"]
    assert heartbeat["execution_authority"] == "NONE"
    assert heartbeat["broker_writes"] == 0


def test_intraday_refresh_plan_rotates_ranked_challengers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output" / "portfolio"
    path.mkdir(parents=True)
    (path / "opportunity_ranking.json").write_text(
        json.dumps(
            {
                "opportunities": [
                    {"ticker": f"CHALLENGER{index}"}
                    for index in range(40)
                ]
            }
        ),
        encoding="utf-8",
    )

    first = service._intraday_refresh_plan(
        tmp_path, {"intraday_refresh_cursor": 0}
    )
    second = service._intraday_refresh_plan(
        tmp_path, {"intraday_refresh_cursor": first["next_cursor"]}
    )

    assert first["rotation_status"] == (
        "PRIORITY_AND_ROTATING_CHALLENGER_BATCH_GO"
    )
    assert first["core_symbol_count"] == len(
        service.INTRADAY_CORE_SYMBOLS
    )
    assert len(first["priority_symbols"]) == 25
    assert len(first["challenger_symbols"]) == 5
    assert first["collection_symbol_limit"] == 50
    assert len(first["symbols"]) == 50
    assert first["challenger_symbols"] != second["challenger_symbols"]
    assert first["execution_authority"] == "NONE"
    assert first["broker_calls"] == 0


def test_intraday_refresh_plan_extends_pool_with_refreshable_signals(
    tmp_path: Path,
) -> None:
    portfolio = tmp_path / "output" / "portfolio"
    portfolio.mkdir(parents=True)
    (portfolio / "opportunity_ranking.json").write_text(
        json.dumps({"opportunities": [{"ticker": "RANKED"}]}),
        encoding="utf-8",
    )
    signals = tmp_path / "output" / "signals"
    signals.mkdir(parents=True)
    (signals / "latest_signals.json").write_text(
        json.dumps(
            {
                "signals": [
                    {
                        "ticker": "FRESH_HIGH",
                        "data_freshness": "FRESH",
                        "expiration_timestamp": "2099-01-01T00:00:00+00:00",
                        "confidence_score": "0.90",
                    },
                    {
                        "ticker": "FRESH_LOW",
                        "data_freshness": "FRESH",
                        "expiration_timestamp": "2099-01-01T00:00:00+00:00",
                        "confidence_score": "0.60",
                    },
                    {
                        "ticker": "STALE_REFERENCE",
                        "data_freshness": "STALE",
                        "original_action": "WATCHLIST",
                        "price_validity_status": (
                            "CURRENT_MARKET_REFERENCE_UNAVAILABLE_OR_STALE"
                        ),
                        "expiration_timestamp": "2099-01-01T00:00:00+00:00",
                        "confidence_score": "1.00",
                    },
                    {
                        "ticker": "STALE_OTHER_REASON",
                        "data_freshness": "STALE",
                        "original_action": "WATCHLIST",
                        "price_validity_status": "CURRENT_PRICE_BREACHED_SIGNAL_STOP",
                        "expiration_timestamp": "2099-01-01T00:00:00+00:00",
                        "confidence_score": "1.00",
                    },
                    {
                        "ticker": "EXPIRED",
                        "data_freshness": "FRESH",
                        "expiration_timestamp": "2020-01-01T00:00:00+00:00",
                        "confidence_score": "1.00",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = service._intraday_refresh_plan(
        tmp_path, {"intraday_refresh_cursor": 0}
    )

    assert result["opportunity_pool_count"] == 1
    assert result["signal_pool_count"] == 3
    assert result["stale_signal_refresh_symbol_count"] == 1
    assert result["priority_symbols"][:4] == [
        "RANKED",
        "STALE_REFERENCE",
        "FRESH_HIGH",
        "FRESH_LOW",
    ]
    assert "STALE_REFERENCE" in result["symbols"]
    assert "STALE_OTHER_REASON" not in result["symbols"]
    assert "EXPIRED" not in result["symbols"]


def test_intraday_refresh_plan_accepts_list_signal_artifact(
    tmp_path: Path,
) -> None:
    signals = tmp_path / "output/signals"
    signals.mkdir(parents=True)
    (signals / "latest_signals.json").write_text(
        json.dumps(
            [
                {
                    "ticker": "LIST_SIGNAL",
                    "data_freshness": "STALE",
                    "original_action": "WATCHLIST",
                    "price_validity_status": (
                        "CURRENT_MARKET_REFERENCE_UNAVAILABLE_OR_STALE"
                    ),
                    "expiration_timestamp": "2099-01-01T00:00:00+00:00",
                    "confidence_score": "0.75",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = service._intraday_refresh_plan(tmp_path, {})

    assert "LIST_SIGNAL" in result["symbols"]
    assert result["signal_pool_count"] == 1
    assert result["stale_signal_refresh_symbol_count"] == 1


def test_intraday_refresh_plan_always_includes_observed_positions(
    tmp_path: Path,
) -> None:
    portfolio = tmp_path / "output/portfolio"
    portfolio.mkdir(parents=True)
    (portfolio / "current_allocation.json").write_text(
        json.dumps(
            {
                "positions": [
                    {"ticker": "HELD"},
                    {"symbol": "AAPL"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = service._intraday_refresh_plan(
        tmp_path, {"intraday_refresh_cursor": 0}
    )

    assert result["position_symbols"] == ["HELD", "AAPL"]
    assert result["position_symbol_count"] == 2
    assert "HELD" in result["symbols"]
    assert result["symbols"].count("AAPL") == 1
    assert result["execution_authority"] == "NONE"
    assert result["broker_calls"] == 0


def test_daily_performance_capture_is_verified_and_append_only(
    tmp_path: Path,
) -> None:
    phase9 = tmp_path / "output" / "ibkr" / "phase9"
    phase9.mkdir(parents=True)
    (phase9 / "position-ledger-audit.json").write_text(
        json.dumps(
            {
                "partial_close_projection": {
                    "last_updated_at": "2026-01-02T12:00:00+00:00",
                    "realized_pnl_eur": "1.25",
                }
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "output" / "capital"
    target.mkdir(parents=True)
    (target / "daily_profit_target.json").write_text(
        json.dumps({"input_source": "OPERATOR_SUPPLIED", "net_daily_pnl_eur": 999}),
        encoding="utf-8",
    )

    first = service._record_daily_performance(tmp_path)
    second = service._record_daily_performance(tmp_path)
    history = (
        tmp_path / "data" / "performance" / "private" / "daily-pnl.jsonl"
    ).read_text(encoding="utf-8").splitlines()

    assert first["records_appended"] == 1
    assert second["records_appended"] == 0
    assert len(history) == 1
    row = json.loads(history[0])
    assert row["net_pnl_eur"] == 1.25
    assert row["account_identifier_stored"] is False
    assert first["operator_supplied_values_counted"] == 0
    assert first["broker_calls"] == 0


def test_survivor_shadow_observer_runs_before_portfolio_without_authority(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".venv-ibkr" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-ibkr" / "Scripts" / "python.exe").touch()
    monkeypatch.setattr(service, "phase1_freeze_status", _phase1)
    monkeypatch.setattr(
        service, "_refresh_due", lambda _value, **_kwargs: True
    )
    monkeypatch.setattr(service, "_research_due", lambda _root: False)
    calls: list[tuple[str, ...]] = []

    def command_step(_root, arguments, **_kwargs):
        calls.append(arguments)
        return {"status": "GO", "schema": "step"}

    monkeypatch.setattr(service, "_command_step", command_step)

    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=1,
    )

    cycle = result["cycles"][0]
    survivor_call = ("research", "phase11-14", "observe")
    portfolio_call = ("portfolio", "plan")
    assert survivor_call in calls
    assert calls.index(survivor_call) < calls.index(portfolio_call)
    assert ("telegram", "send-shadow-digest") in calls
    assert cycle["survivor_shadow_observation"]["status"] == "GO"
    assert (
        service._state(tmp_path)[
            "last_survivor_shadow_observation_refresh"
        ]
        is not None
    )
    assert cycle["execution_authority"] == "NONE"
    assert cycle["broker_writes"] == 0


def test_atomic_json_tolerates_parallel_publishers(tmp_path: Path) -> None:
    path = tmp_path / "output" / "operations" / "parallel.json"

    def publish(index: int) -> None:
        service._atomic_json(path, {"index": index, "status": "GO"})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(publish, range(40)))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "GO"
    assert payload["index"] in range(40)
    assert list(path.parent.glob("*.tmp")) == []


def test_required_step_timeout_degrades_cycle_and_refresh_state(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".venv-ibkr" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-ibkr" / "Scripts" / "python.exe").touch()
    monkeypatch.setattr(service, "phase1_freeze_status", _phase1)
    monkeypatch.setattr(
        service, "_refresh_due", lambda _value, **_kwargs: True
    )
    monkeypatch.setattr(service, "_research_due", lambda _root: False)

    def command_step(_root, arguments, **_kwargs):
        if arguments[:2] == ("data", "multitimeframe"):
            return {"status": "TIMEOUT"}
        if arguments == ("daily", "--no-autopilot"):
            return {"status": "TIMEOUT"}
        return {"status": "GO", "schema": "step"}

    monkeypatch.setattr(service, "_command_step", command_step)

    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=1,
    )

    cycle = result["cycles"][0]
    assert result["status"] == "DEGRADED"
    assert cycle["status"] == "DEGRADED"
    assert "OPERATIONAL_STEP_MARKET_DATA_TIMEOUT" in cycle["blockers"]
    assert "OPERATIONAL_STEP_DAILY_TIMEOUT" in cycle["blockers"]
    assert cycle["execution_authority"] == "NONE"
    assert cycle["broker_writes"] == 0
    state = service._state(tmp_path)
    assert state["last_data_refresh"] is None
    assert state["last_cycle_id"] == cycle["cycle_id"]
    assert state["last_cycle_status"] == "DEGRADED"
    machine_status = json.loads(
        (tmp_path / "output" / "operations" / "machine-status.json").read_text(
            encoding="utf-8"
        )
    )
    assert machine_status["status"] == "DEGRADED"
    assert machine_status["last_cycle_id"] == cycle["cycle_id"]
    assert "OPERATIONAL_STEP_DAILY_TIMEOUT" in machine_status[
        "last_cycle_blockers"
    ]


def test_mtf_telegram_failure_is_recorded_but_non_blocking(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".venv-ibkr" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-ibkr" / "Scripts" / "python.exe").touch()
    monkeypatch.setattr(service, "phase1_freeze_status", _phase1)
    monkeypatch.setattr(
        service, "_refresh_due", lambda _value, **_kwargs: True
    )
    monkeypatch.setattr(service, "_research_due", lambda _root: False)

    def command_step(_root, arguments, **_kwargs):
        if arguments == ("telegram", "send-pit-mtf-signals"):
            return {
                "status": "DEGRADED",
                "failure_is_non_blocking": True,
            }
        return {"status": "GO", "schema": "step"}

    monkeypatch.setattr(service, "_command_step", command_step)
    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=1,
    )

    cycle = result["cycles"][0]
    assert cycle["status"] == "GO"
    assert cycle["multitimeframe_pit_notification"]["status"] == (
        "DEGRADED"
    )
    assert not any(
        "MTF_PIT_NOTIFICATION" in blocker
        for blocker in cycle["blockers"]
    )
    assert cycle["execution_authority"] == "NONE"
    assert cycle["broker_writes"] == 0


def test_failed_macro_refresh_does_not_advance_refresh_clock(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".venv-ibkr" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-ibkr" / "Scripts" / "python.exe").touch()
    monkeypatch.setattr(service, "phase1_freeze_status", _phase1)
    monkeypatch.setattr(
        service, "_refresh_due", lambda _value, **_kwargs: True
    )
    monkeypatch.setattr(service, "_research_due", lambda _root: False)

    def command_step(_root, arguments, **_kwargs):
        if arguments == ("macro", "update"):
            return {
                "status": "ERROR",
                "schema": "macro_update_v2",
            }
        return {"status": "GO", "schema": "step"}

    monkeypatch.setattr(service, "_command_step", command_step)
    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=1,
    )

    assert result["status"] == "DEGRADED"
    assert service._state(tmp_path)["last_macro_refresh"] is None
    assert "OPERATIONAL_STEP_MACRO_ERROR" in result["cycles"][0]["blockers"]


def test_partial_macro_context_is_non_blocking_and_advances_clock(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".venv-ibkr" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-ibkr" / "Scripts" / "python.exe").touch()
    monkeypatch.setattr(service, "phase1_freeze_status", _phase1)
    monkeypatch.setattr(
        service, "_refresh_due", lambda _value, **_kwargs: True
    )
    monkeypatch.setattr(service, "_research_due", lambda _root: False)

    def command_step(_root, arguments, **_kwargs):
        if arguments == ("macro", "update"):
            return {
                "status": "DATA_INCOMPLETE",
                "schema": "macro_update_v2",
                "collection_status": "GO",
                "execution_authority": "NONE",
                "broker_calls": 0,
                "order_calls": 0,
            }
        return {"status": "GO", "schema": "step"}

    monkeypatch.setattr(service, "_command_step", command_step)
    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=1,
    )

    cycle = result["cycles"][0]
    assert result["status"] == "GO"
    assert cycle["status"] == "GO"
    assert cycle["macro"]["status"] == "DATA_INCOMPLETE"
    assert (
        cycle["macro_context_policy"]
        == "OPTIONAL_PARTIAL_CONTEXT_NON_BLOCKING"
    )
    assert cycle["blockers"] == []
    assert service._state(tmp_path)["last_macro_refresh"] is not None
    assert cycle["execution_authority"] == "NONE"
    assert cycle["broker_writes"] == 0


def test_hourly_macro_and_news_refresh_are_bounded_and_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".venv-ibkr" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-ibkr" / "Scripts" / "python.exe").touch()
    monkeypatch.setattr(service, "phase1_freeze_status", _phase1)
    monkeypatch.setattr(
        service, "_refresh_due", lambda _value, **_kwargs: True
    )
    monkeypatch.setattr(service, "_research_due", lambda _root: False)
    commands: list[tuple[str, ...]] = []

    def command_step(_root, arguments, **_kwargs):
        commands.append(arguments)
        if arguments == ("macro", "update"):
            return {
                "status": "GO",
                "schema": "macro_update_v2",
                "collection_status": "GO",
                "execution_authority": "NONE",
                "broker_calls": 0,
                "order_calls": 0,
            }
        return {
            "status": "GO",
            "schema": "step",
            "broker_calls": 0,
            "order_calls": 0,
        }

    monkeypatch.setattr(service, "_command_step", command_step)
    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=1,
    )

    cycle = result["cycles"][0]
    state = service._state(tmp_path)
    assert ("macro", "update") in commands
    assert ("telegram", "market-digest-preview") in commands
    assert (
        "market",
        "context",
        "build",
        "--symbols",
        ",".join(service._market_context_symbols(list(service.INTRADAY_CORE_SYMBOLS))),
        "--max-expirations",
        "4",
    ) in commands
    assert (
        "market",
        "context",
        "cot-update",
        "--start",
        "2018-01-01",
    ) in commands
    assert ("market", "context", "transmission") in commands
    assert (
        "market",
        "context",
        "observe",
        "--max-symbols",
        "20",
        "--depth-symbols",
        "5",
    ) in commands
    assert ("research", "registry", "roles") in commands
    assert ("p3", "publish") in commands
    assert cycle["news_digest"]["status"] == "GO"
    assert cycle["market_context"]["status"] == "GO"
    assert cycle["cot_context"]["status"] == "GO"
    assert cycle["asset_context"]["status"] == "GO"
    assert cycle["entry_observer"]["status"] == "GO"
    assert cycle["role_leaderboards"]["status"] == "GO"
    assert cycle["market_context_policy"] == (
        "CONTEXT_ONLY_NO_STANDALONE_ENTRY_AUTHORITY"
    )
    assert state["last_macro_refresh"] is not None
    assert state["last_news_refresh"] is not None
    assert state["last_market_context_refresh"] is not None
    assert state["last_cot_context_refresh"] is not None
    assert state["last_asset_context_refresh"] is not None
    assert state["last_entry_observer_refresh"] is not None
    assert state["last_role_leaderboards_refresh"] is not None
    assert state["last_p3_evidence_refresh"] is not None
    assert cycle["execution_authority"] == "NONE"
    assert cycle["broker_writes"] == 0


def test_successful_market_data_refresh_forces_downstream_signal_refresh(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        service,
        "_refresh_due",
        lambda _value, **_kwargs: False,
    )

    assert service._downstream_refresh_due(
        {"status": "GO"},
        "2026-08-04T14:32:00+00:00",
        hours=1,
    )
    assert not service._downstream_refresh_due(
        {"status": "NOT_DUE"},
        "2026-08-04T14:32:00+00:00",
        hours=1,
    )


def test_signals_only_uses_proven_live_read_only_reconciliation(
    tmp_path: Path,
) -> None:
    live = tmp_path / "output" / "ibkr" / "live"
    live.mkdir(parents=True)
    (live / "status.json").write_text(
        json.dumps(
            {
                "account_reconciliation": "LIVE_RECONCILED_EMPTY",
                "execution_authority": "NONE",
            }
        ),
        encoding="utf-8",
    )

    assert service._reconciliation_arguments(
        tmp_path, "SIGNALS_ONLY"
    ) == ("live", "reconcile")
    assert service._reconciliation_arguments(
        tmp_path, "PAPER_AUTOMATIC"
    ) == ("ibkr", "phase9", "reconcile")


def test_current_partial_news_context_is_non_blocking() -> None:
    payload = {
        "schema": "telegram_market_digest_preview_v1",
        "status": "PARTIAL",
        "digest": {
            "news_source_status": {"status": "GO"},
            "news_freshness_status": "CURRENT_WITHIN_72H",
            "order_calls": 0,
            "automatic_execution": False,
        },
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }

    assert service._news_context_usable(payload) is True
    assert service._step_blockers("NEWS_DIGEST", payload) == []
    stale = {
        **payload,
        "digest": {
            **payload["digest"],
            "news_freshness_status": "STALE",
        },
    }
    assert service._news_context_usable(stale) is False
    assert service._step_blockers("NEWS_DIGEST", stale) == [
        "OPERATIONAL_STEP_NEWS_DIGEST_PARTIAL"
    ]


def test_expected_pit_research_block_is_not_an_operational_failure() -> None:
    payload = {
        "schema": "bounded_research_autopilot_cycle_v1",
        "status": "DATA_BLOCKED",
        "campaign": {
            "complete_trial_count": 0,
            "eligibility": {"status": "PIT_ELIGIBILITY_UNAVAILABLE"},
        },
        "survivor_recovery": {"status": "GO", "survivor_count": 16},
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }

    assert service._research_data_blocked_expected(payload) is True
    assert service._step_blockers("RESEARCH", payload) == []


def test_unsafe_research_data_block_remains_operationally_visible() -> None:
    payload = {
        "schema": "bounded_research_autopilot_cycle_v1",
        "status": "DATA_BLOCKED",
        "campaign": {
            "complete_trial_count": 0,
            "eligibility": {"status": "PIT_ELIGIBILITY_UNAVAILABLE"},
        },
        "survivor_recovery": {"status": "GO"},
        "execution_authority": "PAPER",
        "broker_calls": 1,
        "orders_generated": 1,
    }

    assert service._research_data_blocked_expected(payload) is False
    assert service._step_blockers("RESEARCH", payload) == [
        "OPERATIONAL_STEP_RESEARCH_DATA_BLOCKED"
    ]


def test_not_due_dynamic_does_not_erase_lifecycle_state(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".venv-ibkr" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-ibkr" / "Scripts" / "python.exe").touch()
    monkeypatch.setattr(service, "phase1_freeze_status", _phase1)
    monkeypatch.setattr(service, "_refresh_due", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(service, "_research_due", lambda _root: False)
    monkeypatch.setattr(
        service,
        "_command_step",
        lambda *_args, **_kwargs: {"status": "GO", "schema": "step"},
    )
    lifecycle_path = (
        tmp_path / "data" / "operations" / "private" / "signal-states.json"
    )
    lifecycle_path.parent.mkdir(parents=True)
    lifecycle_path.write_text(
        json.dumps({"states": {"STRATEGY:TEST": "BUY"}}),
        encoding="utf-8",
    )

    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=1,
    )

    assert result["status"] == "GO"
    assert result["cycles"][0]["dynamic"]["status"] == "NOT_DUE"
    assert result["cycles"][0]["signal_lifecycle"]["status"] == "NOT_DUE"
    persisted = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    assert persisted["states"] == {"STRATEGY:TEST": "BUY"}


def test_explicit_component_go_marker_is_success() -> None:
    payload = {"status": {"status": "DYNAMIC_MULTI_STRATEGY_ENGINE_GO"}}
    assert service._is_success_status(payload) is True
    assert service._step_blockers("DYNAMIC", payload) == []
    assert service._is_success_status({"status": "NO_GO"}) is False


def test_paper_activation_requires_canary_and_exact_phrase(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        service,
        "_execution_preflight",
        lambda *_args: {
            "status": "NO_GO",
            "blockers": ["PHASE9_FILL_CLOSE_CANARY_REQUIRED"],
        },
    )
    result = service.execution_command(
        tmp_path,
        "activate-paper",
        approval="wrong",
    )
    assert result["status"] == "NO_GO"
    assert result["execution_authority"] == "NONE"
    assert "PHASE9_FILL_CLOSE_CANARY_REQUIRED" in result["blockers"]
    assert "EXACT_PAPER_ACTIVATION_PHRASE_REQUIRED" in result["blockers"]


def test_live_mode_cannot_be_selected_without_activation(tmp_path: Path) -> None:
    state = service._state(tmp_path)
    assert (
        service._effective_mode(state, "LIVE_CANARY_AUTOMATIC")
        == "SIGNALS_ONLY"
    )


def test_bounded_runner_uses_persisted_requested_mode(
    tmp_path: Path, monkeypatch
) -> None:
    state = service._state(tmp_path)
    state.update(
        enabled=True,
        paused=False,
        requested_mode="CONTROLLED_LIVE",
        live_enabled=True,
        activated_at="2026-07-29T00:00:00+00:00",
    )
    service._save_state(tmp_path, state)
    monkeypatch.setattr(
        service,
        "_cycle",
        lambda _root, _state, requested: {
            "status": "GO",
            "requested_mode": requested,
            "execution_authority": "NONE",
        },
    )

    result = service._run_cycles(
        tmp_path,
        state,
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=60,
    )

    assert result["status"] == "GO"
    assert result["cycles"][0]["requested_mode"] == "CONTROLLED_LIVE"


def test_explicit_bounded_run_overrides_stale_persisted_live_mode(
    tmp_path: Path, monkeypatch
) -> None:
    state = service._state(tmp_path)
    state.update(
        enabled=True,
        paused=False,
        mode="CONTROLLED_LIVE",
        requested_mode="CONTROLLED_LIVE",
        live_enabled=True,
        activated_at="2026-07-29T00:00:00+00:00",
    )
    service._save_state(tmp_path, state)
    monkeypatch.setattr(
        service,
        "_cycle",
        lambda _root, _state, requested: {
            "status": "GO",
            "requested_mode": requested,
            "execution_authority": "NONE",
        },
    )

    result = service.machine_command(
        tmp_path,
        "run",
        mode="SIGNALS_ONLY",
        max_cycles=1,
        interval_seconds=60,
    )

    assert result["cycles"][0]["requested_mode"] == "SIGNALS_ONLY"
    persisted = service._state(tmp_path)
    assert persisted["mode"] == "SIGNALS_ONLY"
    assert persisted["requested_mode"] == "SIGNALS_ONLY"
    assert persisted["live_enabled"] is False


def test_machine_stop_clears_stale_execution_mode_flags(
    tmp_path: Path,
) -> None:
    state = service._state(tmp_path)
    state.update(
        enabled=True,
        paused=True,
        mode="CONTROLLED_LIVE",
        requested_mode="CONTROLLED_LIVE",
        paper_enabled=True,
        live_enabled=True,
    )
    service._save_state(tmp_path, state)

    result = service.machine_command(tmp_path, "stop")

    assert result["enabled"] is False
    assert result["paused"] is False
    assert result["mode"] == "SIGNALS_ONLY"
    assert result["requested_mode"] == "SIGNALS_ONLY"
    assert result["paper_enabled"] is False
    assert result["live_enabled"] is False


def test_explicit_live_run_enables_live_mode_only_with_current_authority(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        service,
        "authority_status",
        lambda _root: {"execution_authority": "AUTONOMOUS_LEVEL_ONE"},
    )
    monkeypatch.setattr(
        service,
        "_cycle",
        lambda _root, _state, requested: {
            "status": "GO",
            "requested_mode": requested,
            "execution_authority": "AUTONOMOUS_LEVEL_ONE",
        },
    )

    result = service.machine_command(
        tmp_path,
        "run",
        mode="CONTROLLED_LIVE",
        max_cycles=1,
        interval_seconds=60,
    )

    assert result["cycles"][0]["requested_mode"] == "CONTROLLED_LIVE"
    persisted = service._state(tmp_path)
    assert persisted["live_enabled"] is True
    assert persisted["paper_enabled"] is False


def test_signal_lifecycle_only_marks_false_to_true_as_fresh(
    tmp_path: Path,
) -> None:
    dynamic = {
        "signals": {
            "signals": [
                {
                    "ticker": "TEST",
                    "strategy_id": "STRATEGY",
                    "action": "BUY",
                }
            ]
        }
    }
    first = service._signal_lifecycle(tmp_path, dynamic)
    second = service._signal_lifecycle(tmp_path, dynamic)
    assert first["fresh_entry_count"] == 1
    assert first["rows"][0]["lifecycle_status"] == "FRESH_ENTRY"
    assert second["fresh_entry_count"] == 0
    assert (
        second["rows"][0]["lifecycle_status"]
        == "ACTIVE_STATE_NO_NEW_ENTRY"
    )


def test_paper_close_route_never_bypasses_phase9(tmp_path: Path) -> None:
    result = service.positions_command(
        tmp_path,
        "close",
        environment="paper",
        symbol="TEST",
        approval="ignored",
    )
    assert result["status"] == "OPERATOR_ACTION_REQUIRED"
    assert result["automatic_close_submission"] is False
    assert result["paper_place_order_calls"] == 0


def test_machine_stop_publishes_stopped_heartbeat(tmp_path: Path) -> None:
    result = service.machine_command(tmp_path, "stop")
    heartbeat = json.loads(
        (tmp_path / "runtime" / "heartbeat.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["enabled"] is False
    assert heartbeat["runtime_status"] == "STOPPED"
    assert heartbeat["execution_authority"] == "NONE"


def test_phase9_operator_wrapper_uses_only_canonical_cli() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root / "scripts" / "run_phase9_manual_canary.ps1"
    ).read_text(encoding="utf-8")

    assert '"main.py"' in source
    assert '"ibkr", "phase9", "prepare"' in source
    assert '"ibkr", "phase9", "approve"' in source
    assert '"ibkr", "phase9", "submit"' in source
    assert "Type the exact approval challenge" in source
    assert "SUBMIT PAPER INTENT" in source
    assert "[ValidateRange(15, 300)]" in source
    assert "[switch]$PreflightOnly" in source
    assert 'mode = "PREFLIGHT_ONLY"' in source
    assert "intent_created = $false" in source
    assert '"ibkr", "phase9", "reconcile"' in source
    assert "automatic_cancellation = $false" in source
    for forbidden in (
        "placeOrder",
        "cancelOrder",
        "reqIds",
        "reqGlobalCancel",
        "reqAutoOpenOrders",
        "exerciseOptions",
    ):
        assert forbidden not in source


def test_windows_stop_script_bounds_exact_runtime_child_cleanup() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts" / "stop_bot.ps1").read_text(
        encoding="utf-8"
    )

    assert "$ResolvedMain = [regex]::Escape((Resolve-Path $Main).Path)" in source
    assert '$_.CommandLine -match "\\srun\\s+--mode\\s"' in source
    assert "$Deadline = (Get-Date).AddSeconds(75)" in source
    assert "Stop-Process -Id $_.ProcessId -Force" in source
    assert "remaining_runtime_processes = 0" in source
