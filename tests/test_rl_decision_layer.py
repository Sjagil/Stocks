from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocks.rl.contracts import (
    HardGates,
    PositionState,
    RLAction,
    RLMode,
    RLRuntimeConfig,
    action_mask,
    decision_type_for_action,
)
from stocks.rl.data import (
    MARKET_FEATURES,
    CausalFeatureScaler,
    DatasetContract,
    build_causal_swing_frame,
    build_causal_multitimeframe_context,
    load_multitimeframe_frames,
)
from stocks.rl.environment import (
    FinanceSwingEnv,
    OpportunitySelectionEnv,
    SwingEnvironmentConfig,
)
from stocks.rl.evaluation import (
    cost_stress_evaluation,
    evaluate_policy,
    fixed_sizing_policy,
)
from stocks.rl.experience import (
    ExperienceDecision,
    ExperienceOutcome,
    ExperienceStore,
)
from stocks.rl.registry import PolicyRegistry, PolicyState
from stocks.rl.reward import (
    RLRewardConfig,
    RewardInput,
    calculate_reward,
)


@pytest.fixture
def causal_frame() -> pd.DataFrame:
    rows = 620
    index = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(0.0006 * index + 0.04 * np.sin(index / 17.0))
    prices = pd.DataFrame(
        {
            "session_date": pd.date_range("2020-01-01", periods=rows, freq="B", tz="UTC"),
            "open": close * (1.0 + 0.001 * np.sin(index / 7.0)),
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "volume": 1_000_000 + 100_000 * np.cos(index / 13.0),
        }
    )
    contract = DatasetContract(
        symbol="TEST",
        shariah_point_in_time_verified=True,
        point_in_time_universe_verified=True,
        survivorship_verified=True,
        evidence_scope="PRODUCTION_EVIDENCE",
    )
    return build_causal_swing_frame(prices, contract, spread_bps=4.0)


def _environment(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    config: SwingEnvironmentConfig | None = None,
) -> FinanceSwingEnv:
    scaler = CausalFeatureScaler.fit(frame.iloc[:250])
    evaluation = frame.iloc[250:360].reset_index(drop=True)
    return FinanceSwingEnv(
        evaluation,
        scaler=scaler,
        config=config or SwingEnvironmentConfig(),
        seed=seed,
    )


def _first_open_action(env: FinanceSwingEnv) -> int:
    for _ in range(len(env.frame) - 2):
        mask = env.action_masks()
        if mask[RLAction.OPEN_NORMAL]:
            return int(RLAction.OPEN_NORMAL)
        env.step(int(RLAction.HOLD))
    raise AssertionError("fixture did not produce a qualified opportunity")


def test_environment_no_lookahead(causal_frame: pd.DataFrame) -> None:
    first = _environment(causal_frame)
    second_frame = causal_frame.copy()
    second_frame.loc[:, "outcome_next_return"] *= -100.0
    second_frame.loc[:, "outcome_next_high_return"] += 5.0
    second_frame.loc[:, "outcome_next_low_return"] -= 5.0
    second = _environment(second_frame)
    observation_a, _ = first.reset()
    observation_b, _ = second.reset()
    np.testing.assert_array_equal(observation_a, observation_b)


def test_multitimeframe_join_is_backward_only() -> None:
    decisions = pd.Series(pd.to_datetime(["2026-01-05T16:00:00Z"]))
    bars = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-01-05T14:00:00Z",
                    "2026-01-05T15:00:00Z",
                    "2026-01-05T17:00:00Z",
                ]
            ),
            "close": [100.0, 101.0, 500.0],
        }
    )
    context = build_causal_multitimeframe_context(
        decisions, {"1h": bars}
    )
    assert context.iloc[0]["return_1h"] == pytest.approx(0.01)
    assert pd.isna(context.iloc[0]["return_15m"])
    assert context.iloc[0]["source_available_at_1h"] == pd.Timestamp(
        "2026-01-05T16:00:00Z"
    )


def test_multitimeframe_join_waits_for_bar_close_and_excludes_partial() -> None:
    decisions = pd.Series(
        pd.to_datetime(
            [
                "2026-01-05T15:59:59Z",
                "2026-01-05T16:00:00Z",
                "2026-01-05T18:00:00Z",
            ]
        )
    )
    bars = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-01-05T13:00:00Z",
                    "2026-01-05T14:00:00Z",
                    "2026-01-05T15:00:00Z",
                    "2026-01-05T16:00:00Z",
                ]
            ),
            "close": [100.0, 101.0, 102.0, 999.0],
            "partial_bucket": [False, False, False, True],
        }
    )
    context = build_causal_multitimeframe_context(decisions, {"1h": bars})
    assert context.iloc[0]["return_1h"] == pytest.approx(0.01)
    assert context.iloc[1]["return_1h"] == pytest.approx(102.0 / 101.0 - 1.0)
    assert context.iloc[2]["return_1h"] == pytest.approx(102.0 / 101.0 - 1.0)
    assert (
        context["source_available_at_1h"]
        <= pd.to_datetime(decisions, utc=True)
    ).all()


def test_two_hour_context_is_first_class() -> None:
    decisions = pd.Series(pd.to_datetime(["2026-01-05T18:00:00Z"]))
    bars = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2026-01-05T12:00:00Z", "2026-01-05T14:00:00Z"]
            ),
            "close": [100.0, 104.0],
            "partial_bucket": [False, False],
        }
    )
    context = build_causal_multitimeframe_context(decisions, {"2h": bars})
    assert context.iloc[0]["return_2h"] == pytest.approx(0.04)
    assert "return_2h" in CausalFeatureScaler.fit(
        build_causal_swing_frame(
            pd.DataFrame(
                {
                    "session_date": pd.date_range(
                        "2020-01-01", periods=80, freq="B", tz="UTC"
                    ),
                    "open": np.arange(100.0, 180.0),
                    "high": np.arange(101.0, 181.0),
                    "low": np.arange(99.0, 179.0),
                    "close": np.arange(100.0, 180.0),
                    "volume": np.full(80, 1_000_000),
                }
            ),
            DatasetContract(symbol="TEST"),
            retain_unresolved_outcome=True,
        )
    ).feature_names


def test_daily_primary_decision_is_stamped_after_regular_session_close() -> None:
    prices = pd.DataFrame(
        {
            "session_date": pd.date_range("2026-07-01", periods=65, freq="B"),
            "open": np.arange(100.0, 165.0),
            "high": np.arange(101.0, 166.0),
            "low": np.arange(99.0, 164.0),
            "close": np.arange(100.0, 165.0),
            "volume": np.full(65, 1_000_000),
        }
    )
    frame = build_causal_swing_frame(
        prices,
        DatasetContract(symbol="TEST"),
        retain_unresolved_outcome=True,
    )
    assert frame.iloc[0]["timestamp"].hour in {20, 21}
    assert frame.iloc[0]["timestamp"].minute == 0


def test_rl_features_preserve_explicit_primary_timeframe_identity() -> None:
    prices = pd.DataFrame(
        {
            "session_date": pd.date_range(
                "2026-07-01T13:30:00Z", periods=80, freq="15min"
            ),
            "open": np.arange(100.0, 180.0),
            "high": np.arange(101.0, 181.0),
            "low": np.arange(99.0, 179.0),
            "close": np.arange(100.0, 180.0),
            "volume": np.full(80, 1_000_000),
        }
    )
    frame = build_causal_swing_frame(
        prices,
        DatasetContract(symbol="TEST", timeframe="15m"),
        retain_unresolved_outcome=True,
    )
    assert "return_1" not in MARKET_FEATURES
    assert "primary_return_1bar" in MARKET_FEATURES
    np.testing.assert_allclose(
        frame["return_15m"], frame["primary_return_1bar"], equal_nan=True
    )
    assert frame["return_1d"].isna().all()


def test_multitimeframe_loader_rejects_forbidden_five_minute_derivation(
    tmp_path: Path,
) -> None:
    forbidden = (
        tmp_path
        / "data/research/multitimeframe/private/provider=TEST/symbol=SPUS/"
        "interval=15m/source_interval=5m/bars.parquet"
    )
    allowed = (
        tmp_path
        / "data/research/multitimeframe/private/provider=TEST/symbol=SPUS/"
        "interval=1h/source_interval=1h/bars.parquet"
    )
    payload = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2026-01-01T14:00:00Z"]),
            "close": [100.0],
        }
    )
    forbidden.parent.mkdir(parents=True)
    allowed.parent.mkdir(parents=True)
    payload.to_parquet(forbidden, index=False)
    payload.to_parquet(allowed, index=False)
    frames = load_multitimeframe_frames(tmp_path, "SPUS")
    assert "15m" not in frames
    assert frames["1h"].attrs["source_path"].endswith(
        "interval=1h/source_interval=1h/bars.parquet"
    )


def test_multitimeframe_features_have_explicit_missingness(
    causal_frame: pd.DataFrame,
) -> None:
    scaler = CausalFeatureScaler.fit(causal_frame.iloc[:250])
    _, missingness = scaler.transform_row(causal_frame.iloc[300])
    index = scaler.feature_names.index("return_15m")
    assert missingness[index] == 1.0


def test_reward_profit_positive() -> None:
    reward = calculate_reward(RewardInput(net_return=0.01))
    assert reward.total > 0


def test_reward_loss_negative_and_asymmetric() -> None:
    profit = calculate_reward(RewardInput(net_return=0.01)).total
    loss = calculate_reward(RewardInput(net_return=-0.01)).total
    assert loss < 0
    assert abs(loss) > profit


def test_drawdown_penalty_is_convex() -> None:
    small = calculate_reward(RewardInput(drawdown=0.02))
    large = calculate_reward(RewardInput(drawdown=0.04))
    assert abs(large.components["drawdown_penalty"]) > 2 * abs(
        small.components["drawdown_penalty"]
    )


def test_cost_penalty() -> None:
    reward = calculate_reward(RewardInput(transaction_cost_return=0.002))
    assert reward.components["transaction_cost_penalty"] < 0


def test_turnover_penalty() -> None:
    reward = calculate_reward(RewardInput(turnover=2.0))
    assert reward.components["turnover_penalty"] < 0


def test_invalid_action_mask_without_position() -> None:
    mask = action_mask(PositionState(), HardGates())
    assert mask[RLAction.HOLD]
    assert not mask[RLAction.CLOSE]
    assert not mask[RLAction.REDUCE_50]


def test_risk_gate_cannot_be_overridden(causal_frame: pd.DataFrame) -> None:
    env = _environment(causal_frame)
    env.reset()
    env.frame.loc[env._index, "gate_risk_budget"] = False
    mask = env.action_masks()
    assert not any(mask[action] for action in (1, 2, 3))
    _, reward, _, _, info = env.step(int(RLAction.OPEN_LARGE))
    assert info["effective_action"] == "HOLD"
    assert info["invalid_action"] is True
    assert reward <= -RLRewardConfig().invalid_action_penalty


def test_shariah_gate_cannot_be_overridden(causal_frame: pd.DataFrame) -> None:
    env = _environment(causal_frame)
    env.reset()
    env.frame.loc[env._index, "gate_shariah"] = False
    mask = env.action_masks()
    assert not any(mask[action] for action in (1, 2, 3))


def test_execution_authority_gate_can_be_required(causal_frame: pd.DataFrame) -> None:
    config = SwingEnvironmentConfig(require_execution_authority=True)
    env = _environment(causal_frame, config=config)
    env.reset()
    assert not any(env.action_masks()[action] for action in (1, 2, 3))


def test_episode_terminal_reward_and_mark_to_market(causal_frame: pd.DataFrame) -> None:
    env = _environment(causal_frame)
    env.reset()
    action = _first_open_action(env)
    _, _, terminal, _, info = env.step(action)
    assert info["portfolio_return"] != 0
    while not terminal:
        _, _, terminal, _, info = env.step(int(RLAction.HOLD))
    assert info["episode_end_mark_to_market"] is True


def test_trade_episode_decision_types(causal_frame: pd.DataFrame) -> None:
    assert decision_type_for_action(RLAction.HOLD, position_open=False) == (
        "SKIP_DECISION"
    )
    assert decision_type_for_action(RLAction.HOLD, position_open=True) == (
        "HOLD_DECISION"
    )
    assert decision_type_for_action(RLAction.OPEN_SMALL, position_open=False) == (
        "ENTRY_DECISION"
    )
    assert decision_type_for_action(RLAction.REDUCE_25, position_open=True) == (
        "MANAGEMENT_DECISION"
    )
    assert decision_type_for_action(RLAction.CLOSE, position_open=True) == (
        "EXIT_DECISION"
    )
    env = _environment(causal_frame)
    env.reset()
    _, _, _, _, info = env.step(int(RLAction.HOLD))
    assert info["decision_type"] == "SKIP_DECISION"
    assert "episode_end_open_position_penalty" in info["reward_components"]


def test_open_position_state_tracks_mfe_mae(causal_frame: pd.DataFrame) -> None:
    env = _environment(causal_frame)
    env.reset()
    action = _first_open_action(env)
    env.step(action)
    context = env.decision_context
    assert context["position"]["weight"] > 0
    assert env.episode_summary["returns"]


def test_policy_registry_and_hash_verification(tmp_path: Path) -> None:
    model = tmp_path / "source.zip"
    model.write_bytes(b"not-a-real-model-but-an-immutable-test-artifact")
    registry = PolicyRegistry(tmp_path)
    record = registry.register(
        "ppo_v001",
        model_source=model,
        config={},
        reward_config={},
        feature_schema={},
        training_metadata={},
        evaluation={"status": "SHADOW_ONLY"},
        regime_performance={},
        cost_stress={},
        state=PolicyState.CHALLENGER,
    )
    assert record["state"] == "CHALLENGER"
    assert registry.verify("ppo_v001")["status"] == "GO"
    (tmp_path / record["path"] / "config.json").write_text(
        '{"tampered":true}', encoding="utf-8"
    )
    verification = registry.verify("ppo_v001")
    assert verification["status"] == "NO_GO"
    assert "POLICY_HASH_MISMATCH" in verification["blockers"]


def test_challenger_cannot_self_promote(tmp_path: Path) -> None:
    model = tmp_path / "source.zip"
    model.write_bytes(b"model")
    registry = PolicyRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.register(
            "bad",
            model_source=model,
            config={},
            reward_config={},
            feature_schema={},
            training_metadata={},
            evaluation={},
            regime_performance={},
            cost_stress={},
            state=PolicyState.ACTIVE,
        )


def test_shadow_has_no_execution_authority(causal_frame: pd.DataFrame) -> None:
    env = _environment(causal_frame)
    _, info = env.reset()
    assert info["execution_authority"] == "NONE"
    assert info["broker_writes"] == 0


def test_live_default_disabled() -> None:
    config = RLRuntimeConfig()
    config.validate()
    assert config.mode is RLMode.SHADOW_ONLY
    assert config.live_enabled is False
    with pytest.raises(ValueError):
        RLRuntimeConfig(mode=RLMode.LIVE).validate()


def test_retraining_idempotency(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path)
    assert store.claim_training_job("nightly-2026-08-11", {"rows": 100}) is True
    assert store.claim_training_job("nightly-2026-08-11", {"rows": 100}) is False
    assert (
        store.finish_training_job(
            "nightly-2026-08-11", status="COMPLETED", payload={"model": "v1"}
        )
        is True
    )
    assert store.claim_training_job("nightly-2026-08-11", {}) is False


def test_restart_recovery(tmp_path: Path) -> None:
    first = ExperienceStore(tmp_path)
    first.append_episode_event("episode-1", "OPENED", {"episode_id": "episode-1"})
    second = ExperienceStore(tmp_path)
    recovered = second.recover_open_episodes()
    assert recovered == [{"episode_id": "episode-1"}]


def test_experience_store_records_hold_and_outcome(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path)
    decision = ExperienceDecision(
        timestamp="2026-08-11T12:00:00+00:00",
        policy_version="ppo_v001",
        state_hash="ABC",
        observation=[0.0, 1.0],
        available_actions=[action.name for action in RLAction],
        action_mask=[1, 1, 1, 1, 0, 0, 0, 0],
        chosen_action="HOLD",
        action_probability=0.5,
        portfolio_state={"cash": 1.0},
        market_regime="BULL",
        signal_id="signal-1",
        strategy_id="strategy-1",
        asset="SPUS",
        episode_id="episode-1",
        decision_type="ENTRY_DECISION",
    )
    assert store.append_decision(decision) is True
    assert store.append_decision(decision) is False
    outcome = ExperienceOutcome(
        decision_id=decision.decision_id,
        episode_id="episode-1",
        timestamp="2026-08-12T12:00:00+00:00",
        reward=1.0,
        reward_components={"net_return_reward": 1.0},
        realized_pnl=10.0,
        unrealized_pnl=0.0,
        fees=0.1,
        slippage=0.2,
        mfe=0.02,
        mae=-0.01,
        holding_duration=1,
        outcome={"status": "CLOSED"},
    )
    assert store.append_outcome(outcome) is True
    statistics = store.statistics()
    assert statistics["hold_or_skip_count"] == 1
    assert statistics["net_pnl"] == pytest.approx(10.0)
    assert statistics["maximum_drawdown"] == pytest.approx(0.0)
    assert statistics["reward_by_regime"] == {"BULL": pytest.approx(1.0)}
    assert statistics["actions_by_regime"] == {"BULL": {"HOLD": 1}}


def test_missing_features_are_explicit(causal_frame: pd.DataFrame) -> None:
    scaler = CausalFeatureScaler.fit(causal_frame.iloc[:250])
    row = causal_frame.iloc[300]
    _, missingness = scaler.transform_row(row)
    feature_index = scaler.feature_names.index("news_sentiment")
    assert missingness[feature_index] == 1.0


def test_reward_decomposition_sums_exactly() -> None:
    reward = calculate_reward(
        RewardInput(
            net_return=0.01,
            transaction_cost_return=0.001,
            drawdown=0.02,
            turnover=1.0,
        )
    )
    assert reward.total == pytest.approx(sum(reward.components.values()))


def test_same_seed_reproducible(causal_frame: pd.DataFrame) -> None:
    first = _environment(causal_frame, seed=77)
    second = _environment(causal_frame, seed=77)
    first.reset(seed=77)
    second.reset(seed=77)
    action_a = _first_open_action(first)
    action_b = _first_open_action(second)
    result_a = first.step(action_a)
    result_b = second.step(action_b)
    assert result_a[1] == result_b[1]
    assert result_a[4]["transaction_cost_return"] == result_b[4][
        "transaction_cost_return"
    ]


def test_cost_stress_increases_costs(causal_frame: pd.DataFrame) -> None:
    scaler = CausalFeatureScaler.fit(causal_frame.iloc[:250])
    frame = causal_frame.iloc[250:360].reset_index(drop=True)
    report = cost_stress_evaluation(
        frame,
        scaler=scaler,
        base_config=SwingEnvironmentConfig(),
        policy=fixed_sizing_policy,
        policy_name="FIXED",
        seed=42,
        multipliers=(1.0, 2.0),
    )
    assert report["scenarios"]["2.00x"]["fees_and_slippage"] > report[
        "scenarios"
    ]["1.00x"]["fees_and_slippage"]


def test_regime_evaluation(causal_frame: pd.DataFrame) -> None:
    scaler = CausalFeatureScaler.fit(causal_frame.iloc[:250])
    result = evaluate_policy(
        causal_frame.iloc[250:360].reset_index(drop=True),
        scaler=scaler,
        environment_config=SwingEnvironmentConfig(),
        policy=fixed_sizing_policy,
        policy_name="FIXED",
        seed=42,
    )
    assert result["regime_performance"]
    assert all("net_return" in row for row in result["regime_performance"].values())


def test_opportunity_selection_masks_invalid_candidates() -> None:
    rows = []
    for date in pd.date_range("2026-01-01", periods=3, tz="UTC"):
        for rank in (1, 2):
            rows.append(
                {
                    "timestamp": date,
                    "rank": rank,
                    "asset": f"A{rank}",
                    "setup_score": 0.8,
                    "expected_return": 0.02,
                    "expected_risk": 0.01,
                    "expected_rr": 2.0,
                    "historical_expectancy": 0.01,
                    "regime_expectancy": 0.01,
                    "signal_confidence": 0.8,
                    "liquidity_score": 1.0,
                    "gate_risk_budget": True,
                    "gate_shariah": rank == 1,
                    "gate_tradeable": True,
                    "gate_liquidity": True,
                    "outcome_next_return": 0.01,
                }
            )
    env = OpportunitySelectionEnv(pd.DataFrame(rows), top_n=2)
    env.reset()
    assert env.action_masks().tolist() == [True, True, False]


def test_reward_config_rejects_non_convex_drawdown() -> None:
    with pytest.raises(ValueError):
        replace(RLRewardConfig(), drawdown_convex_power=1.0).validate()


def test_reward_hacking_never_trade_is_not_free_on_good_opportunity() -> None:
    reward = calculate_reward(RewardInput(skipped_opportunity_return=0.03))
    assert reward.components["skip_quality_reward"] < 0
    assert reward.total < 0


def test_reward_hacking_skip_losing_opportunity_is_rewarded() -> None:
    reward = calculate_reward(RewardInput(skipped_opportunity_return=-0.03))
    assert reward.components["skip_quality_reward"] > 0


def test_reward_hacking_overtrading_and_concentration_are_penalized() -> None:
    moderate = calculate_reward(
        RewardInput(turnover=1.0, concentration_excess=0.1)
    )
    excessive = calculate_reward(
        RewardInput(turnover=4.0, concentration_excess=0.4)
    )
    assert excessive.total < moderate.total < 0


def test_reward_hacking_immediate_round_trip_loses_to_costs(
    causal_frame: pd.DataFrame,
) -> None:
    flat = causal_frame.copy()
    flat.loc[:, "outcome_next_return"] = 0.0
    flat.loc[:, "outcome_next_high_return"] = 0.0
    flat.loc[:, "outcome_next_low_return"] = 0.0
    env = _environment(flat)
    env.reset()
    open_action = _first_open_action(env)
    _, open_reward, _, _, _ = env.step(open_action)
    _, close_reward, _, _, _ = env.step(int(RLAction.CLOSE))
    assert open_reward + close_reward < 0
