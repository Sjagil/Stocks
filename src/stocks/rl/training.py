from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.p3.io import atomic_write_json, file_hash, read_json
from stocks.rl.contracts import PolicyState, stable_hash
from stocks.rl.data import (
    OBSERVATION_FEATURES,
    CausalFeatureScaler,
    DatasetContract,
    dataset_manifest,
)
from stocks.rl.environment import FinanceSwingEnv, SwingEnvironmentConfig
from stocks.rl.evaluation import (
    PolicyDecision,
    bootstrap_probability_of_improvement,
    cost_stress_evaluation,
    evaluate_baselines,
    evaluate_policy,
    evaluate_promotion_gate,
)
from stocks.rl.registry import PolicyRegistry
from stocks.rl.reward import RLRewardConfig


@dataclass(frozen=True)
class PPOTrainingConfig:
    version: str = "PPO_MASKABLE_BASELINE_V1"
    seeds: tuple[int, ...] = (11, 29, 47)
    total_timesteps: int = 5_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.005
    vf_coef: float = 0.50
    policy_hidden: tuple[int, ...] = (64, 64)
    minimum_train_rows: int = 756
    validation_rows: int = 126
    test_rows: int = 126
    embargo_rows: int = 5
    walk_forward_step_rows: int = 252
    maximum_folds: int = 3
    training_window_policy: str = "EXPANDING_ALL_HISTORY_WITH_RECENT_APPEND"
    minimum_distinct_market_regimes: int = 3
    minimum_observations_per_market_regime: int = 50
    require_macro_regime_for_promotion: bool = True
    device: str = "cpu"

    def validate(self) -> None:
        if len(set(self.seeds)) < 3:
            raise ValueError("PPO baseline requires at least three distinct seeds")
        if self.total_timesteps < 1_000:
            raise ValueError("PPO training budget is too small")
        if self.n_steps < 8 or self.batch_size < 2:
            raise ValueError("invalid PPO rollout configuration")
        if self.minimum_train_rows < 252:
            raise ValueError("PPO chronological training window is too short")
        if min(self.validation_rows, self.test_rows, self.embargo_rows) < 1:
            raise ValueError("PPO temporal validation windows must be positive")
        if self.training_window_policy != "EXPANDING_ALL_HISTORY_WITH_RECENT_APPEND":
            raise ValueError("continual PPO must retain expanding historical replay")
        if self.minimum_distinct_market_regimes < 2:
            raise ValueError("PPO regime diversity requirement is too small")
        if self.minimum_observations_per_market_regime < 1:
            raise ValueError("PPO regime sample requirement must be positive")


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    def to_dict(self, frame: pd.DataFrame) -> dict[str, Any]:
        def timestamp(index: int) -> str:
            return pd.Timestamp(frame.iloc[index]["timestamp"]).isoformat()

        return {
            **asdict(self),
            "train_start_timestamp": timestamp(self.train_start),
            "train_end_timestamp": timestamp(self.train_end - 1),
            "validation_start_timestamp": timestamp(self.validation_start),
            "validation_end_timestamp": timestamp(self.validation_end - 1),
            "test_start_timestamp": timestamp(self.test_start),
            "test_end_timestamp": timestamp(self.test_end - 1),
        }


class SB3MaskablePolicyAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        context: dict[str, Any],
        rng: np.random.Generator,
    ) -> PolicyDecision:
        action, _ = self.model.predict(
            observation,
            deterministic=True,
            action_masks=action_mask,
        )
        probabilities = self._probabilities(observation, action_mask)
        return PolicyDecision(int(action), tuple(probabilities))

    def _probabilities(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> list[float]:
        import torch

        observation_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(
                observation_tensor,
                action_masks=np.asarray(action_mask, dtype=bool).reshape(1, -1),
            )
            values = distribution.distribution.probs.detach().cpu().numpy()[0]
        return [float(value) for value in values]


def purged_walk_forward_splits(
    frame: pd.DataFrame,
    config: PPOTrainingConfig,
) -> list[WalkForwardSplit]:
    config.validate()
    splits: list[WalkForwardSplit] = []
    train_end = config.minimum_train_rows
    fold = 1
    while len(splits) < config.maximum_folds:
        validation_start = train_end + config.embargo_rows
        validation_end = validation_start + config.validation_rows
        test_start = validation_end + config.embargo_rows
        test_end = test_start + config.test_rows
        if test_end > len(frame):
            break
        splits.append(
            WalkForwardSplit(
                fold=fold,
                train_start=0,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold += 1
        train_end += config.walk_forward_step_rows
    if not splits:
        raise ValueError("insufficient rows for purged walk-forward PPO training")
    return splits


def train_ppo_experiment(
    project_root: Path,
    frame: pd.DataFrame,
    contract: DatasetContract,
    *,
    training_config: PPOTrainingConfig | None = None,
    environment_config: SwingEnvironmentConfig | None = None,
    reward_config: RLRewardConfig | None = None,
    register_policy: bool = True,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    import torch

    root = project_root.resolve()
    training_config = training_config or PPOTrainingConfig()
    training_config.validate()
    environment_config = environment_config or SwingEnvironmentConfig(
        require_shariah_gate=False,
        require_quote_gate=False,
    )
    environment_config.validate()
    reward_config = reward_config or RLRewardConfig()
    reward_config.validate()
    torch.set_num_threads(1)

    splits = purged_walk_forward_splits(frame, training_config)
    regime_coverage = _regime_coverage(frame, training_config)
    experiment_id = stable_hash(
        {
            "training_config": asdict(training_config),
            "environment_config": asdict(environment_config),
            "reward_config": reward_config.to_dict(),
            "dataset": dataset_manifest(frame, contract),
        }
    )[:20]
    experiment_root = root / "output/rl/experiments" / experiment_id
    experiment_root.mkdir(parents=True, exist_ok=True)
    fold_reports: list[dict[str, Any]] = []
    canonical_candidates: list[dict[str, Any]] = []

    for split in splits:
        train_frame = frame.iloc[split.train_start : split.train_end].reset_index(drop=True)
        validation_frame = frame.iloc[
            split.validation_start : split.validation_end
        ].reset_index(drop=True)
        test_frame = frame.iloc[split.test_start : split.test_end].reset_index(drop=True)
        scaler = CausalFeatureScaler.fit(train_frame)
        fold_root = experiment_root / f"fold_{split.fold:02d}"
        fold_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(fold_root / "feature_scaler.json", scaler.to_dict())
        baseline_report = evaluate_baselines(
            test_frame,
            scaler=scaler,
            environment_config=environment_config,
            seed=training_config.seeds[0],
        )
        seed_reports: list[dict[str, Any]] = []
        for seed in training_config.seeds:
            environment = FinanceSwingEnv(
                train_frame,
                scaler=scaler,
                config=environment_config,
                reward_config=reward_config,
                seed=seed,
            )
            model = MaskablePPO(
                "MlpPolicy",
                environment,
                learning_rate=training_config.learning_rate,
                n_steps=min(training_config.n_steps, len(train_frame) - 1),
                batch_size=training_config.batch_size,
                n_epochs=training_config.n_epochs,
                gamma=training_config.gamma,
                gae_lambda=training_config.gae_lambda,
                clip_range=training_config.clip_range,
                ent_coef=training_config.ent_coef,
                vf_coef=training_config.vf_coef,
                policy_kwargs={"net_arch": list(training_config.policy_hidden)},
                seed=seed,
                device=training_config.device,
                verbose=0,
            )
            model.learn(total_timesteps=training_config.total_timesteps)
            model_path = fold_root / f"ppo_seed_{seed}.zip"
            model.save(model_path)
            adapter = SB3MaskablePolicyAdapter(model)
            validation = evaluate_policy(
                validation_frame,
                scaler=scaler,
                environment_config=environment_config,
                policy=adapter,
                policy_name=f"PPO_SEED_{seed}",
                seed=seed,
            )
            test = evaluate_policy(
                test_frame,
                scaler=scaler,
                environment_config=environment_config,
                policy=adapter,
                policy_name=f"PPO_SEED_{seed}",
                seed=seed,
            )
            cost_stress = cost_stress_evaluation(
                test_frame,
                scaler=scaler,
                base_config=environment_config,
                policy=adapter,
                policy_name=f"PPO_SEED_{seed}",
                seed=seed,
            )
            seed_report = {
                "seed": seed,
                "model_path": model_path.relative_to(root).as_posix(),
                "model_hash": file_hash(model_path),
                "validation": validation,
                "test": test,
                "cost_stress": cost_stress,
            }
            seed_reports.append(seed_report)
        ordered = sorted(
            seed_reports,
            key=lambda row: float(row["validation"]["metrics"]["net_return"]),
        )
        selected = ordered[len(ordered) // 2]
        canonical_candidates.append(
            {
                **selected,
                "fold": split.fold,
                "scaler": scaler,
                "split": split,
                "baseline_report": baseline_report,
            }
        )
        fold_report = {
            "fold": split.to_dict(frame),
            "selection_rule": "MEDIAN_VALIDATION_NET_RETURN_NOT_BEST_SEED",
            "selected_seed": selected["seed"],
            "seed_summary": _seed_summary(seed_reports),
            "seeds": seed_reports,
            "baselines": baseline_report,
        }
        atomic_write_json(fold_root / "evaluation.json", fold_report)
        fold_reports.append(fold_report)

    canonical = canonical_candidates[-1]
    canonical_test = canonical["test"]
    deterministic = canonical["baseline_report"]["baselines"][
        "EXISTING_DETERMINISTIC_ENGINE"
    ]
    challenger_returns = _evaluation_returns_from_model(
        root / canonical["model_path"],
        frame.iloc[
            canonical["split"].test_start : canonical["split"].test_end
        ].reset_index(drop=True),
        scaler=canonical["scaler"],
        environment_config=environment_config,
        seed=int(canonical["seed"]),
    )
    baseline_returns = _evaluation_returns_from_policy(
        frame.iloc[
            canonical["split"].test_start : canonical["split"].test_end
        ].reset_index(drop=True),
        scaler=canonical["scaler"],
        environment_config=environment_config,
        policy_name="EXISTING_DETERMINISTIC_ENGINE",
        seed=int(canonical["seed"]),
    )
    probability = bootstrap_probability_of_improvement(
        challenger_returns,
        baseline_returns,
        seed=int(canonical["seed"]),
    )
    version = f"ppo_stocks_swing_{experiment_id.lower()}"
    promotion = evaluate_promotion_gate(
        challenger_version=version,
        active_version=None,
        challenger=canonical_test,
        baseline=deterministic,
        cost_stress=canonical["cost_stress"],
        episode_count=len(splits),
        bootstrap_probability=probability,
        safety_regression=False,
        data_blockers=[
            *contract.promotion_blockers(),
            *regime_coverage["promotion_blockers"],
        ],
    )
    report: dict[str, Any] = {
        "schema": "rl_ppo_walk_forward_experiment_v1",
        "status": (
            "SHADOW_ONLY" if not promotion.passed else "PROMOTION_ELIGIBLE"
        ),
        "experiment_id": experiment_id,
        "policy_version": version,
        "algorithm": "SB3_CONTRIB_MASKABLE_PPO",
        "canonical_baseline": "PPO",
        "action_masking": True,
        "selection_rule": "LAST_WALK_FORWARD_FOLD_MEDIAN_VALIDATION_SEED",
        "dataset": dataset_manifest(frame, contract),
        "training_config": _jsonable_training_config(training_config),
        "environment_config": environment_config.to_dict(),
        "reward_config": reward_config.to_dict(),
        "walk_forward_folds": fold_reports,
        "selected_fold": canonical["fold"],
        "selected_seed": canonical["seed"],
        "canonical_test_evaluation": canonical_test,
        "deterministic_baseline_evaluation": deterministic,
        "cost_stress": canonical["cost_stress"],
        "bootstrap_probability_of_improvement": probability,
        "promotion_decision": promotion.to_dict(),
        "regime_performance": canonical_test["regime_performance"],
        "regime_coverage": regime_coverage,
        "catastrophic_forgetting_controls": {
            "training_window_policy": training_config.training_window_policy,
            "historical_replay_retained": True,
            "newest_period_only_training_allowed": False,
            "fold_train_start_is_fixed": True,
            "fold_train_end_expands": True,
        },
        "data_promotion_blockers": [
            *contract.promotion_blockers(),
            *regime_coverage["promotion_blockers"],
        ],
        "historical_oos_is_forward": False,
        "forward_evidence_available": False,
        "rl_mode": "SHADOW_ONLY",
        "rl_live_enabled": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "money_control": False,
        "generated_at": _now(),
    }
    report["content_hash"] = stable_hash(report)
    atomic_write_json(experiment_root / "report.json", report)

    if register_policy:
        registry = PolicyRegistry(root)
        registry.register(
            version,
            model_source=root / canonical["model_path"],
            config={
                "algorithm": "SB3_CONTRIB_MASKABLE_PPO",
                "training": _jsonable_training_config(training_config),
                "environment": environment_config.to_dict(),
                "mode": "SHADOW_ONLY",
                "rl_live_enabled": False,
                "execution_authority": "NONE",
            },
            reward_config=reward_config.to_dict(),
            feature_schema={
                "schema": "rl_feature_schema_v1",
                "observation_features": list(OBSERVATION_FEATURES),
                "scaler": canonical["scaler"].to_dict(),
            },
            training_metadata={
                "experiment_id": experiment_id,
                "training_start": canonical["split"].to_dict(frame)[
                    "train_start_timestamp"
                ],
                "training_end": canonical["split"].to_dict(frame)[
                    "train_end_timestamp"
                ],
                "selected_seed": canonical["seed"],
                "selection_rule": report["selection_rule"],
                "dataset_contract": asdict(contract),
                "regime_coverage": regime_coverage,
                "catastrophic_forgetting_controls": report[
                    "catastrophic_forgetting_controls"
                ],
            },
            evaluation=canonical_test,
            regime_performance=canonical_test["regime_performance"],
            cost_stress=canonical["cost_stress"],
            state=PolicyState.CHALLENGER,
        )
        registry.record_promotion_decision(promotion)
        report["registry_verification"] = registry.verify(version)
        atomic_write_json(experiment_root / "report.json", report)
    return report


def load_maskable_policy(path: Path) -> SB3MaskablePolicyAdapter:
    from sb3_contrib import MaskablePPO

    return SB3MaskablePolicyAdapter(MaskablePPO.load(path, device="cpu"))


def train_default_experiment(
    project_root: Path,
    *,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    from stocks.rl.data import load_causal_dataset
    from stocks.rl.data import load_multitimeframe_frames

    root = project_root.resolve()
    policy = read_json(root / "config/rl_policy_v1.json")
    raw_training = read_json(root / "config/rl_training_v1.json")
    scope = policy.get("initial_policy_scope", {})
    symbol = str(scope.get("symbol") or "SPUS").upper()
    path = root / f"data/research/critical_trading/yfinance/{symbol}.parquet"
    if not path.is_file():
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "blockers": [f"RL_PRICE_HISTORY_MISSING:{symbol}"],
            "execution_authority": "NONE",
            "broker_writes": 0,
        }
    fields = {
        name: value
        for name, value in raw_training.items()
        if name in PPOTrainingConfig.__dataclass_fields__
    }
    fields["seeds"] = tuple(int(value) for value in fields.get("seeds", (11, 29, 47)))
    fields["policy_hidden"] = tuple(
        int(value) for value in fields.get("policy_hidden", (64, 64))
    )
    if total_timesteps is not None:
        fields["total_timesteps"] = int(total_timesteps)
    training_config = PPOTrainingConfig(**fields)
    contract = DatasetContract(
        symbol=symbol,
        asset_class=str(scope.get("asset_class") or "ETF"),
        strategy_id="PPO_STOCKS_SWING_DECISION_LAYER_V1",
        timeframe=str(scope.get("timeframe") or "1d"),
        evidence_scope=str(scope.get("evidence_scope") or "UNRESTRICTED_RESEARCH_ONLY"),
        shariah_point_in_time_verified=False,
        point_in_time_universe_verified=False,
        survivorship_verified=False,
        source_name="LOCAL_YFINANCE_RESEARCH_CACHE",
        source_license="PUBLIC_RESEARCH_SOURCE_UNVERIFIED_FOR_PRODUCTION",
        source_version=str(file_hash(path) or "UNHASHED"),
    )
    frame = load_causal_dataset(
        path,
        contract,
        spread_bps=5.0,
        multitimeframe_frames=load_multitimeframe_frames(root, symbol),
    )
    return train_ppo_experiment(
        root,
        frame,
        contract,
        training_config=training_config,
        environment_config=SwingEnvironmentConfig(
            require_shariah_gate=False,
            require_quote_gate=False,
            require_execution_authority=False,
        ),
    )


def _evaluation_returns_from_model(
    model_path: Path,
    frame: pd.DataFrame,
    *,
    scaler: CausalFeatureScaler,
    environment_config: SwingEnvironmentConfig,
    seed: int,
) -> list[float]:
    return _collect_returns(
        frame,
        scaler=scaler,
        environment_config=environment_config,
        policy=load_maskable_policy(model_path),
        seed=seed,
    )


def _evaluation_returns_from_policy(
    frame: pd.DataFrame,
    *,
    scaler: CausalFeatureScaler,
    environment_config: SwingEnvironmentConfig,
    policy_name: str,
    seed: int,
) -> list[float]:
    from stocks.rl.evaluation import BASELINE_POLICIES

    return _collect_returns(
        frame,
        scaler=scaler,
        environment_config=environment_config,
        policy=BASELINE_POLICIES[policy_name],
        seed=seed,
    )


def _collect_returns(
    frame: pd.DataFrame,
    *,
    scaler: CausalFeatureScaler,
    environment_config: SwingEnvironmentConfig,
    policy: Any,
    seed: int,
) -> list[float]:
    env = FinanceSwingEnv(
        frame, scaler=scaler, config=environment_config, seed=seed
    )
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    terminal = False
    while not terminal:
        decision = policy(
            observation, env.action_masks(), env.decision_context, rng
        )
        observation, _, terminal, _, info = env.step(decision.action)
        values.append(float(info["portfolio_return"]))
    return values


def _seed_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    validation = [
        float(row["validation"]["metrics"]["net_return"]) for row in reports
    ]
    test = [float(row["test"]["metrics"]["net_return"]) for row in reports]
    return {
        "count": len(reports),
        "validation_mean": float(np.mean(validation)),
        "validation_median": float(np.median(validation)),
        "validation_worst": float(min(validation)),
        "validation_best": float(max(validation)),
        "test_mean": float(np.mean(test)),
        "test_median": float(np.median(test)),
        "test_worst": float(min(test)),
        "test_best": float(max(test)),
        "best_seed_not_used_as_selection_rule": True,
    }


def _regime_coverage(
    frame: pd.DataFrame, config: PPOTrainingConfig
) -> dict[str, Any]:
    market = frame["market_regime"].dropna().astype(str).value_counts()
    volatility = frame["volatility_regime"].dropna().astype(str).value_counts()
    trend = frame["trend_regime"].dropna().astype(str).value_counts()
    macro_available = int(frame["macro_regime"].notna().sum())
    blockers: list[str] = []
    sufficiently_sampled = int(
        (market >= config.minimum_observations_per_market_regime).sum()
    )
    if sufficiently_sampled < config.minimum_distinct_market_regimes:
        blockers.append("REGIME_DIVERSITY_INSUFFICIENT")
    if config.require_macro_regime_for_promotion and macro_available == 0:
        blockers.append("MACRO_REGIME_HISTORY_MISSING")
    return {
        "schema": "rl_regime_coverage_v1",
        "market_regime_counts": market.to_dict(),
        "volatility_regime_counts": volatility.to_dict(),
        "trend_regime_counts": trend.to_dict(),
        "macro_regime_observations": macro_available,
        "minimum_distinct_market_regimes": (
            config.minimum_distinct_market_regimes
        ),
        "minimum_observations_per_market_regime": (
            config.minimum_observations_per_market_regime
        ),
        "promotion_blockers": blockers,
        "status": "GO" if not blockers else "INSUFFICIENT_EVIDENCE",
    }


def _jsonable_training_config(config: PPOTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["seeds"] = list(config.seeds)
    payload["policy_hidden"] = list(config.policy_hidden)
    return payload


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "PPOTrainingConfig",
    "SB3MaskablePolicyAdapter",
    "WalkForwardSplit",
    "load_maskable_policy",
    "purged_walk_forward_splits",
    "train_ppo_experiment",
    "train_default_experiment",
]
