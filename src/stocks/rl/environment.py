from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from stocks.rl.contracts import (
    HardGates,
    PositionState,
    RLAction,
    action_mask,
    action_name,
    decision_type_for_action,
    stable_hash,
    validate_observation,
)
from stocks.rl.data import (
    DYNAMIC_FEATURES,
    OBSERVATION_FEATURES,
    OUTCOME_COLUMNS,
    STATIC_FEATURES,
    CausalFeatureScaler,
)
from stocks.rl.reward import (
    RLRewardConfig,
    RewardInput,
    calculate_reward,
)


@dataclass(frozen=True)
class SwingEnvironmentConfig:
    version: str = "FINANCE_SWING_ENV_V1"
    starting_equity: float = 100_000.0
    small_weight: float = 0.05
    normal_weight: float = 0.10
    large_weight: float = 0.125
    maximum_position_weight: float = 0.25
    maximum_drawdown: float = 0.10
    commission_bps: float = 1.0
    spread_bps: float = 5.0
    slippage_bps_mean: float = 2.0
    slippage_bps_std: float = 1.0
    market_impact_bps: float = 1.0
    fx_bps: float = 0.0
    cost_stress_multiplier: float = 1.0
    require_shariah_gate: bool = True
    require_quote_gate: bool = True
    require_execution_authority: bool = False
    minimum_setup_score: float = 0.50
    initial_stop_atr: float = 2.0
    initial_target_atr: float = 3.5
    tightened_stop_atr: float = 1.5
    maximum_holding_steps: int = 30

    def validate(self) -> None:
        if self.starting_equity <= 0:
            raise ValueError("starting equity must be positive")
        weights = (self.small_weight, self.normal_weight, self.large_weight)
        if not (0 < weights[0] <= weights[1] <= weights[2]):
            raise ValueError("approved RL weights must be ordered and positive")
        if weights[2] > self.maximum_position_weight or self.maximum_position_weight > 1:
            raise ValueError("RL sizing exceeds the deterministic position cap")
        if not 0 < self.maximum_drawdown < 1:
            raise ValueError("maximum drawdown must be between zero and one")
        costs = (
            self.commission_bps,
            self.spread_bps,
            self.slippage_bps_mean,
            self.slippage_bps_std,
            self.market_impact_bps,
            self.fx_bps,
        )
        if any(value < 0 or not math.isfinite(value) for value in costs):
            raise ValueError("environment costs must be finite and non-negative")
        if self.cost_stress_multiplier < 1.0:
            raise ValueError("cost stress multiplier cannot reduce costs")
        if self.maximum_holding_steps < 1:
            raise ValueError("maximum holding steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


class FinanceSwingEnv(gym.Env[np.ndarray, int]):
    """Causal close-to-next-close trade-level swing environment.

    The environment emits counterfactual proposals only. It has no broker
    object, no order adapter and no execution authority. All position sizes are
    bounded aliases supplied by the deterministic risk layer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        scaler: CausalFeatureScaler | None = None,
        config: SwingEnvironmentConfig | None = None,
        reward_config: RLRewardConfig | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.frame = _validate_frame(frame)
        self.config = config or SwingEnvironmentConfig()
        self.config.validate()
        self.reward_config = reward_config or RLRewardConfig()
        self.reward_config.validate()
        self.scaler = scaler or CausalFeatureScaler.fit(self.frame)
        if tuple(self.scaler.feature_names) != tuple(STATIC_FEATURES):
            raise ValueError("feature scaler schema does not match RL observation schema")
        self.action_space = spaces.Discrete(len(RLAction))
        self.observation_space = spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(len(OBSERVATION_FEATURES),),
            dtype=np.float32,
        )
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._index = 0
        self._equity = self.config.starting_equity
        self._peak_equity = self.config.starting_equity
        self._position = PositionState()
        self._realized_return = 0.0
        self._last_reward = 0.0
        self._last_components: dict[str, float] = {}
        self._episode_id = ""
        self._actions: list[int] = []
        self._returns: list[float] = []
        self._costs: list[float] = []
        self._rewards: list[float] = []
        self._mfe = 0.0
        self._mae = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._index = int((options or {}).get("start_index", 0))
        if not 0 <= self._index < len(self.frame):
            raise ValueError("invalid RL episode start index")
        self._equity = self.config.starting_equity
        self._peak_equity = self.config.starting_equity
        self._position = PositionState()
        self._realized_return = 0.0
        self._last_reward = 0.0
        self._last_components = {}
        self._actions = []
        self._returns = []
        self._costs = []
        self._rewards = []
        self._mfe = 0.0
        self._mae = 0.0
        first = self.frame.iloc[self._index]
        self._episode_id = stable_hash(
            {
                "environment": self.config.version,
                "asset": first["asset"],
                "timestamp": str(first["timestamp"]),
                "seed": self._seed,
            }
        )[:24]
        observation = self._observation()
        return observation, self._info(invalid_action=False)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._index >= len(self.frame) - 1:
            raise RuntimeError("RL outcome is not available for the latest observation")
        row = self.frame.iloc[self._index]
        mask = self.action_masks()
        requested = int(action)
        invalid = requested < 0 or requested >= len(mask) or not bool(mask[requested])
        effective = int(RLAction.HOLD) if invalid else requested
        current_price = float(row["close"])
        atr = max(1e-12, float(row["atr"]))
        previous_weight = self._position.weight
        position_was_open = self._position.is_open
        realized_return = 0.0
        management_quality = 0.0

        if effective in {
            int(RLAction.OPEN_SMALL),
            int(RLAction.OPEN_NORMAL),
            int(RLAction.OPEN_LARGE),
        }:
            target_weight = self._approved_weight(RLAction(effective))
            self._position = PositionState(
                weight=target_weight,
                entry_price=current_price,
                current_price=current_price,
                stop_price=current_price - self.config.initial_stop_atr * atr,
                target_price=current_price + self.config.initial_target_atr * atr,
                peak_price=current_price,
                trough_price=current_price,
                holding_steps=0,
            )
        elif self._position.is_open:
            if effective == int(RLAction.REDUCE_25):
                realized_return = self._position_return(current_price) * 0.25
                self._position = _replace_position(
                    self._position, weight=self._position.weight * 0.75
                )
            elif effective == int(RLAction.REDUCE_50):
                realized_return = self._position_return(current_price) * 0.50
                self._position = _replace_position(
                    self._position, weight=self._position.weight * 0.50
                )
            elif effective == int(RLAction.CLOSE):
                realized_return = self._position_return(current_price)
                self._position = PositionState()
            elif effective == int(RLAction.TIGHTEN_STOP):
                tightened = max(
                    self._position.stop_price,
                    current_price - self.config.tightened_stop_atr * atr,
                )
                management_quality = max(
                    0.0,
                    (tightened - self._position.stop_price) / max(atr, 1e-12),
                )
                self._position = _replace_position(
                    self._position, stop_price=min(current_price, tightened)
                )

        turnover = abs(self._position.weight - previous_weight)
        transaction_cost = self._transaction_cost(turnover)
        market_return = float(row["outcome_next_return"])
        next_high = current_price * (1.0 + float(row["outcome_next_high_return"]))
        next_low = current_price * (1.0 + float(row["outcome_next_low_return"]))
        next_price = current_price * (1.0 + market_return)
        forced_close = False
        applied_market_return = market_return

        if self._position.is_open:
            if next_low <= self._position.stop_price:
                applied_market_return = self._position.stop_price / current_price - 1.0
                next_price = self._position.stop_price
                forced_close = True
            elif self._position.holding_steps + 1 >= self.config.maximum_holding_steps:
                forced_close = True

        portfolio_return = self._position.weight * applied_market_return - transaction_cost
        self._equity *= max(0.0, 1.0 + portfolio_return)
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = max(0.0, 1.0 - self._equity / self._peak_equity)
        self._realized_return += realized_return

        if self._position.is_open:
            peak = max(self._position.peak_price, next_high)
            trough = min(self._position.trough_price, next_low)
            self._mfe = max(
                self._mfe, peak / max(self._position.entry_price, 1e-12) - 1.0
            )
            self._mae = min(
                self._mae, trough / max(self._position.entry_price, 1e-12) - 1.0
            )
            if forced_close:
                realized_return += self._position_return(next_price)
                close_cost = self._transaction_cost(self._position.weight)
                transaction_cost += close_cost
                portfolio_return -= close_cost
                self._equity *= max(0.0, 1.0 - close_cost)
                self._position = PositionState()
            else:
                self._position = _replace_position(
                    self._position,
                    current_price=next_price,
                    peak_price=peak,
                    trough_price=trough,
                    holding_steps=self._position.holding_steps + 1,
                )

        terminal = self._index >= len(self.frame) - 2
        episode_end_open = terminal and self._position.is_open
        if episode_end_open:
            close_cost = self._transaction_cost(self._position.weight)
            transaction_cost += close_cost
            portfolio_return -= close_cost
            self._equity *= max(0.0, 1.0 - close_cost)
            realized_return += self._position_return(next_price)
            self._position = PositionState()

        skipped_return = (
            market_return
            if previous_weight <= 1e-12
            and effective == int(RLAction.HOLD)
            and float(row["setup_score"]) >= self.config.minimum_setup_score
            else 0.0
        )
        if effective in {int(RLAction.REDUCE_25), int(RLAction.REDUCE_50), int(RLAction.CLOSE)}:
            management_quality += max(0.0, -market_return) * 100.0

        breakdown = calculate_reward(
            RewardInput(
                net_return=portfolio_return,
                realized_return=realized_return,
                transaction_cost_return=transaction_cost,
                drawdown=drawdown,
                downside_return=min(0.0, portfolio_return),
                risk_excess=max(
                    0.0, self._position.weight - self.config.maximum_position_weight
                ),
                concentration_excess=max(
                    0.0, self._position.weight - self.config.maximum_position_weight
                ),
                turnover=turnover,
                bad_execution=0.0,
                invalid_action=invalid,
                good_management=management_quality,
                skipped_opportunity_return=skipped_return,
                episode_end_open_position=episode_end_open,
            ),
            self.reward_config,
        )
        self._last_reward = breakdown.total
        self._last_components = breakdown.components
        self._actions.append(effective)
        self._returns.append(portfolio_return)
        self._costs.append(transaction_cost)
        self._rewards.append(breakdown.total)
        self._index += 1
        observation = self._terminal_observation() if terminal else self._observation()
        info = self._info(invalid_action=invalid)
        info.update(
            {
                "requested_action": action_name(requested),
                "effective_action": action_name(effective),
                "decision_type": decision_type_for_action(
                    effective, position_open=position_was_open
                ),
                "terminal_event_type": (
                    "EXIT_DECISION"
                    if forced_close or episode_end_open
                    else None
                ),
                "reward_components": breakdown.components,
                "portfolio_return": portfolio_return,
                "market_return": market_return,
                "transaction_cost_return": transaction_cost,
                "turnover": turnover,
                "realized_return": realized_return,
                "forced_close": forced_close,
                "episode_end_mark_to_market": terminal,
            }
        )
        return observation, breakdown.total, terminal, False, info

    def action_masks(self) -> np.ndarray:
        row = self.frame.iloc[self._index]
        gates = self._gates(row)
        return np.asarray(
            action_mask(
                self._position,
                gates,
                require_execution_authority=self.config.require_execution_authority,
                stop_can_tighten=self._position.is_open,
            ),
            dtype=bool,
        )

    def _gates(self, row: pd.Series) -> HardGates:
        return HardGates(
            risk_budget=(
                bool(row["gate_risk_budget"])
                and float(row["setup_score"]) >= self.config.minimum_setup_score
                and float(row["signal_direction"]) > 0
            ),
            shariah=(
                bool(row["gate_shariah"])
                if self.config.require_shariah_gate
                else True
            ),
            tradeable=bool(row["gate_tradeable"]),
            liquidity=bool(row["gate_liquidity"]),
            portfolio_capacity=bool(row["gate_portfolio_capacity"]),
            data_fresh=bool(row["gate_data_fresh"]),
            quote_valid=(
                bool(row["gate_quote_valid"])
                if self.config.require_quote_gate
                else True
            ),
            execution_authorized=False,
        )

    def _approved_weight(self, action: RLAction) -> float:
        values = {
            RLAction.OPEN_SMALL: self.config.small_weight,
            RLAction.OPEN_NORMAL: self.config.normal_weight,
            RLAction.OPEN_LARGE: self.config.large_weight,
        }
        return min(self.config.maximum_position_weight, values[action])

    def _position_return(self, price: float) -> float:
        if not self._position.is_open:
            return 0.0
        return price / max(self._position.entry_price, 1e-12) - 1.0

    def _transaction_cost(self, turnover: float) -> float:
        if turnover <= 0:
            return 0.0
        sampled_slippage = max(
            0.0,
            float(
                self._rng.normal(
                    self.config.slippage_bps_mean,
                    self.config.slippage_bps_std,
                )
            ),
        )
        bps = (
            self.config.commission_bps
            + self.config.spread_bps
            + sampled_slippage
            + self.config.market_impact_bps
            + self.config.fx_bps
        ) * self.config.cost_stress_multiplier
        return turnover * bps / 10_000.0

    def _observation(self) -> np.ndarray:
        row = self.frame.iloc[self._index]
        static, missingness = self.scaler.transform_row(row)
        current_price = float(row["close"])
        atr = max(float(row["atr"]), 1e-12)
        position_return = self._position_return(current_price)
        drawdown = max(0.0, 1.0 - self._equity / self._peak_equity)
        dynamic = {
            "cash_pct": 1.0 - self._position.weight,
            "current_exposure": self._position.weight,
            "open_risk": (
                self._position.weight
                * max(0.0, current_price - self._position.stop_price)
                / max(current_price, 1e-12)
                if self._position.is_open
                else 0.0
            ),
            "number_positions": 1.0 if self._position.is_open else 0.0,
            "unrealized_pnl": position_return * self._position.weight,
            "realized_daily_pnl": self._realized_return,
            "portfolio_drawdown": drawdown,
            "correlation_concentration": self._position.weight,
            "sector_concentration": self._position.weight,
            "asset_class_concentration": self._position.weight,
            "position_entry_price_ratio": (
                self._position.entry_price / current_price - 1.0
                if self._position.is_open
                else 0.0
            ),
            "position_unrealized_return": position_return,
            "position_atr_since_entry": (
                (current_price - self._position.entry_price) / atr
                if self._position.is_open
                else 0.0
            ),
            "position_mfe": self._mfe,
            "position_mae": self._mae,
            "position_holding_duration": self._position.holding_steps / max(
                1, self.config.maximum_holding_steps
            ),
            "position_distance_stop": (
                (current_price - self._position.stop_price) / atr
                if self._position.is_open
                else 0.0
            ),
            "position_distance_target": (
                (self._position.target_price - current_price) / atr
                if self._position.is_open
                else 0.0
            ),
            "position_trailing_stop_status": (
                1.0 if self._position.is_open and self._position.stop_price >= self._position.entry_price else 0.0
            ),
        }
        values = static + [dynamic[name] for name in DYNAMIC_FEATURES] + missingness
        validate_observation(values)
        return np.asarray(np.clip(values, -20.0, 20.0), dtype=np.float32)

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _info(self, *, invalid_action: bool) -> dict[str, Any]:
        row = self.frame.iloc[min(self._index, len(self.frame) - 1)]
        mask = self.action_masks() if self._index < len(self.frame) else np.zeros(len(RLAction), dtype=bool)
        return {
            "schema": "finance_swing_env_info_v1",
            "episode_id": self._episode_id,
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "asset": str(row["asset"]),
            "strategy_id": str(row["strategy_id"]),
            "market_regime": float(row["market_regime"]),
            "equity": self._equity,
            "position_weight": self._position.weight,
            "available_actions": [
                action_name(index) for index, allowed in enumerate(mask) if allowed
            ],
            "action_mask": mask.astype(int).tolist(),
            "invalid_action": invalid_action,
            "reward": self._last_reward,
            "reward_components": dict(self._last_components),
            "mfe": self._mfe,
            "mae": self._mae,
            "holding_duration": self._position.holding_steps,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "broker_writes": 0,
            "money_control": False,
        }

    @property
    def episode_summary(self) -> dict[str, Any]:
        returns = np.asarray(self._returns, dtype=float)
        return {
            "episode_id": self._episode_id,
            "steps": len(self._actions),
            "actions": list(self._actions),
            "returns": returns.tolist(),
            "costs": list(self._costs),
            "ending_equity": self._equity,
            "net_return": self._equity / self.config.starting_equity - 1.0,
            "total_cost_return": float(sum(self._costs)),
            "mean_reward": float(np.mean(self._rewards)) if self._rewards else 0.0,
            "total_reward": float(sum(self._rewards)),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }

    @property
    def decision_context(self) -> dict[str, Any]:
        row = self.frame.iloc[min(self._index, len(self.frame) - 1)]
        return {
            "row": row.to_dict(),
            "position": asdict(self._position),
            "equity": self._equity,
            "peak_equity": self._peak_equity,
            "drawdown": max(0.0, 1.0 - self._equity / self._peak_equity),
            "action_mask": self.action_masks().astype(int).tolist(),
            "execution_authority": "NONE",
        }


class OpportunitySelectionEnv(gym.Env[np.ndarray, int]):
    """Select one candidate from a prequalified top-N set, or explicitly skip."""

    metadata = {"render_modes": []}
    CANDIDATE_FEATURES = (
        "setup_score",
        "expected_return",
        "expected_risk",
        "expected_rr",
        "historical_expectancy",
        "regime_expectancy",
        "signal_confidence",
        "liquidity_score",
    )

    def __init__(
        self,
        candidates: pd.DataFrame,
        *,
        top_n: int = 10,
        cost_bps: float = 10.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if top_n < 1:
            raise ValueError("top_n must be positive")
        required = {
            "timestamp",
            "rank",
            "outcome_next_return",
            "gate_risk_budget",
            "gate_shariah",
            "gate_tradeable",
            "gate_liquidity",
            *self.CANDIDATE_FEATURES,
        }
        if not required.issubset(candidates.columns):
            raise ValueError("opportunity selection dataset schema incomplete")
        self.top_n = top_n
        self.cost_return = max(0.0, cost_bps) / 10_000.0
        ordered = candidates.copy()
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
        ordered = ordered.sort_values(["timestamp", "rank"])
        self._groups = [
            group.head(top_n).reset_index(drop=True)
            for _, group in ordered.groupby("timestamp", sort=True)
        ]
        if len(self._groups) < 2:
            raise ValueError("opportunity environment needs at least two periods")
        self.action_space = spaces.Discrete(top_n + 1)
        self.observation_space = spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(top_n * len(self.CANDIDATE_FEATURES) + top_n,),
            dtype=np.float32,
        )
        self._index = 0
        self._seed = seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._index = int((options or {}).get("start_index", 0))
        return self._observation(), self._info("SKIP", 0.0)

    def action_masks(self) -> np.ndarray:
        group = self._groups[self._index]
        mask = np.zeros(self.top_n + 1, dtype=bool)
        mask[0] = True
        for index, row in group.iterrows():
            mask[index + 1] = all(
                bool(row[name])
                for name in (
                    "gate_risk_budget",
                    "gate_shariah",
                    "gate_tradeable",
                    "gate_liquidity",
                )
            )
        return mask

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        mask = self.action_masks()
        invalid = action < 0 or action >= len(mask) or not mask[action]
        effective = 0 if invalid else int(action)
        group = self._groups[self._index]
        if effective == 0:
            best_valid = [
                float(row["outcome_next_return"])
                for index, row in group.iterrows()
                if mask[index + 1]
            ]
            reward = -max([0.0, *best_valid]) * 5.0
            label = "SKIP"
        else:
            row = group.iloc[effective - 1]
            reward = float(row["outcome_next_return"]) - self.cost_return
            label = str(row.get("asset") or f"CANDIDATE_{effective}")
        if invalid:
            reward -= 2.0
        self._index += 1
        terminal = self._index >= len(self._groups) - 1
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminal
            else self._observation()
        )
        return observation, float(reward), terminal, False, self._info(label, reward)

    def _observation(self) -> np.ndarray:
        group = self._groups[self._index]
        values: list[float] = []
        presence: list[float] = []
        for index in range(self.top_n):
            if index < len(group):
                row = group.iloc[index]
                values.extend(
                    _finite_or_zero(row[name]) for name in self.CANDIDATE_FEATURES
                )
                presence.append(1.0)
            else:
                values.extend([0.0] * len(self.CANDIDATE_FEATURES))
                presence.append(0.0)
        return np.asarray(np.clip(values + presence, -20, 20), dtype=np.float32)

    def _info(self, choice: str, reward: float) -> dict[str, Any]:
        group = self._groups[min(self._index, len(self._groups) - 1)]
        return {
            "timestamp": pd.Timestamp(group.iloc[0]["timestamp"]).isoformat(),
            "choice": choice,
            "reward": float(reward),
            "action_mask": self.action_masks().astype(int).tolist()
            if self._index < len(self._groups)
            else [0] * (self.top_n + 1),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp",
        "asset",
        "strategy_id",
        "close",
        "high",
        "low",
        "atr",
        "setup_score",
        "gate_risk_budget",
        "gate_shariah",
        "gate_tradeable",
        "gate_liquidity",
        "gate_portfolio_capacity",
        "gate_data_fresh",
        "gate_quote_valid",
        *STATIC_FEATURES,
        *OUTCOME_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"RL environment frame columns missing: {missing}")
    if len(frame) < 3:
        raise ValueError("RL environment requires at least three causal rows")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError("RL environment rows must be chronological")
    if result["timestamp"].duplicated().any():
        raise ValueError("RL environment timestamps must be unique")
    return result.reset_index(drop=True)


def _replace_position(position: PositionState, **changes: Any) -> PositionState:
    values = asdict(position)
    values.update(changes)
    return PositionState(**values)


def _finite_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


__all__ = [
    "FinanceSwingEnv",
    "OpportunitySelectionEnv",
    "SwingEnvironmentConfig",
]
