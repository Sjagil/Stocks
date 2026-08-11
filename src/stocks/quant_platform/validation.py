from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray


class WalkForwardValidator:
    """Causal rolling/expanding splits with purge and embargo support."""

    def __init__(
        self,
        *,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        purge_size: int = 0,
        embargo_size: int = 0,
        expanding: bool = False,
    ):
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size and test_size must be positive")
        if purge_size < 0 or embargo_size < 0:
            raise ValueError("purge_size and embargo_size cannot be negative")
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.expanding = expanding

    def split(self, observations: int | pd.Index) -> list[WalkForwardFold]:
        count = observations if isinstance(observations, int) else len(observations)
        folds: list[WalkForwardFold] = []
        test_start = self.train_size + self.purge_size
        fold = 1
        while test_start + self.test_size <= count:
            train_end = test_start - self.purge_size
            train_start = 0 if self.expanding else train_end - self.train_size
            if train_start >= 0:
                test_end = test_start + self.test_size
                folds.append(
                    WalkForwardFold(
                        fold=fold,
                        train_indices=np.arange(train_start, train_end),
                        test_indices=np.arange(test_start, test_end),
                    )
                )
                fold += 1
            test_start += self.step_size + self.embargo_size
        return folds

    def evaluate(
        self,
        data: pd.DataFrame,
        *,
        parameter_grid: Iterable[dict[str, Any]],
        train_score: Callable[[pd.DataFrame, dict[str, Any]], float],
        test_score: Callable[[pd.DataFrame, dict[str, Any]], float],
    ) -> pd.DataFrame:
        parameters = list(parameter_grid)
        if not parameters:
            raise ValueError("parameter_grid cannot be empty")
        rows: list[dict[str, Any]] = []
        for fold in self.split(len(data)):
            train = data.iloc[fold.train_indices]
            test = data.iloc[fold.test_indices]
            scored = [(float(train_score(train, values)), values) for values in parameters]
            best_train_score, best = max(scored, key=lambda item: item[0])
            rows.append(
                {
                    "fold": fold.fold,
                    "train_start": train.index[0],
                    "train_end": train.index[-1],
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
                    "parameters": best,
                    "train_score": best_train_score,
                    "test_score": float(test_score(test, best)),
                }
            )
        return pd.DataFrame(rows)


def stability_report(results: pd.DataFrame) -> dict[str, Any]:
    required = {"train_score", "test_score", "parameters"}
    if not required <= set(results):
        raise ValueError("results require train_score, test_score and parameters")
    train = pd.to_numeric(results["train_score"], errors="coerce")
    test = pd.to_numeric(results["test_score"], errors="coerce")
    parameter_changes = sum(
        left != right
        for left, right in zip(results["parameters"], results["parameters"].iloc[1:], strict=False)
    )
    degradation = test - train
    return {
        "folds": len(results),
        "mean_train_score": float(train.mean()),
        "mean_test_score": float(test.mean()),
        "mean_out_of_sample_degradation": float(degradation.mean()),
        "positive_test_fraction": float((test > 0).mean()),
        "parameter_change_fraction": float(parameter_changes / max(len(results) - 1, 1)),
        "test_score_dispersion": float(test.std(ddof=0)),
        "research_only": True,
        "execution_authority": "NONE",
    }


def cost_stress(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: Iterable[float],
) -> pd.DataFrame:
    aligned = pd.concat(
        [
            pd.to_numeric(gross_returns, errors="coerce").rename("gross"),
            pd.to_numeric(turnover, errors="coerce").rename("turnover"),
        ],
        axis=1,
    ).dropna()
    rows = []
    for bps in cost_bps:
        value = float(bps)
        if value < 0:
            raise ValueError("cost_bps cannot be negative")
        net = aligned["gross"] - aligned["turnover"] * value / 10_000.0
        rows.append(
            {
                "cost_bps": value,
                "total_net_return": float((1 + net).prod() - 1),
                "mean_net_return": float(net.mean()),
                "positive_period_fraction": float((net > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def parameter_sensitivity(
    scores: pd.DataFrame,
    *,
    parameter_columns: Iterable[str],
    score_column: str = "score",
) -> dict[str, Any]:
    parameters = list(parameter_columns)
    missing = sorted(set([*parameters, score_column]) - set(scores))
    if missing:
        raise ValueError(f"missing sensitivity columns: {', '.join(missing)}")
    score = pd.to_numeric(scores[score_column], errors="coerce")
    best_index = score.idxmax()
    best = scores.loc[best_index]
    threshold = float(score.max() - score.std(ddof=0))
    plateau = scores.loc[score >= threshold]
    return {
        "best_parameters": {column: best[column] for column in parameters},
        "best_score": float(best[score_column]),
        "plateau_threshold": threshold,
        "plateau_fraction": float(len(plateau) / len(scores)),
        "score_range": float(score.max() - score.min()),
        "stable_plateau": bool(len(plateau) >= max(3, round(len(scores) * 0.2))),
    }
