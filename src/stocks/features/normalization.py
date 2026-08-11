from __future__ import annotations

from statistics import mean, median, pstdev


def z_score(value: float, population: list[float]) -> float:
    if not population:
        raise ValueError("population is required")
    sigma = pstdev(population)
    if sigma == 0:
        return 0.0
    return (value - mean(population)) / sigma


def robust_z_score(value: float, population: list[float]) -> float:
    if not population:
        raise ValueError("population is required")
    center = median(population)
    absolute_deviations = [abs(item - center) for item in population]
    mad = median(absolute_deviations)
    if mad == 0:
        return 0.0
    return (value - center) / (1.4826 * mad)


def winsorize(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ValueError("lower cannot exceed upper")
    return min(max(value, lower), upper)


def positive_normalize(raw_weights: dict[str, float]) -> dict[str, float]:
    positive = {key: max(value, 0.0) for key, value in raw_weights.items()}
    total = sum(positive.values())
    if total == 0:
        return {key: 0.0 for key in raw_weights}
    return {key: value / total for key, value in positive.items()}
