from __future__ import annotations

import math
from collections import defaultdict


def winsorize(values: list[float], lower: float = 0.05, upper: float = 0.95) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    lo = ordered[min(len(ordered) - 1, max(0, int(math.floor((len(ordered) - 1) * lower))))]
    hi = ordered[min(len(ordered) - 1, max(0, int(math.ceil((len(ordered) - 1) * upper))))]
    return [min(max(value, lo), hi) for value in values]


def zscores(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def neutralize_by_group(rows: list[dict[str, object]], value_key: str, group_key: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(float(str(row[value_key])))
    group_z = {group: zscores(winsorize(values)) for group, values in grouped.items()}
    offsets: dict[str, int] = defaultdict(int)
    result = []
    for row in rows:
        group = str(row[group_key])
        index = offsets[group]
        result.append(group_z[group][index])
        offsets[group] += 1
    return result
