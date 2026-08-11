from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_statistical_artifacts(project_root: Path) -> dict[str, Any]:
    output = project_root / "output" / "ui" / "analytics"
    output.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    strategy = (
        project_root
        / "output"
        / "research"
        / "phase11_10"
        / "architecture-summary.csv"
    )
    if strategy.is_file():
        frame = pd.read_csv(strategy)
        eligible = frame.loc[
            frame["median_fill_count"].ge(20)
            & frame["median_oos_CAGR"].gt(0)
        ].copy()
        family_order = (
            eligible.groupby("entry_strategy")["median_oos_Sharpe"]
            .median()
            .sort_values(ascending=False)
            .head(18)
            .index
        )
        eligible = eligible.loc[
            eligible["entry_strategy"].isin(family_order)
        ]
        for metric, filename, title, center in (
            (
                "median_oos_Sharpe",
                "strategy-timeframe-sharpe.png",
                "Median nested OOS Sharpe by strategy and entry timeframe",
                0.0,
            ),
            (
                "cost_50bps_median_pf",
                "strategy-timeframe-cost-pf.png",
                "Median portfolio PF after 50 bps per side",
                1.0,
            ),
        ):
            pivot = eligible.pivot_table(
                index="entry_strategy",
                columns="lower_timeframe",
                values=metric,
                aggfunc="median",
            ).reindex(family_order)
            _heatmap(
                pivot,
                output / filename,
                title=title,
                center=center,
                fmt=".2f",
            )
            generated.append(filename)
    correlation = (
        project_root
        / "output"
        / "portfolio"
        / "correlation_matrix.parquet"
    )
    if correlation.is_file():
        frame = pd.read_parquet(correlation)
        if "ticker" in frame:
            frame = frame.set_index("ticker")
        symbols = list(frame.index[:20])
        matrix = frame.loc[symbols, symbols].apply(
            pd.to_numeric, errors="coerce"
        )
        _heatmap(
            matrix,
            output / "portfolio-correlation.png",
            title="Current research opportunity correlation",
            center=0.0,
            fmt=".2f",
        )
        generated.append("portfolio-correlation.png")
    return {
        "status": "GO" if generated else "NO_DATA",
        "generated": generated,
        "output": "output/ui/analytics",
    }


def _heatmap(
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str,
    center: float,
    fmt: str,
) -> None:
    if frame.empty:
        return
    sns.set_theme(style="dark")
    height = max(4.5, 0.42 * len(frame.index) + 1.8)
    width = max(7.5, 1.2 * len(frame.columns) + 3.5)
    figure, axis = plt.subplots(figsize=(width, height), dpi=150)
    figure.patch.set_facecolor("#101416")
    axis.set_facecolor("#171c1f")
    sns.heatmap(
        frame,
        ax=axis,
        annot=True,
        fmt=fmt,
        cmap="vlag",
        center=center,
        linewidths=0.5,
        linecolor="#2d363a",
        cbar_kws={"shrink": 0.7},
    )
    axis.set_title(title, color="#e8edee", pad=14, fontsize=12)
    axis.set_xlabel("Entry timeframe", color="#94a1a6")
    axis.set_ylabel("Strategy family", color="#94a1a6")
    axis.tick_params(colors="#c8d1d4", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#2d363a")
    figure.tight_layout()
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)
