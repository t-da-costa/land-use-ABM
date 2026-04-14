##################################################################
## Plotting helpers for the ABM-LU model outputs.
##################################################################

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


LAND_USE_CODES = {"I": 0, "O": 1, "S": 2}
LAND_USE_LABELS = {"I": "Intensive", "O": "Organic", "S": "Conservation"}
LAND_USE_COLORS = ["#b54b4b", "#4b8f5c", "#d4a84f"]


def _extract_grids(landscape) -> tuple[np.ndarray, np.ndarray]:
    q_grid = np.zeros((landscape.n_rows, landscape.n_cols), dtype=float)
    owner_grid = np.full((landscape.n_rows, landscape.n_cols), fill_value=-1, dtype=int)

    for plot in landscape.plots.values():
        q_grid[plot.x, plot.y] = plot.q
        owner_grid[plot.x, plot.y] = -1 if plot.owner is None else int(plot.owner)

    return q_grid, owner_grid


def land_use_array_to_grid(landscape, land_use_by_plot: np.ndarray) -> np.ndarray:
    grid = np.zeros((landscape.n_rows, landscape.n_cols), dtype=int)
    for plot_id, plot in landscape.plots.items():
        grid[plot.x, plot.y] = LAND_USE_CODES[str(land_use_by_plot[plot_id])]
    return grid


def plot_quality_with_farm_borders(
    landscape,
    figsize: tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    border_color: str = "white",
    border_linewidth: float = 1.0,
    show_axes: bool = False,
    title: str = "Initial plot quality and farm boundaries",
    savepath: Optional[Path] = None,
):
    q_grid, owner_grid = _extract_grids(landscape)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        q_grid,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    n_rows, n_cols = owner_grid.shape
    for x in range(n_rows):
        for y in range(n_cols - 1):
            if owner_grid[x, y] != owner_grid[x, y + 1]:
                ax.plot(
                    [y + 0.5, y + 0.5],
                    [x - 0.5, x + 0.5],
                    color=border_color,
                    linewidth=border_linewidth,
                )

    for x in range(n_rows - 1):
        for y in range(n_cols):
            if owner_grid[x, y] != owner_grid[x + 1, y]:
                ax.plot(
                    [y - 0.5, y + 0.5],
                    [x + 0.5, x + 0.5],
                    color=border_color,
                    linewidth=border_linewidth,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Initial plot quality q")

    ax.set_title(title)
    ax.set_aspect("equal")
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(summary_df, scenario_configs, savepath: Path) -> None:
    metrics = [
        ("avg_quality", "Average land quality"),
        ("avg_environment", "Average environmental quality"),
        ("total_output", "Total output"),
        ("mean_farmer_profit_before_subsidy", "Mean farmer profit before subsidy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for ax, (column, title) in zip(axes, metrics):
        for scenario in scenario_configs:
            scenario_df = summary_df[summary_df["scenario"] == scenario.name]
            ax.plot(
                scenario_df["step"],
                scenario_df[column],
                label=scenario.label,
                color=scenario.color,
                linewidth=2.0,
            )
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.25, linestyle=":")

    axes[0].legend(loc="best", frameon=False)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_land_use_shares(summary_df, scenario_configs, savepath: Path) -> None:
    metrics = [
        ("share_I", "Intensive share"),
        ("share_O", "Organic share"),
        ("share_S", "Conservation share"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    for ax, (column, title) in zip(axes, metrics):
        for scenario in scenario_configs:
            scenario_df = summary_df[summary_df["scenario"] == scenario.name]
            ax.plot(
                scenario_df["step"],
                scenario_df[column],
                label=scenario.label,
                color=scenario.color,
                linewidth=2.0,
            )
        ax.set_ylabel("Share")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25, linestyle=":")

    axes[-1].set_xlabel("Step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.subplots_adjust(top=0.9, hspace=0.2)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_land_use_maps(representative_runs: Iterable[dict], scenario_configs, savepath: Path) -> None:
    scenario_lookup = {scenario.name: scenario for scenario in scenario_configs}
    runs = list(representative_runs)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 5.0))
    if len(runs) == 1:
        axes = [axes]

    cmap = ListedColormap(LAND_USE_COLORS)

    for ax, run in zip(axes, runs):
        land_use_grid = land_use_array_to_grid(run["landscape"], run["final_land_use"])
        ax.imshow(
            land_use_grid,
            origin="lower",
            cmap=cmap,
            vmin=-0.5,
            vmax=2.5,
            interpolation="nearest",
        )
        ax.set_title(scenario_lookup[run["scenario"]].label)
        ax.set_xticks([])
        ax.set_yticks([])

    legend_handles = [
        Patch(facecolor=LAND_USE_COLORS[LAND_USE_CODES[code]], edgecolor="none", label=LAND_USE_LABELS[code])
        for code in ("I", "O", "S")
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.14, wspace=0.12)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
