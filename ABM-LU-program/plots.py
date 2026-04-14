##################################################################
## This module defines the graphing functions for visualizing the ABM-LU model.
##################################################################

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


####===============================================================####
### Defining the graphing functions
####===============================================================####


def _extract_grids(landscape) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 2D arrays for plot quality and ownership from a Landscape object.

    Returns
    -------
    q_grid : np.ndarray
        Array of shape (n_rows, n_cols) with plot quality values.
    owner_grid : np.ndarray
        Array of shape (n_rows, n_cols) with farmer IDs.
    """
    q_grid = np.zeros((landscape.n_rows, landscape.n_cols), dtype=float)
    owner_grid = np.full((landscape.n_rows, landscape.n_cols), fill_value=-1, dtype=int)

    for plot in landscape.plots.values():
        q_grid[plot.x, plot.y] = plot.q
        owner_grid[plot.x, plot.y] = -1 if plot.owner is None else int(plot.owner)

    return q_grid, owner_grid



def plot_quality_with_farm_borders(
    landscape,
    figsize: Tuple[float, float] = (8, 8),
    cmap: str = "viridis",
    border_color: str = "white",
    border_linewidth: float = 1.0,
    show_axes: bool = False,
    title: str = "Initial plot quality and farm boundaries",
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Plot initial land quality as a heatmap and overlay farm boundaries.

    Parameters
    ----------
    landscape : Landscape
        Landscape object from landscape.py.
    figsize : tuple[float, float]
        Matplotlib figure size.
    cmap : str
        Colormap for plot quality.
    border_color : str
        Color used for farm-boundary overlays.
    border_linewidth : float
        Width of boundary lines.
    show_axes : bool
        If False, remove axis ticks and labels for a cleaner map.
    title : str
        Figure title.
    savepath : str | None
        If provided, save the figure to this path.
    show : bool
        If True, display the figure with plt.show().

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    q_grid, owner_grid = _extract_grids(landscape)

    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap of plot quality
    im = ax.imshow(
        q_grid,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    # Draw farm boundaries by checking ownership changes across neighboring cells.
    n_rows, n_cols = owner_grid.shape

    # Vertical boundaries: compare left-right neighbors
    for x in range(n_rows):
        for y in range(n_cols - 1):
            if owner_grid[x, y] != owner_grid[x, y + 1]:
                ax.plot(
                    [y + 0.5, y + 0.5],
                    [x - 0.5, x + 0.5],
                    color=border_color,
                    linewidth=border_linewidth,
                )

    # Horizontal boundaries: compare up-down neighbors
    for x in range(n_rows - 1):
        for y in range(n_cols):
            if owner_grid[x, y] != owner_grid[x + 1, y]:
                ax.plot(
                    [y - 0.5, y + 0.5],
                    [x + 0.5, x + 0.5],
                    color=border_color,
                    linewidth=border_linewidth,
                )

    # Outer border of the landscape
    ax.plot([-0.5, n_cols - 0.5], [-0.5, -0.5], color=border_color, linewidth=border_linewidth)
    ax.plot([-0.5, n_cols - 0.5], [n_rows - 0.5, n_rows - 0.5], color=border_color, linewidth=border_linewidth)
    ax.plot([-0.5, -0.5], [-0.5, n_rows - 0.5], color=border_color, linewidth=border_linewidth)
    ax.plot([n_cols - 0.5, n_cols - 0.5], [-0.5, n_rows - 0.5], color=border_color, linewidth=border_linewidth)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Initial plot quality q")

    ax.set_title(title)
    ax.set_aspect("equal")

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

####===============================================================####
### Initial Land Use Map
####===============================================================####

def plot_initial_land_use_with_farm_borders(
    landscape,
    land_use_by_plot: dict,
    figsize: Tuple[float, float] = (8, 8),
    color_I: str = "#d73027",
    color_O: str = "#1a9850",
    color_S: str = "#4575b4",
    border_color: str = "white",
    border_linewidth: float = 1.0,
    show_axes: bool = False,
    title: str = "Initial land-use allocation and farm boundaries",
    savepath: Optional[str] = None,
    show: bool = True,
):
    """
    Plot the initialized land-use map and overlay farm boundaries.

    Parameters
    ----------
    landscape : Landscape
        Landscape object from landscape.py.
    land_use_by_plot : dict[int, str]
        Mapping plot_id -> land use in {'I', 'O', 'S'}.
    figsize : tuple[float, float]
        Matplotlib figure size.
    color_I, color_O, color_S : str
        Colors used for intensive, organic, and conservation land uses.
    border_color : str
        Color used for farm-boundary overlays.
    border_linewidth : float
        Width of boundary lines.
    show_axes : bool
        If False, remove axis ticks and labels for a cleaner map.
    title : str
        Figure title.
    savepath : str | None
        If provided, save the figure to this path.
    show : bool
        If True, display the figure with plt.show().

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    _, owner_grid = _extract_grids(landscape)

    land_use_grid = np.empty((landscape.n_rows, landscape.n_cols), dtype=object)
    land_use_to_code = {"S": 0, "O": 1, "I": 2}

    coded_grid = np.full((landscape.n_rows, landscape.n_cols), fill_value=np.nan, dtype=float)

    for plot in landscape.plots.values():
        if plot.plot_id not in land_use_by_plot:
            raise KeyError(f"Plot {plot.plot_id} is missing from land_use_by_plot")
        land_use = land_use_by_plot[plot.plot_id]
        if land_use not in land_use_to_code:
            raise ValueError("land_use_by_plot values must be one of {'I', 'O', 'S'}")
        land_use_grid[plot.x, plot.y] = land_use
        coded_grid[plot.x, plot.y] = land_use_to_code[land_use]

    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    cmap = ListedColormap([color_S, color_O, color_I])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        coded_grid,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    n_rows, n_cols = owner_grid.shape

    # Vertical boundaries: compare left-right neighbors
    for x in range(n_rows):
        for y in range(n_cols - 1):
            if owner_grid[x, y] != owner_grid[x, y + 1]:
                ax.plot(
                    [y + 0.5, y + 0.5],
                    [x - 0.5, x + 0.5],
                    color=border_color,
                    linewidth=border_linewidth,
                )

    # Horizontal boundaries: compare up-down neighbors
    for x in range(n_rows - 1):
        for y in range(n_cols):
            if owner_grid[x, y] != owner_grid[x + 1, y]:
                ax.plot(
                    [y - 0.5, y + 0.5],
                    [x + 0.5, x + 0.5],
                    color=border_color,
                    linewidth=border_linewidth,
                )

    # Outer border of the landscape
    ax.plot([-0.5, n_cols - 0.5], [-0.5, -0.5], color=border_color, linewidth=border_linewidth)
    ax.plot([-0.5, n_cols - 0.5], [n_rows - 0.5, n_rows - 0.5], color=border_color, linewidth=border_linewidth)
    ax.plot([-0.5, -0.5], [-0.5, n_rows - 0.5], color=border_color, linewidth=border_linewidth)
    ax.plot([n_cols - 0.5, n_cols - 0.5], [-0.5, n_rows - 0.5], color=border_color, linewidth=border_linewidth)

    legend_handles = [
        Patch(facecolor=color_S, edgecolor="none", label="Conservation"),
        Patch(facecolor=color_O, edgecolor="none", label="Organic"),
        Patch(facecolor=color_I, edgecolor="none", label="Intensive"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    ax.set_title(title)
    ax.set_aspect("equal")

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax





