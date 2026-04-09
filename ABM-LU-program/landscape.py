##################################################################
## This module builds the spatial landscape with autocorrelated 
# land quality and allocates plots to farmers.
##################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "landscape.py requires scipy. Install it with `pip install scipy`."
    ) from exc


PlotID = int
FarmerID = int
Coord = Tuple[int, int]


@dataclass
class Plot:
    """
    One spatial plot in the landscape.

    Attributes
    ----------
    plot_id : int
        Unique plot identifier.
    x, y : int
        Grid coordinates.
    q : float
        Initial land quality in [0, 1].
    owner : int | None
        Farmer who owns the plot. None means unassigned.
    """

    plot_id: PlotID
    x: int
    y: int
    q: float
    owner: Optional[FarmerID] = None


class Landscape:
    """
    Spatial landscape with autocorrelated land quality and clustered farm allocation.

    This class implements the "Landscape" paragraph of the proposal:
    - a rectangular grid of K plots,
    - spatially autocorrelated initial land quality q_k,
    - farm sizes drawn from a lognormal distribution,
    - spatially clustered allocation of plots to farmers.
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        n_farmers: int,
        q_sigma: float,
        q_length_scale: float,
        farm_mu: float,
        farm_sigma: float,
        seed: Optional[int] = None,
    ) -> None:
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive")
        if n_farmers <= 0:
            raise ValueError("n_farmers must be positive")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_plots = n_rows * n_cols
        self.n_farmers = n_farmers
        self.q_sigma = q_sigma
        self.q_length_scale = q_length_scale
        self.farm_mu = farm_mu
        self.farm_sigma = farm_sigma
        self.rng = np.random.default_rng(seed)

        if self.n_farmers > self.n_plots:
            raise ValueError("Need N <= K: number of farmers cannot exceed number of plots")

        self.plots: Dict[PlotID, Plot] = {}
        self.plot_ids_by_coord: Dict[Coord, PlotID] = {}
        self.farmer_plot_ids: Dict[FarmerID, List[PlotID]] = {}

    
    def build(self) -> None:
        """Build the full landscape and allocate plots to farmers."""
        q_grid = self._generate_autocorrelated_quality()
        self._create_plots_from_quality_grid(q_grid)

        farm_sizes = self._draw_farm_sizes()
        self._allocate_plots_clustered(farm_sizes)

    def to_plot_dict(self) -> Dict[PlotID, Dict[str, float | int | None]]:
        """
        Export plots in a plain-dictionary format convenient for other modules.
        """
        return {
            plot_id: {
                "id": plot.plot_id,
                "x": plot.x,
                "y": plot.y,
                "q": plot.q,
                "owner": plot.owner,
            }
            for plot_id, plot in self.plots.items()
        }

    def get_neighbors(self, plot_id: PlotID) -> List[PlotID]:
        """
        Von Neumann neighborhood (up, down, left, right).
        """
        plot = self.plots[plot_id]
        candidates = [
            (plot.x - 1, plot.y),
            (plot.x + 1, plot.y),
            (plot.x, plot.y - 1),
            (plot.x, plot.y + 1),
        ]
        return [
            self.plot_ids_by_coord[(x, y)]
            for (x, y) in candidates
            if (x, y) in self.plot_ids_by_coord
        ]

    def distance_between_plots(self, plot_id_1: PlotID, plot_id_2: PlotID) -> float:
        p1 = self.plots[plot_id_1]
        p2 = self.plots[plot_id_2]
        return float(np.hypot(p1.x - p2.x, p1.y - p2.y))

    def summary(self) -> Dict[str, object]:
        """Basic diagnostics for quick checking."""
        q_vals = np.array([plot.q for plot in self.plots.values()])
        farm_sizes = {farmer_id: len(plot_ids) for farmer_id, plot_ids in self.farmer_plot_ids.items()}
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_plots": self.n_plots,
            "n_farmers": self.n_farmers,
            "q_min": float(q_vals.min()),
            "q_max": float(q_vals.max()),
            "q_mean": float(q_vals.mean()),
            "farm_sizes": farm_sizes,
        }

    # ------------------------------------------------------------------
    # Quality of land generation
    # ------------------------------------------------------------------
    def _generate_autocorrelated_quality(self) -> np.ndarray:
        """
        Generate q_k from spatially autocorrelated noise.

        We start from iid Gaussian noise epsilon_k ~ N(0, 1), smooth it with a
        Gaussian kernel, then rescale to [0, 1].
        """
        noise = self.rng.normal(loc=0.0, scale=1.0, size=(self.n_rows, self.n_cols))

        # Larger q_length_scale -> smoother landscape -> stronger local autocorrelation
        smoothed = gaussian_filter(noise, sigma=self.q_length_scale)

        # Optional extra scaling before normalization, mostly for transparency
        smoothed = self.q_sigma * smoothed

        q_min = smoothed.min()
        q_max = smoothed.max()
        if np.isclose(q_min, q_max):
            return np.full((self.n_rows, self.n_cols), 0.5)

        q_grid = (smoothed - q_min) / (q_max - q_min)
        return q_grid

    def _create_plots_from_quality_grid(self, q_grid: np.ndarray) -> None:
        plot_id = 0
        for x in range(self.n_rows):
            for y in range(self.n_cols):
                q = float(q_grid[x, y])
                self.plots[plot_id] = Plot(plot_id=plot_id, x=x, y=y, q=q, owner=None)
                self.plot_ids_by_coord[(x, y)] = plot_id
                plot_id += 1

    # ------------------------------------------------------------------
    # Farm-size generation
    # ------------------------------------------------------------------
    def _draw_farm_sizes(self) -> np.ndarray:
        """
        Draw lognormal farm sizes and rescale them so that:
        - each farmer has at least one plot,
        - total plots sum exactly to K.
        """
        raw_sizes = self.rng.lognormal(mean=self.farm_mu, sigma=self.farm_sigma, size=self.n_farmers)

        # Rescale to sum to the total number of plots
        scaled = raw_sizes / raw_sizes.sum() * self.n_plots

        # Floor at 1 plot per farmer
        sizes = np.floor(scaled).astype(int)
        sizes[sizes < 1] = 1

        # Adjust so total equals K exactly
        diff = self.n_plots - int(sizes.sum())

        if diff > 0:
            # Give extra plots to farmers with largest residuals
            residuals = scaled - np.floor(scaled)
            order = np.argsort(-residuals)
            idx = 0
            while diff > 0:
                sizes[order[idx % self.n_farmers]] += 1
                diff -= 1
                idx += 1
        elif diff < 0:
            # Remove plots from farmers with largest sizes, keeping minimum 1
            order = np.argsort(-sizes)
            idx = 0
            while diff < 0:
                farmer_idx = order[idx % self.n_farmers]
                if sizes[farmer_idx] > 1:
                    sizes[farmer_idx] -= 1
                    diff += 1
                idx += 1
                if idx > 10 * self.n_plots:
                    raise RuntimeError("Could not reconcile farm sizes with total number of plots")

        if int(sizes.sum()) != self.n_plots:
            raise RuntimeError("Farm sizes do not sum to total number of plots after adjustment")

        return sizes

    # ------------------------------------------------------------------
    # Clustered ownership allocation
    # ------------------------------------------------------------------
    def _allocate_plots_clustered(self, farm_sizes: np.ndarray) -> None:
        """
        Allocate plots to farmers using a seed-and-expand clustering rule.
        """
        unassigned = set(self.plots.keys())
        self.farmer_plot_ids = {farmer_id: [] for farmer_id in range(self.n_farmers)}

        # Step 1: seed one plot per farmer
        for farmer_id in range(self.n_farmers):
            if not unassigned:
                raise RuntimeError("No unassigned plots left while seeding farmers")
            seed_plot = int(self.rng.choice(list(unassigned)))
            self._assign_plot_to_farmer(seed_plot, farmer_id)
            unassigned.remove(seed_plot)

        # Step 2: expand each farm until target size is reached
        target_sizes = {farmer_id: int(farm_sizes[farmer_id]) for farmer_id in range(self.n_farmers)}

        # Because we already seeded one plot per farmer, the target must be at least 1
        for farmer_id, target in target_sizes.items():
            if target < 1:
                raise RuntimeError("Each farmer must have at least one plot")

        progress = True
        while unassigned and progress:
            progress = False
            for farmer_id in range(self.n_farmers):
                current_size = len(self.farmer_plot_ids[farmer_id])
                target_size = target_sizes[farmer_id]

                if current_size >= target_size:
                    continue

                frontier = self._frontier_unassigned_neighbors(farmer_id, unassigned)
                if frontier:
                    new_plot = int(self.rng.choice(frontier))
                else:
                    # If no neighboring plots remain, fall back to nearest unassigned plot
                    new_plot = self._nearest_unassigned_plot(farmer_id, unassigned)

                self._assign_plot_to_farmer(new_plot, farmer_id)
                unassigned.remove(new_plot)
                progress = True

        if unassigned:
            raise RuntimeError("Some plots remain unassigned after clustered allocation")

    def _assign_plot_to_farmer(self, plot_id: PlotID, farmer_id: FarmerID) -> None:
        self.plots[plot_id].owner = farmer_id
        self.farmer_plot_ids[farmer_id].append(plot_id)

    def _frontier_unassigned_neighbors(
        self,
        farmer_id: FarmerID,
        unassigned: set[PlotID],
    ) -> List[PlotID]:
        frontier: set[PlotID] = set()
        for owned_plot_id in self.farmer_plot_ids[farmer_id]:
            for neigh in self.get_neighbors(owned_plot_id):
                if neigh in unassigned:
                    frontier.add(neigh)
        return list(frontier)

    def _nearest_unassigned_plot(self, farmer_id: FarmerID, unassigned: set[PlotID]) -> PlotID:
        """
        Fallback rule when a farm's frontier is blocked.

        Returns the unassigned plot closest to any plot already owned by the farmer.
        """
        owned = self.farmer_plot_ids[farmer_id]
        if not owned:
            raise RuntimeError("Cannot compute nearest unassigned plot for farmer with no land")

        best_plot: Optional[PlotID] = None
        best_dist = np.inf

        for candidate in unassigned:
            candidate_dist = min(self.distance_between_plots(candidate, owned_plot) for owned_plot in owned)
            if candidate_dist < best_dist:
                best_dist = candidate_dist
                best_plot = candidate

        if best_plot is None:
            raise RuntimeError("Could not find a nearest unassigned plot")
        return best_plot


def build_landscape(
    n_rows: int,
    n_cols: int,
    n_farmers: int,
    q_sigma: float,
    q_length_scale: float,
    farm_mu: float,
    farm_sigma: float,
    seed: Optional[int] = None,
) -> Landscape:
    """
    Convenience constructor.
    """
    landscape = Landscape(
        n_rows=n_rows,
        n_cols=n_cols,
        n_farmers=n_farmers,
        q_sigma=q_sigma,
        q_length_scale=q_length_scale,
        farm_mu=farm_mu,
        farm_sigma=farm_sigma,
        seed=seed,
    )
    landscape.build()
    return landscape