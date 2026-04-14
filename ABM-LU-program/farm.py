##################################################################
## This module initializes the type of land use of each plot in the landscape. 
##################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


LandUse = str  # expected values: "I", "O", "S"


@dataclass(frozen=True)
class PortfolioRule:
    """
    Initialization rule for one farm type.

    Parameters
    ----------
    share_of_farms : float
        Share of farms *within a size category* receiving this portfolio.
    share_I : float
        Share of owned plots allocated to intensive agriculture.
    share_O : float
        Share of owned plots allocated to organic agriculture.

    Notes
    -----
    The conservation share is implied as:
        share_S = 1 - share_I - share_O
    and must therefore be non-negative.
    """

    share_of_farms: float
    share_I: float
    share_O: float

    @property
    def share_S(self) -> float:
        return 1.0 - self.share_I - self.share_O


class FarmInitializer:
    """
    Initialize plot-level land use from farm size classes and portfolio rules.

    This module is intentionally general. The user can specify:
    - cutoffs defining small / medium / large farms from farm-size quantiles,
    - a list of portfolio rules for each size category,
    - how plots are selected within a farm when assigning I / O / S.

    The default within-farm allocation is quality-based:
    - highest-q plots receive intensive use first,
    - lowest-q plots receive conservation first,
    - remaining plots receive organic use.

    This makes it easy to reproduce initialization schemes such as:
    - large farms: 100% I or 90% I / 10% S,
    - small farms: 100% I or 100% O,
    - medium farms: mixtures of those patterns.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)


    def initialize_land_use_by_farm_size(
        self,
        farmer_plot_ids: Dict[int, List[int]],
        plots_by_id: Dict[int, dict],
        small_cutoff: float,
        large_cutoff: float,
        small_rules: Sequence[PortfolioRule],
        medium_rules: Sequence[PortfolioRule],
        large_rules: Sequence[PortfolioRule],
        allocation_mode: str = "quality_based",
    ) -> Dict[int, LandUse]:
        """
        Build an initial plot-level land-use allocation.

        Parameters
        ----------
        farmer_plot_ids : dict[int, list[int]]
            Mapping farmer_id -> list of owned plot IDs.
        plots_by_id : dict[int, dict]
            Mapping plot_id -> plot dictionary, expected to contain at least 'q'.
        small_cutoff : float
            Quantile cutoff defining small farms. Example: 0.50 means the bottom
            50% of farms by size are classified as small.
        large_cutoff : float
            Quantile cutoff defining large farms. Example: 0.90 means the top 10%
            of farms by size are classified as large.
        small_rules, medium_rules, large_rules : sequence[PortfolioRule]
            Portfolio mixtures for each size class.
        allocation_mode : str
            How plots are assigned within a farm. Supported values:
            - 'quality_based'
            - 'random'

        Returns
        -------
        dict[int, str]
            Mapping plot_id -> initial land use in {'I','O','S'}.
        """
        self._validate_cutoffs(small_cutoff, large_cutoff)
        self._validate_rule_set(small_rules, "small_rules")
        self._validate_rule_set(medium_rules, "medium_rules")
        self._validate_rule_set(large_rules, "large_rules")
        self._validate_allocation_mode(allocation_mode)

        size_class_by_farmer = self.classify_farms_by_size(
            farmer_plot_ids=farmer_plot_ids,
            small_cutoff=small_cutoff,
            large_cutoff=large_cutoff,
        )

        land_use_by_plot: Dict[int, LandUse] = {}

        for farmer_id, plot_ids in farmer_plot_ids.items():
            size_class = size_class_by_farmer[farmer_id]
            if size_class == "small":
                rules = small_rules
            elif size_class == "medium":
                rules = medium_rules
            else:
                rules = large_rules

            chosen_rule = self._draw_rule(rules)
            farmer_allocation = self.allocate_portfolio_within_farm(
                plot_ids=plot_ids,
                plots_by_id=plots_by_id,
                share_I=chosen_rule.share_I,
                share_O=chosen_rule.share_O,
                allocation_mode=allocation_mode,
            )
            land_use_by_plot.update(farmer_allocation)

        return land_use_by_plot

    def classify_farms_by_size(
        self,
        farmer_plot_ids: Dict[int, List[int]],
        small_cutoff: float,
        large_cutoff: float,
    ) -> Dict[int, str]:
        """
        Classify farms as 'small', 'medium', or 'large' using quantiles of farm size.
        """
        self._validate_cutoffs(small_cutoff, large_cutoff)

        farmer_ids = list(farmer_plot_ids.keys())
        sizes = np.array([len(farmer_plot_ids[fid]) for fid in farmer_ids], dtype=float)

        small_threshold = float(np.quantile(sizes, small_cutoff))
        large_threshold = float(np.quantile(sizes, large_cutoff))

        size_class_by_farmer: Dict[int, str] = {}
        for fid in farmer_ids:
            size = len(farmer_plot_ids[fid])
            if size <= small_threshold:
                size_class_by_farmer[fid] = "small"
            elif size >= large_threshold:
                size_class_by_farmer[fid] = "large"
            else:
                size_class_by_farmer[fid] = "medium"

        return size_class_by_farmer

    def allocate_portfolio_within_farm(
        self,
        plot_ids: Sequence[int],
        plots_by_id: Dict[int, dict],
        share_I: float,
        share_O: float,
        allocation_mode: str = "quality_based",
    ) -> Dict[int, LandUse]:
        """
        Allocate I / O / S to one farm's plots from the requested shares.

        Parameters
        ----------
        plot_ids : sequence[int]
            Owned plots of one farmer.
        plots_by_id : dict[int, dict]
            Plot mapping containing at least 'q'.
        share_I : float
            Share of plots allocated to intensive use.
        share_O : float
            Share of plots allocated to organic use.
        allocation_mode : str
            - 'quality_based': assign I to best plots, S to worst plots, O to middle
            - 'random': assign all land uses randomly across owned plots

        Returns
        -------
        dict[int, str]
            Mapping plot_id -> assigned land use.
        """
        self._validate_shares(share_I, share_O)
        self._validate_allocation_mode(allocation_mode)

        n_plots = len(plot_ids)
        if n_plots == 0:
            return {}

        n_I, n_O, n_S = self._shares_to_counts(n_plots, share_I, share_O)

        if allocation_mode == "random":
            ordered_plot_ids = list(plot_ids)
            self.rng.shuffle(ordered_plot_ids)
            I_ids = ordered_plot_ids[:n_I]
            O_ids = ordered_plot_ids[n_I:n_I + n_O]
            S_ids = ordered_plot_ids[n_I + n_O:]
        else:  # allocation_mode == 'quality_based'
            sorted_plot_ids = sorted(plot_ids, key=lambda pid: plots_by_id[pid]["q"])
            # Lowest-q plots go to conservation, highest-q plots go to intensive,
            # and the middle part goes to organic.
            S_ids = sorted_plot_ids[:n_S]
            I_ids = sorted_plot_ids[-n_I:] if n_I > 0 else []
            middle_start = n_S
            middle_end = len(sorted_plot_ids) - n_I
            O_ids = sorted_plot_ids[middle_start:middle_end]
            # Reorder I_ids so the very best plots are intensive.
            I_ids = sorted(I_ids, key=lambda pid: plots_by_id[pid]["q"], reverse=True)

        allocation: Dict[int, LandUse] = {}
        for pid in I_ids:
            allocation[pid] = "I"
        for pid in O_ids:
            allocation[pid] = "O"
        for pid in S_ids:
            allocation[pid] = "S"

        if len(allocation) != n_plots:
            raise RuntimeError("Initial farm allocation did not cover all owned plots")

        return allocation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _draw_rule(self, rules: Sequence[PortfolioRule]) -> PortfolioRule:
        shares = np.array([rule.share_of_farms for rule in rules], dtype=float)
        shares = shares / shares.sum()
        idx = int(self.rng.choice(len(rules), p=shares))
        return rules[idx]

    @staticmethod
    def _shares_to_counts(n_plots: int, share_I: float, share_O: float) -> Tuple[int, int, int]:
        share_S = 1.0 - share_I - share_O

        raw = np.array([share_I, share_O, share_S], dtype=float) * n_plots
        counts = np.floor(raw).astype(int)
        remainder = n_plots - int(counts.sum())

        if remainder > 0:
            residuals = raw - np.floor(raw)
            order = np.argsort(-residuals)
            for idx in order[:remainder]:
                counts[idx] += 1

        n_I, n_O, n_S = map(int, counts)
        if n_I + n_O + n_S != n_plots:
            raise RuntimeError("Shares could not be converted consistently into plot counts")
        return n_I, n_O, n_S

    @staticmethod
    def _validate_cutoffs(small_cutoff: float, large_cutoff: float) -> None:
        if not (0.0 < small_cutoff < 1.0):
            raise ValueError("small_cutoff must lie strictly between 0 and 1")
        if not (0.0 < large_cutoff < 1.0):
            raise ValueError("large_cutoff must lie strictly between 0 and 1")
        if small_cutoff >= large_cutoff:
            raise ValueError("small_cutoff must be strictly smaller than large_cutoff")

    @staticmethod
    def _validate_shares(share_I: float, share_O: float) -> None:
        if share_I < 0 or share_O < 0:
            raise ValueError("share_I and share_O must be non-negative")
        if share_I + share_O > 1.0:
            raise ValueError("share_I + share_O must be <= 1")

    def _validate_rule_set(self, rules: Sequence[PortfolioRule], name: str) -> None:
        if len(rules) == 0:
            raise ValueError(f"{name} cannot be empty")
        total_share = 0.0
        for rule in rules:
            if rule.share_of_farms < 0:
                raise ValueError(f"{name} contains a negative farm-share weight")
            self._validate_shares(rule.share_I, rule.share_O)
            total_share += rule.share_of_farms
        if not np.isclose(total_share, 1.0):
            raise ValueError(f"{name} must sum to 1 across share_of_farms")

    @staticmethod
    def _validate_allocation_mode(allocation_mode: str) -> None:
        if allocation_mode not in {"quality_based", "random"}:
            raise ValueError("allocation_mode must be either 'quality_based' or 'random'")


# ----------------------------------------------------------------------
# Convenience constructor
# ----------------------------------------------------------------------
def build_farm_initializer(seed: Optional[int] = None) -> FarmInitializer:
    return FarmInitializer(seed=seed)