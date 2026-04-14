##################################################################
## This module defines the subsidy designs of the model.
##################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from production import ProductionModel

import parameters as p


LandUse = str  # expected values: "I", "O", "S"
SubsidyType = str  # expected values: "practice", "conversion", "results"


@dataclass
class SubsidyParameters:
    """
    Parameters for the three subsidy designs.
    """

    # Practice-based subsidy
    s_O: float = getattr(p, "s_O", 0.10)
    s_S: float = getattr(p, "s_S", 0.22)

    # Conversion subsidy
    s_C: float = getattr(p, "s_C", 0.25)

    # Results-based subsidy
    beta: float = getattr(p, "beta", 0.15)
    B_bar: float = getattr(p, "B_bar", 0.0)
    results_mode: str = getattr(p, "results_mode", "continuous")  # or "threshold"


class SubsidyModel:
    """
    Implementing the three policy designs from the proposal.

    `plot_subsidy(...)` returns the subsidy
    attached to one plot under a given subsidy type.
    """

    def __init__(self, params: Optional[SubsidyParameters] = None) -> None:
        self.params = params or SubsidyParameters()

    # ------------------------------------------------------------------
    # Plot-level subsidies
    # ------------------------------------------------------------------
    def plot_subsidy(
        self,
        subsidy_type: SubsidyType,
        plot: dict,
        land_use: LandUse,
        previous_land_use: Optional[LandUse] = None,
        farmer: Optional[object] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Return the plot-level subsidy under the requested subsidy design.

        Parameters
        ----------
        subsidy_type : str
            One of {'practice', 'conversion', 'results'}.
        plot : dict
            Plot dictionary. For results-based payments, expected to contain 'E'.
        land_use : str
            Current land use in {'I', 'O', 'S'}.
        previous_land_use : str | None
            Previous-period land use, needed for conversion subsidy.
        farmer : object | None
            Included for interface compatibility.
        t : int | None
            Included for interface compatibility.

        Returns
        -------
        float
            Plot-level subsidy amount.
        """
        self._validate_subsidy_type(subsidy_type)
        self._validate_land_use(land_use)

        if subsidy_type == "practice":
            return self.practice_based_plot_subsidy(land_use)
        if subsidy_type == "conversion":
            return self.conversion_plot_subsidy(land_use, previous_land_use)
        return self.results_based_plot_subsidy(plot)

    def practice_based_plot_subsidy(self, land_use: LandUse) -> float:
        """
        Practice-based subsidy:
        - organic plots receive s_O
        - conservation plots receive s_S
        - intensive plots receive 0
        """
        self._validate_land_use(land_use)

        if land_use == "O":
            return float(self.params.s_O)
        if land_use == "S":
            return float(self.params.s_S)
        return 0.0

    def conversion_plot_subsidy(
        self,
        land_use: LandUse,
        previous_land_use: Optional[LandUse],
    ) -> float:
        """
        Conversion-to-organic subsidy:
        pay s_C only when a plot switches from I to O.
        """
        self._validate_land_use(land_use)
        if previous_land_use is not None:
            self._validate_land_use(previous_land_use)

        if previous_land_use == "I" and land_use == "O":
            return float(self.params.s_C)
        return 0.0

    def results_based_plot_subsidy(self, plot: dict) -> float:
        """
        Results-based subsidy using the plot's environmental state E.

        Two supported modes:
        - continuous: beta * E
        - threshold: beta * 1{E >= B_bar}
        """
        if "E" not in plot:
            raise KeyError("Results-based subsidy requires plot['E']")

        E = float(plot["E"])

        if self.params.results_mode == "continuous":
            return float(self.params.beta * E)
        if self.params.results_mode == "threshold":
            return float(self.params.beta * (1.0 if E >= self.params.B_bar else 0.0))

        raise ValueError("results_mode must be either 'continuous' or 'threshold'")

    # ------------------------------------------------------------------
    # Farmer-level aggregates
    # ------------------------------------------------------------------
    def aggregate_subsidy_for_farmer(
        self,
        subsidy_type: SubsidyType,
        farmer: object,
        plots_by_id: Dict[int, dict],
        land_use_by_plot: Dict[int, LandUse],
        previous_land_use_by_plot: Optional[Dict[int, LandUse]] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Aggregate subsidy for one farmer by summing plot-level subsidies over all
        plots owned by that farmer.
        """
        self._validate_subsidy_type(subsidy_type)
        if farmer is None or not hasattr(farmer, "plot_ids"):
            raise ValueError("farmer must have a 'plot_ids' attribute")

        total_subsidy = 0.0

        for plot_id in farmer.plot_ids:
            if plot_id not in plots_by_id:
                raise KeyError(f"Plot {plot_id} is missing from plots_by_id")
            if plot_id not in land_use_by_plot:
                raise KeyError(f"Plot {plot_id} is missing from land_use_by_plot")

            plot = plots_by_id[plot_id]
            land_use = land_use_by_plot[plot_id]
            previous_land_use = None
            if previous_land_use_by_plot is not None:
                previous_land_use = previous_land_use_by_plot.get(plot_id)

            total_subsidy += self.plot_subsidy(
                subsidy_type=subsidy_type,
                plot=plot,
                land_use=land_use,
                previous_land_use=previous_land_use,
                farmer=farmer,
                t=t,
                **kwargs,
            )

        return total_subsidy

    def plot_profit_after_subsidy(
        self,
        production_model: ProductionModel,
        subsidy_type: SubsidyType,
        plot: dict,
        land_use: LandUse,
        previous_land_use: Optional[LandUse] = None,
        farmer: Optional[object] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Compute plot-level profit after subsidy.

        This is defined as:
            profit before subsidy + plot-level subsidy

        The current plot-level profit before subsidy is delegated to production.py.
        """
        if production_model is None:
            raise ValueError("production_model cannot be None")
        if "q" not in plot:
            raise KeyError("Plot dictionary must contain a 'q' key")

        q = float(plot["q"])
        profit_before = production_model.profit_before_subsidy(
            land_use=land_use,
            q=q,
            plot=plot,
            farmer=farmer,
            t=t,
            **kwargs,
        )
        subsidy_val = self.plot_subsidy(
            subsidy_type=subsidy_type,
            plot=plot,
            land_use=land_use,
            previous_land_use=previous_land_use,
            farmer=farmer,
            t=t,
            **kwargs,
        )
        return profit_before + subsidy_val

    def aggregate_profit_after_subsidy_for_farmer(
        self,
        production_model: ProductionModel,
        subsidy_type: SubsidyType,
        farmer: object,
        plots_by_id: Dict[int, dict],
        land_use_by_plot: Dict[int, LandUse],
        previous_land_use_by_plot: Optional[Dict[int, LandUse]] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Compute aggregate profit after subsidy for one farmer by summing
        plot-level profit after subsidy over all plots owned by that farmer.
        """
        self._validate_subsidy_type(subsidy_type)
        if production_model is None:
            raise ValueError("production_model cannot be None")
        if farmer is None or not hasattr(farmer, "plot_ids"):
            raise ValueError("farmer must have a 'plot_ids' attribute")

        total_profit_after_subsidy = 0.0

        for plot_id in farmer.plot_ids:
            if plot_id not in plots_by_id:
                raise KeyError(f"Plot {plot_id} is missing from plots_by_id")
            if plot_id not in land_use_by_plot:
                raise KeyError(f"Plot {plot_id} is missing from land_use_by_plot")

            plot = plots_by_id[plot_id]
            land_use = land_use_by_plot[plot_id]
            previous_land_use = None
            if previous_land_use_by_plot is not None:
                previous_land_use = previous_land_use_by_plot.get(plot_id)

            total_profit_after_subsidy += self.plot_profit_after_subsidy(
                production_model=production_model,
                subsidy_type=subsidy_type,
                plot=plot,
                land_use=land_use,
                previous_land_use=previous_land_use,
                farmer=farmer,
                t=t,
                **kwargs,
            )

        return total_profit_after_subsidy

    # ------------------------------------------------------------------
    # Convenience method for regime logic
    # ------------------------------------------------------------------
    def subsidy_if_policy_regime(
        self,
        regime: str,
        subsidy_type: SubsidyType,
        plot: dict,
        land_use: LandUse,
        previous_land_use: Optional[LandUse] = None,
        farmer: Optional[object] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Return the relevant plot-level subsidy only if regime == 'policy'.
        Otherwise return 0.
        """
        if regime != "policy":
            return 0.0
        return self.plot_subsidy(
            subsidy_type=subsidy_type,
            plot=plot,
            land_use=land_use,
            previous_land_use=previous_land_use,
            farmer=farmer,
            t=t,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_land_use(land_use: LandUse) -> None:
        if land_use not in {"I", "O", "S"}:
            raise ValueError("land_use must be one of {'I', 'O', 'S'}")

    @staticmethod
    def _validate_subsidy_type(subsidy_type: SubsidyType) -> None:
        if subsidy_type not in {"practice", "conversion", "results"}:
            raise ValueError(
                "subsidy_type must be one of {'practice', 'conversion', 'results'}"
            )


# ----------------------------------------------------------------------
# Convenience constructor
# ----------------------------------------------------------------------
def build_subsidy_model(
    params: Optional[SubsidyParameters] = None,
) -> SubsidyModel:
    """Convenience constructor."""
    return SubsidyModel(params=params)