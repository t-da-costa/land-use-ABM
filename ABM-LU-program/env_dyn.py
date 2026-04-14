##################################################################
## This module defines the environmental and land-quality dynamics.
##################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import parameters as p


LandUse = str  # expected values: "I", "O", "S"


@dataclass
class EnvironmentalParameters:
    """
    Parameters governing environmental dynamics.
    -----
    - r_S > 0 : environmental improvement under conservation/set-aside
    - r_O > 0 : environmental improvement under organic use
    - d_I > 0 : environmental degradation under intensive use
    - theta > 0 : translation from environmental state to land-quality change
    - initial_E : default initial environmental quality if not already set on plots
    """

    r_S: float = getattr(p, "r_S", 0.06)
    r_O: float = getattr(p, "r_O", 0.03)
    d_I: float = getattr(p, "d_I", 0.07)
    theta: float = getattr(p, "theta", 0.02)
    initial_E: float = getattr(p, "initial_E", 0.0)

    q_min: float = 0.0
    q_max: float = 1.0


class EnvironmentalDynamics:
    """
    Environmental and land-quality transition module.

    The implementation is modular:
    - it updates one plot at a time or all plots at once,
    - it reads the current land use from a plot-level mapping,
    - it keeps q bounded in [0, 1] by truncation.
    """

    def __init__(self, params: Optional[EnvironmentalParameters] = None) -> None:
        self.params = params or EnvironmentalParameters()


    def initialize_environmental_state(self, plots_by_id: Dict[int, dict]) -> None:
        """
        Ensure every plot has an environmental state variable `E`.

        If a plot does not yet contain an 'E' key, it is initialized at
        parameters.initial_E.
        """
        for plot in plots_by_id.values():
            if "E" not in plot:
                plot["E"] = float(self.params.initial_E)

    def environmental_increment(self, land_use: LandUse) -> float:
        """
        Return the one-period increment in environmental quality for a land use.
        """
        self._validate_land_use(land_use)

        if land_use == "S":
            return float(self.params.r_S)
        if land_use == "O":
            return float(self.params.r_O)
        return -float(self.params.d_I)  # land_use == "I"

    def update_environment_for_plot(
        self,
        plot: dict,
        land_use: LandUse,
    ) -> float:
        """
        Update E for one plot and return the new value.

        Parameters
        ----------
        plot : dict
            Plot dictionary expected to contain an 'E' key. If absent, it is
            initialized at parameters.initial_E.
        land_use : str
            Current land use in {'I', 'O', 'S'}.
        """
        self._validate_land_use(land_use)

        if "E" not in plot:
            plot["E"] = float(self.params.initial_E)

        plot["E"] = float(plot["E"] + self.environmental_increment(land_use))
        return float(plot["E"])

    def update_land_quality_for_plot(self, plot: dict) -> float:
        """
        Update q for one plot using the current environmental state E.

        q_{k,t+1} = q_{k,t} + theta * E_{k,t}

        The updated q is truncated to [q_min, q_max].
        """
        if "q" not in plot:
            raise KeyError("Plot dictionary must contain a 'q' key")
        if "E" not in plot:
            plot["E"] = float(self.params.initial_E)

        q_new = float(plot["q"] + self.params.theta * plot["E"])
        q_new = min(max(q_new, self.params.q_min), self.params.q_max)
        plot["q"] = q_new
        return q_new

    def update_one_step(
        self,
        plots_by_id: Dict[int, dict],
        land_use_by_plot: Dict[int, LandUse],
    ) -> None:
        """
        Advance all plots by one time step.

        1. update environmental quality E using current land use,
        2. update land quality q using the updated E.
        """
        self.initialize_environmental_state(plots_by_id)

        for plot_id, plot in plots_by_id.items():
            if plot_id not in land_use_by_plot:
                raise KeyError(f"Plot {plot_id} is missing from land_use_by_plot")
            land_use = land_use_by_plot[plot_id]
            self.update_environment_for_plot(plot=plot, land_use=land_use)

        for plot in plots_by_id.values():
            self.update_land_quality_for_plot(plot)

    def environmental_summary(self, plots_by_id: Dict[int, dict]) -> dict:
        """
        Return simple diagnostics on environmental and land-quality states.
        """
        self.initialize_environmental_state(plots_by_id)

        q_vals = [float(plot["q"]) for plot in plots_by_id.values()]
        e_vals = [float(plot["E"]) for plot in plots_by_id.values()]

        return {
            "mean_q": sum(q_vals) / len(q_vals),
            "min_q": min(q_vals),
            "max_q": max(q_vals),
            "mean_E": sum(e_vals) / len(e_vals),
            "min_E": min(e_vals),
            "max_E": max(e_vals),
        }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_land_use(land_use: LandUse) -> None:
        if land_use not in {"I", "O", "S"}:
            raise ValueError("land_use must be one of {'I', 'O', 'S'}")


# ----------------------------------------------------------------------
# Convenience constructor
# ----------------------------------------------------------------------
def build_environmental_dynamics(
    params: Optional[EnvironmentalParameters] = None,
) -> EnvironmentalDynamics:
    """Convenience constructor."""
    return EnvironmentalDynamics(params=params)