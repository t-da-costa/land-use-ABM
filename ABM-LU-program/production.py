from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import parameters as p


LandUse = str  # expected values: "I", "O", "S"


@dataclass
class ProductionParameters:
    """
    Parameters for yields and baseline production costs.

    The defaults are read from parameters.py when available; otherwise,
    conservative placeholders are used so the module remains runnable.
    """

    # Yield parameters from the proposal
    alpha_I: float = getattr(p, "alpha_I", 1.20)
    alpha_O: float = getattr(p, "alpha_O", 0.90)

    gamma_I: float = getattr(p, "gamma_I", 1.00)
    gamma_O: float = getattr(p, "gamma_O", 1.20)

    # Baseline cost parameters from the proposal
    c_I: float = getattr(p, "c_I", 0.40)
    c_O: float = getattr(p, "c_O", 0.30)
    c_S: float = getattr(p, "c_S", 0.00)

    kappa_I: float = getattr(p, "kappa_I", 0.15)
    kappa_O: float = getattr(p, "kappa_O", 0.25)
    kappa_S: float = getattr(p, "kappa_S", 0.00)

    sigma_eta_I: float = getattr(p, "sigma_eta_I", 0.010)
    sigma_eta_O: float = getattr(p, "sigma_eta_O", 0.015)
    seed: Optional[int] = getattr(p, "seed", None)


class ProductionModel:
    """
    Production and baseline cost module.

    for LandUse in {I, O}:
      y(LandUse, q) = alpha_LandUse * q**gamma_LandUse 
      c(LandUse, q) = c_LandUse + kappa_LandUse * (1 - q) 

      y(S, q) = 0, c(S, q) = 0

    Farmer-specific heterogeneity is supported through optional technology-specificvcost shocks eta_{i,LandUse}, which can either be passed explicitly or read from attributes on the farmer object.
    """

    def __init__(self, params: Optional[ProductionParameters] = None) -> None:
        self.params = params or ProductionParameters()
        self.rng = np.random.default_rng(self.params.seed)

    # ------------------------------------------------------------------
    # Yield functions y(a_{i,k,t}, q_{k,t})
    # ------------------------------------------------------------------
    def yield_fn(
        self,
        land_use: LandUse,
        q: float,
        plot: Optional[dict] = None,
        farmer: Optional[object] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """Evaluate plot-level output under the chosen land use."""
        self._validate_land_use(land_use)
        q = self._validate_q(q)

        if land_use == "I":
            value = self.params.alpha_I * (q ** self.params.gamma_I)
        elif land_use == "O":
            value = self.params.alpha_O * (q ** self.params.gamma_O)
        else:  # land_use == "S"
            value = self.params.alpha_S * (q ** self.params.gamma_S)

        return self._validate_numeric_output(value, "yield")

    # ------------------------------------------------------------------
    # Baseline production costs c(a, q)
    # ------------------------------------------------------------------
    def baseline_cost(
        self,
        land_use: LandUse,
        q: float,
        plot: Optional[dict] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Baseline plot-level cost before farmer-specific heterogeneity.
        """
        self._validate_land_use(land_use)
        q = self._validate_q(q)

        if land_use == "I":
            value = self.params.c_I + self.params.kappa_I * (1.0 - q)
        elif land_use == "O":
            value = self.params.c_O + self.params.kappa_O * (1.0 - q)
        else:  # land_use == "S"
            value = self.params.c_S + self.params.kappa_S * (1.0 - q)

        value = self._validate_numeric_output(value, "baseline cost")
        if value < 0:
            raise ValueError("Baseline production cost must be non-negative")
        return value

    def farmer_cost_shock(
        self,
        land_use: LandUse,
        farmer: Optional[object] = None,
        eta_I: Optional[float] = None,
        eta_O: Optional[float] = None,
    ) -> float:
        """
        Farmer-specific technology / management cost shock.
        """
        self._validate_land_use(land_use)

        if land_use == "S":
            return 0.0

        if land_use == "I":
            if eta_I is not None:
                return float(eta_I)
            if farmer is not None:
                self.ensure_farmer_cost_shocks(farmer)
                return float(farmer.eta_I)
            return 0.0

        if land_use == "O":
            if eta_O is not None:
                return float(eta_O)
            if farmer is not None:
                self.ensure_farmer_cost_shocks(farmer)
                return float(farmer.eta_O)
            return 0.0

        raise ValueError("Unreachable branch in farmer_cost_shock")

    def initialize_farmer_cost_shocks(self, farmer: object) -> None:
        """
        Draw farmer-specific cost shocks at initialization.

        Each farmer has a technology-specific cost components eta_{i,LandUse} drawn once from a normal distribution with mean 0 and standard deviation sigma_eta.
        """
        if farmer is None:
            raise ValueError("farmer cannot be None when initializing cost shocks")

        farmer.eta_I = float(self.rng.normal(loc=0.0, scale=self.params.sigma_eta))
        farmer.eta_O = float(self.rng.normal(loc=0.0, scale=self.params.sigma_eta))

    def ensure_farmer_cost_shocks(self, farmer: Optional[object]) -> None:
        """
        Make sure a farmer has eta_I and eta_O attributes.

        If they do not exist yet, they are drawn once and then kept fixed for all future periods.
        """
        if farmer is None:
            return
        if not hasattr(farmer, "eta_I") or not hasattr(farmer, "eta_O"):
            self.initialize_farmer_cost_shocks(farmer)

    def cost(
        self,
        land_use: LandUse,
        q: float,
        plot: Optional[dict] = None,
        farmer: Optional[object] = None,
        t: Optional[int] = None,
        eta_I: Optional[float] = None,
        eta_O: Optional[float] = None,
        **kwargs,
    ) -> float:
        """
        Total production cost including farmer-specific shocks.
        """
        baseline = self.baseline_cost(
            land_use=land_use,
            q=q,
            plot=plot,
            t=t,
            **kwargs,
        )
        shock = self.farmer_cost_shock(
            land_use=land_use,
            farmer=farmer,
            eta_I=eta_I,
            eta_O=eta_O,
        )
        total_cost = baseline + shock
        if total_cost < 0:
            total_cost = 0.0  # enforce non-negative total cost
            # raise ValueError("Total production cost must be non-negative")
        return total_cost
    # ------------------------------------------------------------------
    # Convenience accounting functions
    # ------------------------------------------------------------------
    def revenue(
        self,
        land_use: LandUse,
        q: float,
        price: Optional[float] = None,
        plot: Optional[dict] = None,
        farmer: Optional[object] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        if price is None:
            price = getattr(p, "output_price", 1.0)
        if price < 0:
            raise ValueError("Price must be non-negative")
        return price * self.yield_fn(
            land_use=land_use,
            q=q,
            plot=plot,
            farmer=farmer,
            t=t,
            **kwargs,
        )

    def profit_before_subsidy(
        self,
        land_use: LandUse,
        q: float,
        price: Optional[float] = None,
        plot: Optional[dict] = None,
        farmer: Optional[object] = None,
        t: Optional[int] = None,
        eta_I: Optional[float] = None,
        eta_O: Optional[float] = None,
        **kwargs,
    ) -> float:
        return self.revenue(
            land_use=land_use,
            q=q,
            plot=plot,
            farmer=farmer,
            t=t,
            **kwargs,
        ) - self.cost(
            land_use=land_use,
            q=q,
            plot=plot,
            farmer=farmer,
            t=t,
            eta_I=eta_I,
            eta_O=eta_O,
            **kwargs,
        )


    def aggregate_revenue_for_farmer(
        self,
        farmer: object,
        plots_by_id: dict,
        land_use_by_plot: dict,
        price: Optional[float] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Compute aggregate revenue for one farmer by summing plot-level revenue
        over all plots owned by that farmer.

        Parameters
        ----------
        farmer : object
            Farmer object expected to have a `plot_ids` attribute.
        plots_by_id : dict
            Mapping plot_id -> plot dictionary.
        land_use_by_plot : dict
            Mapping plot_id -> current land use.
        price : float | None
            Output price. If None, uses parameters.output_price.
        t : int | None
            Optional time index.

        Returns
        -------
        float
            Total revenue for the farmer.
        """
        if farmer is None or not hasattr(farmer, "plot_ids"):
            raise ValueError("farmer must have a 'plot_ids' attribute")

        total_revenue = 0.0
        for plot_id in farmer.plot_ids:
            if plot_id not in plots_by_id:
                raise KeyError(f"Plot {plot_id} is missing from plots_by_id")
            if plot_id not in land_use_by_plot:
                raise KeyError(f"Plot {plot_id} is missing from land_use_by_plot")

            plot = plots_by_id[plot_id]
            land_use = land_use_by_plot[plot_id]
            q = plot["q"]

            total_revenue += self.revenue(
                land_use=land_use,
                q=q,
                price=price,
                plot=plot,
                farmer=farmer,
                t=t,
                **kwargs,
            )

        return total_revenue

    def aggregate_profit_before_subsidy_for_farmer(
        self,
        farmer: object,
        plots_by_id: dict,
        land_use_by_plot: dict,
        price: Optional[float] = None,
        t: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Compute aggregate profit before subsidy for one farmer by summing
        plot-level profit over all plots owned by that farmer.

        Parameters
        ----------
        farmer : object
            Farmer object expected to have a `plot_ids` attribute.
        plots_by_id : dict
            Mapping plot_id -> plot dictionary.
        land_use_by_plot : dict
            Mapping plot_id -> current land use.
        price : float | None
            Output price. If None, uses parameters.output_price.
        t : int | None
            Optional time index.

        Returns
        -------
        float
            Total profit before subsidy for the farmer.
        """
        if farmer is None or not hasattr(farmer, "plot_ids"):
            raise ValueError("farmer must have a 'plot_ids' attribute")

        total_profit = 0.0
        for plot_id in farmer.plot_ids:
            if plot_id not in plots_by_id:
                raise KeyError(f"Plot {plot_id} is missing from plots_by_id")
            if plot_id not in land_use_by_plot:
                raise KeyError(f"Plot {plot_id} is missing from land_use_by_plot")

            plot = plots_by_id[plot_id]
            land_use = land_use_by_plot[plot_id]
            q = plot["q"]

            total_profit += self.profit_before_subsidy(
                land_use=land_use,
                q=q,
                price=price,
                plot=plot,
                farmer=farmer,
                t=t,
                **kwargs,
            )

        return total_profit


    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_land_use(land_use: LandUse) -> None:
        if land_use not in {"I", "O", "S"}:
            raise ValueError("land_use must be one of {'I', 'O', 'S'}")

    @staticmethod
    def _validate_q(q: float) -> float:
        if not isinstance(q, (int, float)):
            raise TypeError("q must be numeric")
        if q < 0 or q > 1:
            raise ValueError("q must lie in [0, 1]")
        return float(q)

    @staticmethod
    def _validate_numeric_output(value: float, name: str) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} function must return a numeric value")
        return float(value)


# ----------------------------------------------------------------------
# Convenience constructor
# ----------------------------------------------------------------------
def build_production_model(
    params: Optional[ProductionParameters] = None,
) -> ProductionModel:
    """Convenience constructor."""
    return ProductionModel(params=params)
