##################################################################
## Production and cost functions for the ABM-LU model.
##################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from parameters import DEFAULT_PARAMETERS, ModelParameters


LandUse = str  # expected values: "I", "O", "S"


@dataclass
class FarmerCostShocks:
    """Farmer-specific technology cost components."""

    eta_I: float
    eta_O: float


class ProductionModel:
    """Implements the proposal's yield, revenue, and cost equations."""

    def __init__(self, params: Optional[ModelParameters] = None) -> None:
        self.params = params or DEFAULT_PARAMETERS

    def yield_fn(self, land_use: LandUse, q: float) -> float:
        self._validate_land_use(land_use)
        q = self._validate_q(q)

        if land_use == "I":
            return float(self.params.alpha_I * (q ** self.params.gamma_I))
        if land_use == "O":
            return float(self.params.alpha_O * (q ** self.params.gamma_O))
        return float(self.params.alpha_S * (q ** self.params.gamma_S))

    def baseline_cost(self, land_use: LandUse, q: float) -> float:
        self._validate_land_use(land_use)
        q = self._validate_q(q)

        if land_use == "I":
            return float(self.params.c_I + self.params.kappa_I * (1.0 - q))
        if land_use == "O":
            return float(self.params.c_O + self.params.kappa_O * (1.0 - q))
        return float(self.params.c_S + self.params.kappa_S * (1.0 - q))

    def cost(
        self,
        land_use: LandUse,
        q: float,
        shocks: Optional[FarmerCostShocks] = None,
    ) -> float:
        total_cost = self.baseline_cost(land_use, q)

        if shocks is not None:
            if land_use == "I":
                total_cost += shocks.eta_I
            elif land_use == "O":
                total_cost += shocks.eta_O

        return float(max(total_cost, 0.0))

    def revenue(self, land_use: LandUse, q: float) -> float:
        if self.params.output_price < 0:
            raise ValueError("output_price must be non-negative")
        return float(self.params.output_price * self.yield_fn(land_use, q))

    def profit_before_subsidy(
        self,
        land_use: LandUse,
        q: float,
        shocks: Optional[FarmerCostShocks] = None,
    ) -> float:
        return float(self.revenue(land_use, q) - self.cost(land_use, q, shocks=shocks))

    def draw_cost_shocks(self, rng: np.random.Generator) -> FarmerCostShocks:
        """Draw farmer-specific cost shocks once at initialization."""

        return FarmerCostShocks(
            eta_I=float(rng.normal(loc=0.0, scale=self.params.sigma_eta_I)),
            eta_O=float(rng.normal(loc=0.0, scale=self.params.sigma_eta_O)),
        )

    @staticmethod
    def _validate_land_use(land_use: LandUse) -> None:
        if land_use not in {"I", "O", "S"}:
            raise ValueError("land_use must be one of {'I', 'O', 'S'}")

    @staticmethod
    def _validate_q(q: float) -> float:
        if not isinstance(q, (int, float)):
            raise TypeError("q must be numeric")
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must lie in [0, 1]")
        return float(q)
