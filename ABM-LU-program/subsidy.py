##################################################################
## Subsidy rules for the ABM-LU model.
##################################################################

from __future__ import annotations

from typing import Optional

from parameters import DEFAULT_PARAMETERS, ModelParameters


class SubsidyModel:
    """Implements the proposal's three stylized subsidy designs."""

    def __init__(self, params: Optional[ModelParameters] = None) -> None:
        self.params = params or DEFAULT_PARAMETERS

    def payment(
        self,
        subsidy_type: str | None,
        current_land_use: str,
        previous_land_use: str,
        next_environment: float,
        conversion_support_remaining: int = 0,
    ) -> float:
        if subsidy_type is None:
            return 0.0
        if subsidy_type == "practice":
            return self.practice_payment(current_land_use)
        if subsidy_type == "conversion":
            return self.conversion_payment(
                current_land_use=current_land_use,
                previous_land_use=previous_land_use,
                conversion_support_remaining=conversion_support_remaining,
            )
        if subsidy_type == "results":
            return self.results_payment(next_environment)
        raise ValueError("subsidy_type must be one of {None, 'practice', 'conversion', 'results'}")

    def practice_payment(self, land_use: str) -> float:
        if land_use == "O":
            return float(self.params.s_O)
        if land_use == "S":
            return float(self.params.s_S)
        return 0.0

    def conversion_payment(
        self,
        current_land_use: str,
        previous_land_use: str,
        conversion_support_remaining: int = 0,
    ) -> float:
        if current_land_use != "O":
            return 0.0
        if previous_land_use == "I":
            return float(self.params.s_C)
        if conversion_support_remaining > 0:
            return float(self.params.s_C)
        return 0.0

    def next_conversion_support(
        self,
        current_land_use: str,
        previous_land_use: str,
        conversion_support_remaining: int = 0,
    ) -> int:
        if current_land_use == "O" and previous_land_use == "I":
            return max(self.params.conversion_duration - 1, 0)
        if current_land_use == "O" and conversion_support_remaining > 0:
            return conversion_support_remaining - 1
        return 0

    def results_payment(self, next_environment: float) -> float:
        if self.params.results_payment_mode == "threshold":
            return float(self.params.beta if next_environment >= self.params.results_threshold else 0.0)
        if self.params.results_payment_mode == "continuous":
            return float(self.params.beta * max(next_environment, 0.0))
        raise ValueError("results_payment_mode must be 'continuous' or 'threshold'")
