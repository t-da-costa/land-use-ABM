##################################################################
## Environmental and land-quality dynamics for the ABM-LU model.
##################################################################

from __future__ import annotations

from typing import Optional

import numpy as np

from parameters import DEFAULT_PARAMETERS, ModelParameters


class EnvironmentalDynamics:
    """
    Updates environmental quality E and productive land quality q.

    The proposal states that land use affects the environmental state first,
    and land quality then responds to the ecological condition. To make the
    policy feedback operational, the implementation uses E_{t+1} when updating
    q_{t+1}.
    """

    def __init__(self, params: Optional[ModelParameters] = None) -> None:
        self.params = params or DEFAULT_PARAMETERS

    def environmental_delta(self, land_use: str) -> float:
        if land_use == "I":
            return -self.params.d_I
        if land_use == "O":
            return self.params.r_O
        if land_use == "S":
            return self.params.r_S
        raise ValueError("land_use must be one of {'I', 'O', 'S'}")

    def next_environment(self, current_environment: float, land_use: str) -> float:
        next_environment = current_environment + self.environmental_delta(land_use)
        return float(
            np.clip(
                next_environment,
                self.params.environmental_min,
                self.params.environmental_max,
            )
        )

    def next_quality(self, current_quality: float, next_environment: float) -> float:
        next_quality = current_quality + self.params.theta * next_environment
        return float(np.clip(next_quality, 0.0, 1.0))

    def transition(self, current_quality: float, current_environment: float, land_use: str) -> tuple[float, float]:
        next_environment = self.next_environment(current_environment, land_use)
        next_quality = self.next_quality(current_quality, next_environment)
        return next_environment, next_quality
