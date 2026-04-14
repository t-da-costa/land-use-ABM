##################################################################
## Central parameter definitions for the ABM-LU project.
##################################################################

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "results" / "andrew-ethan"
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"


@dataclass(frozen=True)
class ModelParameters:
    """Top-level parameter bundle for the land-use ABM."""

    seed: int = 67
    n_replicates: int = 5
    n_steps: int = 25

    ##### LANDSCAPE PARAMETERS #####
    n_rows: int = 20
    n_cols: int = 20
    n_farmers: int = 30
    farm_mu: float = 0.0
    farm_sigma: float = 1.0
    q_sigma: float = 1.0
    q_length_scale: float = 3.0

    ##### INITIAL CONDITIONS #####
    initial_land_use: str = "I"
    initial_environmental_quality: float = 0.0

    ##### PRODUCTION PARAMETERS #####
    output_price: float = 1.0
    alpha_I: float = 1.2
    organic_to_intensive_yield_ratio: float = 0.75
    gamma_I: float = 1.0
    gamma_O: float = 1.0
    alpha_S: float = 0.0
    gamma_S: float = 1.0

    ##### PRODUCTION COST PARAMETERS #####
    c_I: float = 0.35
    kappa_I: float = 0.30
    c_O: float = 0.28
    kappa_O: float = 0.20
    c_S: float = 0.0
    kappa_S: float = 0.0

    ##### FARMER HETEROGENEITY #####
    sigma_eta_I: float = 0.010
    sigma_eta_O: float = 0.015

    ##### ENVIRONMENTAL DYNAMICS #####
    r_S: float = 0.06
    r_O: float = 0.03
    d_I: float = 0.05
    theta: float = 0.02
    environmental_min: float = -1.0
    environmental_max: float = 1.0

    ##### PRACTICE-BASED SUBSIDY #####
    s_O: float = 0.02
    s_S: float = 0.06

    ##### CONVERSION SUBSIDY #####
    s_C: float = 0.08
    conversion_duration: int = 3

    ##### RESULTS-BASED SUBSIDY #####
    beta: float = 0.08
    results_threshold: float = 0.20
    results_payment_mode: str = "continuous"  # choose from: "continuous", "threshold"

    @property
    def alpha_O(self) -> float:
        return self.organic_to_intensive_yield_ratio * self.alpha_I

    def with_overrides(self, **overrides: object) -> "ModelParameters":
        return replace(self, **overrides)


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for one scenario shown in the project outputs."""

    name: str
    label: str
    regime: str
    subsidy_type: str | None = None
    color: str = "#4c566a"


DEFAULT_PARAMETERS = ModelParameters()

CORE_SCENARIOS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig(
        name="productivist",
        label="Productivist",
        regime="productivist",
        subsidy_type=None,
        color="#b54b4b",
    ),
    ScenarioConfig(
        name="market",
        label="Market",
        regime="market",
        subsidy_type=None,
        color="#2a6f97",
    ),
    ScenarioConfig(
        name="policy_practice",
        label="Policy: practice subsidy",
        regime="policy",
        subsidy_type="practice",
        color="#4b8f5c",
    ),
)

ROBUSTNESS_SCENARIOS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig(
        name="market",
        label="Market",
        regime="market",
        subsidy_type=None,
        color="#2a6f97",
    ),
    ScenarioConfig(
        name="policy_conversion",
        label="Policy: conversion subsidy",
        regime="policy",
        subsidy_type="conversion",
        color="#c17c2f",
    ),
    ScenarioConfig(
        name="policy_results",
        label="Policy: results subsidy",
        regime="policy",
        subsidy_type="results",
        color="#5b7c99",
    ),
)

def replicate_seeds(params: ModelParameters = DEFAULT_PARAMETERS) -> tuple[int, ...]:
    """Use deterministic consecutive seeds for replicate runs."""

    return tuple(params.seed + offset for offset in range(params.n_replicates))
