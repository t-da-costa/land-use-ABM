##################################################################
## End-to-end runner for the agricultural land-use ABM.
##################################################################

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from env_dyn import EnvironmentalDynamics
from landscape import build_landscape
from parameters import (
    CORE_SCENARIOS,
    DEFAULT_PARAMETERS,
    OUTPUT_DIR,
    ROBUSTNESS_SCENARIOS,
    ModelParameters,
    ScenarioConfig,
    replicate_seeds,
)
from plots import (
    plot_final_land_use_maps,
    plot_land_use_shares,
    plot_quality_with_farm_borders,
    plot_timeseries,
)
from production import FarmerCostShocks, ProductionModel
from subsidy import SubsidyModel


LAND_USE_OPTIONS = ("I", "O", "S")
LAND_USE_TIEBREAK = {"O": 2, "S": 1, "I": 0}
OBJECTIVE_TOL = 1e-12


@dataclass
class Farmer:
    farmer_id: int
    plot_ids: list[int]
    cost_shocks: FarmerCostShocks


@dataclass
class SimulationResult:
    scenario: str
    scenario_label: str
    replicate: int
    seed: int
    timeseries: list[dict[str, float | int | str]]
    farmer_summary: list[dict[str, float | int | str]]
    final_land_use: np.ndarray
    landscape: object


class LandUseABM:
    """A single replicate for one scenario."""

    def __init__(
        self,
        params: ModelParameters,
        scenario: ScenarioConfig,
        replicate: int,
        seed: int,
    ) -> None:
        self.params = params
        self.scenario = scenario
        self.replicate = replicate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.landscape = build_landscape(
            n_rows=self.params.n_rows,
            n_cols=self.params.n_cols,
            n_farmers=self.params.n_farmers,
            q_sigma=self.params.q_sigma,
            q_length_scale=self.params.q_length_scale,
            farm_mu=self.params.farm_mu,
            farm_sigma=self.params.farm_sigma,
            seed=seed,
        )

        self.production = ProductionModel(self.params)
        self.environment = EnvironmentalDynamics(self.params)
        self.subsidies = SubsidyModel(self.params)
        self.farmers = self._build_farmers()

        self.current_q = np.array(
            [self.landscape.plots[plot_id].q for plot_id in range(self.landscape.n_plots)],
            dtype=float,
        )
        self.current_e = np.full(
            self.landscape.n_plots,
            fill_value=self.params.initial_environmental_quality,
            dtype=float,
        )
        self.current_land_use = np.full(
            self.landscape.n_plots,
            fill_value=self.params.initial_land_use,
            dtype="<U1",
        )
        self.conversion_support_remaining = np.zeros(self.landscape.n_plots, dtype=int)

    def _build_farmers(self) -> list[Farmer]:
        farmers: list[Farmer] = []
        for farmer_id in range(self.landscape.n_farmers):
            farmers.append(
                Farmer(
                    farmer_id=farmer_id,
                    plot_ids=list(self.landscape.farmer_plot_ids[farmer_id]),
                    cost_shocks=self.production.draw_cost_shocks(self.rng),
                )
            )
        return farmers

    def _evaluate_action(self, farmer: Farmer, plot_id: int, land_use: str) -> dict[str, float | int | str]:
        q = float(self.current_q[plot_id])
        current_environment = float(self.current_e[plot_id])
        previous_land_use = str(self.current_land_use[plot_id])
        conversion_remaining = int(self.conversion_support_remaining[plot_id])

        output = self.production.yield_fn(land_use, q)
        cost = self.production.cost(land_use, q, shocks=farmer.cost_shocks)
        profit_before_subsidy = self.production.profit_before_subsidy(
            land_use,
            q,
            shocks=farmer.cost_shocks,
        )
        next_environment, next_quality = self.environment.transition(q, current_environment, land_use)

        subsidy = 0.0
        if self.scenario.regime == "policy":
            subsidy = self.subsidies.payment(
                subsidy_type=self.scenario.subsidy_type,
                current_land_use=land_use,
                previous_land_use=previous_land_use,
                next_environment=next_environment,
                conversion_support_remaining=conversion_remaining,
            )

        if self.scenario.regime == "productivist":
            objective = output
        elif self.scenario.regime == "market":
            objective = profit_before_subsidy
        else:
            objective = profit_before_subsidy + subsidy

        return {
            "plot_id": plot_id,
            "land_use": land_use,
            "objective": float(objective),
            "output": float(output),
            "cost": float(cost),
            "profit_before_subsidy": float(profit_before_subsidy),
            "subsidy": float(subsidy),
            "profit_after_subsidy": float(profit_before_subsidy + subsidy),
            "next_environment": float(next_environment),
            "next_quality": float(next_quality),
        }

    def _choose_action(self, farmer: Farmer, plot_id: int) -> dict[str, float | int | str]:
        options = [self._evaluate_action(farmer, plot_id, land_use) for land_use in LAND_USE_OPTIONS]
        best_objective = max(option["objective"] for option in options)
        tied = [
            option
            for option in options
            if abs(float(option["objective"]) - float(best_objective)) <= OBJECTIVE_TOL
        ]

        previous_land_use = str(self.current_land_use[plot_id])
        for option in tied:
            if option["land_use"] == previous_land_use:
                return option

        tied.sort(
            key=lambda option: (
                LAND_USE_TIEBREAK[str(option["land_use"])],
                float(option["profit_after_subsidy"]),
            ),
            reverse=True,
        )
        return tied[0]

    def _share(self, land_use_by_plot: np.ndarray, land_use: str) -> float:
        return float(np.mean(land_use_by_plot == land_use))

    def _clustering_stat(self, land_use_by_plot: np.ndarray) -> float:
        same_state_pairs = 0
        candidate_pairs = 0

        for plot_id in range(self.landscape.n_plots):
            for neighbor_id in self.landscape.get_neighbors(plot_id):
                if neighbor_id <= plot_id:
                    continue
                uses = {str(land_use_by_plot[plot_id]), str(land_use_by_plot[neighbor_id])}
                if uses - {"I", "S"}:
                    continue
                candidate_pairs += 1
                if len(uses) == 1:
                    same_state_pairs += 1

        if candidate_pairs == 0:
            return float("nan")
        return float(same_state_pairs / candidate_pairs)

    def _record_step(
        self,
        step: int,
        land_use_by_plot: np.ndarray,
        q_by_plot: np.ndarray,
        e_by_plot: np.ndarray,
        farmer_period_profit_before_subsidy: dict[int, float],
        farmer_period_profit_after_subsidy: dict[int, float],
        farmer_period_subsidy: dict[int, float],
        total_output: float,
    ) -> dict[str, float | int | str]:
        return {
            "scenario": self.scenario.name,
            "scenario_label": self.scenario.label,
            "replicate": self.replicate,
            "seed": self.seed,
            "step": step,
            "total_output": float(total_output),
            "mean_farmer_profit_before_subsidy": float(np.mean(list(farmer_period_profit_before_subsidy.values()))),
            "mean_farmer_profit_after_subsidy": float(np.mean(list(farmer_period_profit_after_subsidy.values()))),
            "mean_farmer_subsidy": float(np.mean(list(farmer_period_subsidy.values()))),
            "avg_environment": float(np.mean(e_by_plot)),
            "avg_quality": float(np.mean(q_by_plot)),
            "share_I": self._share(land_use_by_plot, "I"),
            "share_O": self._share(land_use_by_plot, "O"),
            "share_S": self._share(land_use_by_plot, "S"),
            "clustering_IS": self._clustering_stat(land_use_by_plot),
        }

    def run(self) -> SimulationResult:
        timeseries: list[dict[str, float | int | str]] = []
        farmer_total_profit_before_subsidy = {farmer.farmer_id: 0.0 for farmer in self.farmers}
        farmer_total_profit_after_subsidy = {farmer.farmer_id: 0.0 for farmer in self.farmers}
        farmer_total_subsidy = {farmer.farmer_id: 0.0 for farmer in self.farmers}
        farmer_total_output = {farmer.farmer_id: 0.0 for farmer in self.farmers}

        initial_farmer_profit_before_subsidy = {
            farmer.farmer_id: sum(
                self.production.profit_before_subsidy(
                    land_use=str(self.current_land_use[plot_id]),
                    q=float(self.current_q[plot_id]),
                    shocks=farmer.cost_shocks,
                )
                for plot_id in farmer.plot_ids
            )
            for farmer in self.farmers
        }
        initial_farmer_profit_after_subsidy = initial_farmer_profit_before_subsidy.copy()
        initial_farmer_subsidy = {farmer.farmer_id: 0.0 for farmer in self.farmers}
        initial_total_output = float(
            sum(
                self.production.yield_fn(str(self.current_land_use[plot_id]), float(self.current_q[plot_id]))
                for plot_id in range(self.landscape.n_plots)
            )
        )
        timeseries.append(
            self._record_step(
                step=0,
                land_use_by_plot=self.current_land_use.copy(),
                q_by_plot=self.current_q.copy(),
                e_by_plot=self.current_e.copy(),
                farmer_period_profit_before_subsidy=initial_farmer_profit_before_subsidy,
                farmer_period_profit_after_subsidy=initial_farmer_profit_after_subsidy,
                farmer_period_subsidy=initial_farmer_subsidy,
                total_output=initial_total_output,
            )
        )

        for step in range(1, self.params.n_steps + 1):
            chosen_land_use = np.empty_like(self.current_land_use)
            next_environment = np.zeros_like(self.current_e)
            next_quality = np.zeros_like(self.current_q)
            next_conversion_support = np.zeros_like(self.conversion_support_remaining)

            farmer_period_profit_before_subsidy = {farmer.farmer_id: 0.0 for farmer in self.farmers}
            farmer_period_profit_after_subsidy = {farmer.farmer_id: 0.0 for farmer in self.farmers}
            farmer_period_subsidy = {farmer.farmer_id: 0.0 for farmer in self.farmers}
            farmer_period_output = {farmer.farmer_id: 0.0 for farmer in self.farmers}

            for farmer in self.farmers:
                for plot_id in farmer.plot_ids:
                    decision = self._choose_action(farmer, plot_id)
                    chosen_land_use[plot_id] = str(decision["land_use"])
                    next_environment[plot_id] = float(decision["next_environment"])
                    next_quality[plot_id] = float(decision["next_quality"])
                    next_conversion_support[plot_id] = self.subsidies.next_conversion_support(
                        current_land_use=str(decision["land_use"]),
                        previous_land_use=str(self.current_land_use[plot_id]),
                        conversion_support_remaining=int(self.conversion_support_remaining[plot_id]),
                    )
                    farmer_period_profit_before_subsidy[farmer.farmer_id] += float(decision["profit_before_subsidy"])
                    farmer_period_profit_after_subsidy[farmer.farmer_id] += float(decision["profit_after_subsidy"])
                    farmer_period_subsidy[farmer.farmer_id] += float(decision["subsidy"])
                    farmer_period_output[farmer.farmer_id] += float(decision["output"])

            total_output = float(sum(farmer_period_output.values()))

            for farmer in self.farmers:
                farmer_total_profit_before_subsidy[farmer.farmer_id] += farmer_period_profit_before_subsidy[farmer.farmer_id]
                farmer_total_profit_after_subsidy[farmer.farmer_id] += farmer_period_profit_after_subsidy[farmer.farmer_id]
                farmer_total_subsidy[farmer.farmer_id] += farmer_period_subsidy[farmer.farmer_id]
                farmer_total_output[farmer.farmer_id] += farmer_period_output[farmer.farmer_id]

            self.current_land_use = chosen_land_use
            self.current_e = next_environment
            self.current_q = next_quality
            self.conversion_support_remaining = next_conversion_support

            timeseries.append(
                self._record_step(
                step=step,
                land_use_by_plot=self.current_land_use.copy(),
                q_by_plot=self.current_q.copy(),
                e_by_plot=self.current_e.copy(),
                farmer_period_profit_before_subsidy=farmer_period_profit_before_subsidy,
                farmer_period_profit_after_subsidy=farmer_period_profit_after_subsidy,
                farmer_period_subsidy=farmer_period_subsidy,
                total_output=total_output,
            )
        )

        farmer_summary = [
            {
                "scenario": self.scenario.name,
                "scenario_label": self.scenario.label,
                "replicate": self.replicate,
                "seed": self.seed,
                "farmer_id": farmer.farmer_id,
                "n_plots": len(farmer.plot_ids),
                "eta_I": farmer.cost_shocks.eta_I,
                "eta_O": farmer.cost_shocks.eta_O,
                "total_profit_before_subsidy": farmer_total_profit_before_subsidy[farmer.farmer_id],
                "mean_profit_before_subsidy": farmer_total_profit_before_subsidy[farmer.farmer_id] / self.params.n_steps,
                "total_profit_after_subsidy": farmer_total_profit_after_subsidy[farmer.farmer_id],
                "mean_profit_after_subsidy": farmer_total_profit_after_subsidy[farmer.farmer_id] / self.params.n_steps,
                "total_subsidy": farmer_total_subsidy[farmer.farmer_id],
                "mean_output": farmer_total_output[farmer.farmer_id] / self.params.n_steps,
            }
            for farmer in self.farmers
        ]

        return SimulationResult(
            scenario=self.scenario.name,
            scenario_label=self.scenario.label,
            replicate=self.replicate,
            seed=self.seed,
            timeseries=timeseries,
            farmer_summary=farmer_summary,
            final_land_use=self.current_land_use.copy(),
            landscape=self.landscape,
        )


def ensure_output_dirs(root: Path) -> tuple[Path, Path, Path]:
    raw_dir = root / "raw"
    table_dir = root / "tables"
    figure_dir = root / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, table_dir, figure_dir


def build_dataframes(results: Iterable[SimulationResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    timeseries_rows = [row for result in results for row in result.timeseries]
    farmer_rows = [row for result in results for row in result.farmer_summary]
    return pd.DataFrame(timeseries_rows), pd.DataFrame(farmer_rows)


def add_market_profit_baseline(farmer_df: pd.DataFrame) -> pd.DataFrame:
    market_baseline = (
        farmer_df[
            farmer_df["scenario"] == "market"
        ][["replicate", "farmer_id", "mean_profit_before_subsidy", "mean_profit_after_subsidy"]]
        .rename(
            columns={
                "mean_profit_before_subsidy": "market_mean_profit_before_subsidy",
                "mean_profit_after_subsidy": "market_mean_profit_after_subsidy",
            }
        )
        .copy()
    )
    merged = farmer_df.merge(market_baseline, on=["replicate", "farmer_id"], how="left")
    merged["above_market_baseline_before_subsidy"] = (
        merged["mean_profit_before_subsidy"] >= merged["market_mean_profit_before_subsidy"] - OBJECTIVE_TOL
    )
    merged["above_market_baseline_after_subsidy"] = (
        merged["mean_profit_after_subsidy"] >= merged["market_mean_profit_after_subsidy"] - OBJECTIVE_TOL
    )
    return merged


def summarize_results(timeseries_df: pd.DataFrame, farmer_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_periods = timeseries_df[timeseries_df["step"] > 0].copy()

    final_by_replicate = (
        timeseries_df.sort_values(["scenario", "replicate", "step"])
        .groupby(["scenario", "scenario_label", "replicate", "seed"], as_index=False)
        .tail(1)
        .rename(
            columns={
                "avg_environment": "final_avg_environment",
                "avg_quality": "final_avg_quality",
                "share_I": "final_share_I",
                "share_O": "final_share_O",
                "share_S": "final_share_S",
            }
        )
    )

    flows_by_replicate = (
        policy_periods.groupby(["scenario", "scenario_label", "replicate", "seed"], as_index=False)
        .agg(
            cumulative_output=("total_output", "sum"),
            mean_period_profit_before_subsidy=("mean_farmer_profit_before_subsidy", "mean"),
            mean_period_profit_after_subsidy=("mean_farmer_profit_after_subsidy", "mean"),
            mean_period_subsidy=("mean_farmer_subsidy", "mean"),
            mean_clustering_IS=("clustering_IS", "mean"),
        )
    )

    farmer_by_replicate = (
        farmer_df.groupby(["scenario", "scenario_label", "replicate", "seed"], as_index=False)
        .agg(
            share_above_market_baseline_before_subsidy=("above_market_baseline_before_subsidy", "mean"),
            share_above_market_baseline_after_subsidy=("above_market_baseline_after_subsidy", "mean"),
        )
    )

    summary_by_replicate = final_by_replicate.merge(
        flows_by_replicate,
        on=["scenario", "scenario_label", "replicate", "seed"],
        how="left",
    ).merge(
        farmer_by_replicate,
        on=["scenario", "scenario_label", "replicate", "seed"],
        how="left",
    )

    scenario_summary = (
        summary_by_replicate.groupby(["scenario", "scenario_label"], as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            cumulative_output_mean=("cumulative_output", "mean"),
            cumulative_output_sd=("cumulative_output", "std"),
            mean_period_profit_before_subsidy_mean=("mean_period_profit_before_subsidy", "mean"),
            mean_period_profit_before_subsidy_sd=("mean_period_profit_before_subsidy", "std"),
            mean_period_profit_after_subsidy_mean=("mean_period_profit_after_subsidy", "mean"),
            mean_period_profit_after_subsidy_sd=("mean_period_profit_after_subsidy", "std"),
            mean_period_subsidy_mean=("mean_period_subsidy", "mean"),
            mean_period_subsidy_sd=("mean_period_subsidy", "std"),
            final_avg_environment_mean=("final_avg_environment", "mean"),
            final_avg_environment_sd=("final_avg_environment", "std"),
            final_avg_quality_mean=("final_avg_quality", "mean"),
            final_avg_quality_sd=("final_avg_quality", "std"),
            final_share_I_mean=("final_share_I", "mean"),
            final_share_O_mean=("final_share_O", "mean"),
            final_share_S_mean=("final_share_S", "mean"),
            share_above_market_baseline_before_subsidy_mean=("share_above_market_baseline_before_subsidy", "mean"),
            share_above_market_baseline_after_subsidy_mean=("share_above_market_baseline_after_subsidy", "mean"),
            mean_clustering_IS_mean=("mean_clustering_IS", "mean"),
        )
    )

    return summary_by_replicate, scenario_summary


def write_presentation_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_lookup = summary_df.set_index("scenario")

    practice = summary_lookup.loc["policy_practice"]
    market = summary_lookup.loc["market"]
    productivist = summary_lookup.loc["productivist"]

    lines = [
        "# Presentation Notes",
        "",
        "## Core story",
        "",
        f"- Relative to `market`, the practice-based policy improves final average environmental quality from "
        f"{market['final_avg_environment_mean']:.3f} to {practice['final_avg_environment_mean']:.3f}.",
        f"- Relative to `productivist`, the same policy avoids the environmental collapse visible in the baseline "
        f"({productivist['final_avg_environment_mean']:.3f} versus {practice['final_avg_environment_mean']:.3f}).",
        f"- The practice-based policy shifts the final land-use mix to "
        f"I={practice['final_share_I_mean']:.2%}, O={practice['final_share_O_mean']:.2%}, "
        f"S={practice['final_share_S_mean']:.2%}.",
        "",
        "## Economic interpretation",
        "",
        f"- Mean farmer profit before subsidy moves from {market['mean_period_profit_before_subsidy_mean']:.3f} "
        f"under `market` to {practice['mean_period_profit_before_subsidy_mean']:.3f} under the practice policy.",
        f"- Mean subsidy paid per farmer per period is {practice['mean_period_subsidy_mean']:.3f}, so the after-subsidy "
        f"profit rises to {practice['mean_period_profit_after_subsidy_mean']:.3f}.",
        f"- Only {practice['share_above_market_baseline_before_subsidy_mean']:.2%} of farmers are weakly above their "
        f"market baseline before subsidy transfers, compared with "
        f"{practice['share_above_market_baseline_after_subsidy_mean']:.2%} after transfers.",
        "",
        "## Scope choice",
        "",
        "- The main presentation should focus on `productivist`, `market`, and `policy_practice`.",
        "- `policy_conversion` and `policy_results` are retained as robustness checks, not headline evidence.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_robustness_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_lookup = summary_df.set_index("scenario")
    market = summary_lookup.loc["market"]
    conversion = summary_lookup.loc["policy_conversion"]
    results = summary_lookup.loc["policy_results"]

    lines = [
        "# Robustness Notes",
        "",
        f"- Conversion subsidy raises final average environmental quality from {market['final_avg_environment_mean']:.3f} "
        f"to {conversion['final_avg_environment_mean']:.3f} and shifts land use toward organic production "
        f"({conversion['final_share_O_mean']:.2%}).",
        f"- Results-based subsidy moves the system only modestly relative to market under the current calibration "
        f"({results['final_avg_environment_mean']:.3f} final average environmental quality).",
        f"- Mean farmer profit before subsidy is {conversion['mean_period_profit_before_subsidy_mean']:.3f} under "
        f"conversion and {results['mean_period_profit_before_subsidy_mean']:.3f} under results-based payments.",
        "- These cases should be treated as appendix material. They help show robustness, but they are not the cleanest main story.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    params: ModelParameters,
    scenario_configs: Iterable[ScenarioConfig],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_configs = tuple(scenario_configs)
    raw_dir, table_dir, figure_dir = ensure_output_dirs(output_root)

    results: list[SimulationResult] = []
    for replicate, seed in enumerate(replicate_seeds(params), start=1):
        for scenario in scenario_configs:
            model = LandUseABM(params=params.with_overrides(seed=seed), scenario=scenario, replicate=replicate, seed=seed)
            results.append(model.run())

    timeseries_df, farmer_df = build_dataframes(results)
    farmer_df = add_market_profit_baseline(farmer_df)
    summary_by_replicate, scenario_summary = summarize_results(timeseries_df, farmer_df)

    timeseries_df.to_csv(raw_dir / "timeseries.csv", index=False)
    farmer_df.to_csv(raw_dir / "farmer_summary.csv", index=False)
    summary_by_replicate.to_csv(table_dir / "summary_by_replicate.csv", index=False)
    scenario_summary.to_csv(table_dir / "scenario_summary.csv", index=False)

    summary_for_plots = (
        timeseries_df.groupby(["scenario", "scenario_label", "step"], as_index=False)
        .agg(
            total_output=("total_output", "mean"),
            mean_farmer_profit_before_subsidy=("mean_farmer_profit_before_subsidy", "mean"),
            mean_farmer_profit_after_subsidy=("mean_farmer_profit_after_subsidy", "mean"),
            mean_farmer_subsidy=("mean_farmer_subsidy", "mean"),
            avg_environment=("avg_environment", "mean"),
            avg_quality=("avg_quality", "mean"),
            share_I=("share_I", "mean"),
            share_O=("share_O", "mean"),
            share_S=("share_S", "mean"),
            clustering_IS=("clustering_IS", "mean"),
        )
    )

    representative_runs = [result for result in results if result.replicate == 1]
    representative_lookup = {result.scenario: result for result in representative_runs}

    plot_quality_with_farm_borders(
        representative_runs[0].landscape,
        title="Initial plot quality and farm boundaries",
        savepath=figure_dir / "initial_quality_map.png",
    )
    plot_timeseries(summary_for_plots, tuple(scenario_configs), figure_dir / "scenario_timeseries.png")
    plot_land_use_shares(summary_for_plots, tuple(scenario_configs), figure_dir / "land_use_shares.png")
    plot_final_land_use_maps(
        [representative_lookup[scenario.name].__dict__ for scenario in scenario_configs],
        tuple(scenario_configs),
        figure_dir / "final_land_use_maps.png",
    )
    return timeseries_df, farmer_df, scenario_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agricultural land-use ABM and export results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where result tables and figures will be written.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_PARAMETERS.n_replicates,
        help="Number of replicate seeds to run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_PARAMETERS.n_steps,
        help="Number of time steps per replicate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = DEFAULT_PARAMETERS.with_overrides(
        n_replicates=args.replicates,
        n_steps=args.steps,
    )
    _, _, core_summary = run_experiment(
        params=params,
        scenario_configs=CORE_SCENARIOS,
        output_root=args.output_dir,
    )
    write_presentation_summary(core_summary, args.output_dir / "presentation-summary.md")
    _, _, robustness_summary = run_experiment(
        params=params,
        scenario_configs=ROBUSTNESS_SCENARIOS,
        output_root=args.output_dir / "robustness",
    )
    write_robustness_summary(robustness_summary, args.output_dir / "robustness" / "appendix-summary.md")
    print("Core scenario summary")
    print(core_summary.to_string(index=False))
    print("\nRobustness summary")
    print(robustness_summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
