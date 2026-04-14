from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ABM-LU-program"))

from abm_model import run_experiment
from parameters import CORE_SCENARIOS, DEFAULT_PARAMETERS


class ExperimentSmokeTest(unittest.TestCase):
    def test_small_experiment_runs_and_writes_outputs(self) -> None:
        params = DEFAULT_PARAMETERS.with_overrides(
            seed=101,
            n_replicates=1,
            n_steps=3,
            n_rows=8,
            n_cols=8,
            n_farmers=8,
        )

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            output_dir = Path(tmp_dir)
            timeseries_df, farmer_df, summary_df = run_experiment(
                params=params,
                scenario_configs=CORE_SCENARIOS,
                output_root=output_dir,
            )

            self.assertEqual(timeseries_df["replicate"].nunique(), 1)
            self.assertEqual(summary_df["scenario"].nunique(), len(CORE_SCENARIOS))
            self.assertEqual(farmer_df["scenario"].nunique(), len(CORE_SCENARIOS))
            self.assertIn("mean_profit_before_subsidy", farmer_df.columns)
            self.assertIn("share_above_market_baseline_before_subsidy_mean", summary_df.columns)
            self.assertTrue((output_dir / "raw" / "timeseries.csv").exists())
            self.assertTrue((output_dir / "tables" / "scenario_summary.csv").exists())
            self.assertTrue((output_dir / "figures" / "scenario_timeseries.png").exists())


if __name__ == "__main__":
    unittest.main()
