import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from scripts.q4_optimization import (
    choose_reference_policies,
    configure_paths,
    empirical_pareto_mask,
    exposure_metrics,
    generate_local_grid,
    predict_bootstrap_quantiles,
    standardized_nearest_distance,
    theoretical_charge_time,
)


class Q4OptimizationTests(unittest.TestCase):
    def test_configure_paths_updates_all_q4_inputs_and_output(self):
        source = Path("D:/temporary/source_results")
        output = Path("D:/temporary/q4_results")
        paths = configure_paths(source, output)
        self.assertEqual(paths["source_results"], source)
        self.assertEqual(paths["output_dir"], output)
        self.assertEqual(
            paths["q1_battery"], source / "问题1" / "q1_03_四十九块电池指标与基准EOL.csv"
        )
        self.assertEqual(
            paths["q3_oof"], source / "问题3" / "q3_06_外层40折逐点预测.csv"
        )

    def test_theoretical_charge_time_matches_two_stage_formula(self):
        self.assertAlmostEqual(theoretical_charge_time(3.6, 80, 3.6), 13.3333333333, places=9)
        self.assertAlmostEqual(theoretical_charge_time(5.3, 54, 4.0), 10.0132075472, places=9)

    def test_pareto_mask_requires_no_worse_and_one_strict_improvement(self):
        frame = pd.DataFrame(
            {
                "time": [10.0, 10.5, 9.8, 10.0],
                "rate": [2.0, 1.8, 2.4, 2.0],
            }
        )
        # Points 0 and 3 are identical and both non-dominated; point 1 trades
        # slower time for lower degradation, while point 2 trades the reverse.
        np.testing.assert_array_equal(
            empirical_pareto_mask(frame, "time", "rate"),
            np.array([True, True, True, True]),
        )
        dominated = pd.concat(
            [frame, pd.DataFrame({"time": [10.6], "rate": [2.5]})], ignore_index=True
        )
        self.assertFalse(empirical_pareto_mask(dominated, "time", "rate")[-1])

    def test_standardized_distance_is_zero_at_observed_design(self):
        observed = np.array(
            [[5.6, 36.0, 4.3], [5.3, 54.0, 4.0], [4.8, 80.0, 4.8], [3.7, 31.0, 5.9]]
        )
        distance, threshold = standardized_nearest_distance(observed, observed)
        np.testing.assert_allclose(distance, 0.0)
        self.assertGreater(threshold, 0.0)

    def test_local_grid_respects_hull_distance_time_and_resolution(self):
        observed = np.array(
            [
                [5.6, 36.0, 4.3],
                [5.3, 54.0, 4.0],
                [4.8, 80.0, 4.8],
                [5.0, 67.0, 4.0],
                [5.6, 19.0, 4.6],
                [3.7, 31.0, 5.9],
            ]
        )
        grid, distance_threshold = generate_local_grid(observed, time_budget=10.0132075472)
        self.assertGreater(len(grid), 0)
        self.assertTrue((grid["T_theoretical_0_80"] <= 10.0132075472 + 1e-10).all())
        self.assertTrue((grid["nearest_standardized_distance"] <= distance_threshold + 1e-10).all())
        np.testing.assert_allclose(grid["C1"] * 10, np.round(grid["C1"] * 10))
        np.testing.assert_allclose(grid["C2"] * 10, np.round(grid["C2"] * 10))
        np.testing.assert_allclose(grid["Q1"], np.round(grid["Q1"]))
        observed_exposure = exposure_metrics(observed[:, 0], observed[:, 1], observed[:, 2])
        observed_ad = np.column_stack([observed_exposure["A"], observed_exposure["D_50"]])
        candidate_ad = grid[["A", "D_50"]].to_numpy(float)
        self.assertTrue((Delaunay(observed_ad, qhull_options="QJ").find_simplex(candidate_ad) >= 0).all())

    def test_bootstrap_quantiles_use_all_model_draws(self):
        candidate = pd.DataFrame({"A_z": [0.0, 1.0], "D_z": [0.0, -1.0]})
        models = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 3.0],
                [3.0, 2.0, 3.0],
            ]
        )
        result = predict_bootstrap_quantiles(candidate, models, quantiles=(0.5, 0.9))
        self.assertAlmostEqual(result.loc[0, "q50"], 2.0)
        self.assertAlmostEqual(result.loc[1, "q50"], 1.0)
        self.assertAlmostEqual(result.loc[0, "q90"], 2.8)
        self.assertAlmostEqual(result.loc[1, "q90"], 1.8)

    def test_reference_policies_follow_current_q1_results(self):
        policy_stats = pd.DataFrame(
            {
                "policy": ["long_new", "middle", "short_new"],
                "life_median": [1800.0, 1200.0, 600.0],
                "stable_rate_median": [1.5e-5, 2.0e-5, 8.0e-5],
            }
        )
        roles = choose_reference_policies(policy_stats, "middle")
        self.assertEqual(roles["typical_long_EOL"], "long_new")
        self.assertEqual(roles["typical_short"], "short_new")
        self.assertEqual(roles["early_degradation_representative"], "long_new")


if __name__ == "__main__":
    unittest.main()
