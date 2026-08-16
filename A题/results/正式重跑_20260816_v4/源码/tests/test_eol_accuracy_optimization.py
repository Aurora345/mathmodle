import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eol_accuracy_optimization import (
    acceleration_feature_sets,
    build_accuracy_comparison,
    choose_structure_one_se,
    early_feature_row,
    fit_monotone_quadratic,
    nested_pooling_predictions,
    nested_ridge_lobo,
    predict_eol_model,
    shrink_log_eol,
)


class EOLAccuracyOptimizationTests(unittest.TestCase):
    def test_accuracy_comparison_reports_relative_direction(self):
        eol = pd.DataFrame(
            {
                "model": ["power", "quadratic"],
                "pooled_RMSE": [1.0, 1.1],
                "median_eol_relative_update": [0.2, 0.1],
                "mean_eol_relative_update": [0.3, 0.2],
                "eol_spearman_150_200": [0.9, 0.8],
            }
        )
        heads = pd.DataFrame(
            {
                "feature_set": ["直接延续", "现有SOH趋势_Policy", "+静态_残差_Policy"],
                "future_slope_RMSE": [2.0, 1.5, 1.4],
            }
        )
        pool = pd.DataFrame(
            {"individual_relative_error": [0.2, 0.4], "pooled_relative_error": [0.1, 0.3]}
        )
        comparison = build_accuracy_comparison(eol, heads, pool)
        median_row = comparison[comparison["metric"].eq("EOL更新中位相对差")].iloc[0]
        self.assertAlmostEqual(median_row["relative_change_percent"], -50.0)
    def test_acceleration_ablation_contains_policy_only_increment(self):
        sets = acceleration_feature_sets(["policy_A", "policy_B"])
        self.assertIn("现有SOH趋势_Policy", sets)
        self.assertNotIn("initial_capacity", sets["现有SOH趋势_Policy"])
        self.assertNotIn("SOH_residual_MAD", sets["现有SOH趋势_Policy"])
        self.assertIn("policy_A", sets["现有SOH趋势_Policy"])

    def test_script_help_runs_from_repository_root(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "eol_accuracy_optimization.py"), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_early_feature_row_uses_summary_index_as_battery_id(self):
        cycle = np.arange(1, 151)
        group = pd.DataFrame(
            {
                "cycle": cycle,
                "SOH_sg": 1.0 - 1e-4 * cycle,
                "SOH_clean": 1.0 - 1e-4 * cycle,
                "IR_clean": 0.015 + 1e-6 * cycle,
            }
        )
        summary_row = pd.Series(
            {"policy": "A", "initial_capacity": 1.05}, name=7
        )
        features = early_feature_row(group, summary_row)
        self.assertEqual(features["battery_id"], 7)

    def test_monotone_quadratic_recovers_decreasing_accelerating_curve(self):
        cycle = np.arange(21.0, 201.0)
        observed = 1.002 - 1.5e-5 * cycle - 2.0e-8 * cycle**2
        fit = fit_monotone_quadratic(cycle, observed)
        self.assertGreaterEqual(fit["b"], 0.0)
        self.assertGreaterEqual(fit["c"], 0.0)
        self.assertAlmostEqual(fit["a"], 1.002, places=8)
        self.assertAlmostEqual(fit["b"], 1.5e-5, places=10)
        self.assertAlmostEqual(fit["c"], 2.0e-8, places=12)
        self.assertAlmostEqual(
            float(predict_eol_model("quadratic", fit, np.array([fit["life"]]))[0]),
            0.8,
            places=8,
        )

    def test_one_se_choice_prefers_eol_stability_among_short_term_near_optima(self):
        table = pd.DataFrame(
            {
                "model": ["linear", "power", "quadratic"],
                "mean_battery_MSE": [3.0e-7, 2.0e-7, 2.3e-7],
                "SE_battery_MSE": [1.0e-8, 5.0e-8, 4.0e-8],
                "median_eol_relative_update": [0.05, 0.18, 0.08],
                "nonfinite_eol_count": [0, 0, 0],
                "boundary_count": [0, 0, 10],
            }
        )
        chosen, audited = choose_structure_one_se(table)
        self.assertEqual(chosen, "quadratic")
        self.assertFalse(bool(audited.loc[audited.model.eq("linear"), "eligible_one_SE"].iloc[0]))
        self.assertTrue(bool(audited.loc[audited.model.eq("power"), "eligible_one_SE"].iloc[0]))
        self.assertTrue(bool(audited.loc[audited.model.eq("quadratic"), "eligible_one_SE"].iloc[0]))

    def test_log_eol_shrinkage_has_interpretable_endpoints(self):
        self.assertAlmostEqual(shrink_log_eol(4000.0, 1000.0, 1.0), 4000.0)
        self.assertAlmostEqual(shrink_log_eol(4000.0, 1000.0, 0.0), 1000.0)
        self.assertAlmostEqual(shrink_log_eol(4000.0, 1000.0, 0.5), 2000.0)

    def test_nested_pooling_selects_individual_when_individual_is_exact(self):
        frame = pd.DataFrame(
            {
                "battery_id": np.arange(1, 7),
                "policy": ["A", "A", "A", "B", "B", "B"],
                "life_150": [1000.0, 2000.0, 4000.0, 1200.0, 2400.0, 4800.0],
                "life_200": [1000.0, 2000.0, 4000.0, 1200.0, 2400.0, 4800.0],
            }
        )
        predictions, _ = nested_pooling_predictions(frame, weights=[0.0, 0.5, 1.0])
        self.assertTrue((predictions["selected_weight"] == 1.0).all())
        np.testing.assert_allclose(predictions["pooled_life"], predictions["life_200"])

    def test_nested_ridge_lobo_learns_simple_linear_signal(self):
        x = np.linspace(-2.0, 2.0, 12)
        frame = pd.DataFrame({"battery_id": np.arange(12), "x": x, "target": 2.0 * x + 1.0})
        predictions = nested_ridge_lobo(frame, ["x"], "target", alpha_grid=[0.0, 0.01, 0.1])
        rmse = float(np.sqrt(np.mean((predictions["predicted"] - predictions["observed"]) ** 2)))
        self.assertLess(rmse, 0.05)


if __name__ == "__main__":
    unittest.main()
