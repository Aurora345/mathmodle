import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.eol_v4_final_arbitration import (
    choose_joint_candidate_one_se,
    fit_centered_monotone_quadratic,
    mean_delta_lobo,
    predict_centered_quadratic,
)


class EOLV4FinalArbitrationTests(unittest.TestCase):
    def test_centered_quadratic_recovers_stable_segment_parameters(self):
        n0 = 21
        cycle = np.arange(n0, 201, dtype=float)
        shifted = cycle - n0
        observed = 1.001 - 2.0e-5 * shifted - 5.0e-8 * shifted**2
        fit = fit_centered_monotone_quadratic(cycle, observed, n0)
        self.assertAlmostEqual(fit["A"], 1.001, places=8)
        self.assertAlmostEqual(fit["B"], 2.0e-5, places=10)
        self.assertAlmostEqual(fit["C"], 5.0e-8, places=12)
        self.assertGreaterEqual(fit["B"], 0.0)
        self.assertGreaterEqual(fit["C"], 0.0)
        self.assertAlmostEqual(
            float(predict_centered_quadratic(fit, np.array([fit["life"]]))[0]),
            0.8,
            places=8,
        )

    def test_joint_one_se_can_select_new_n0_and_centered_structure(self):
        summary = pd.DataFrame(
            {
                "n0": [21, 31, 41],
                "model": ["power", "centered_quadratic", "linear"],
                "mean_battery_MSE": [2.0e-7, 2.4e-7, 5.0e-7],
                "SE_battery_MSE": [5.0e-8, 4.0e-8, 5.0e-8],
                "median_eol_relative_update": [0.20, 0.08, 0.05],
                "nonfinite_eol_count": [0, 0, 0],
                "boundary_count": [0, 4, 0],
            }
        )
        n0, model, audited = choose_joint_candidate_one_se(summary)
        self.assertEqual(n0, 31)
        self.assertEqual(model, "centered_quadratic")
        self.assertFalse(bool(audited.loc[audited.model.eq("linear"), "eligible_one_SE"].iloc[0]))

    def test_global_and_policy_mean_delta_are_leave_one_battery_out(self):
        frame = pd.DataFrame(
            {
                "battery_id": [1, 2, 3, 4],
                "policy": ["A", "A", "B", "B"],
                "delta_slope": [1.0, 3.0, 10.0, 14.0],
            }
        )
        predictions = mean_delta_lobo(frame)
        first = predictions[predictions["battery_id"].eq(1)].iloc[0]
        self.assertAlmostEqual(first["global_mean_prediction"], 9.0)
        self.assertAlmostEqual(first["policy_mean_prediction"], 3.0)

    def test_script_help_runs_from_repository_root(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "eol_v4_final_arbitration.py"), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
