import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis_pipeline import (
    fit_centered_monotone_quadratic,
    fit_eol_candidate,
    model_predict,
    select_q1_candidate,
)
from scripts.formal_v4_pipeline import audit_life_propagation, build_formal_q3_eol_table


class FormalV4PipelineTests(unittest.TestCase):
    def test_centered_quadratic_is_available_through_formal_eol_dispatch(self):
        n0 = 31
        cycle = np.arange(n0, 201, dtype=float)
        shifted = cycle - n0
        soh = 1.001 - 2.0e-5 * shifted - 5.0e-8 * shifted**2

        direct = fit_centered_monotone_quadratic(cycle, soh, n0)
        dispatched = fit_eol_candidate("centered_quadratic", cycle, soh, n0)

        self.assertAlmostEqual(direct["a"], 1.001, places=8)
        self.assertAlmostEqual(dispatched["b"], 2.0e-5, places=10)
        self.assertAlmostEqual(dispatched["c"], 5.0e-8, places=12)
        self.assertAlmostEqual(
            float(model_predict("centered_quadratic", dispatched, np.array([dispatched["life"]]))[0]),
            0.8,
            places=8,
        )

    def test_q1_one_se_prioritizes_truncation_stability_before_boundary(self):
        rows = []
        for battery_id in range(1, 5):
            rows.extend(
                [
                    {
                        "battery_id": battery_id,
                        "n0": 21,
                        "model": "power",
                        "MSE": 0.8 if battery_id % 2 else 1.2,
                        "MAE": 1.0,
                        "E200": 1.0,
                        "success": True,
                        "life": 1000.0 + battery_id,
                        "life_200": 800.0 + battery_id,
                        "eol_relative_update": 0.25,
                        "fit150_boundary": False,
                        "fit200_boundary": False,
                        "b_at_lower_bound": False,
                        "c_near_bound": False,
                        "c": 1.0,
                    },
                    {
                        "battery_id": battery_id,
                        "n0": 31,
                        "model": "centered_quadratic",
                        "MSE": 1.01,
                        "MAE": 1.0,
                        "E200": 1.0,
                        "success": True,
                        "life": 900.0 + battery_id,
                        "life_200": 860.0 + battery_id,
                        "eol_relative_update": 0.04,
                        "fit150_boundary": True,
                        "fit200_boundary": False,
                        "b_at_lower_bound": True,
                        "c_near_bound": False,
                        "c": 1.0e-7,
                    },
                ]
            )

        n0, model, audited = select_q1_candidate(pd.DataFrame(rows))

        self.assertEqual((n0, model), (31, "centered_quadratic"))
        self.assertTrue(audited.loc[audited["selected_one_SE"], "eligible_one_SE"].all())

    def test_formal_q3_table_uses_pooled_point_estimate_and_pooled_interval(self):
        base = pd.DataFrame(
            {
                "battery_id": [2],
                "policy": ["P"],
                "life_q3": [1800.0],
                "life_boot_low": [1400.0],
                "life_boot_median": [1750.0],
                "life_boot_high": [2300.0],
                "model_a": [1.0],
                "model_b": [2.0e-5],
                "model_c": [5.0e-8],
            }
        )
        pooled = pd.DataFrame(
            {
                "battery_id": [2],
                "v4_individual_EOL": [1800.0],
                "peer_geometric_median_EOL": [1200.0],
                "individual_weight": [0.75],
                "v4_partially_pooled_EOL": [1626.278],
                "fit_A": [1.0],
                "fit_B": [2.0e-5],
                "fit_C": [5.0e-8],
            }
        )

        formal = build_formal_q3_eol_table(base, pooled)

        self.assertAlmostEqual(float(formal.loc[0, "life_q3"]), 1626.278, places=3)
        self.assertEqual(float(formal.loc[0, "life_q3_individual"]), 1800.0)
        self.assertLess(float(formal.loc[0, "life_boot_low"]), float(formal.loc[0, "life_boot_high"]))
        self.assertEqual(formal.loc[0, "selected_model"], "centered_quadratic")
        self.assertEqual(int(formal.loc[0, "selected_n0"]), 31)

    def test_formal_orchestrator_help_runs(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "formal_v4_pipeline.py"), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("formal-output-dir", result.stdout)

    def test_cross_question_life_audit_accepts_consistent_tables(self):
        q1_battery = pd.DataFrame(
            {"battery_id": [1, 2], "policy": ["P", "P"], "life_150": [1000.0, 1200.0]}
        )
        q2_design = pd.DataFrame({"policy": ["P"], "life_150": [1100.0]})
        q3_formal = pd.DataFrame(
            {
                "battery_id": [1],
                "life_q3_individual": [1000.0],
                "peer_geometric_median_EOL": [800.0],
                "individual_weight": [0.75],
                "life_q3": [945.741609],
            }
        )
        q4_existing = pd.DataFrame({"policy": ["P"], "EOL_base": [1100.0]})

        audit = audit_life_propagation(q1_battery, q2_design, q3_formal, q4_existing)

        self.assertTrue(audit["passed"].all())
        self.assertEqual(set(audit["link"]), {"Q1→Q2", "Q1→Q4", "Q3 pooling"})


if __name__ == "__main__":
    unittest.main()
