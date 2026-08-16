from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = ROOT / "results" / "重跑_20260816_四份MD同步版_v1"
OUTPUT_DIR = ROOT / "results" / "问题4_20260816_四份MD同步版_v2"
Q1_BATTERY = SOURCE_RESULTS / "问题1" / "q1_03_四十九块电池指标与基准EOL.csv"
Q1_POLICY = SOURCE_RESULTS / "问题1" / "q1_08_九种策略分布统计.csv"
Q2_DESIGN = SOURCE_RESULTS / "问题2" / "q2_04_策略参数与SOC暴露.csv"
Q2_LOPO = SOURCE_RESULTS / "问题2" / "q2_08_AD回归_LOPO汇总.csv"
Q3_OOF = SOURCE_RESULTS / "问题3" / "q3_06_外层40折逐点预测.csv"

RNG_SEED = 20260816
BOOTSTRAP_DRAWS = 5_000
TIME_TOLERANCE = 1e-10


def configure_paths(source_results: Path, output_dir: Path) -> dict[str, Path]:
    """Configure one explicit Q1--Q3 run as Q4 input and return the resolved path set."""
    global SOURCE_RESULTS, OUTPUT_DIR, Q1_BATTERY, Q1_POLICY, Q2_DESIGN, Q2_LOPO, Q3_OOF
    SOURCE_RESULTS = Path(source_results)
    OUTPUT_DIR = Path(output_dir)
    Q1_BATTERY = SOURCE_RESULTS / "问题1" / "q1_03_四十九块电池指标与基准EOL.csv"
    Q1_POLICY = SOURCE_RESULTS / "问题1" / "q1_08_九种策略分布统计.csv"
    Q2_DESIGN = SOURCE_RESULTS / "问题2" / "q2_04_策略参数与SOC暴露.csv"
    Q2_LOPO = SOURCE_RESULTS / "问题2" / "q2_08_AD回归_LOPO汇总.csv"
    Q3_OOF = SOURCE_RESULTS / "问题3" / "q3_06_外层40折逐点预测.csv"
    return {
        "source_results": SOURCE_RESULTS,
        "output_dir": OUTPUT_DIR,
        "q1_battery": Q1_BATTERY,
        "q1_policy": Q1_POLICY,
        "q2_design": Q2_DESIGN,
        "q2_lopo": Q2_LOPO,
        "q3_oof": Q3_OOF,
    }


def theoretical_charge_time(c1, q1_percent, c2):
    """Ideal two-stage constant-current time from 0% to 80% SOC, in minutes."""
    c1 = np.asarray(c1, dtype=float)
    q = np.asarray(q1_percent, dtype=float) / 100.0
    c2 = np.asarray(c2, dtype=float)
    return 60.0 * (q / c1 + (0.8 - q) / c2)


def exposure_metrics(c1, q1_percent, c2, split: float = 0.5) -> dict[str, np.ndarray]:
    c1 = np.asarray(c1, dtype=float)
    q = np.asarray(q1_percent, dtype=float) / 100.0
    c2 = np.asarray(c2, dtype=float)
    t = theoretical_charge_time(c1, q1_percent, c2)
    avg = (c1 * q + c2 * (0.8 - q)) / 0.8
    low_len = np.minimum(q, split)
    e_low = (c1 * low_len + c2 * (split - low_len)) / split
    high_start = np.maximum(q, split)
    e_high = (c1 * np.maximum(q - split, 0.0) + c2 * (0.8 - high_start)) / (0.8 - split)
    return {
        "T_theoretical_0_80": t,
        "A": avg,
        "E_L_50": e_low,
        "E_H_50": e_high,
        "D_50": e_high - e_low,
    }


def empirical_pareto_mask(frame: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    """Return non-dominated mask when both objectives are minimized."""
    values = frame[[x_col, y_col]].to_numpy(float)
    mask = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        no_worse = np.all(values <= point + 1e-12, axis=1)
        strictly_better = np.any(values < point - 1e-12, axis=1)
        no_worse[index] = False
        strictly_better[index] = False
        if np.any(no_worse & strictly_better):
            mask[index] = False
    return mask


def choose_reference_policies(
    policy_stats: pd.DataFrame, recommended_existing_policy: str
) -> dict[str, str]:
    """从本轮Q1策略汇总动态确定对照策略，避免沿用旧寿命模型下的角色。"""
    required = {"policy", "life_median", "stable_rate_median"}
    missing = required.difference(policy_stats.columns)
    if missing:
        raise ValueError(f"Q1 policy table missing columns: {sorted(missing)}")
    valid_life = policy_stats.dropna(subset=["life_median"])
    valid_rate = policy_stats.dropna(subset=["stable_rate_median"])
    if valid_life.empty or valid_rate.empty:
        raise ValueError("Q1 policy table has no finite lifetime/rate reference")
    return {
        "recommended_existing": recommended_existing_policy,
        "typical_long_EOL": str(valid_life.loc[valid_life["life_median"].idxmax(), "policy"]),
        "early_degradation_representative": str(
            valid_rate.loc[valid_rate["stable_rate_median"].idxmin(), "policy"]
        ),
        "typical_short": str(valid_life.loc[valid_life["life_median"].idxmin(), "policy"]),
    }


def standardized_nearest_distance(
    candidates: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, float]:
    observed = np.asarray(observed, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    mean = observed.mean(axis=0)
    scale = observed.std(axis=0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("Observed design dimensions must all vary.")
    observed_z = (observed - mean) / scale
    candidate_z = (candidates - mean) / scale
    distances = np.linalg.norm(candidate_z[:, None, :] - observed_z[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    observed_distances = np.linalg.norm(
        observed_z[:, None, :] - observed_z[None, :, :], axis=2
    )
    np.fill_diagonal(observed_distances, np.inf)
    threshold = float(observed_distances.min(axis=1).max())
    return nearest, threshold


def generate_local_grid(
    observed: np.ndarray,
    time_budget: float,
    c_step: float = 0.1,
    q_step: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    """Create an implementable grid inside the raw-parameter convex hull and local radius."""
    observed = np.asarray(observed, dtype=float)
    c1_values = np.arange(observed[:, 0].min(), observed[:, 0].max() + c_step / 2, c_step)
    q_values = np.arange(observed[:, 1].min(), observed[:, 1].max() + q_step / 2, q_step)
    c2_values = np.arange(observed[:, 2].min(), observed[:, 2].max() + c_step / 2, c_step)
    c1_grid, q_grid, c2_grid = np.meshgrid(c1_values, q_values, c2_values, indexing="ij")
    candidates = np.column_stack([c1_grid.ravel(), q_grid.ravel(), c2_grid.ravel()])
    candidates[:, 0] = np.round(candidates[:, 0], 1)
    candidates[:, 1] = np.round(candidates[:, 1], 0)
    candidates[:, 2] = np.round(candidates[:, 2], 1)

    hull = Delaunay(observed, qhull_options="QJ")
    inside = hull.find_simplex(candidates, tol=1e-9) >= 0
    candidates = candidates[inside]
    candidates = np.unique(np.vstack([candidates, observed]), axis=0)

    nearest, threshold = standardized_nearest_distance(candidates, observed)
    candidate_metrics = exposure_metrics(candidates[:, 0], candidates[:, 1], candidates[:, 2])
    observed_metrics = exposure_metrics(observed[:, 0], observed[:, 1], observed[:, 2])
    observed_ad = np.column_stack([observed_metrics["A"], observed_metrics["D_50"]])
    candidate_ad = np.column_stack([candidate_metrics["A"], candidate_metrics["D_50"]])
    ad_inside = Delaunay(observed_ad, qhull_options="QJ").find_simplex(candidate_ad, tol=1e-9) >= 0
    keep = (
        (nearest <= threshold + 1e-10)
        & (candidate_metrics["T_theoretical_0_80"] <= time_budget + TIME_TOLERANCE)
        & ad_inside
    )
    candidates = candidates[keep]
    nearest = nearest[keep]
    metrics = exposure_metrics(candidates[:, 0], candidates[:, 1], candidates[:, 2])
    frame = pd.DataFrame(candidates, columns=["C1", "Q1", "C2"])
    for name, values in metrics.items():
        frame[name] = values
    frame["nearest_standardized_distance"] = nearest
    frame = frame.sort_values(["C1", "Q1", "C2"]).reset_index(drop=True)
    return frame, threshold


def predict_bootstrap_quantiles(
    candidates: pd.DataFrame,
    models: np.ndarray,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.95),
    chunk_size: int = 2_000,
) -> pd.DataFrame:
    """Evaluate bootstrap linear models without allocating a full candidate-by-draw matrix."""
    model_array = np.asarray(models, dtype=float)
    if model_array.ndim != 2 or model_array.shape[1] != 3:
        raise ValueError("models must have columns [intercept, beta_A, beta_D].")
    x = candidates[["A_z", "D_z"]].to_numpy(float)
    output = {f"q{int(round(q * 100))}": np.empty(len(candidates)) for q in quantiles}
    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        prediction = (
            model_array[:, 0, None]
            + model_array[:, 1, None] * x[None, start:stop, 0]
            + model_array[:, 2, None] * x[None, start:stop, 1]
        )
        for quantile in quantiles:
            output[f"q{int(round(quantile * 100))}"][start:stop] = np.quantile(
                prediction, quantile, axis=0
            )
    return pd.DataFrame(output, index=candidates.index)


def bootstrap_rate_models(
    battery: pd.DataFrame,
    design: pd.DataFrame,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Resample batteries within each policy and refit the six-position A,D model."""
    policy_order = design["policy"].tolist()
    groups = {
        policy: battery.loc[battery["policy"].eq(policy), "stable_rate"].dropna().to_numpy(float)
        for policy in policy_order
    }
    if any(len(values) == 0 for values in groups.values()):
        raise ValueError("Every design policy must have at least one battery.")
    x_augmented = np.column_stack(
        [np.ones(len(design)), design["A_z"].to_numpy(float), design["D_z"].to_numpy(float)]
    )
    x_pinv = np.linalg.pinv(x_augmented)
    rng = np.random.default_rng(seed)
    responses = np.empty((draws, len(policy_order)))
    for column, policy in enumerate(policy_order):
        values = groups[policy]
        sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
        responses[:, column] = np.median(sampled, axis=1)
    return responses @ x_pinv.T


def aggregate_existing_strategies(battery: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    grouped = battery.groupby("policy", as_index=False).agg(
        N=("battery_id", "size"),
        T_obs=("charge_time_150", "median"),
        T_obs_q1=("charge_time_150", lambda x: x.quantile(0.25)),
        T_obs_q3=("charge_time_150", lambda x: x.quantile(0.75)),
        r_stable=("stable_rate", "median"),
        r_stable_q1=("stable_rate", lambda x: x.quantile(0.25)),
        r_stable_q3=("stable_rate", lambda x: x.quantile(0.75)),
        EOL_base=("life_150", "median"),
    )
    parameters = policy[["policy", "C1", "Q1", "C2", "dataset_id"]].drop_duplicates("policy")
    result = grouped.merge(parameters, on="policy", how="left")
    result["pareto_observed"] = empirical_pareto_mask(result, "T_obs", "r_stable")
    return result


def bootstrap_pareto_frequency(battery: pd.DataFrame, policies: list[str], draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = {policy: battery[battery["policy"].eq(policy)] for policy in policies}
    counts = np.zeros(len(policies), dtype=int)
    for _ in range(draws):
        rows = []
        for policy in policies:
            group = groups[policy]
            sample = group.iloc[rng.integers(0, len(group), size=len(group))]
            rows.append(
                {
                    "T": float(sample["charge_time_150"].median()),
                    "R": float(sample["stable_rate"].median()),
                }
            )
        counts += empirical_pareto_mask(pd.DataFrame(rows), "T", "R")
    return counts / draws


def q3_consistency_table(
    battery: pd.DataFrame, policy_stats: pd.DataFrame, oof: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    cycle_200 = oof[oof["cycle"].eq(200)][["battery_id", "observed", "predicted"]].copy()
    merged = cycle_200.merge(
        battery[["battery_id", "policy", "SOH150", "stable_rate"]], on="battery_id", how="left"
    )
    merged["observed_rate_151_200"] = (merged["SOH150"] - merged["observed"]) / 50.0
    merged["predicted_rate_151_200"] = (merged["SOH150"] - merged["predicted"]) / 50.0
    summary = merged.groupby("policy", as_index=False).agg(
        N_complete=("battery_id", "size"),
        SOH200_observed=("observed", "median"),
        SOH200_oof=("predicted", "median"),
        future_rate_observed=("observed_rate_151_200", "median"),
        future_rate_oof=("predicted_rate_151_200", "median"),
    )
    summary = summary.merge(
        policy_stats[["policy", "stable_rate_median"]], on="policy", how="left"
    )
    summary["future_observed_rank"] = summary["future_rate_observed"].rank(method="min")
    summary["future_oof_rank"] = summary["future_rate_oof"].rank(method="min")
    observed_rho = spearmanr(summary["stable_rate_median"], summary["future_rate_observed"])
    predicted_rho = spearmanr(summary["stable_rate_median"], summary["future_rate_oof"])
    metrics = {
        "N_policies": int(len(summary)),
        "stable_vs_future_observed_spearman": float(observed_rho.statistic),
        "stable_vs_future_observed_p": float(observed_rho.pvalue),
        "stable_vs_future_oof_spearman": float(predicted_rho.statistic),
        "stable_vs_future_oof_p": float(predicted_rho.pvalue),
    }
    return summary, metrics


def add_design_metrics(frame: pd.DataFrame, a_mean: float, a_scale: float, d_mean: float, d_scale: float) -> pd.DataFrame:
    result = frame.copy()
    metrics = exposure_metrics(result["C1"], result["Q1"], result["C2"])
    for name, values in metrics.items():
        result[name] = values
    result["A_z"] = (result["A"] - a_mean) / a_scale
    result["D_z"] = (result["D_50"] - d_mean) / d_scale
    return result


def design_prediction(design_row: pd.Series, models: np.ndarray) -> np.ndarray:
    return (
        models[:, 0]
        + models[:, 1] * float(design_row["A_z"])
        + models[:, 2] * float(design_row["D_z"])
    )


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    battery = pd.read_csv(Q1_BATTERY)
    policy_stats = pd.read_csv(Q1_POLICY)
    q2_design = pd.read_csv(Q2_DESIGN)
    q2_lopo = pd.read_csv(Q2_LOPO)
    q3_oof = pd.read_csv(Q3_OOF)

    existing = aggregate_existing_strategies(battery, policy_stats)
    existing["pareto_bootstrap_probability"] = bootstrap_pareto_frequency(
        battery, existing["policy"].tolist(), BOOTSTRAP_DRAWS, RNG_SEED
    )

    time_diagnostic = existing[["policy", "C1", "Q1", "C2", "T_obs"]].copy()
    inferred = time_diagnostic["C1"].isna() & time_diagnostic["policy"].eq("80PER_3_6C")
    time_diagnostic.loc[inferred, "C1"] = time_diagnostic.loc[inferred, "C2"]
    time_diagnostic.loc[inferred, "Q1"] = 80.0
    time_diagnostic["parameter_source"] = np.where(inferred, "single_stage_inferred", "provided")
    time_diagnostic["T_theoretical_0_80"] = theoretical_charge_time(
        time_diagnostic["C1"], time_diagnostic["Q1"], time_diagnostic["C2"]
    )
    time_diagnostic["time_residual_obs_minus_theory"] = (
        time_diagnostic["T_obs"] - time_diagnostic["T_theoretical_0_80"]
    )
    finite_time = time_diagnostic.dropna(subset=["T_theoretical_0_80"])
    time_metrics = {
        "N_strategies": int(len(finite_time)),
        "median_residual_min": float(finite_time["time_residual_obs_minus_theory"].median()),
        "MAE_min": float(finite_time["time_residual_obs_minus_theory"].abs().mean()),
        "max_abs_error_min": float(finite_time["time_residual_obs_minus_theory"].abs().max()),
        "spearman": float(spearmanr(finite_time["T_theoretical_0_80"], finite_time["T_obs"]).statistic),
    }

    new = q2_design[q2_design["dataset_id"].eq(3)].copy().reset_index(drop=True)
    observed_theta = new[["C1", "Q1", "C2"]].to_numpy(float)
    time_budget = float(new["T_theoretical_0_80"].max())
    local_grid, distance_threshold = generate_local_grid(observed_theta, time_budget)

    a_mean = float(new["A"].mean())
    a_scale = float(new["A"].std(ddof=0))
    d_mean = float(new["D_50"].mean())
    d_scale = float(new["D_50"].std(ddof=0))
    new["A_z"] = (new["A"] - a_mean) / a_scale
    new["D_z"] = (new["D_50"] - d_mean) / d_scale
    local_grid["A_z"] = (local_grid["A"] - a_mean) / a_scale
    local_grid["D_z"] = (local_grid["D_50"] - d_mean) / d_scale

    battery_new = battery[battery["dataset_id"].eq(3)].copy()
    selection_models = bootstrap_rate_models(
        battery_new, new, BOOTSTRAP_DRAWS, RNG_SEED + 1
    )
    validation_models = bootstrap_rate_models(
        battery_new, new, BOOTSTRAP_DRAWS, RNG_SEED + 2
    )
    full_response = new["stable_rate_median"].to_numpy(float)
    x_augmented = np.column_stack(
        [np.ones(len(new)), new["A_z"].to_numpy(float), new["D_z"].to_numpy(float)]
    )
    point_model = np.linalg.pinv(x_augmented) @ full_response

    grid_quantiles = predict_bootstrap_quantiles(local_grid, selection_models)
    local_grid = pd.concat([local_grid, grid_quantiles.add_prefix("selection_")], axis=1)
    local_grid["point_prediction"] = (
        point_model[0] + point_model[1] * local_grid["A_z"] + point_model[2] * local_grid["D_z"]
    )
    optimized = local_grid.loc[local_grid["selection_q90"].idxmin()].copy()

    observed_design = new[
        ["policy", "C1", "Q1", "C2", "T_theoretical_0_80", "A", "D_50", "A_z", "D_z"]
    ].copy()
    observed_quantiles = predict_bootstrap_quantiles(observed_design, selection_models)
    observed_design = pd.concat([observed_design, observed_quantiles.add_prefix("selection_")], axis=1)
    best_existing = observed_design.loc[observed_design["selection_q90"].idxmin()].copy()

    optimized_validation = design_prediction(optimized, validation_models)
    existing_validation = design_prediction(best_existing, validation_models)
    paired_delta = optimized_validation - existing_validation
    lopo_mae = float(q2_lopo.loc[q2_lopo["response"].eq("rate_response"), "MAE"].iloc[0])
    validation_comparison = pd.DataFrame(
        [
            {
                "design": "best_existing",
                "policy": best_existing["policy"],
                "C1": best_existing["C1"],
                "Q1": best_existing["Q1"],
                "C2": best_existing["C2"],
                "T_theoretical_0_80": best_existing["T_theoretical_0_80"],
                "A": best_existing["A"],
                "D_50": best_existing["D_50"],
                "nearest_standardized_distance": 0.0,
                "validation_q50": np.quantile(existing_validation, 0.5),
                "validation_q90": np.quantile(existing_validation, 0.9),
                "validation_q95": np.quantile(existing_validation, 0.95),
            },
            {
                "design": "optimized_local_candidate",
                "policy": "NEW_LOCAL_CANDIDATE",
                "C1": optimized["C1"],
                "Q1": optimized["Q1"],
                "C2": optimized["C2"],
                "T_theoretical_0_80": optimized["T_theoretical_0_80"],
                "A": optimized["A"],
                "D_50": optimized["D_50"],
                "nearest_standardized_distance": optimized["nearest_standardized_distance"],
                "validation_q50": np.quantile(optimized_validation, 0.5),
                "validation_q90": np.quantile(optimized_validation, 0.9),
                "validation_q95": np.quantile(optimized_validation, 0.95),
            },
        ]
    )
    median_improvement = float(np.median(existing_validation) - np.median(optimized_validation))
    q90_paired_delta = float(np.quantile(paired_delta, 0.9))
    probability_improved = float(np.mean(paired_delta < 0))
    recommend_new = bool(q90_paired_delta < 0 and median_improvement > lopo_mae)

    q3_table, q3_metrics = q3_consistency_table(battery, policy_stats, q3_oof)
    recommended_existing_policy = str(best_existing["policy"])
    recommendation_q3 = q3_table[q3_table["policy"].eq(recommended_existing_policy)]
    if not recommendation_q3.empty:
        q3_metrics["recommended_policy"] = recommended_existing_policy
        q3_metrics["recommended_future_observed_rate"] = float(
            recommendation_q3["future_rate_observed"].iloc[0]
        )
        q3_metrics["recommended_future_oof_rate"] = float(
            recommendation_q3["future_rate_oof"].iloc[0]
        )
        q3_metrics["recommended_future_observed_rank"] = float(
            recommendation_q3["future_observed_rank"].iloc[0]
        )

    named_policies = choose_reference_policies(policy_stats, recommended_existing_policy)
    comparison_rows = []
    policy_lookup = existing.set_index("policy")
    # The A,D response model was deliberately fit only on dataset 3 / NEWSTRUCTURE.
    # Do not reuse its predictions for an identical numeric protocol from another category.
    design_lookup = new.set_index("policy")
    seen_policies: set[str] = set()
    for role, policy_name in named_policies.items():
        # 正式长寿命策略若也具有最慢早期退化，只保留正式角色，避免重复对照行。
        if policy_name in seen_policies:
            continue
        seen_policies.add(policy_name)
        row = policy_lookup.loc[policy_name]
        comparison = {
            "role": role,
            "policy": policy_name,
            "is_experimental": True,
            "C1": row["C1"],
            "Q1": row["Q1"],
            "C2": row["C2"],
            "T_obs": row["T_obs"],
            "r_stable_observed": row["r_stable"],
            "EOL_base": row["EOL_base"],
            "pareto_observed": row["pareto_observed"],
        }
        if policy_name in design_lookup.index:
            design = design_lookup.loc[policy_name]
            design_frame = add_design_metrics(
                pd.DataFrame([{"C1": design["C1"], "Q1": design["Q1"], "C2": design["C2"]}]),
                a_mean,
                a_scale,
                d_mean,
                d_scale,
            )
            prediction = design_prediction(design_frame.iloc[0], validation_models)
            comparison.update(
                {
                    "T_theoretical_0_80": design_frame["T_theoretical_0_80"].iloc[0],
                    "A": design_frame["A"].iloc[0],
                    "D_50": design_frame["D_50"].iloc[0],
                    "robust_q50": np.quantile(prediction, 0.5),
                    "robust_q90": np.quantile(prediction, 0.9),
                    "robust_q95": np.quantile(prediction, 0.95),
                }
            )
        comparison_rows.append(comparison)
    candidate_comparison = validation_comparison.iloc[1].to_dict()
    candidate_comparison.update(
        {
            "role": "optimized_local_candidate",
            "is_experimental": False,
            "T_obs": np.nan,
            "r_stable_observed": np.nan,
            "EOL_base": np.nan,
            "pareto_observed": False,
            "robust_q50": candidate_comparison.pop("validation_q50"),
            "robust_q90": candidate_comparison.pop("validation_q90"),
            "robust_q95": candidate_comparison.pop("validation_q95"),
        }
    )
    comparison_rows.append(candidate_comparison)
    recommendation_comparison = pd.DataFrame(comparison_rows)

    bootstrap_models = pd.DataFrame(
        np.vstack([selection_models, validation_models]),
        columns=["intercept", "beta_A", "beta_D"],
    )
    bootstrap_models.insert(
        0,
        "split",
        ["selection"] * len(selection_models) + ["validation"] * len(validation_models),
    )
    bootstrap_models.insert(
        1,
        "draw",
        list(range(1, len(selection_models) + 1)) + list(range(1, len(validation_models) + 1)),
    )
    paired = pd.DataFrame(
        {
            "draw": np.arange(1, len(validation_models) + 1),
            "best_existing_prediction": existing_validation,
            "optimized_candidate_prediction": optimized_validation,
            "candidate_minus_existing": paired_delta,
        }
    )

    existing.to_csv(OUTPUT_DIR / "q4_01_九策略经验Pareto.csv", index=False)
    time_diagnostic.to_csv(OUTPUT_DIR / "q4_02_充电时间理论实测校验.csv", index=False)
    local_grid.to_csv(OUTPUT_DIR / "q4_03_局部可行网格与选择目标.csv", index=False)
    validation_comparison.to_csv(OUTPUT_DIR / "q4_04_已有最优与局部候选验证比较.csv", index=False)
    bootstrap_models.to_csv(OUTPUT_DIR / "q4_05_鲁棒退化模型Bootstrap系数.csv", index=False)
    recommendation_comparison.to_csv(OUTPUT_DIR / "q4_06_推荐与典型长短策略比较.csv", index=False)
    q3_table.to_csv(OUTPUT_DIR / "q4_07_Q3短期未来一致性审计.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "q4_08_局部候选配对Bootstrap差异.csv", index=False)

    summary = {
        "source_results": str(SOURCE_RESULTS.relative_to(ROOT)),
        "time_model": time_metrics,
        "empirical_pareto_policies": existing.loc[existing["pareto_observed"], "policy"].tolist(),
        "time_budget_min": time_budget,
        "local_domain": {
            "parameter_resolution": {"C1_C": 0.1, "Q1_percent": 1.0, "C2_C": 0.1},
            "raw_convex_hull": True,
            "AD_convex_hull": True,
            "nearest_distance_threshold": distance_threshold,
            "feasible_candidate_count": int(len(local_grid)),
        },
        "robust_optimization": {
            "selection_bootstrap_draws": BOOTSTRAP_DRAWS,
            "validation_bootstrap_draws": BOOTSTRAP_DRAWS,
            "best_existing_policy": recommended_existing_policy,
            "optimized_candidate": optimized[
                [
                    "C1",
                    "Q1",
                    "C2",
                    "T_theoretical_0_80",
                    "A",
                    "D_50",
                    "nearest_standardized_distance",
                    "selection_q50",
                    "selection_q90",
                    "selection_q95",
                ]
            ].to_dict(),
            "validation_median_improvement": median_improvement,
            "validation_q90_candidate_minus_existing": q90_paired_delta,
            "validation_probability_candidate_better": probability_improved,
            "q2_lopo_mae_resolution": lopo_mae,
            "recommend_new_candidate": recommend_new,
            "final_recommendation": (
                "NEW_LOCAL_CANDIDATE" if recommend_new else recommended_existing_policy
            ),
        },
        "q3_consistency": q3_metrics,
        "claim_boundary": (
            "The optimized local candidate is an interpolation hypothesis. Q3 can update it only after "
            "150 observed cycles; no true 80% EOL labels are available."
        ),
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, ensure_ascii=False, indent=2)

    pareto_text = "、".join(summary["empirical_pareto_policies"])
    optimized_text = (
        f"{optimized['C1']:.1f}C–{optimized['Q1']:.0f}%–{optimized['C2']:.1f}C"
    )
    report = f"""# 问题四正式计算摘要

## 1. 充电时间口径

逐循环 `chargetime` 的量级与0–80%两阶段理论时间一致，而显著小于再加入80–100%理想1C阶段后至少增加12 min的完整充电时间。因此本问把它解释为约0–80%快充时间。九策略理论—实测残差中位数为 {time_metrics['median_residual_min']:.4f} min，平均绝对误差为 {time_metrics['MAE_min']:.4f} min；该证据支持字段口径判断，但不把理论式当成无误差的实测预测器。

## 2. 现有策略经验Pareto

按观测充电时间和稳定退化速率同时最小化，经验非支配策略为：{pareto_text}。

## 3. 局部鲁棒优化

- 参数域：同时位于六个 dataset 3 / NEWSTRUCTURE 实验点的原始参数凸包与 `(A,D)` 特征凸包内，并限制标准化最近邻距离不超过真实点中最大的最近邻距离。
- 可实施网格：C1、C2 按0.1C，Q1按1%离散，共保留 {len(local_grid)} 个满足局部域及理论时间不超过 {time_budget:.4f} min 的候选。
- 目标：最小化策略内电池重采样所得退化预测的90%分位数；5000次Bootstrap用于选择，另5000次独立Bootstrap用于审计。
- 模型选择的最佳已有策略：{recommended_existing_policy}。
- 局部网格候选：{optimized_text}。
- 验证Bootstrap下，候选相对最佳已有策略的退化中位改善为 {median_improvement:.3e}，候选更优概率为 {100 * probability_improved:.1f}%，配对差的90%分位数为 {q90_paired_delta:.3e}；Q2策略级LOPO MAE为 {lopo_mae:.3e}。
- 是否正式推荐新候选：{'是' if recommend_new else '否'}。最终推荐：{summary['robust_optimization']['final_recommendation']}。

## 4. Q3一致性审计

九策略稳定退化速率与第151–200圈真实退化速率的策略级Spearman相关为 {q3_metrics['stable_vs_future_observed_spearman']:.3f}，与外层OOF预测退化速率的相关为 {q3_metrics['stable_vs_future_oof_spearman']:.3f}。这只用于检查真实已有策略的短期未来方向是否矛盾；全新候选没有前150圈SOH，不能直接输入Q3。

## 5. 结论边界

Q4以直接观测稳定退化速率为主要寿命损伤指标，不用无真实标签的远期EOL作为连续优化目标。任何新参数组合仅是已有实验邻域内的待验证插值候选；获得至少150圈实测数据后，才可调用Q3进行在线短期健康更新。
"""
    (OUTPUT_DIR / "问题4正式计算摘要.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用指定的前三问结果运行问题四")
    parser.add_argument("--source-results", type=Path, default=SOURCE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    source_results = args.source_results if args.source_results.is_absolute() else ROOT / args.source_results
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    configure_paths(source_results.resolve(), output_dir.resolve())
    main()
