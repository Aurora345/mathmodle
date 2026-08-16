"""v4最终裁决：中心化单调二次、n0×结构联合选择与简单加速度基线。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from scripts.analysis_pipeline import fit_linear, fit_power
    from scripts.eol_accuracy_optimization import (
        acceleration_feature_sets,
        build_slope_feature_table,
        choose_alpha_full,
        fixed_pooling_weight_diagnostics,
        nested_pooling_predictions,
        nested_ridge_lobo,
        safe_spearman,
        shrink_log_eol,
    )
except ModuleNotFoundError:
    from analysis_pipeline import fit_linear, fit_power
    from eol_accuracy_optimization import (
        acceleration_feature_sets,
        build_slope_feature_table,
        choose_alpha_full,
        fixed_pooling_weight_diagnostics,
        nested_pooling_predictions,
        nested_ridge_lobo,
        safe_spearman,
        shrink_log_eol,
    )


ROOT = Path(__file__).resolve().parents[1]


def fit_centered_monotone_quadratic(
    cycle: np.ndarray,
    soh: np.ndarray,
    n0: int,
) -> dict[str, float | bool | int]:
    x = np.asarray(cycle, dtype=float)
    y = np.asarray(soh, dtype=float)
    shifted = x - float(n0)
    design = np.column_stack([np.ones(len(x)), -shifted, -(shifted**2)])
    result = lsq_linear(
        design,
        y,
        bounds=(np.array([0.8, 0.0, 0.0]), np.array([1.2, np.inf, np.inf])),
        lsmr_tol="auto",
    )
    A, B, C = (float(value) for value in result.x)
    if C > 0 and A > 0.8:
        discriminant = B * B + 4.0 * C * (A - 0.8)
        shifted_life = (-B + math.sqrt(discriminant)) / (2.0 * C)
    elif B > 0 and A > 0.8:
        shifted_life = (A - 0.8) / B
    else:
        shifted_life = math.nan
    life = float(n0) + shifted_life
    if not np.isfinite(life) or life <= float(np.max(x)):
        life = math.nan
    return {
        "A": A,
        "B": B,
        "C": C,
        "n0": int(n0),
        "life": float(life),
        "success": bool(result.success),
        "B_at_lower_bound": bool(B <= 1e-12),
        "C_at_lower_bound": bool(C <= 1e-14),
    }


def predict_centered_quadratic(parameters: dict, cycle: np.ndarray) -> np.ndarray:
    shifted = np.asarray(cycle, dtype=float) - float(parameters["n0"])
    return (
        float(parameters["A"])
        - float(parameters["B"]) * shifted
        - float(parameters["C"]) * shifted**2
    )


def choose_joint_candidate_one_se(summary: pd.DataFrame) -> tuple[int, str, pd.DataFrame]:
    audited = summary.copy()
    best = audited.sort_values(["mean_battery_MSE", "n0", "model"]).iloc[0]
    threshold = float(best["mean_battery_MSE"] + best["SE_battery_MSE"])
    audited["one_SE_threshold"] = threshold
    audited["eligible_one_SE"] = audited["mean_battery_MSE"].le(threshold + 1e-18)
    eligible = audited[audited["eligible_one_SE"]].copy()
    complexity = {"linear": 0, "power": 1, "centered_quadratic": 1}
    eligible["complexity"] = eligible["model"].map(complexity)
    selected = eligible.sort_values([
        "nonfinite_eol_count",
        "median_eol_relative_update",
        "boundary_count",
        "complexity",
        "n0",
        "model",
    ]).iloc[0]
    audited["selected"] = audited["n0"].eq(int(selected["n0"])) & audited["model"].eq(selected["model"])
    return int(selected["n0"]), str(selected["model"]), audited


def mean_delta_lobo(feature_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, outer in feature_table.iterrows():
        training = feature_table[~feature_table["battery_id"].eq(outer["battery_id"])]
        peers = training[training["policy"].eq(outer["policy"])]
        rows.append({
            "battery_id": int(outer["battery_id"]),
            "policy": outer["policy"],
            "observed_delta_slope": float(outer["delta_slope"]),
            "global_mean_prediction": float(training["delta_slope"].mean()),
            "policy_mean_prediction": float(peers["delta_slope"].mean()),
        })
    return pd.DataFrame(rows)


def fit_candidate(model: str, cycle: np.ndarray, soh: np.ndarray, n0: int) -> dict:
    if model == "linear":
        fit = fit_linear(cycle, soh)
        return {
            "param_A": float(fit["a"]),
            "param_B": float(fit["b"]),
            "param_C": 1.0,
            "life": float(fit["life"]),
            "success": bool(fit["success"]),
            "boundary": bool(fit["b"] >= 0),
            "n0": int(n0),
        }
    if model == "power":
        fit = fit_power(cycle, soh)
        return {
            "param_A": float(fit["a"]),
            "param_B": float(fit["b"]),
            "param_C": float(fit["c"]),
            "life": float(fit["life"]),
            "success": bool(fit["success"]),
            "boundary": bool(fit.get("b_at_lower_bound", False) or fit.get("c_near_bound", False)),
            "n0": int(n0),
        }
    if model == "centered_quadratic":
        fit = fit_centered_monotone_quadratic(cycle, soh, n0)
        return {
            "param_A": float(fit["A"]),
            "param_B": float(fit["B"]),
            "param_C": float(fit["C"]),
            "life": float(fit["life"]),
            "success": bool(fit["success"]),
            "boundary": bool(fit["B_at_lower_bound"] or fit["C_at_lower_bound"]),
            "n0": int(n0),
        }
    raise ValueError(f"Unknown candidate: {model}")


def predict_candidate(model: str, parameters: dict, cycle: np.ndarray) -> np.ndarray:
    x = np.asarray(cycle, dtype=float)
    if model == "linear":
        return float(parameters["param_A"]) + float(parameters["param_B"]) * x
    if model == "power":
        return float(parameters["param_A"]) - float(parameters["param_B"]) * x ** float(parameters["param_C"])
    if model == "centered_quadratic":
        shifted = x - float(parameters["n0"])
        return (
            float(parameters["param_A"])
            - float(parameters["param_B"]) * shifted
            - float(parameters["param_C"]) * shifted**2
        )
    raise ValueError(f"Unknown candidate: {model}")


def build_joint_candidate_detail(
    clean: pd.DataFrame,
    stable_starts: list[int],
) -> pd.DataFrame:
    rows = []
    for battery_id, group in clean.groupby("battery_id", sort=True):
        group = group.sort_values("cycle")
        if int(group["cycle"].max()) < 200:
            continue
        future = group[group["cycle"].between(151, 200)]
        x_future = future["cycle"].to_numpy(float)
        y_future = future["SOH_sg"].to_numpy(float)
        for n0 in stable_starts:
            fit150_data = group[group["cycle"].between(n0, 150)]
            fit200_data = group[group["cycle"].between(n0, 200)]
            x150 = fit150_data["cycle"].to_numpy(float)
            y150 = fit150_data["SOH_sg"].to_numpy(float)
            x200 = fit200_data["cycle"].to_numpy(float)
            y200 = fit200_data["SOH_sg"].to_numpy(float)
            for model in ["linear", "power", "centered_quadratic"]:
                fit150 = fit_candidate(model, x150, y150, n0)
                fit200 = fit_candidate(model, x200, y200, n0)
                prediction = predict_candidate(model, fit150, x_future)
                error = prediction - y_future
                life150 = float(fit150["life"])
                life200 = float(fit200["life"])
                valid = np.isfinite(life150) and np.isfinite(life200) and life200 > 0
                rows.append({
                    "battery_id": int(battery_id),
                    "policy": group["policy"].iloc[0],
                    "n0": int(n0),
                    "model": model,
                    "MAE_151_200": float(np.mean(np.abs(error))),
                    "MSE_151_200": float(np.mean(error**2)),
                    "RMSE_151_200": float(np.sqrt(np.mean(error**2))),
                    "E200": float(abs(error[-1])),
                    "life_150": life150,
                    "life_200": life200,
                    "eol_relative_update": abs(life150 - life200) / life200 if valid else math.nan,
                    "fit150_A": fit150["param_A"],
                    "fit150_B": fit150["param_B"],
                    "fit150_C": fit150["param_C"],
                    "fit200_A": fit200["param_A"],
                    "fit200_B": fit200["param_B"],
                    "fit200_C": fit200["param_C"],
                    "fit150_boundary": bool(fit150["boundary"]),
                    "fit200_boundary": bool(fit200["boundary"]),
                })
    return pd.DataFrame(rows)


def summarize_joint_candidates(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n0, model), group in detail.groupby(["n0", "model"], sort=True):
        mse = group["MSE_151_200"].to_numpy(float)
        valid = group.dropna(subset=["life_150", "life_200", "eol_relative_update"])
        rows.append({
            "n0": int(n0),
            "model": model,
            "N_batteries": int(len(group)),
            "mean_battery_MSE": float(np.mean(mse)),
            "SE_battery_MSE": float(np.std(mse, ddof=1) / np.sqrt(len(mse))),
            "pooled_RMSE": float(np.sqrt(np.mean(mse))),
            "pooled_MAE": float(group["MAE_151_200"].mean()),
            "median_E200": float(group["E200"].median()),
            "median_eol_relative_update": float(valid["eol_relative_update"].median()),
            "mean_eol_relative_update": float(valid["eol_relative_update"].mean()),
            "eol_spearman_150_200": safe_spearman(valid["life_150"], valid["life_200"]),
            "nonfinite_eol_count": int(len(group) - len(valid)),
            "boundary_count": int((group["fit150_boundary"] | group["fit200_boundary"]).sum()),
        })
    return pd.DataFrame(rows)


def nested_joint_selection(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outer_id in sorted(detail["battery_id"].unique()):
        inner_detail = detail[~detail["battery_id"].eq(outer_id)]
        inner_summary = summarize_joint_candidates(inner_detail)
        selected_n0, selected_model, audited = choose_joint_candidate_one_se(inner_summary)
        outer = detail[
            detail["battery_id"].eq(outer_id)
            & detail["n0"].eq(selected_n0)
            & detail["model"].eq(selected_model)
        ].iloc[0]
        best = audited.sort_values(["mean_battery_MSE", "n0", "model"]).iloc[0]
        rows.append({
            "held_out_battery": int(outer_id),
            "selected_n0": selected_n0,
            "selected_model": selected_model,
            "inner_best_n0": int(best["n0"]),
            "inner_best_model": str(best["model"]),
            "inner_one_SE_threshold": float(best["one_SE_threshold"]),
            "outer_MAE": float(outer["MAE_151_200"]),
            "outer_MSE": float(outer["MSE_151_200"]),
            "outer_RMSE": float(outer["RMSE_151_200"]),
            "outer_E200": float(outer["E200"]),
            "outer_eol_relative_update": float(outer["eol_relative_update"]),
        })
    return pd.DataFrame(rows)


def acceleration_baseline_predictions(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    complete = feature_table.dropna(subset=["delta_slope", "future_slope"]).reset_index(drop=True)
    means = mean_delta_lobo(complete)
    policy_columns = [column for column in complete.columns if column.startswith("policy_")]
    ridge_columns = acceleration_feature_sets(policy_columns)["现有SOH趋势_Policy"]
    ridge = nested_ridge_lobo(
        complete, ridge_columns, "delta_slope", alpha_grid=[0.01, 0.1, 1.0, 10.0, 100.0]
    ).rename(columns={"observed": "observed_delta_slope", "predicted": "predicted_delta_slope"})
    base = complete[["battery_id", "policy", "SOH_recent_slope", "future_slope", "delta_slope"]]
    frames = []
    specifications = [
        ("Direct_zero_delta", 0, pd.Series(0.0, index=base.index)),
        ("Global_mean_delta", 1, means.set_index("battery_id").loc[base["battery_id"], "global_mean_prediction"].reset_index(drop=True)),
        ("Policy_mean_delta", 2, means.set_index("battery_id").loc[base["battery_id"], "policy_mean_prediction"].reset_index(drop=True)),
        ("Ridge_SOH_Policy", 3, ridge.set_index("battery_id").loc[base["battery_id"], "predicted_delta_slope"].reset_index(drop=True)),
    ]
    for name, complexity, predicted_delta in specifications:
        frame = base.copy().reset_index(drop=True)
        frame["model"] = name
        frame["complexity_order"] = complexity
        frame["predicted_delta_slope"] = np.asarray(predicted_delta, dtype=float)
        frame["predicted_future_slope"] = frame["SOH_recent_slope"] + frame["predicted_delta_slope"]
        frame["future_slope_error"] = frame["predicted_future_slope"] - frame["future_slope"]
        frame["observed_accelerating"] = frame["delta_slope"].lt(0)
        frame["predicted_accelerating"] = frame["predicted_delta_slope"].lt(0)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    summary_rows = []
    for model, group in predictions.groupby("model", sort=False):
        squared = group["future_slope_error"] ** 2
        summary_rows.append({
            "model": model,
            "complexity_order": int(group["complexity_order"].iloc[0]),
            "N": int(len(group)),
            "future_slope_MAE": float(group["future_slope_error"].abs().mean()),
            "future_slope_RMSE": float(np.sqrt(squared.mean())),
            "mean_future_slope_MSE": float(squared.mean()),
            "SE_future_slope_MSE": float(squared.std(ddof=1) / np.sqrt(len(squared))),
            "future_slope_spearman": safe_spearman(group["predicted_future_slope"], group["future_slope"]),
            "acceleration_sign_accuracy": float(
                (group["observed_accelerating"] == group["predicted_accelerating"]).mean()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    best = summary.sort_values(["mean_future_slope_MSE", "complexity_order"]).iloc[0]
    threshold = float(best["mean_future_slope_MSE"] + best["SE_future_slope_MSE"])
    summary["one_SE_threshold"] = threshold
    summary["eligible_one_SE"] = summary["mean_future_slope_MSE"].le(threshold + 1e-30)
    selected = summary[summary["eligible_one_SE"]].sort_values("complexity_order").iloc[0]
    selected_model = str(selected["model"])
    summary["selected_one_SE"] = summary["model"].eq(selected_model)
    return predictions, summary, selected_model


def predict_test_acceleration(
    feature_table: pd.DataFrame,
    selected_model: str,
) -> pd.DataFrame:
    complete = feature_table.dropna(subset=["delta_slope"]).copy().reset_index(drop=True)
    test = feature_table[feature_table["delta_slope"].isna()].copy().reset_index(drop=True)
    if selected_model == "Direct_zero_delta":
        test["predicted_delta_slope"] = 0.0
    elif selected_model == "Global_mean_delta":
        test["predicted_delta_slope"] = float(complete["delta_slope"].mean())
    elif selected_model == "Policy_mean_delta":
        policy_means = complete.groupby("policy")["delta_slope"].mean()
        test["predicted_delta_slope"] = test["policy"].map(policy_means).astype(float)
    else:
        policy_columns = [column for column in complete.columns if column.startswith("policy_")]
        columns = acceleration_feature_sets(policy_columns)["现有SOH趋势_Policy"]
        alpha = choose_alpha_full(complete, columns, "delta_slope", [0.01, 0.1, 1.0, 10.0, 100.0])
        scaler = StandardScaler().fit(complete[columns])
        model = Ridge(alpha=alpha).fit(
            scaler.transform(complete[columns]), complete["delta_slope"].to_numpy(float)
        )
        test["predicted_delta_slope"] = model.predict(scaler.transform(test[columns]))
    test["selected_acceleration_model"] = selected_model
    test["predicted_future_slope"] = test["SOH_recent_slope"] + test["predicted_delta_slope"]
    return test[[
        "battery_id", "policy", "selected_acceleration_model", "SOH_recent_slope",
        "predicted_delta_slope", "predicted_future_slope",
    ]]


def build_test_eol(
    clean: pd.DataFrame,
    q3_predictions: pd.DataFrame,
    baseline_eol: pd.DataFrame,
    selected_detail: pd.DataFrame,
    selected_n0: int,
    selected_model: str,
    final_weight: float,
) -> pd.DataFrame:
    baseline = baseline_eol.set_index("battery_id")
    rows = []
    for battery_id in sorted(q3_predictions["battery_id"].unique()):
        observed = clean[
            clean["battery_id"].eq(battery_id) & clean["cycle"].between(selected_n0, 150)
        ].sort_values("cycle")
        predicted = q3_predictions[q3_predictions["battery_id"].eq(battery_id)].sort_values("cycle")
        cycle = np.concatenate([observed["cycle"].to_numpy(float), predicted["cycle"].to_numpy(float)])
        soh = np.concatenate([observed["SOH_sg"].to_numpy(float), predicted["SOH_pred"].to_numpy(float)])
        fit = fit_candidate(selected_model, cycle, soh, selected_n0)
        individual = float(fit["life"])
        policy = str(predicted["policy"].iloc[0])
        peers = selected_detail[selected_detail["policy"].eq(policy)]["life_200"].dropna()
        peer_life = float(np.exp(np.median(np.log(peers.to_numpy(float)))))
        pooled = shrink_log_eol(individual, peer_life, final_weight)
        rows.append({
            "battery_id": int(battery_id),
            "policy": policy,
            "selected_n0": selected_n0,
            "selected_model": selected_model,
            "pre_pool_individual_EOL": float(baseline.loc[battery_id, "life_q3"]),
            "v4_individual_EOL": individual,
            "peer_geometric_median_EOL": peer_life,
            "individual_weight": final_weight,
            "v4_partially_pooled_EOL": pooled,
            "fit_A": float(fit["param_A"]),
            "fit_B": float(fit["param_B"]),
            "fit_C": float(fit["param_C"]),
        })
    return pd.DataFrame(rows)


def make_comparison(
    joint_summary: pd.DataFrame,
    selected_n0: int,
    selected_model: str,
    nested_pool: pd.DataFrame,
    acceleration_summary: pd.DataFrame,
    selected_acceleration: str,
) -> pd.DataFrame:
    baseline = joint_summary[
        joint_summary["n0"].eq(21) & joint_summary["model"].eq("power")
    ].iloc[0]
    selected = joint_summary[
        joint_summary["n0"].eq(selected_n0) & joint_summary["model"].eq(selected_model)
    ].iloc[0]
    direct = acceleration_summary[acceleration_summary["model"].eq("Direct_zero_delta")].iloc[0]
    acceleration = acceleration_summary[acceleration_summary["model"].eq(selected_acceleration)].iloc[0]

    def row(category, metric, baseline_name, baseline_value, new_name, new_value, lower):
        change = (float(new_value) - float(baseline_value)) / abs(float(baseline_value)) * 100
        return {
            "category": category,
            "metric": metric,
            "baseline_name": baseline_name,
            "baseline_value": float(baseline_value),
            "v4_name": new_name,
            "v4_value": float(new_value),
            "relative_change_percent": change,
            "better_direction": "lower" if lower else "higher",
            "improved": bool(new_value < baseline_value if lower else new_value > baseline_value),
        }

    v4_name = f"n0={selected_n0}|{selected_model}"
    return pd.DataFrame([
        row("EOL结构", "151-200 SOH RMSE", "n0=21|power", baseline["pooled_RMSE"], v4_name, selected["pooled_RMSE"], True),
        row("EOL结构", "EOL更新中位相对差", "n0=21|power", baseline["median_eol_relative_update"], v4_name, selected["median_eol_relative_update"], True),
        row("EOL结构", "EOL更新平均相对差", "n0=21|power", baseline["mean_eol_relative_update"], v4_name, selected["mean_eol_relative_update"], True),
        row("EOL结构", "EOL排序Spearman", "n0=21|power", baseline["eol_spearman_150_200"], v4_name, selected["eol_spearman_150_200"], False),
        row("部分池化", "嵌套LOBO更新中位相对差", "individual", nested_pool["individual_relative_error"].median(), "partial_pool", nested_pool["pooled_relative_error"].median(), True),
        row("部分池化", "嵌套LOBO更新平均相对差", "individual", nested_pool["individual_relative_error"].mean(), "partial_pool", nested_pool["pooled_relative_error"].mean(), True),
        row("加速度", "未来斜率RMSE", "Direct_zero_delta", direct["future_slope_RMSE"], selected_acceleration, acceleration["future_slope_RMSE"], True),
    ])


def write_report(output_dir: Path, summary: dict) -> None:
    selected = summary["joint_selection"]
    pool = summary["partial_pooling"]
    acceleration = summary["acceleration"]
    comparison = {row["metric"]: row for row in summary["comparison"]}
    lines = [
        "# v4最终裁决结果",
        "",
        "## 最终联合选择",
        "",
        f"- 15个候选联合比较后选择 `n0={selected['selected_n0']} | {selected['selected_model']}`。",
        f"- 相对旧的 `n0=21 | power`，151--200圈RMSE变化 {comparison['151-200 SOH RMSE']['relative_change_percent']:.2f}%。",
        f"- 外层联合嵌套LOBO RMSE：{selected['baseline_nested_RMSE']:.8f}→{selected['nested_metrics']['RMSE']:.8f}（变化 {selected['nested_RMSE_relative_change_percent']:.2f}%）。",
        f"- EOL更新中位相对差变化 {comparison['EOL更新中位相对差']['relative_change_percent']:.2f}%，但平均相对差变化 {comparison['EOL更新平均相对差']['relative_change_percent']:+.2f}%。",
        f"- EOL排序Spearman从 {comparison['EOL排序Spearman']['baseline_value']:.3f} 降至 {comparison['EOL排序Spearman']['v4_value']:.3f}。因此中心化二次模型的改进不是所有稳定性指标都同步变好。",
        f"- 外层40折联合选择计数：{json.dumps(selected['nested_selection_counts'], ensure_ascii=False)}。",
        "",
        "## 部分池化",
        "",
        f"- 最终个体权重为 {pool['final_individual_weight']:.2f}。",
        f"- 嵌套LOBO中位更新差：{100*pool['nested_individual_median']:.2f}%→{100*pool['nested_pooled_median']:.2f}%。",
        f"- 嵌套LOBO平均更新差：{100*pool['nested_individual_mean']:.2f}%→{100*pool['nested_pooled_mean']:.2f}%。",
        "- 未池化个体模型的平均更新差受少数极端漂移电池影响；0.75个体+0.25同策略收缩同时降低中位数和平均数，说明其主要作用是压制尾部不稳定。",
        "- 该收缩仅用于Q3测试电池EOL，不回填Q1的统一前150圈基准寿命。",
        "",
        "## 加速度辅助预测",
        "",
        f"- 四个候选中one-SE选择 `{acceleration['selected_model']}`，未来斜率RMSE为 {acceleration['selected_RMSE']:.6g}。",
        "- 加速度结果只用于近期退化速度辅助解释，不再用于Power/Quadratic二元门控。",
        "",
        "## 证据边界",
        "",
        "151--200圈SOH与未来斜率有真实观测；80% EOL没有真实标签。这里的EOL改善表示150圈与200圈截断下的内部结构稳定性提高，不等于真实寿命准确率得到验证。",
    ]
    (output_dir / "v4最终裁决与准确性判断.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="v4中心化二次与联合嵌套选择")
    parser.add_argument(
        "--source-results",
        type=Path,
        default=ROOT / "results" / "重跑_20260816_四份MD同步版_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "寿命预测v4最终裁决_20260816",
    )
    args = parser.parse_args()
    source_results = args.source_results if args.source_results.is_absolute() else ROOT / args.source_results
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(source_results / "问题1" / "q1_01_逐循环清洗数据.csv")
    source_summary = json.loads((source_results / "summary.json").read_text(encoding="utf-8"))
    battery_summary = pd.read_csv(ROOT / "battery_summary.csv")
    q3_predictions = pd.read_csv(source_results / "问题3" / "q3_14_九块测试电池151_200正式预测.csv")
    baseline_eol = pd.read_csv(source_results / "问题3" / "q3_15_九块测试电池EOL与统计区间.csv")

    detail = build_joint_candidate_detail(clean, [11, 21, 31, 41, 51])
    joint_summary_raw = summarize_joint_candidates(detail)
    selected_n0, selected_model, joint_summary = choose_joint_candidate_one_se(joint_summary_raw)
    nested = nested_joint_selection(detail)
    detail.to_csv(output_dir / "v4_01_四十块电池15候选逐电池结果.csv", index=False)
    joint_summary.to_csv(output_dir / "v4_02_十五候选联合oneSE与稳定性.csv", index=False)
    nested.to_csv(output_dir / "v4_03_联合嵌套LOBO逐折.csv", index=False)
    nested_summary = pd.DataFrame([{
        "outer_folds": int(len(nested)),
        "MAE": float(nested["outer_MAE"].mean()),
        "RMSE": float(np.sqrt(nested["outer_MSE"].mean())),
        "median_E200": float(nested["outer_E200"].median()),
        "median_eol_relative_update": float(nested["outer_eol_relative_update"].median()),
        "mean_eol_relative_update": float(nested["outer_eol_relative_update"].mean()),
    }])
    nested_summary.to_csv(output_dir / "v4_04_联合嵌套LOBO汇总.csv", index=False)

    selected_detail = detail[
        detail["n0"].eq(selected_n0) & detail["model"].eq(selected_model)
    ][["battery_id", "policy", "life_150", "life_200"]].dropna()
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    fixed_pool_detail, fixed_pool_summary, final_weight = fixed_pooling_weight_diagnostics(selected_detail, weights)
    nested_pool, nested_pool_inner = nested_pooling_predictions(selected_detail, weights)
    fixed_pool_detail.to_csv(output_dir / "v4_05_Q3部分池化固定权重逐电池.csv", index=False)
    fixed_pool_summary.to_csv(output_dir / "v4_06_Q3部分池化固定权重汇总.csv", index=False)
    nested_pool.to_csv(output_dir / "v4_07_Q3部分池化嵌套LOBO逐电池.csv", index=False)
    nested_pool_inner.to_csv(output_dir / "v4_08_Q3部分池化嵌套内层选权.csv", index=False)

    feature_table = build_slope_feature_table(clean, battery_summary)
    acceleration_predictions, acceleration_summary, selected_acceleration = acceleration_baseline_predictions(feature_table)
    acceleration_predictions.to_csv(output_dir / "v4_09_加速度四基线LOBO逐电池.csv", index=False)
    acceleration_summary.to_csv(output_dir / "v4_10_加速度四基线oneSE汇总.csv", index=False)
    test_acceleration = predict_test_acceleration(feature_table, selected_acceleration)
    test_acceleration.to_csv(output_dir / "v4_11_九块测试电池未来斜率预测.csv", index=False)

    test_eol = build_test_eol(
        clean, q3_predictions, baseline_eol,
        detail[detail["n0"].eq(selected_n0) & detail["model"].eq(selected_model)],
        selected_n0, selected_model, final_weight,
    )
    test_eol.to_csv(output_dir / "v4_12_九块测试电池EOL与部分池化.csv", index=False)

    comparison = make_comparison(
        joint_summary, selected_n0, selected_model, nested_pool,
        acceleration_summary, selected_acceleration,
    )
    comparison.to_csv(output_dir / "v4_13_旧基线与v4关键指标对比.csv", index=False)

    selected_row = joint_summary[joint_summary["selected"]].iloc[0]
    selected_acceleration_row = acceleration_summary[acceleration_summary["selected_one_SE"]].iloc[0]
    selection_counts = {
        f"n0={int(n0)}|{model}": int(count)
        for (n0, model), count in nested.groupby(["selected_n0", "selected_model"]).size().items()
    }
    baseline_nested_rmse = float(source_summary["q1"]["nested_RMSE"])
    v4_nested_rmse = float(nested_summary.iloc[0]["RMSE"])
    summary = {
        "source_results": str(source_results.relative_to(ROOT)),
        "selection_rule": "151-200 mean battery MSE one-SE; then nonfinite, median EOL truncation update, boundary, complexity, n0",
        "joint_selection": {
            "selected_n0": selected_n0,
            "selected_model": selected_model,
            "selected_metrics": {key: (value.item() if isinstance(value, np.generic) else value) for key, value in selected_row.to_dict().items()},
            "nested_selection_counts": selection_counts,
            "nested_metrics": nested_summary.iloc[0].to_dict(),
            "baseline_nested_RMSE": baseline_nested_rmse,
            "nested_RMSE_relative_change_percent": 100.0 * (v4_nested_rmse / baseline_nested_rmse - 1.0),
        },
        "partial_pooling": {
            "scope": "Q3 test-cell EOL only; not Q1 baseline life",
            "candidate_weights": weights,
            "final_individual_weight": final_weight,
            "nested_selected_weight_counts": {
                str(weight): int(count)
                for weight, count in nested_pool["selected_weight"].value_counts().sort_index().items()
            },
            "nested_individual_median": float(nested_pool["individual_relative_error"].median()),
            "nested_pooled_median": float(nested_pool["pooled_relative_error"].median()),
            "nested_individual_mean": float(nested_pool["individual_relative_error"].mean()),
            "nested_pooled_mean": float(nested_pool["pooled_relative_error"].mean()),
        },
        "acceleration": {
            "selected_model": selected_acceleration,
            "selected_RMSE": float(selected_acceleration_row["future_slope_RMSE"]),
            "selected_MAE": float(selected_acceleration_row["future_slope_MAE"]),
            "selected_sign_accuracy": float(selected_acceleration_row["acceleration_sign_accuracy"]),
            "formal_eol_gate_used": False,
            "all_models": acceleration_summary.to_dict("records"),
        },
        "comparison": comparison.to_dict("records"),
        "claim_boundary": "No true 80% EOL labels; EOL metrics are truncation-stability diagnostics, not true lifetime accuracy.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    write_report(output_dir, summary)
    print(json.dumps({
        "output_dir": str(output_dir),
        "selected_n0": selected_n0,
        "selected_model": selected_model,
        "nested_selection_counts": selection_counts,
        "partial_pool_weight": final_weight,
        "selected_acceleration": selected_acceleration,
        "selected_acceleration_RMSE": float(selected_acceleration_row["future_slope_RMSE"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
