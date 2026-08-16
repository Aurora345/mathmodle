"""寿命外推小优化实验：三结构、截断稳定性、同策略部分池化与斜率辅助头。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from scripts.analysis_pipeline import fit_linear, fit_power, ts_slope
except ModuleNotFoundError:
    from analysis_pipeline import fit_linear, fit_power, ts_slope


ROOT = Path(__file__).resolve().parents[1]


def fit_monotone_quadratic(cycle: np.ndarray, soh: np.ndarray) -> dict[str, float | bool]:
    """拟合 SOH=a-b*n-c*n^2，约束 b,c>=0，并返回80%交点。"""
    x = np.asarray(cycle, dtype=float)
    y = np.asarray(soh, dtype=float)
    design = np.column_stack([np.ones(len(x)), -x, -(x**2)])
    result = lsq_linear(
        design,
        y,
        bounds=(np.array([0.8, 0.0, 0.0]), np.array([1.2, np.inf, np.inf])),
        lsmr_tol="auto",
    )
    a, b, c = (float(value) for value in result.x)
    if c > 0 and a > 0.8:
        discriminant = b * b + 4.0 * c * (a - 0.8)
        life = (-b + math.sqrt(discriminant)) / (2.0 * c)
    elif b > 0 and a > 0.8:
        life = (a - 0.8) / b
    else:
        life = math.nan
    if not np.isfinite(life) or life <= float(np.max(x)):
        life = math.nan
    return {
        "a": a,
        "b": b,
        "c": c,
        "life": float(life),
        "success": bool(result.success),
        "b_at_lower_bound": bool(b <= 1e-12),
        "c_at_lower_bound": bool(c <= 1e-14),
    }


def predict_eol_model(model: str, parameters: dict, cycle: np.ndarray) -> np.ndarray:
    x = np.asarray(cycle, dtype=float)
    if model == "linear":
        return float(parameters["a"]) + float(parameters["b"]) * x
    if model == "power":
        return float(parameters["a"]) - float(parameters["b"]) * x ** float(parameters["c"])
    if model == "quadratic":
        return float(parameters["a"]) - float(parameters["b"]) * x - float(parameters["c"]) * x**2
    raise ValueError(f"Unknown EOL model: {model}")


def choose_structure_one_se(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """短期MSE一标准误近优后，按EOL稳定性、无效数和边界数裁决。"""
    audited = summary.copy()
    best = audited.sort_values(["mean_battery_MSE", "model"]).iloc[0]
    threshold = float(best["mean_battery_MSE"] + best["SE_battery_MSE"])
    audited["one_SE_threshold"] = threshold
    audited["eligible_one_SE"] = audited["mean_battery_MSE"].le(threshold + 1e-18)
    eligible = audited[audited["eligible_one_SE"]].copy()
    chosen = eligible.sort_values(
        ["nonfinite_eol_count", "median_eol_relative_update", "boundary_count", "model"]
    ).iloc[0]["model"]
    audited["selected"] = audited["model"].eq(chosen)
    return str(chosen), audited


def shrink_log_eol(individual_life: float, peer_life: float, individual_weight: float) -> float:
    if individual_life <= 0 or peer_life <= 0:
        return math.nan
    weight = float(individual_weight)
    return float(math.exp(weight * math.log(individual_life) + (1.0 - weight) * math.log(peer_life)))


def nested_pooling_predictions(
    life_table: pd.DataFrame,
    weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """外层留一电池；每个外层训练集中再用其余电池选择收缩权重。"""
    required = {"battery_id", "policy", "life_150", "life_200"}
    missing = required - set(life_table.columns)
    if missing:
        raise ValueError(f"Missing pooling columns: {sorted(missing)}")
    frame = life_table.dropna(subset=["life_150", "life_200"]).copy()
    prediction_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for outer_id in frame["battery_id"].tolist():
        outer = frame[frame["battery_id"].eq(outer_id)].iloc[0]
        training = frame[~frame["battery_id"].eq(outer_id)].copy()
        candidates = []
        for weight in weights:
            errors = []
            for inner_id in training["battery_id"].tolist():
                inner = training[training["battery_id"].eq(inner_id)].iloc[0]
                peers = training[
                    ~training["battery_id"].eq(inner_id) & training["policy"].eq(inner["policy"])
                ]["life_200"].dropna()
                if peers.empty:
                    continue
                peer_life = float(np.exp(np.median(np.log(peers.to_numpy(float)))))
                prediction = shrink_log_eol(float(inner["life_150"]), peer_life, float(weight))
                errors.append(abs(prediction - float(inner["life_200"])) / float(inner["life_200"]))
            median_error = float(np.median(errors)) if errors else math.inf
            mean_error = float(np.mean(errors)) if errors else math.inf
            candidates.append((median_error, mean_error, -float(weight), float(weight), len(errors)))
        selected = min(candidates)
        selected_weight = float(selected[3])
        for median_error, mean_error, _, weight, count in candidates:
            diagnostic_rows.append({
                "outer_battery_id": int(outer_id),
                "weight": float(weight),
                "inner_N": int(count),
                "inner_median_relative_error": float(median_error),
                "inner_mean_relative_error": float(mean_error),
                "selected": bool(weight == selected_weight),
            })
        peers = training[training["policy"].eq(outer["policy"])]["life_200"].dropna()
        peer_life = (
            float(np.exp(np.median(np.log(peers.to_numpy(float))))) if not peers.empty
            else float(outer["life_150"])
        )
        pooled = shrink_log_eol(float(outer["life_150"]), peer_life, selected_weight)
        reference = float(outer["life_200"])
        prediction_rows.append({
            "battery_id": int(outer_id),
            "policy": outer["policy"],
            "life_150": float(outer["life_150"]),
            "peer_life_200": peer_life,
            "selected_weight": selected_weight,
            "pooled_life": pooled,
            "life_200": reference,
            "individual_relative_error": abs(float(outer["life_150"]) - reference) / reference,
            "pooled_relative_error": abs(pooled - reference) / reference,
        })
    return pd.DataFrame(prediction_rows), pd.DataFrame(diagnostic_rows)


def nested_ridge_lobo(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alpha_grid: list[float],
) -> pd.DataFrame:
    """外层留一预测，内层LOO选择Ridge惩罚；所有标准化仅在训练折内拟合。"""
    data = frame.dropna(subset=[*feature_columns, target_column]).reset_index(drop=True)
    rows: list[dict] = []
    for outer_index in range(len(data)):
        outer = data.iloc[[outer_index]]
        training = data.drop(index=outer_index).reset_index(drop=True)
        alpha_scores = []
        for alpha in alpha_grid:
            inner_errors = []
            for inner_index in range(len(training)):
                inner_validation = training.iloc[[inner_index]]
                inner_training = training.drop(index=inner_index)
                scaler = StandardScaler().fit(inner_training[feature_columns])
                model = Ridge(alpha=float(alpha)).fit(
                    scaler.transform(inner_training[feature_columns]),
                    inner_training[target_column].to_numpy(float),
                )
                predicted = float(model.predict(scaler.transform(inner_validation[feature_columns]))[0])
                observed = float(inner_validation[target_column].iloc[0])
                inner_errors.append((predicted - observed) ** 2)
            alpha_scores.append((float(np.mean(inner_errors)), float(alpha)))
        selected_alpha = min(alpha_scores)[1]
        scaler = StandardScaler().fit(training[feature_columns])
        model = Ridge(alpha=selected_alpha).fit(
            scaler.transform(training[feature_columns]),
            training[target_column].to_numpy(float),
        )
        predicted = float(model.predict(scaler.transform(outer[feature_columns]))[0])
        rows.append({
            "battery_id": int(outer["battery_id"].iloc[0]),
            "observed": float(outer[target_column].iloc[0]),
            "predicted": predicted,
            "selected_alpha": selected_alpha,
        })
    return pd.DataFrame(rows)


def fit_eol_model(model: str, cycle: np.ndarray, soh: np.ndarray) -> dict:
    if model == "linear":
        result = fit_linear(cycle, soh)
        result["boundary"] = bool(result["b"] >= 0)
        return result
    if model == "power":
        result = fit_power(cycle, soh)
        result["boundary"] = bool(
            result.get("b_at_lower_bound", False) or result.get("c_near_bound", False)
        )
        return result
    if model == "quadratic":
        result = fit_monotone_quadratic(cycle, soh)
        result["boundary"] = bool(
            result.get("b_at_lower_bound", False) or result.get("c_at_lower_bound", False)
        )
        return result
    raise ValueError(f"Unknown EOL model: {model}")


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy)
    if valid.sum() < 3 or np.unique(xx[valid]).size < 2 or np.unique(yy[valid]).size < 2:
        return math.nan
    return float(spearmanr(xx[valid], yy[valid]).statistic)


def build_eol_model_diagnostics(clean: pd.DataFrame, n0: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict] = []
    for battery_id, group in clean.groupby("battery_id", sort=True):
        group = group.sort_values("cycle")
        if int(group["cycle"].max()) < 200:
            continue
        train150 = group[group["cycle"].between(n0, 150)]
        train200 = group[group["cycle"].between(n0, 200)]
        future = group[group["cycle"].between(151, 200)]
        x150 = train150["cycle"].to_numpy(float)
        y150 = train150["SOH_sg"].to_numpy(float)
        x200 = train200["cycle"].to_numpy(float)
        y200 = train200["SOH_sg"].to_numpy(float)
        x_future = future["cycle"].to_numpy(float)
        y_future = future["SOH_sg"].to_numpy(float)
        for model in ["linear", "power", "quadratic"]:
            fit150 = fit_eol_model(model, x150, y150)
            fit200 = fit_eol_model(model, x200, y200)
            prediction = predict_eol_model(model, fit150, x_future)
            error = prediction - y_future
            life150 = float(fit150["life"])
            life200 = float(fit200["life"])
            finite_pair = np.isfinite(life150) and np.isfinite(life200) and life200 > 0
            detail_rows.append({
                "battery_id": int(battery_id),
                "policy": group["policy"].iloc[0],
                "model": model,
                "MAE_151_200": float(np.mean(np.abs(error))),
                "MSE_151_200": float(np.mean(error**2)),
                "RMSE_151_200": float(np.sqrt(np.mean(error**2))),
                "E200": float(abs(error[-1])),
                "life_150": life150,
                "life_200": life200,
                "eol_absolute_update": abs(life150 - life200) if finite_pair else math.nan,
                "eol_relative_update": abs(life150 - life200) / life200 if finite_pair else math.nan,
                "fit150_a": float(fit150["a"]),
                "fit150_b": float(fit150["b"]),
                "fit150_c": float(fit150["c"]),
                "fit200_a": float(fit200["a"]),
                "fit200_b": float(fit200["b"]),
                "fit200_c": float(fit200["c"]),
                "fit150_boundary": bool(fit150.get("boundary", False)),
                "fit200_boundary": bool(fit200.get("boundary", False)),
            })
    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for model, group in detail.groupby("model", sort=False):
        mse = group["MSE_151_200"].to_numpy(float)
        valid_eol = group.dropna(subset=["life_150", "life_200", "eol_relative_update"])
        summary_rows.append({
            "model": model,
            "N_batteries": int(len(group)),
            "mean_battery_MSE": float(np.mean(mse)),
            "SE_battery_MSE": float(np.std(mse, ddof=1) / np.sqrt(len(mse))),
            "pooled_RMSE": float(np.sqrt(np.mean(mse))),
            "pooled_MAE": float(group["MAE_151_200"].mean()),
            "median_E200": float(group["E200"].median()),
            "median_eol_relative_update": float(valid_eol["eol_relative_update"].median()),
            "mean_eol_relative_update": float(valid_eol["eol_relative_update"].mean()),
            "eol_spearman_150_200": safe_spearman(valid_eol["life_150"], valid_eol["life_200"]),
            "nonfinite_eol_count": int(len(group) - len(valid_eol)),
            "boundary_count": int((group["fit150_boundary"] | group["fit200_boundary"]).sum()),
        })
    return detail, pd.DataFrame(summary_rows)


def fixed_pooling_weight_diagnostics(
    life_table: pd.DataFrame,
    weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    rows = []
    for weight in weights:
        for _, row in life_table.iterrows():
            peers = life_table[
                ~life_table["battery_id"].eq(row["battery_id"])
                & life_table["policy"].eq(row["policy"])
            ]["life_200"].dropna()
            if peers.empty or not np.isfinite(row["life_150"]) or not np.isfinite(row["life_200"]):
                continue
            peer = float(np.exp(np.median(np.log(peers.to_numpy(float)))))
            prediction = shrink_log_eol(float(row["life_150"]), peer, float(weight))
            reference = float(row["life_200"])
            rows.append({
                "battery_id": int(row["battery_id"]),
                "policy": row["policy"],
                "weight": float(weight),
                "life_150": float(row["life_150"]),
                "peer_life_200": peer,
                "pooled_life": prediction,
                "life_200": reference,
                "relative_error": abs(prediction - reference) / reference,
                "log_error": math.log(prediction) - math.log(reference),
            })
    detail = pd.DataFrame(rows)
    summary_rows = []
    for weight, group in detail.groupby("weight", sort=True):
        summary_rows.append({
            "weight": float(weight),
            "N": int(len(group)),
            "median_relative_error": float(group["relative_error"].median()),
            "mean_relative_error": float(group["relative_error"].mean()),
            "log_RMSE": float(np.sqrt(np.mean(group["log_error"] ** 2))),
            "spearman": safe_spearman(group["pooled_life"], group["life_200"]),
        })
    summary = pd.DataFrame(summary_rows)
    selected = summary.sort_values(
        ["median_relative_error", "mean_relative_error", "weight"], ascending=[True, True, False]
    ).iloc[0]
    summary["selected_for_final_fit"] = summary["weight"].eq(float(selected["weight"]))
    return detail, summary, float(selected["weight"])


def early_feature_row(group: pd.DataFrame, summary_row: pd.Series, length: int = 150) -> dict:
    early = group[group["cycle"].le(length)].sort_values("cycle")
    recent = early[early["cycle"].between(length - 49, length)]
    x = early["cycle"].to_numpy(float)
    soh = early["SOH_sg"].to_numpy(float)
    recent_x = recent["cycle"].to_numpy(float)
    recent_soh = recent["SOH_sg"].to_numpy(float)
    residual = recent["SOH_clean"].to_numpy(float) - recent_soh
    residual_center = float(np.median(residual))
    residual_mad = float(1.4826 * np.median(np.abs(residual - residual_center)))
    recent_ir = recent["IR_clean"].to_numpy(float)
    coupling = safe_spearman(recent_ir, recent_soh)
    if not np.isfinite(coupling):
        coupling = 0.0
    global_slope = ts_slope(x, soh)
    recent_slope = ts_slope(recent_x, recent_soh)
    return {
        "battery_id": int(summary_row.get("battery_id", summary_row.name)),
        "policy": summary_row["policy"],
        "initial_capacity": float(summary_row["initial_capacity"]),
        "initial_abs_IR": float(np.median(early["IR_clean"].to_numpy(float)[:10])),
        "SOH_state": float(np.median(soh[-10:])),
        "SOH_AUC": float(trapezoid(soh, x) / (x[-1] - x[0])),
        "SOH_global_slope": global_slope,
        "SOH_recent_slope": recent_slope,
        "SOH_delta_slope": recent_slope - global_slope,
        "SOH_residual_MAD": residual_mad,
        "IR_SOH_spearman": coupling,
    }


def build_slope_feature_table(clean: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    summary_by_id = summary.set_index("battery_id")
    for battery_id, group in clean.groupby("battery_id", sort=True):
        base = early_feature_row(group, summary_by_id.loc[battery_id])
        if int(group["cycle"].max()) >= 200:
            future = group[group["cycle"].between(151, 200)].sort_values("cycle")
            future_slope = ts_slope(
                future["cycle"].to_numpy(float), future["SOH_sg"].to_numpy(float)
            )
            base["future_slope"] = future_slope
            base["delta_slope"] = future_slope - float(base["SOH_recent_slope"])
        rows.append(base)
    feature_table = pd.DataFrame(rows)
    dummies = pd.get_dummies(feature_table["policy"], prefix="policy", dtype=float)
    return pd.concat([feature_table, dummies], axis=1)


def acceleration_feature_sets(policy_columns: list[str]) -> dict[str, list[str]]:
    base = ["SOH_recent_slope", "SOH_state", "SOH_AUC", "SOH_global_slope", "SOH_delta_slope"]
    static = ["initial_capacity", "initial_abs_IR"]
    rough = ["SOH_residual_MAD"]
    coupling = ["IR_SOH_spearman"]
    return {
        "直接延续": [],
        "现有SOH趋势": base,
        "现有SOH趋势_Policy": base + policy_columns,
        "+初始容量_绝对IR": base + static,
        "+初始容量_绝对IR_Policy": base + static + policy_columns,
        "+残差MAD": base + rough,
        "+残差MAD_Policy": base + rough + policy_columns,
        "+静态_残差": base + static + rough,
        "+静态_残差_Policy": base + static + rough + policy_columns,
        "+静态_残差_IR耦合": base + static + rough + coupling,
        "+静态_残差_IR耦合_Policy": base + static + rough + coupling + policy_columns,
    }


def evaluate_acceleration_heads(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, list[str]]]:
    complete = feature_table.dropna(subset=["delta_slope", "future_slope"]).reset_index(drop=True)
    feature_sets = acceleration_feature_sets(
        [column for column in complete.columns if column.startswith("policy_")]
    )
    prediction_frames = []
    for name, columns in feature_sets.items():
        if not columns:
            predictions = complete[["battery_id", "delta_slope"]].rename(
                columns={"delta_slope": "observed"}
            )
            predictions["predicted"] = 0.0
            predictions["selected_alpha"] = math.nan
        else:
            predictions = nested_ridge_lobo(
                complete, columns, "delta_slope", alpha_grid=[0.01, 0.1, 1.0, 10.0, 100.0]
            )
        predictions = predictions.merge(
            complete[["battery_id", "policy", "SOH_recent_slope", "future_slope"]],
            on="battery_id",
            how="left",
            validate="one_to_one",
        )
        predictions["feature_set"] = name
        predictions["predicted_future_slope"] = predictions["SOH_recent_slope"] + predictions["predicted"]
        predictions["future_slope_error"] = predictions["predicted_future_slope"] - predictions["future_slope"]
        predictions["observed_accelerating"] = predictions["observed"].lt(0)
        predictions["predicted_accelerating"] = predictions["predicted"].lt(0)
        prediction_frames.append(predictions)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    complexity = {name: index for index, name in enumerate(feature_sets)}
    summary_rows = []
    for name, group in all_predictions.groupby("feature_set", sort=False):
        squared = group["future_slope_error"] ** 2
        summary_rows.append({
            "feature_set": name,
            "feature_count": len(feature_sets[name]),
            "complexity_order": complexity[name],
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
    head_summary = pd.DataFrame(summary_rows)
    best = head_summary.sort_values(["mean_future_slope_MSE", "complexity_order"]).iloc[0]
    threshold = float(best["mean_future_slope_MSE"] + best["SE_future_slope_MSE"])
    head_summary["one_SE_threshold"] = threshold
    head_summary["eligible_one_SE"] = head_summary["mean_future_slope_MSE"].le(threshold + 1e-30)
    selected = head_summary[head_summary["eligible_one_SE"]].sort_values("complexity_order").iloc[0]
    selected_name = str(selected["feature_set"])
    head_summary["selected_one_SE"] = head_summary["feature_set"].eq(selected_name)
    return all_predictions, head_summary, selected_name, feature_sets


def choose_alpha_full(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    alpha_grid: list[float],
) -> float:
    scores = []
    for alpha in alpha_grid:
        errors = []
        for index in range(len(frame)):
            validation = frame.iloc[[index]]
            training = frame.drop(index=index)
            scaler = StandardScaler().fit(training[feature_columns])
            model = Ridge(alpha=float(alpha)).fit(
                scaler.transform(training[feature_columns]), training[target_column].to_numpy(float)
            )
            prediction = float(model.predict(scaler.transform(validation[feature_columns]))[0])
            errors.append((prediction - float(validation[target_column].iloc[0])) ** 2)
        scores.append((float(np.mean(errors)), float(alpha)))
    return min(scores)[1]


def final_acceleration_predictions(
    feature_table: pd.DataFrame,
    selected_feature_set: str,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    complete = feature_table.dropna(subset=["delta_slope"]).copy()
    test = feature_table[feature_table["delta_slope"].isna()].copy()
    columns = feature_sets[selected_feature_set]
    if not columns:
        test["predicted_delta_slope"] = 0.0
        test["selected_alpha"] = math.nan
    else:
        alpha = choose_alpha_full(complete.reset_index(drop=True), columns, "delta_slope", [0.01, 0.1, 1.0, 10.0, 100.0])
        scaler = StandardScaler().fit(complete[columns])
        model = Ridge(alpha=alpha).fit(
            scaler.transform(complete[columns]), complete["delta_slope"].to_numpy(float)
        )
        test["predicted_delta_slope"] = model.predict(scaler.transform(test[columns]))
        test["selected_alpha"] = alpha
    test["predicted_future_slope"] = test["SOH_recent_slope"] + test["predicted_delta_slope"]
    test["predicted_accelerating"] = test["predicted_delta_slope"].lt(0)
    test["feature_set"] = selected_feature_set
    return test[[
        "battery_id", "policy", "feature_set", "selected_alpha", "SOH_recent_slope",
        "predicted_delta_slope", "predicted_future_slope", "predicted_accelerating",
    ]]


def acceleration_gate_audit(
    eol_detail: pd.DataFrame,
    acceleration_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    selected_head = acceleration_predictions[
        acceleration_predictions["feature_set"].eq(acceleration_predictions["feature_set"].iloc[0])
    ]
    rows = []
    for _, head in selected_head.iterrows():
        chosen_model = "quadratic" if bool(head["predicted_accelerating"]) else "power"
        eol = eol_detail[
            eol_detail["battery_id"].eq(head["battery_id"]) & eol_detail["model"].eq(chosen_model)
        ].iloc[0]
        rows.append({
            "battery_id": int(head["battery_id"]),
            "policy": eol["policy"],
            "predicted_delta_slope": float(head["predicted"]),
            "observed_delta_slope": float(head["observed"]),
            "gated_model": chosen_model,
            "life_150": float(eol["life_150"]),
            "life_200": float(eol["life_200"]),
            "eol_relative_update": float(eol["eol_relative_update"]),
        })
    audit = pd.DataFrame(rows)
    result = {
        "median_eol_relative_update": float(audit["eol_relative_update"].median()),
        "mean_eol_relative_update": float(audit["eol_relative_update"].mean()),
        "spearman": safe_spearman(audit["life_150"], audit["life_200"]),
        "quadratic_gate_count": int(audit["gated_model"].eq("quadratic").sum()),
    }
    return audit, result


def build_test_eol_predictions(
    clean: pd.DataFrame,
    q3_predictions: pd.DataFrame,
    baseline_eol: pd.DataFrame,
    eol_detail: pd.DataFrame,
    selected_model: str,
    final_weight: float,
    test_acceleration: pd.DataFrame,
    n0: int = 21,
) -> pd.DataFrame:
    rows = []
    baseline_by_id = baseline_eol.set_index("battery_id")
    acceleration_by_id = test_acceleration.set_index("battery_id")
    for battery_id in sorted(q3_predictions["battery_id"].unique()):
        observed = clean[
            clean["battery_id"].eq(battery_id) & clean["cycle"].between(n0, 150)
        ].sort_values("cycle")
        predicted = q3_predictions[q3_predictions["battery_id"].eq(battery_id)].sort_values("cycle")
        cycle = np.concatenate([observed["cycle"].to_numpy(float), predicted["cycle"].to_numpy(float)])
        soh = np.concatenate([observed["SOH_sg"].to_numpy(float), predicted["SOH_pred"].to_numpy(float)])
        policy = str(predicted["policy"].iloc[0])
        fits = {model: fit_eol_model(model, cycle, soh) for model in ["linear", "power", "quadratic"]}
        individual = float(fits[selected_model]["life"])
        peers = eol_detail[
            eol_detail["model"].eq(selected_model) & eol_detail["policy"].eq(policy)
        ]["life_200"].dropna()
        peer_life = float(np.exp(np.median(np.log(peers.to_numpy(float)))))
        pooled = shrink_log_eol(individual, peer_life, final_weight)
        predicted_accelerating = bool(acceleration_by_id.loc[battery_id, "predicted_accelerating"])
        gated_model = "quadratic" if predicted_accelerating else "power"
        rows.append({
            "battery_id": int(battery_id),
            "policy": policy,
            "current_power_EOL_baseline": float(baseline_by_id.loc[battery_id, "life_q3"]),
            "linear_EOL": float(fits["linear"]["life"]),
            "power_EOL": float(fits["power"]["life"]),
            "quadratic_EOL": float(fits["quadratic"]["life"]),
            "selected_structure": selected_model,
            "selected_individual_EOL": individual,
            "peer_geometric_median_EOL": peer_life,
            "individual_weight": final_weight,
            "partially_pooled_EOL": pooled,
            "acceleration_head_feature_set": acceleration_by_id.loc[battery_id, "feature_set"],
            "predicted_delta_slope": float(acceleration_by_id.loc[battery_id, "predicted_delta_slope"]),
            "acceleration_gated_structure": gated_model,
            "acceleration_gated_EOL": float(fits[gated_model]["life"]),
        })
    return pd.DataFrame(rows)


def serializable_record(record: dict) -> dict:
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in record.items()
    }


def build_accuracy_comparison(
    eol_summary: pd.DataFrame,
    slope_summary: pd.DataFrame,
    nested_pool: pd.DataFrame,
) -> pd.DataFrame:
    eol = eol_summary.set_index("model")
    heads = slope_summary.set_index("feature_set")
    new_policy = slope_summary[
        slope_summary["feature_set"].str.startswith("+")
        & slope_summary["feature_set"].str.endswith("_Policy")
    ].sort_values("future_slope_RMSE").iloc[0]

    def row(
        category: str,
        metric: str,
        baseline_name: str,
        baseline_value: float,
        optimized_name: str,
        optimized_value: float,
        lower_is_better: bool,
    ) -> dict:
        relative = (optimized_value - baseline_value) / abs(baseline_value) * 100.0
        improved = optimized_value < baseline_value if lower_is_better else optimized_value > baseline_value
        return {
            "category": category,
            "metric": metric,
            "baseline_name": baseline_name,
            "baseline_value": float(baseline_value),
            "optimized_name": optimized_name,
            "optimized_value": float(optimized_value),
            "relative_change_percent": float(relative),
            "better_direction": "lower" if lower_is_better else "higher",
            "improved": bool(improved),
        }

    rows = [
        row("EOL结构", "151-200 SOH RMSE", "power", eol.loc["power", "pooled_RMSE"],
            "quadratic", eol.loc["quadratic", "pooled_RMSE"], True),
        row("EOL结构", "EOL更新中位相对差", "power", eol.loc["power", "median_eol_relative_update"],
            "quadratic", eol.loc["quadratic", "median_eol_relative_update"], True),
        row("EOL结构", "EOL更新平均相对差", "power", eol.loc["power", "mean_eol_relative_update"],
            "quadratic", eol.loc["quadratic", "mean_eol_relative_update"], True),
        row("EOL结构", "EOL排序Spearman", "power", eol.loc["power", "eol_spearman_150_200"],
            "quadratic", eol.loc["quadratic", "eol_spearman_150_200"], False),
        row("部分池化", "嵌套LOBO更新中位相对差", "quadratic_individual",
            nested_pool["individual_relative_error"].median(), "quadratic_partial_pool",
            nested_pool["pooled_relative_error"].median(), True),
        row("部分池化", "嵌套LOBO更新平均相对差", "quadratic_individual",
            nested_pool["individual_relative_error"].mean(), "quadratic_partial_pool",
            nested_pool["pooled_relative_error"].mean(), True),
        row("加速度头", "未来斜率RMSE", "直接延续", heads.loc["直接延续", "future_slope_RMSE"],
            "现有SOH趋势_Policy", heads.loc["现有SOH趋势_Policy", "future_slope_RMSE"], True),
        row("新增特征", "同Policy基线上的最佳原始RMSE", "现有SOH趋势_Policy",
            heads.loc["现有SOH趋势_Policy", "future_slope_RMSE"], str(new_policy["feature_set"]),
            float(new_policy["future_slope_RMSE"]), True),
    ]
    return pd.DataFrame(rows)


def write_summary_markdown(output_dir: Path, summary: dict) -> None:
    model_rows = summary["eol_models"]
    pool = summary["partial_pooling"]
    head = summary["acceleration_head"]
    comparisons = {row["metric"]: row for row in summary["accuracy_comparison"]}
    selected = summary["selected_eol_model"]
    lines = [
        "# 寿命预测小优化实验结果",
        "",
        "## 结论",
        "",
        f"- 固定稳定段起点 n0=21 后，一标准误与截断稳定性联合规则选择 `{selected}`。",
        f"- Quadratic相对Power的151--200圈RMSE增加 {comparisons['151-200 SOH RMSE']['relative_change_percent']:.2f}%，但仍处于一标准误近优集合；EOL更新中位相对差降低 {-comparisons['EOL更新中位相对差']['relative_change_percent']:.2f}%。",
        f"- 同策略部分池化的最终个体权重为 {pool['final_individual_weight']:.2f}。嵌套LOBO中位差没有改善（{100*pool['nested_individual_median']:.2f}%→{100*pool['nested_pooled_median']:.2f}%），但平均差降低 {-comparisons['嵌套LOBO更新平均相对差']['relative_change_percent']:.2f}%。",
        f"- 加速度辅助头的一标准误选择为 `{head['selected_feature_set']}`；未来斜率RMSE为 {head['selected_future_slope_RMSE']:.6g}，较直接延续降低 {-comparisons['未来斜率RMSE']['relative_change_percent']:.2f}%。",
        f"- 新增静态量与残差MAD在相同Policy基线上最多只带来 {-comparisons['同Policy基线上的最佳原始RMSE']['relative_change_percent']:.2f}% 的原始RMSE变化，未通过一标准误规则替代较简单的Policy趋势头。",
        "",
        "## 三种EOL结构",
        "",
        "| 模型 | 151--200 RMSE | EOL更新中位相对差 | Spearman | one-SE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['pooled_RMSE']:.8f} | {100*row['median_eol_relative_update']:.2f}% | "
            f"{row['eol_spearman_150_200']:.3f} | {'是' if row['eligible_one_SE'] else '否'} |"
        )
    lines.extend([
        "",
        "## 准确性解释边界",
        "",
        "本实验使用40块完整电池的151--200圈真实SOH评价短期预测，并使用同一电池分别看到150圈和200圈时的EOL更新幅度评价远期结构稳定性。附件没有真实80% EOL标签，因此任何EOL改善只能称为内部稳定性改善，不能称为真实寿命准确率已被验证。",
        "",
        "全部逐电池结果、权重诊断、消融结果和九块测试电池候选寿命均保存在本目录CSV文件中。",
    ])
    (output_dir / "优化结果与准确性判断.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="独立寿命预测小优化实验")
    parser.add_argument(
        "--source-results",
        type=Path,
        default=ROOT / "results" / "重跑_20260816_四份MD同步版_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "寿命预测小优化_20260816_v1",
    )
    args = parser.parse_args()
    source_results = args.source_results if args.source_results.is_absolute() else ROOT / args.source_results
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(source_results / "问题1" / "q1_01_逐循环清洗数据.csv")
    battery_summary = pd.read_csv(ROOT / "battery_summary.csv")
    q3_predictions = pd.read_csv(source_results / "问题3" / "q3_14_九块测试电池151_200正式预测.csv")
    baseline_eol = pd.read_csv(source_results / "问题3" / "q3_15_九块测试电池EOL与统计区间.csv")

    eol_detail, eol_summary_raw = build_eol_model_diagnostics(clean, n0=21)
    selected_model, eol_summary = choose_structure_one_se(eol_summary_raw)
    eol_detail.to_csv(output_dir / "opt_01_四十块电池三结构短期误差与EOL更新.csv", index=False)
    eol_summary.to_csv(output_dir / "opt_02_三结构oneSE与截断稳定性汇总.csv", index=False)

    selected_life = eol_detail[eol_detail["model"].eq(selected_model)][
        ["battery_id", "policy", "life_150", "life_200"]
    ].dropna()
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    fixed_detail, fixed_summary, final_weight = fixed_pooling_weight_diagnostics(selected_life, weights)
    nested_pool, nested_weight_diagnostics = nested_pooling_predictions(selected_life, weights)
    fixed_detail.to_csv(output_dir / "opt_03_同策略固定权重逐电池诊断.csv", index=False)
    fixed_summary.to_csv(output_dir / "opt_04_同策略固定权重汇总.csv", index=False)
    nested_pool.to_csv(output_dir / "opt_05_部分池化嵌套LOBO逐电池.csv", index=False)
    nested_weight_diagnostics.to_csv(output_dir / "opt_06_部分池化嵌套LOBO内层选权.csv", index=False)

    slope_features = build_slope_feature_table(clean, battery_summary)
    slope_features.to_csv(output_dir / "opt_07_新增静态与粗糙度特征.csv", index=False)
    slope_predictions, slope_summary, selected_head, feature_sets = evaluate_acceleration_heads(slope_features)
    slope_predictions.to_csv(output_dir / "opt_08_未来斜率加速度头_LOBO逐电池.csv", index=False)
    slope_summary.to_csv(output_dir / "opt_09_未来斜率加速度头_消融汇总.csv", index=False)

    selected_training_head = slope_predictions[slope_predictions["feature_set"].eq(selected_head)].copy()
    gate_detail, gate_summary = acceleration_gate_audit(eol_detail, selected_training_head)
    gate_detail.to_csv(output_dir / "opt_10_加速度辅助头_EOL结构门控审计.csv", index=False)

    test_acceleration = final_acceleration_predictions(slope_features, selected_head, feature_sets)
    test_acceleration.to_csv(output_dir / "opt_11_九块测试电池未来斜率辅助预测.csv", index=False)
    test_eol = build_test_eol_predictions(
        clean, q3_predictions, baseline_eol, eol_detail, selected_model, final_weight,
        test_acceleration, n0=21,
    )
    test_eol.to_csv(output_dir / "opt_12_九块测试电池优化EOL候选与基线.csv", index=False)

    accuracy_comparison = build_accuracy_comparison(eol_summary, slope_summary, nested_pool)
    accuracy_comparison.to_csv(output_dir / "opt_13_基线与优化关键指标对比.csv", index=False)

    selected_head_row = slope_summary[slope_summary["selected_one_SE"]].iloc[0]
    summary = {
        "source_results": str(source_results.relative_to(ROOT)),
        "n0": 21,
        "selection_rule": "151-200 mean battery MSE one-SE, then nonfinite/boundary/EOL truncation stability",
        "selected_eol_model": selected_model,
        "eol_models": [serializable_record(row) for row in eol_summary.to_dict("records")],
        "partial_pooling": {
            "candidate_weights": weights,
            "final_individual_weight": final_weight,
            "nested_selected_weight_counts": {
                str(weight): int(count)
                for weight, count in nested_pool["selected_weight"].value_counts().sort_index().items()
            },
            "nested_individual_median": float(nested_pool["individual_relative_error"].median()),
            "nested_individual_mean": float(nested_pool["individual_relative_error"].mean()),
            "nested_pooled_median": float(nested_pool["pooled_relative_error"].median()),
            "nested_pooled_mean": float(nested_pool["pooled_relative_error"].mean()),
        },
        "acceleration_head": {
            "selected_feature_set": selected_head,
            "selected_future_slope_RMSE": float(selected_head_row["future_slope_RMSE"]),
            "selected_future_slope_MAE": float(selected_head_row["future_slope_MAE"]),
            "selected_acceleration_sign_accuracy": float(selected_head_row["acceleration_sign_accuracy"]),
            "all_feature_sets": [serializable_record(row) for row in slope_summary.to_dict("records")],
            "eol_gate": gate_summary,
        },
        "accuracy_comparison": [
            serializable_record(row) for row in accuracy_comparison.to_dict("records")
        ],
        "claim_boundary": "No true 80% EOL labels are available; EOL comparisons measure truncation stability, not true lifetime accuracy.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_summary_markdown(output_dir, summary)
    print(json.dumps({
        "output_dir": str(output_dir),
        "selected_eol_model": selected_model,
        "final_individual_weight": final_weight,
        "nested_individual_median": summary["partial_pooling"]["nested_individual_median"],
        "nested_pooled_median": summary["partial_pooling"]["nested_pooled_median"],
        "selected_acceleration_head": selected_head,
        "selected_future_slope_RMSE": summary["acceleration_head"]["selected_future_slope_RMSE"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
