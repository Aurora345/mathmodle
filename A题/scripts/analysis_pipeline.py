from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import savgol_filter
from scipy.stats import kruskal, norm, rankdata, spearmanr, theilslopes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
Q1_RESULTS = RESULTS / "问题1"
Q2_RESULTS = RESULTS / "问题2"
Q3_RESULTS = RESULTS / "问题3"
RNG_SEED = 20260814
Q3_RECENT_WINDOW = 30
STABLE_STARTS = [11, 21, 31, 41, 51]
STATE_TRAJECTORY_FAMILIES = ["PERSIST", "ANCHOR_GLOBAL", "ANCHOR_POLICY"]
STATE_DIAGNOSTIC_FAMILIES = ["FULL_GLOBAL", "FULL_POLICY"]
STATE_FAMILIES = [*STATE_TRAJECTORY_FAMILIES, *STATE_DIAGNOSTIC_FAMILIES]
STATE_CHANNELS = {
    "IR": ("IR_clean", "mean_IR"),
    "Tavg": ("Tavg_clean", "mean_Tavg"),
    "ChargeTime": ("chargetime_clean", "mean_chargetime"),
}


def configure_output_dir(path: Path) -> None:
    global RESULTS, Q1_RESULTS, Q2_RESULTS, Q3_RESULTS
    RESULTS = path
    Q1_RESULTS = RESULTS / "问题1"
    Q2_RESULTS = RESULTS / "问题2"
    Q3_RESULTS = RESULTS / "问题3"
    for directory in [RESULTS, Q1_RESULTS, Q2_RESULTS, Q3_RESULTS]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_sg(values: np.ndarray, window: int = 11, polyorder: int = 2, deriv: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.zeros_like(values) if deriv else values.copy()
    win = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    win = max(win, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    if win > len(values):
        win = len(values) if len(values) % 2 == 1 else len(values) - 1
    if win <= polyorder:
        return np.gradient(values) if deriv == 1 else (np.gradient(np.gradient(values)) if deriv == 2 else values.copy())
    return savgol_filter(values, window_length=win, polyorder=polyorder, deriv=deriv, delta=1.0, mode="interp")


def hampel_candidates(values: np.ndarray, scale_floor: float, half_window: int = 5, k: float = 5.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    flags = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        lo, hi = max(0, i - half_window), min(len(values), i + half_window + 1)
        local = values[lo:hi]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        threshold = max(1.4826 * k * mad, scale_floor)
        flags[i] = abs(values[i] - med) > threshold
    return flags


def isolated_repair(values: np.ndarray, candidates: np.ndarray, neighbor_tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    repaired = values.copy()
    confirmed = np.zeros(len(values), dtype=bool)
    for i in np.where(candidates)[0]:
        if i == 0 or i == len(values) - 1:
            continue
        if abs(values[i - 1] - values[i + 1]) <= neighbor_tolerance:
            repaired[i] = (values[i - 1] + values[i + 1]) / 2
            confirmed[i] = True
    return repaired, confirmed


def preprocess(
    summary: pd.DataFrame,
    cycle: pd.DataFrame,
    k: float = 5.0,
    sg_window: int = 11,
    capacity_floor_fraction: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_idx = summary.set_index("battery_id")
    blocks = []
    audits = []
    for bid, group in cycle.groupby("battery_id", sort=True):
        g = group.sort_values("cycle").copy()
        initial = float(summary_idx.loc[bid, "initial_capacity"])
        g["capacity_clean"] = g["capacity"].astype(float)
        g["IR_clean"] = g["IR"].astype(float)
        g["Tavg_clean"] = g["Tavg"].astype(float)
        g["chargetime_clean"] = g["chargetime"].astype(float)
        for name in ["capacity", "IR", "Tavg", "chargetime"]:
            g[f"{name}_candidate"] = False
            g[f"{name}_repaired"] = False

        # 150/151 边界分段，防止验证段信息进入前150圈处理。
        for mask in [g["cycle"].le(150), g["cycle"].gt(150)]:
            idx = g.index[mask].to_numpy()
            if len(idx) == 0:
                continue

            cap = g.loc[idx, "capacity_clean"].to_numpy()
            cap_cand = hampel_candidates(cap, capacity_floor_fraction * initial, half_window=5, k=k)
            cap_fix, cap_rep = isolated_repair(cap, cap_cand, 0.005 * initial)
            g.loc[idx, "capacity_candidate"] = cap_cand
            g.loc[idx, "capacity_repaired"] = cap_rep
            g.loc[idx, "capacity_clean"] = cap_fix

            ir = g.loc[idx, "IR_clean"].to_numpy()
            ir_invalid = ~np.isfinite(ir) | (ir <= 0)
            ir_cand = hampel_candidates(ir, 0.02 * np.nanmedian(ir), half_window=5, k=k) | ir_invalid
            ir_fix, ir_rep = isolated_repair(ir, ir_cand, 0.02 * np.nanmedian(ir))
            g.loc[idx, "IR_candidate"] = ir_cand
            g.loc[idx, "IR_repaired"] = ir_rep
            g.loc[idx, "IR_clean"] = ir_fix

            temp = g.loc[idx, "Tavg_clean"].to_numpy()
            temp_cand = hampel_candidates(temp, 1.0, half_window=5, k=k)
            temp_fix, temp_rep = isolated_repair(temp, temp_cand, 0.8)
            g.loc[idx, "Tavg_candidate"] = temp_cand
            g.loc[idx, "Tavg_repaired"] = temp_rep
            g.loc[idx, "Tavg_clean"] = temp_fix

            charge = g.loc[idx, "chargetime_clean"].to_numpy()
            charge_invalid = ~np.isfinite(charge) | (charge <= 0)
            charge_cand = hampel_candidates(charge, 0.5, half_window=5, k=k) | charge_invalid
            # 正值候选仅标记；只有物理无效且局部两侧一致时才修复。
            charge_fix, charge_rep = isolated_repair(charge, charge_invalid, 0.5)
            g.loc[idx, "chargetime_candidate"] = charge_cand
            g.loc[idx, "chargetime_repaired"] = charge_rep
            g.loc[idx, "chargetime_clean"] = charge_fix

        g["SOH_clean"] = g["capacity_clean"] / initial
        g["SOH_sg"] = np.nan
        for mask in [g["cycle"].le(150), g["cycle"].gt(150)]:
            idx = g.index[mask]
            if len(idx):
                g.loc[idx, "SOH_sg"] = safe_sg(g.loc[idx, "SOH_clean"].to_numpy(), window=sg_window)

        for name in ["capacity", "IR", "Tavg", "chargetime"]:
            audits.append({
                "battery_id": bid,
                "variable": name,
                "candidate_count": int(g[f"{name}_candidate"].sum()),
                "repaired_count": int(g[f"{name}_repaired"].sum()),
            })
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(audits)


def ts_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(theilslopes(np.asarray(y, float), np.asarray(x, float)).slope)


def q1_features(summary: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bid, g in clean.groupby("battery_id", sort=True):
        e = g[g["cycle"].le(150)].sort_values("cycle")
        x = e["cycle"].to_numpy(float)
        y = e["SOH_sg"].to_numpy(float)
        ir_rel = e["IR_clean"].to_numpy(float) / np.median(e["IR_clean"].iloc[:10])
        rows.append({
            "battery_id": bid,
            "charge_time_150": float(np.median(e["chargetime_clean"])),
            "SOH150": float(np.median(e.loc[e["cycle"].between(141, 150), "SOH_sg"])),
            "SOH_global_slope": ts_slope(x, y),
            "SOH_AUC": float(trapezoid(y, x) / (x[-1] - x[0])),
            "IR_rel_slope": ts_slope(x, ir_rel),
            "IR_rel_150": float(np.median(ir_rel[-10:])),
            "Tavg_150": float(np.median(e["Tavg_clean"])),
            "Tavg_slope": ts_slope(x, e["Tavg_clean"].to_numpy(float)),
            "charge_slope": ts_slope(x, e["chargetime_clean"].to_numpy(float)),
        })
    return summary.merge(pd.DataFrame(rows), on="battery_id", how="left", validate="one_to_one")


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept = np.polyfit(x, y, 1)
    life = (0.8 - intercept) / slope if slope < 0 else np.nan
    valid_life = np.isfinite(life) and life > max(x)
    return {"a": float(intercept), "b": float(slope), "c": 1.0, "life": float(life) if valid_life else np.nan, "success": True}


def fit_power(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def predict(z: np.ndarray, xx: np.ndarray) -> np.ndarray:
        return z[0] - np.exp(z[1]) * np.power(xx, np.exp(z[2]))

    best = None
    for c0 in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0]:
        a0 = max(float(y[0]), 1.0)
        b0 = max((a0 - float(y[-1])) / (float(x[-1]) ** c0), 1e-8)
        res = least_squares(
            lambda z: predict(z, x) - y,
            x0=[a0, math.log(b0), math.log(c0)],
            bounds=([0.8, math.log(1e-12), math.log(0.1)], [1.2, math.log(0.1), math.log(5.0)]),
            max_nfev=20000,
        )
        mse = float(np.mean(res.fun**2))
        if best is None or mse < best[0]:
            best = (mse, res)
    assert best is not None
    res = best[1]
    a, b, c = float(res.x[0]), float(np.exp(res.x[1])), float(np.exp(res.x[2]))
    life = ((a - 0.8) / b) ** (1 / c) if a > 0.8 and b > 0 else np.nan
    valid_life = np.isfinite(life) and life > max(x)
    return {
        "a": a,
        "b": b,
        "c": c,
        "life": float(life) if valid_life else np.nan,
        "success": bool(res.success),
        "b_at_lower_bound": b <= 1.02e-12,
        "c_near_bound": c <= 0.105 or c >= 4.95,
    }


def fit_power_warm(x: np.ndarray, y: np.ndarray, start: dict) -> dict:
    """Bootstrap专用单初值幂律拟合；只优化实现，不改变模型、边界或目标函数。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def predict(z: np.ndarray) -> np.ndarray:
        return z[0] - np.exp(z[1]) * np.power(x, np.exp(z[2]))

    z0 = np.array([
        float(start["a"]),
        math.log(max(float(start["b"]), 1e-12)),
        math.log(min(max(float(start["c"]), 0.1), 5.0)),
    ])
    result = least_squares(
        lambda z: predict(z) - y,
        x0=z0,
        bounds=([0.8, math.log(1e-12), math.log(0.1)], [1.2, math.log(0.1), math.log(5.0)]),
        max_nfev=1000,
    )
    a, b, c = float(result.x[0]), float(np.exp(result.x[1])), float(np.exp(result.x[2]))
    life = ((a - 0.8) / b) ** (1 / c) if a > 0.8 and b > 0 else np.nan
    valid_life = np.isfinite(life) and life > max(x)
    return {
        "a": a, "b": b, "c": c,
        "life": float(life) if valid_life else np.nan,
        "success": bool(result.success),
        "b_at_lower_bound": b <= 1.02e-12,
        "c_near_bound": c <= 0.105 or c >= 4.95,
    }


def fit_centered_monotone_quadratic(x: np.ndarray, y: np.ndarray, n0: int) -> dict:
    """拟合 A-B(n-n0)-C(n-n0)^2，并只约束稳定段 n>=n0 单调下降。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    shifted = x - float(n0)
    design = np.column_stack([np.ones(len(x)), -shifted, -(shifted**2)])
    result = lsq_linear(
        design,
        y,
        bounds=(np.array([0.8, 0.0, 0.0]), np.array([1.2, np.inf, np.inf])),
        lsmr_tol="auto",
    )
    a, b, c = (float(value) for value in result.x)
    if c > 0 and a > 0.8:
        discriminant = b * b + 4.0 * c * (a - 0.8)
        shifted_life = (-b + math.sqrt(discriminant)) / (2.0 * c)
    elif b > 0 and a > 0.8:
        shifted_life = (a - 0.8) / b
    else:
        shifted_life = np.nan
    life = float(n0) + shifted_life
    valid_life = np.isfinite(life) and life > max(x)
    return {
        "a": a,
        "b": b,
        "c": c,
        "n0": int(n0),
        "life": float(life) if valid_life else np.nan,
        "success": bool(result.success),
        "b_at_lower_bound": bool(b <= 1e-12),
        "c_near_bound": bool(c <= 1e-14),
    }


def fit_eol_candidate(model: str, x: np.ndarray, y: np.ndarray, n0: int) -> dict:
    if model == "linear":
        result = fit_linear(x, y)
    elif model == "power":
        result = fit_power(x, y)
    elif model == "centered_quadratic":
        result = fit_centered_monotone_quadratic(x, y, n0)
    else:
        raise ValueError(f"Unknown EOL model: {model}")
    result["n0"] = int(n0)
    result["boundary"] = bool(
        result.get("b_at_lower_bound", False) or result.get("c_near_bound", False)
    )
    return result


def model_predict(model: str, params: dict, x: np.ndarray) -> np.ndarray:
    if model == "linear":
        return params["a"] + params["b"] * np.asarray(x, float)
    if model == "power":
        return params["a"] - params["b"] * np.asarray(x, float) ** params["c"]
    if model == "centered_quadratic":
        shifted = np.asarray(x, float) - float(params["n0"])
        return params["a"] - params["b"] * shifted - params["c"] * shifted**2
    raise ValueError(f"Unknown EOL model: {model}")


def summarize_q1_candidates(validation_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n0, model), group in validation_rows.groupby(["n0", "model"], sort=True):
        battery_mse = group["MSE"].to_numpy(float)
        finite_life = group["life"].replace([np.inf, -np.inf], np.nan).dropna()
        valid_pair = group.dropna(subset=["life", "life_200", "eol_relative_update"])
        rows.append({
            "n0": int(n0),
            "model": model,
            "mean_battery_MSE": float(np.mean(battery_mse)),
            "SE_battery_MSE": float(np.std(battery_mse, ddof=1) / np.sqrt(len(battery_mse))),
            "pooled_MAE": float(group["MAE"].mean()),
            "pooled_RMSE": float(np.sqrt(group["MSE"].mean())),
            "MAE_pp": float(100 * group["MAE"].mean()),
            "RMSE_pp": float(100 * np.sqrt(group["MSE"].mean())),
            "median_battery_MAE": float(group["MAE"].median()),
            "median_E200": float(group["E200"].median()),
            "failure_count": int((~group["success"].astype(bool)).sum()),
            "nonfinite_eol_count": int(group["life"].isna().sum()),
            "b_lower_bound_count": int(group["b_at_lower_bound"].astype(bool).sum()),
            "c_near_bound_count": int(group["c_near_bound"].astype(bool).sum()),
            "boundary_count": int((group["fit150_boundary"] | group["fit200_boundary"]).sum()),
            "median_eol_relative_update": float(valid_pair["eol_relative_update"].median()),
            "mean_eol_relative_update": float(valid_pair["eol_relative_update"].mean()),
            "eol_spearman_150_200": float(spearmanr(valid_pair["life"], valid_pair["life_200"]).statistic),
            "life_median": float(finite_life.median()) if len(finite_life) else np.nan,
            "life_log_SD": float(np.log(finite_life).std(ddof=1)) if len(finite_life) > 1 else np.nan,
            "c_median": float(group["c"].median()),
            "c_IQR": float(group["c"].quantile(0.75) - group["c"].quantile(0.25)),
        })
    return pd.DataFrame(rows)


def select_q1_candidate(validation_rows: pd.DataFrame) -> tuple[int, str, pd.DataFrame]:
    """短期MSE one-SE近优后，按无效EOL、截断稳定性、边界及简约性裁决。"""
    table = summarize_q1_candidates(validation_rows)
    best = table.sort_values(["mean_battery_MSE", "n0", "model"]).iloc[0]
    threshold = float(best["mean_battery_MSE"] + best["SE_battery_MSE"])
    table["best_candidate"] = (table["n0"].eq(best["n0"]) & table["model"].eq(best["model"]))
    table["one_SE_threshold"] = threshold
    table["eligible_one_SE"] = table["mean_battery_MSE"].le(threshold + 1e-18)
    eligible = table[table["eligible_one_SE"]].copy()
    eligible["complexity"] = eligible["model"].map({
        "linear": 0, "power": 1, "centered_quadratic": 1,
    })
    chosen = eligible.sort_values([
        "nonfinite_eol_count", "median_eol_relative_update", "boundary_count", "complexity", "n0", "model"
    ]).iloc[0]
    table["selected_one_SE"] = table["n0"].eq(chosen["n0"]) & table["model"].eq(chosen["model"])
    return int(chosen["n0"]), str(chosen["model"]), table


def q1_model_analysis(
    clean: pd.DataFrame,
    battery: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, str, pd.DataFrame, pd.DataFrame]:
    val_rows, pred_rows, param_rows = [], [], []
    for bid, group in clean.groupby("battery_id", sort=True):
        group = group.sort_values("cycle")
        test = group[group["cycle"].between(151, 200)]
        for n0 in STABLE_STARTS:
            train = group[group["cycle"].between(n0, 150)]
            xtr, ytr = train["cycle"].to_numpy(float), train["SOH_sg"].to_numpy(float)
            stable_slope = ts_slope(xtr, ytr)
            for model in ["linear", "power", "centered_quadratic"]:
                params = fit_eol_candidate(model, xtr, ytr, n0)
                common = {
                    "battery_id": int(bid), "n0": n0, "fit_end": 150, "model": model,
                    "stable_slope": stable_slope, **params,
                    "b_at_lower_bound": bool(params.get("b_at_lower_bound", False)),
                    "c_near_bound": bool(params.get("c_near_bound", False)),
                }
                param_rows.append(common)
                if len(test):
                    xte, yte = test["cycle"].to_numpy(float), test["SOH_sg"].to_numpy(float)
                    pred = model_predict(model, params, xte)
                    fit200 = group[group["cycle"].between(n0, 200)].sort_values("cycle")
                    params200 = fit_eol_candidate(
                        model,
                        fit200["cycle"].to_numpy(float),
                        fit200["SOH_sg"].to_numpy(float),
                        n0,
                    )
                    valid_pair = (
                        np.isfinite(params["life"])
                        and np.isfinite(params200["life"])
                        and params200["life"] > 0
                    )
                    mse = float(np.mean((pred - yte) ** 2))
                    val_rows.append({
                        **common,
                        "life_200": params200["life"],
                        "eol_relative_update": (
                            abs(params["life"] - params200["life"]) / params200["life"]
                            if valid_pair else np.nan
                        ),
                        "fit150_boundary": bool(params["boundary"]),
                        "fit200_boundary": bool(params200["boundary"]),
                        "MAE": float(mean_absolute_error(yte, pred)),
                        "MSE": mse,
                        "RMSE": float(np.sqrt(mse)),
                        "E200": abs(float(yte[-1] - pred[-1])),
                    })
                    pred_rows.extend({
                        "battery_id": int(bid), "n0": n0, "model": model,
                        "cycle": int(cycle), "observed": float(observed), "predicted": float(predicted),
                    } for cycle, observed, predicted in zip(xte, yte, pred))

    validation = pd.DataFrame(val_rows)
    selected_n0, selected_model, validation_summary = select_q1_candidate(validation)

    nested_rows = []
    for outer_bid in sorted(validation["battery_id"].unique()):
        inner = validation[~validation["battery_id"].eq(outer_bid)]
        fold_n0, fold_model, fold_table = select_q1_candidate(inner)
        outer = validation[
            validation["battery_id"].eq(outer_bid)
            & validation["n0"].eq(fold_n0)
            & validation["model"].eq(fold_model)
        ].iloc[0]
        best_inner = fold_table[fold_table["best_candidate"]].iloc[0]
        nested_rows.append({
            "held_out_battery": int(outer_bid),
            "selected_n0": fold_n0,
            "selected_model": fold_model,
            "inner_best_n0": int(best_inner["n0"]),
            "inner_best_model": str(best_inner["model"]),
            "inner_one_SE_threshold": float(best_inner["one_SE_threshold"]),
            "MAE": float(outer["MAE"]),
            "MSE": float(outer["MSE"]),
            "RMSE": float(outer["RMSE"]),
            "MAE_pp": float(100 * outer["MAE"]),
            "RMSE_pp": float(100 * outer["RMSE"]),
            "E200": float(outer["E200"]),
            "E200_pp": float(100 * outer["E200"]),
        })
    nested = pd.DataFrame(nested_rows)

    parameters = pd.DataFrame(param_rows)
    chosen = parameters[
        parameters["n0"].eq(selected_n0) & parameters["model"].eq(selected_model)
    ].copy().rename(columns={
        "life": "life_150", "a": "model_a_150", "b": "model_b_150", "c": "model_c_150",
        "stable_slope": "SOH_stable_slope",
    })
    keep = [
        "battery_id", "life_150", "model_a_150", "model_b_150", "model_c_150",
        "SOH_stable_slope", "b_at_lower_bound", "c_near_bound",
    ]
    battery = battery.merge(chosen[keep], on="battery_id", how="left")
    battery["SOH_slope"] = battery["SOH_stable_slope"]
    battery["stable_rate"] = -battery["SOH_stable_slope"]
    battery["selected_n0"] = selected_n0
    battery["selected_model"] = selected_model

    life200_rows = []
    for bid, group in clean.groupby("battery_id", sort=True):
        if group["cycle"].max() < 200:
            continue
        fit = group[group["cycle"].between(selected_n0, 200)].sort_values("cycle")
        x, y = fit["cycle"].to_numpy(float), fit["SOH_sg"].to_numpy(float)
        params = fit_eol_candidate(selected_model, x, y, selected_n0)
        life200_rows.append({"battery_id": int(bid), "life_200_ref": params["life"]})
    battery = battery.merge(pd.DataFrame(life200_rows), on="battery_id", how="left")
    battery["life_abs_diff_150_200"] = (battery["life_150"] - battery["life_200_ref"]).abs()
    battery["life_rel_diff_150_200"] = battery["life_abs_diff_150_200"] / battery["life_200_ref"]

    sensitivity_rows = []
    for bid, group in parameters.groupby("battery_id"):
        finite = group["life"].replace([np.inf, -np.inf], np.nan).dropna()
        sensitivity_rows.append({
            "battery_id": int(bid),
            "candidate_count": len(group),
            "finite_eol_count": len(finite),
            "structural_life_min": float(finite.min()) if len(finite) else np.nan,
            "structural_life_median": float(finite.median()) if len(finite) else np.nan,
            "structural_life_max": float(finite.max()) if len(finite) else np.nan,
            "structural_relative_span": float((finite.max() - finite.min()) / finite.median()) if len(finite) else np.nan,
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    battery = battery.merge(sensitivity, on="battery_id", how="left")
    return battery, validation_summary, pd.DataFrame(pred_rows), selected_n0, selected_model, nested, parameters


def preprocessing_sensitivity(
    summary: pd.DataFrame,
    cycle: pd.DataFrame,
    reference_policy_stats: pd.DataFrame,
    selected_n0: int,
    selected_model: str,
) -> pd.DataFrame:
    reference = reference_policy_stats.set_index("policy")
    rows = []
    for k, window, floor in itertools.product([4.0, 5.0, 6.0], [7, 11, 15], [0.005, 0.01, 0.02]):
        cleaned, audit = preprocess(
            summary, cycle, k=k, sg_window=window, capacity_floor_fraction=floor
        )
        battery_rows = []
        for battery_id, group in cleaned.groupby("battery_id"):
            stable = group[group["cycle"].between(selected_n0, 150)].sort_values("cycle")
            x = stable["cycle"].to_numpy(float)
            y = stable["SOH_sg"].to_numpy(float)
            fit = fit_eol_candidate(selected_model, x, y, selected_n0)
            battery_rows.append({
                "battery_id": battery_id,
                "policy": group["policy"].iloc[0],
                "life": fit["life"],
                "stable_rate": -ts_slope(x, y),
            })
        policy = pd.DataFrame(battery_rows).groupby("policy", as_index=False).agg(
            life_median=("life", "median"), stable_rate_median=("stable_rate", "median")
        ).set_index("policy")
        aligned = reference[["life_median", "stable_rate_median"]].join(
            policy, lsuffix="_ref", rsuffix="_setting", how="inner"
        )
        rows.append({
            "hampel_k": k,
            "sg_window": window,
            "capacity_floor_fraction": floor,
            "capacity_candidates": int(audit.loc[audit["variable"].eq("capacity"), "candidate_count"].sum()),
            "capacity_repaired": int(audit.loc[audit["variable"].eq("capacity"), "repaired_count"].sum()),
            "life_rank_spearman_vs_main": float(spearmanr(
                aligned["life_median_ref"], aligned["life_median_setting"]
            ).statistic),
            "stable_rate_rank_spearman_vs_main": float(spearmanr(
                aligned["stable_rate_median_ref"], aligned["stable_rate_median_setting"]
            ).statistic),
            "long_policy": str(policy["life_median"].idxmax()),
            "short_policy": str(policy["life_median"].idxmin()),
        })
    return pd.DataFrame(rows)


def policy_statistics(battery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, g in battery.groupby("policy", sort=False):
        life = g["life_150"].dropna()
        rows.append({
            "policy": policy,
            "N": len(g),
            "life_median": life.median(),
            "life_q1": life.quantile(0.25),
            "life_q3": life.quantile(0.75),
            "life_iqr": life.quantile(0.75) - life.quantile(0.25),
            "life_min": life.min(),
            "life_max": life.max(),
            "SOH150_median": g["SOH150"].median(),
            "SOH_slope_median": g["SOH_slope"].median(),
            "stable_rate_median": g["stable_rate"].median(),
            "SOH_AUC_median": g["SOH_AUC"].median(),
            "charge_time_median": g["charge_time_150"].median(),
            "IR_slope_median": g["IR_rel_slope"].median(),
            "Tavg_median": g["Tavg_150"].median(),
            "C1": g["C1"].iloc[0], "Q1": g["Q1"].iloc[0], "C2": g["C2"].iloc[0],
            "dataset_id": g["dataset_id"].iloc[0],
        })
    return pd.DataFrame(rows).sort_values("life_median", ascending=False).reset_index(drop=True)


def latex_policy(policy: str) -> str:
    labels = {
        "3_6C-80PER_3_6C": r"3.6C--80\%--3.6C",
        "80PER_3_6C": r"80\%--3.6C",
        "4_8C_80PER_4_8C": r"4.8C--80\%--4.8C",
        "5C_67PER_4C_NEWSTRUCTURE": r"5.0C--67\%--4.0C（新结构）",
        "5_3C_54PER_4C_NEWSTRUCTURE": r"5.3C--54\%--4.0C（新结构）",
        "5_6C_19PER_4_6C_NEWSTRUCTURE": r"5.6C--19\%--4.6C（新结构）",
        "3_7C_31PER_5_9C_NEWSTRUCTURE": r"3.7C--31\%--5.9C（新结构）",
        "5_6C_36PER_4_3C_NEWSTRUCTURE": r"5.6C--36\%--4.3C（新结构）",
        "4_8C_80PER_4_8C_NEWSTRUCTURE": r"4.8C--80\%--4.8C（新结构）",
    }
    return labels.get(str(policy), str(policy).replace("_", r"\_"))


def write_q1_battery_table(battery: pd.DataFrame) -> None:
    """生成附录用49块电池关键指标表，保证正文展示与CSV同源。"""
    lines = [
        r"\begin{longtable}{rrlrrrr}",
        r"\caption{49块电池的策略、充电时间、SOH与基准估计寿命}\label{tab:q1battery}\\",
        r"\toprule",
        r"电池 & 数据集 & 快充策略 & $T_{1:150}$/min & $S_{150}$ & $10^4k^{\TS}$ & $L$/圈 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{c}{表\thetable\ （续）}\\",
        r"\toprule",
        r"电池 & 数据集 & 快充策略 & $T_{1:150}$/min & $S_{150}$ & $10^4k^{\TS}$ & $L$/圈 \\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{7}{r}{续下页}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    for _, row in battery.sort_values("battery_id").iterrows():
        lines.append(
            f"{int(row.battery_id)} & {int(row.dataset_id)} & {latex_policy(row.policy)} & "
            f"{row.charge_time_150:.3f} & {row.SOH150:.5f} & {1e4 * row.SOH_slope:.3f} & {row.life_150:.0f} \\\\"
        )
    lines.append(r"\end{longtable}")
    (Q1_RESULTS / "q1_battery_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def q1_typical_comparison(policy_stats: pd.DataFrame) -> pd.DataFrame:
    selected = policy_stats.iloc[[0, -1]].copy()
    selected.insert(0, "category", ["EOL_long_candidate", "EOL_short_candidate"])
    return selected[[
        "category", "policy", "dataset_id", "C1", "Q1", "C2", "N",
        "charge_time_median", "SOH150_median", "SOH_slope_median",
        "stable_rate_median", "SOH_AUC_median", "IR_slope_median", "Tavg_median", "life_median",
        "life_q1", "life_q3",
    ]]


def q1_typical_consistency_audit(policy_stats: pd.DataFrame) -> pd.DataFrame:
    """不造综合分数，只并列呈现EOL排序与三个早期退化指标是否同向。"""
    frame = policy_stats.copy()
    frame["life_rank_desc"] = frame["life_median"].rank(ascending=False, method="min").astype(int)
    frame["stable_rate_rank_asc"] = frame["stable_rate_median"].rank(ascending=True, method="min").astype(int)
    frame["SOH150_rank_desc"] = frame["SOH150_median"].rank(ascending=False, method="min").astype(int)
    frame["AUC_rank_desc"] = frame["SOH_AUC_median"].rank(ascending=False, method="min").astype(int)
    life_long = frame["life_median"].idxmax()
    life_short = frame["life_median"].idxmin()
    frame["EOL_extreme_role"] = ""
    frame.loc[life_long, "EOL_extreme_role"] = "EOL_long_candidate"
    frame.loc[life_short, "EOL_extreme_role"] = "EOL_short_candidate"
    med_rate = frame["stable_rate_median"].median()
    med_soh = frame["SOH150_median"].median()
    med_auc = frame["SOH_AUC_median"].median()
    frame["early_metrics_direction_consistent"] = pd.Series(
        [pd.NA] * len(frame), index=frame.index, dtype="boolean"
    )
    frame.loc[life_long, "early_metrics_direction_consistent"] = bool(
        frame.loc[life_long, "stable_rate_median"] <= med_rate
        and frame.loc[life_long, "SOH150_median"] >= med_soh
        and frame.loc[life_long, "SOH_AUC_median"] >= med_auc
    )
    frame.loc[life_short, "early_metrics_direction_consistent"] = bool(
        frame.loc[life_short, "stable_rate_median"] >= med_rate
        and frame.loc[life_short, "SOH150_median"] <= med_soh
        and frame.loc[life_short, "SOH_AUC_median"] <= med_auc
    )
    return frame[[
        "policy", "EOL_extreme_role", "life_median", "life_rank_desc",
        "stable_rate_median", "stable_rate_rank_asc", "SOH150_median", "SOH150_rank_desc",
        "SOH_AUC_median", "AUC_rank_desc", "early_metrics_direction_consistent",
    ]].sort_values("life_rank_desc")


def dunn_holm(battery: pd.DataFrame, response: str) -> pd.DataFrame:
    y = battery[response].to_numpy(float)
    groups = battery["policy"].astype(str).to_numpy()
    ranks = rankdata(y, method="average")
    n = len(y)
    _, tie_counts = np.unique(y, return_counts=True)
    s2 = n * (n + 1) / 12 - np.sum(tie_counts**3 - tie_counts) / (12 * (n - 1))
    rows = []
    policies = list(pd.unique(groups))
    for a, b in itertools.combinations(policies, 2):
        ia, ib = groups == a, groups == b
        z = (ranks[ia].mean() - ranks[ib].mean()) / math.sqrt(s2 * (1 / ia.sum() + 1 / ib.sum()))
        p = 2 * norm.sf(abs(z))
        rows.append({"policy_a": a, "policy_b": b, "z": z, "p_raw": p})
    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(out)
    adjusted = np.maximum.accumulate([(m - i) * p for i, p in enumerate(out["p_raw"])])
    out["p_holm"] = np.minimum(adjusted, 1.0)
    out["significant_0_05"] = out["p_holm"] < 0.05
    out.insert(0, "response", response)
    return out.sort_values(["policy_a", "policy_b"]).reset_index(drop=True)


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        xcols = [c for c in cols if c != col]
        model = LinearRegression().fit(df[xcols], df[col])
        r2 = model.score(df[xcols], df[col])
        rows.append({"variable": col, "R2_aux": r2, "VIF": np.inf if r2 >= 1 else 1 / (1 - r2)})
    return pd.DataFrame(rows)


def exposure(c1: float, q1: float, c2: float, s0: float = 0.5) -> tuple[float, float, float, float, float]:
    q = q1 / 100.0
    theoretical_time = 60 * (q / c1 + (0.8 - q) / c2)
    avg = (c1 * q + c2 * (0.8 - q)) / 0.8
    low_len = min(q, s0)
    e_low = (c1 * low_len + c2 * (s0 - low_len)) / s0
    e_high = (c1 * max(q - s0, 0) + c2 * (0.8 - max(q, s0))) / (0.8 - s0)
    return theoretical_time, avg, e_low, e_high, e_high - e_low


def standardized_regression(df: pd.DataFrame, response: str, predictors: list[str]) -> dict:
    scaler = StandardScaler()
    x = scaler.fit_transform(df[predictors])
    y = df[response].to_numpy(float)
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)
    r2 = float(r2_score(y, pred))
    n, p = len(y), x.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return {
        "intercept": float(model.intercept_),
        "coef": model.coef_.astype(float),
        "r2": r2,
        "adjusted_r2": float(adjusted_r2),
        "resid": y - pred,
        "x": x,
        "y": y,
    }


def exact_r2_permutation_p(x: np.ndarray, y: np.ndarray, observed_r2: float) -> tuple[float, int, int]:
    r2_perm = np.array([
        LinearRegression().fit(x, np.asarray(perm)).score(x, np.asarray(perm))
        for perm in itertools.permutations(y)
    ])
    exceedances = int(np.sum(r2_perm >= observed_r2 - 1e-12))
    return exceedances / len(r2_perm), exceedances, len(r2_perm)


def leave_one_policy_out(df: pd.DataFrame, response: str, predictors: list[str]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for i in range(len(df)):
        train = np.arange(len(df)) != i
        scaler = StandardScaler().fit(df.loc[train, predictors])
        x_train = scaler.transform(df.loc[train, predictors])
        x_test = scaler.transform(df.loc[[i], predictors])
        y = df[response].to_numpy(float)
        model = LinearRegression().fit(x_train, y[train])
        predicted = float(model.predict(x_test)[0])
        rows.append({
            "held_out_policy": df.loc[i, "policy"],
            "observed": y[i],
            "predicted": predicted,
            "error": float(predicted - y[i]),
            "predictor_1": predictors[0],
            "predictor_2": predictors[1],
            "coef_1": float(model.coef_[0]),
            "coef_2": float(model.coef_[1]),
            "train_R2": float(model.score(x_train, y[train])),
        })
    out = pd.DataFrame(rows)
    summary = {
        "response": response,
        "N_policies": len(out),
        "MAE": float(mean_absolute_error(out["observed"], out["predicted"])),
        "RMSE": float(mean_squared_error(out["observed"], out["predicted"]) ** 0.5),
        "spearman": float(spearmanr(out["observed"], out["predicted"]).statistic),
        "predictor_1": predictors[0],
        "predictor_2": predictors[1],
        "coef_1_positive_folds": int((out["coef_1"] > 0).sum()),
        "coef_1_negative_folds": int((out["coef_1"] < 0).sum()),
        "coef_2_positive_folds": int((out["coef_2"] > 0).sum()),
        "coef_2_negative_folds": int((out["coef_2"] < 0).sum()),
        "coef_1_min": float(out["coef_1"].min()),
        "coef_1_max": float(out["coef_1"].max()),
        "coef_2_min": float(out["coef_2"].min()),
        "coef_2_max": float(out["coef_2"].max()),
    }
    return out, summary


def kruskal_strategy_analysis(
    battery: pd.DataFrame,
    response: str,
    rng: np.random.Generator,
    n_permutations: int = 100_000,
) -> dict:
    groups = [group[response].dropna().to_numpy(float) for _, group in battery.groupby("policy")]
    statistic = kruskal(*groups)
    frame = battery.dropna(subset=[response]).copy()
    y = frame[response].to_numpy(float)
    labels = frame["policy"].to_numpy()
    policies = pd.unique(labels)
    h_perm = np.empty(n_permutations)
    for index in range(len(h_perm)):
        permuted = rng.permutation(labels)
        h_perm[index] = kruskal(*[y[permuted == policy] for policy in policies]).statistic
    exceedances = int(np.sum(h_perm >= statistic.statistic - 1e-12))
    permutation_p = (1 + exceedances) / (len(h_perm) + 1)
    eta_h2 = (statistic.statistic - len(groups) + 1) / (len(y) - len(groups))
    return {
        "response": response,
        "H": float(statistic.statistic),
        "p_asymptotic": float(statistic.pvalue),
        "p_permutation": float(permutation_p),
        "p_permutation_percent": float(100 * permutation_p),
        "permutation_exceedances": exceedances,
        "permutation_draws": len(h_perm),
        "eta_H2": float(eta_h2),
        "N": len(y),
        "G": len(groups),
    }


def choose_eol_response(frame: pd.DataFrame) -> tuple[str, str]:
    raw_skew = abs(float(frame["life_median"].skew()))
    log_skew = abs(float(np.log(frame["life_median"]).skew()))
    if np.isfinite(log_skew) and log_skew < raw_skew:
        frame["eol_response"] = np.log(frame["life_median"])
        return "eol_response", "log"
    frame["eol_response"] = frame["life_median"]
    return "eol_response", "raw"


def q2_analysis(battery: pd.DataFrame, policy_stats: pd.DataFrame) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    kw_rows = [
        kruskal_strategy_analysis(battery, "stable_rate", rng),
        kruskal_strategy_analysis(battery, "life_150", rng),
    ]
    kw_table = pd.DataFrame(kw_rows)
    dunn_parts = []
    for row in kw_rows:
        if row["p_permutation"] < 0.05:
            dunn_parts.append(dunn_holm(battery.dropna(subset=[row["response"]]), row["response"]))
    dunn = pd.concat(dunn_parts, ignore_index=True) if dunn_parts else pd.DataFrame(columns=[
        "response", "policy_a", "policy_b", "z", "p_raw", "p_holm", "significant_0_05"
    ])

    corr_rows = []
    complete = battery.dropna(subset=["C1", "Q1", "C2", "stable_rate", "life_150"])
    policy_complete = complete.groupby("policy", as_index=False).agg(
        C1=("C1", "first"), Q1=("Q1", "first"), C2=("C2", "first"),
        stable_rate=("stable_rate", "median"), life_150=("life_150", "median"),
    )
    for level, frame in [("battery", complete), ("policy", policy_complete)]:
        for predictor in ["C1", "Q1", "C2"]:
            for response in ["stable_rate", "life_150"]:
                rho, p_value = spearmanr(frame[predictor], frame[response])
                corr_rows.append({
                    "level": level, "predictor": predictor, "response": response,
                    "rho": float(rho), "p": float(p_value), "N": len(frame),
                })

    complete_policy = policy_stats.dropna(subset=["C1", "Q1", "C2"]).copy().reset_index(drop=True)
    exp_values = complete_policy.apply(
        lambda row: exposure(row.C1, row.Q1, row.C2, 0.5), axis=1, result_type="expand"
    )
    exp_values.columns = ["T_theoretical_0_80", "A", "E_L_50", "E_H_50", "D_50"]
    complete_policy = pd.concat([complete_policy, exp_values], axis=1)
    new = complete_policy[complete_policy["dataset_id"].eq(3)].copy().reset_index(drop=True)
    new["rate_response"] = new["stable_rate_median"]
    eol_response, eol_scale = choose_eol_response(new)

    vif_raw = vif_table(new, ["C1", "Q1", "C2"])
    vif_raw.insert(0, "parameterization", "raw_C1_Q1_C2")
    vif_ad = vif_table(new, ["A", "D_50"])
    vif_ad.insert(0, "parameterization", "reparameterized_A_D")
    vif = pd.concat([vif_raw, vif_ad], ignore_index=True)

    rate_model = standardized_regression(new, "rate_response", ["A", "D_50"])
    eol_model = standardized_regression(new, eol_response, ["A", "D_50"])
    rate_perm = exact_r2_permutation_p(rate_model["x"], rate_model["y"], rate_model["r2"])
    eol_perm = exact_r2_permutation_p(eol_model["x"], eol_model["y"], eol_model["r2"])

    rate_lopo, rate_lopo_summary = leave_one_policy_out(new, "rate_response", ["A", "D_50"])
    eol_lopo, eol_lopo_summary = leave_one_policy_out(new, eol_response, ["A", "D_50"])
    rate_lopo.insert(0, "response", "stable_rate")
    eol_lopo.insert(0, "response", f"EOL_{eol_scale}")
    lopo = pd.concat([rate_lopo, eol_lopo], ignore_index=True)
    lopo_summary_table = pd.DataFrame([
        {"response": "stable_rate", **rate_lopo_summary},
        {"response": f"EOL_{eol_scale}", **eol_lopo_summary},
    ])

    battery_new = battery[battery["dataset_id"].eq(3)].copy()
    policy_ad = new.set_index("policy")[["A", "D_50"]]
    boot_rate = np.empty((5000, 2))
    boot_eol = np.empty((5000, 2))
    for draw in range(5000):
        rows = []
        for policy, group in battery_new.groupby("policy"):
            sample = group.iloc[rng.integers(0, len(group), len(group))]
            rows.append({
                "policy": policy,
                "rate_response": float(np.median(sample["stable_rate"])),
                "life_median": float(np.median(sample["life_150"])),
            })
        temp = pd.DataFrame(rows).set_index("policy").join(policy_ad).reset_index()
        temp["eol_response"] = np.log(temp["life_median"]) if eol_scale == "log" else temp["life_median"]
        boot_rate[draw] = standardized_regression(temp, "rate_response", ["A", "D_50"])["coef"]
        boot_eol[draw] = standardized_regression(temp, "eol_response", ["A", "D_50"])["coef"]
    rate_ci = np.quantile(boot_rate, [0.025, 0.5, 0.975], axis=0)
    eol_ci = np.quantile(boot_eol, [0.025, 0.5, 0.975], axis=0)

    coefficient_table = pd.DataFrame([
        {
            "response": "stable_rate", "response_scale": "raw", "intercept": rate_model["intercept"],
            "beta_A": rate_model["coef"][0], "beta_D": rate_model["coef"][1],
            "R2": rate_model["r2"], "adjusted_R2": rate_model["adjusted_r2"],
            "permutation_p": rate_perm[0], "permutation_p_percent": 100 * rate_perm[0],
            "permutation_exceedances": rate_perm[1], "permutation_total": rate_perm[2],
            "beta_A_ci_low": rate_ci[0, 0], "beta_A_ci_median": rate_ci[1, 0], "beta_A_ci_high": rate_ci[2, 0],
            "beta_D_ci_low": rate_ci[0, 1], "beta_D_ci_median": rate_ci[1, 1], "beta_D_ci_high": rate_ci[2, 1],
        },
        {
            "response": "EOL", "response_scale": eol_scale, "intercept": eol_model["intercept"],
            "beta_A": eol_model["coef"][0], "beta_D": eol_model["coef"][1],
            "R2": eol_model["r2"], "adjusted_R2": eol_model["adjusted_r2"],
            "permutation_p": eol_perm[0], "permutation_p_percent": 100 * eol_perm[0],
            "permutation_exceedances": eol_perm[1], "permutation_total": eol_perm[2],
            "beta_A_ci_low": eol_ci[0, 0], "beta_A_ci_median": eol_ci[1, 0], "beta_A_ci_high": eol_ci[2, 0],
            "beta_D_ci_low": eol_ci[0, 1], "beta_D_ci_median": eol_ci[1, 1], "beta_D_ci_high": eol_ci[2, 1],
        },
    ])

    soc_rows = []
    for s0 in [0.4, 0.45, 0.5, 0.55, 0.6]:
        temp = new[["policy", "C1", "Q1", "C2", "rate_response", "eol_response"]].copy()
        exposures = temp.apply(lambda row: exposure(row.C1, row.Q1, row.C2, s0), axis=1, result_type="expand")
        exposures.columns = ["T", "A", "EL", "EH", "D"]
        temp = pd.concat([temp, exposures], axis=1)
        for response, label in [("rate_response", "stable_rate"), ("eol_response", f"EOL_{eol_scale}")]:
            model = standardized_regression(temp, response, ["EL", "EH"])
            permutation = exact_r2_permutation_p(model["x"], model["y"], model["r2"])
            _, lopo_summary = leave_one_policy_out(temp, response, ["EL", "EH"])
            soc_rows.append({
                "s0": s0, "response": label,
                "beta_low": model["coef"][0], "beta_high": model["coef"][1],
                "R2": model["r2"], "adjusted_R2": model["adjusted_r2"],
                "permutation_p": permutation[0], "permutation_exceedances": permutation[1],
                "permutation_total": permutation[2],
                "LOPO_MAE": lopo_summary["MAE"], "LOPO_RMSE": lopo_summary["RMSE"],
                "low_positive_folds": lopo_summary["coef_1_positive_folds"],
                "high_positive_folds": lopo_summary["coef_2_positive_folds"],
            })

    mechanism_rows = []
    for predictor in ["A", "D_50", "E_H_50"]:
        for response in ["rate_response", "IR_slope_median", "Tavg_median", "charge_time_median"]:
            rho, p_value = spearmanr(new[predictor], new[response])
            mechanism_rows.append({
                "predictor": predictor, "response": response,
                "rho": float(rho), "p": float(p_value), "N_policy_positions": len(new),
            })
    for predictor in ["IR_slope_median", "Tavg_median", "charge_time_median"]:
        rho, p_value = spearmanr(new[predictor], new["rate_response"])
        mechanism_rows.append({
            "predictor": predictor, "response": "rate_response",
            "rho": float(rho), "p": float(p_value), "N_policy_positions": len(new),
        })
    mechanism = pd.DataFrame(mechanism_rows)

    kw_table.to_csv(Q2_RESULTS / "q2_01_九策略双响应_Kruskal置换效应量.csv", index=False)
    dunn.to_csv(Q2_RESULTS / "q2_02_总体显著后的Dunn_Holm.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(Q2_RESULTS / "q2_03_原始参数_Spearman_双响应.csv", index=False)
    complete_policy.to_csv(Q2_RESULTS / "q2_04_策略参数与SOC暴露.csv", index=False)
    vif.to_csv(Q2_RESULTS / "q2_05_VIF_原始参数与AD.csv", index=False)
    coefficient_table.to_csv(Q2_RESULTS / "q2_06_AD主辅响应回归_置换_Bootstrap.csv", index=False)
    lopo.to_csv(Q2_RESULTS / "q2_07_AD回归_LOPO逐折.csv", index=False)
    lopo_summary_table.to_csv(Q2_RESULTS / "q2_08_AD回归_LOPO汇总.csv", index=False)
    pd.DataFrame(soc_rows).to_csv(Q2_RESULTS / "q2_09_SOC分界敏感性_双响应.csv", index=False)
    mechanism.to_csv(Q2_RESULTS / "q2_10_机制通道描述性相关.csv", index=False)
    return {
        "kruskal": kw_table.to_dict("records"),
        "main_rate_regression": coefficient_table.iloc[0].to_dict(),
        "auxiliary_eol_regression": coefficient_table.iloc[1].to_dict(),
        "eol_response_scale": eol_scale,
        "lopo": lopo_summary_table.to_dict("records"),
        "vif_reparameterized": vif_ad.to_dict("records"),
        "mechanism_correlations": mechanism.to_dict("records"),
    }


POLICIES = []


def visible_features(g: pd.DataFrame, length: int, feature_set: str) -> dict:
    e = g[g["cycle"].le(length)].sort_values("cycle")
    x = e["cycle"].to_numpy(float)
    soh = safe_sg(e["SOH_clean"].to_numpy(float), 11, 2)
    soh_d1 = safe_sg(e["SOH_clean"].to_numpy(float), 11, 2, deriv=1)
    soh_d2 = safe_sg(e["SOH_clean"].to_numpy(float), 11, 2, deriv=2)
    recent = x >= max(1, length - (Q3_RECENT_WINDOW - 1))
    last20 = x >= max(1, length - 19)

    ir_raw = e["IR_clean"].to_numpy(float)
    ir_rel = ir_raw / np.median(ir_raw[:10])
    ir_sg_d1 = safe_sg(ir_rel, 11, 2, deriv=1)
    temp = e["Tavg_clean"].to_numpy(float)
    charge = e["chargetime_clean"].to_numpy(float)

    f = {
        "SOH_state": float(np.median(soh[-10:])),
        "SOH_AUC": float(trapezoid(soh, x) / (x[-1] - x[0])),
        "SOH_global": ts_slope(x, soh),
        "SOH_recent": ts_slope(x[recent], soh[recent]),
        "SOH_delta": ts_slope(x[recent], soh[recent]) - ts_slope(x, soh),
        "SOH_d1": float(np.median(soh_d1[last20])),
        "SOH_d2": float(np.median(soh_d2[last20])),
        "IR_state": float(np.median(ir_rel[-10:])),
        "IR_global": ts_slope(x, ir_rel),
        "IR_recent": ts_slope(x[recent], ir_rel[recent]),
        "IR_delta": ts_slope(x[recent], ir_rel[recent]) - ts_slope(x, ir_rel),
        "IR_d1": float(np.median(ir_sg_d1[last20])),
        "T_state": float(np.median(temp)),
        "T_global": ts_slope(x, temp),
        "T_recent": ts_slope(x[recent], temp[recent]),
        "T_delta": ts_slope(x[recent], temp[recent]) - ts_slope(x, temp),
        "charge_state": float(np.median(charge)),
        "charge_global": ts_slope(x, charge),
        "charge_recent": ts_slope(x[recent], charge[recent]),
        "charge_delta": ts_slope(x[recent], charge[recent]) - ts_slope(x, charge),
        "baseline_TS_slope": ts_slope(x[recent], soh[recent]),
        "baseline_D_slope": float(np.median(soh_d1[last20])),
        "baseline_OLS_slope": float(np.polyfit(x[recent], soh[recent], 1)[0]),
        "baseline_state": float(np.median(soh[-10:])),
    }
    cols = {
        "M1": ["SOH_state", "SOH_AUC", "SOH_global", "SOH_recent", "SOH_delta"],
        "M1D": ["SOH_state", "SOH_AUC", "SOH_global", "SOH_recent", "SOH_delta", "SOH_d1"],
        "M1D2": ["SOH_state", "SOH_AUC", "SOH_global", "SOH_recent", "SOH_delta", "SOH_d1", "SOH_d2"],
        "M2": ["SOH_state", "SOH_AUC", "SOH_global", "SOH_recent", "SOH_delta", "SOH_d1", "IR_state", "IR_global", "IR_recent", "IR_delta"],
        "M2D": ["SOH_state", "SOH_AUC", "SOH_global", "SOH_recent", "SOH_delta", "SOH_d1", "IR_state", "IR_global", "IR_recent", "IR_delta", "IR_d1"],
        "M3": ["SOH_state", "SOH_AUC", "SOH_global", "SOH_recent", "SOH_delta", "SOH_d1", "IR_state", "IR_global", "IR_recent", "IR_delta", "T_state", "T_global", "T_recent", "T_delta", "charge_state", "charge_global", "charge_recent", "charge_delta"],
        "M4": ["SOH_state", "SOH_AUC", "SOH_global", "SOH_recent", "SOH_delta", "SOH_d1", "IR_state", "IR_global", "IR_recent", "IR_delta", "T_state", "T_global", "T_recent", "T_delta", "charge_state", "charge_global", "charge_recent", "charge_delta"],
    }[feature_set]
    return {k: f[k] for k in cols} | {
        "baseline_TS_slope": f["baseline_TS_slope"],
        "baseline_D_slope": f["baseline_D_slope"],
        "baseline_OLS_slope": f["baseline_OLS_slope"],
        "baseline_state": f["baseline_state"],
    }


def design_matrix(feature_frame: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
    base_cols = [c for c in feature_frame.columns if c not in [
        "battery_id", "policy", "baseline_TS_slope", "baseline_D_slope", "baseline_OLS_slope", "baseline_state"
    ]]
    x = feature_frame[base_cols].to_numpy(float)
    names = list(base_cols)
    if feature_set == "M4":
        dummies = pd.get_dummies(pd.Categorical(feature_frame["policy"], categories=POLICIES), prefix="policy", dtype=float)
        x = np.column_stack([x, dummies.to_numpy(float)])
        names.extend(dummies.columns.tolist())
    return x, names


def choose_alpha(x: np.ndarray, scores: np.ndarray, alphas: np.ndarray) -> float:
    best = None
    # Ridge 的线性平滑矩阵可直接给出精确LOO残差：e_i/(1-h_ii)。
    # 截距列不惩罚，其余标准化特征统一施加alpha惩罚。
    x1 = np.column_stack([np.ones(len(x)), x])
    for alpha in alphas:
        penalty = np.eye(x1.shape[1]) * float(alpha)
        penalty[0, 0] = 0.0
        inv = np.linalg.pinv(x1.T @ x1 + penalty)
        hat = x1 @ inv @ x1.T
        fitted = hat @ scores
        denom = np.maximum(1.0 - np.diag(hat), 1e-10)
        loo_resid = (scores - fitted) / denom[:, None]
        mse = float(np.mean(loo_resid**2))
        if best is None or mse < best[0]:
            best = (mse, float(alpha))
    return best[1]


def future_target(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = g[g["cycle"].between(151, 200)].sort_values("cycle")
    return t["cycle"].to_numpy(float), t["SOH_sg"].to_numpy(float)


def build_feature_frame(clean: pd.DataFrame, ids: list[int], length: int, feature_set: str) -> pd.DataFrame:
    rows = []
    for bid in ids:
        g = clean[clean["battery_id"].eq(bid)]
        feats = visible_features(g, length, feature_set)
        rows.append({"battery_id": bid, "policy": g["policy"].iloc[0], **feats})
    return pd.DataFrame(rows)


def baseline_curve(row: pd.Series, length: int, future_cycles: np.ndarray, mode: str) -> np.ndarray:
    if mode == "PERSIST":
        slope = 0.0
    elif mode == "TS":
        slope = row["baseline_TS_slope"]
    elif mode == "OLS":
        slope = row["baseline_OLS_slope"]
    else:
        slope = row["baseline_D_slope"]
    return row["baseline_state"] + slope * (future_cycles - length)


def prediction_metrics(predictions: pd.DataFrame) -> dict:
    err = predictions["predicted"] - predictions["observed"]
    battery_mse = predictions.assign(sqerr=err**2).groupby("battery_id")["sqerr"].mean()
    battery_rmse = np.sqrt(battery_mse)
    e200 = predictions[predictions["cycle"].eq(200)].assign(
        abs_error=lambda d: (d["predicted"] - d["observed"]).abs()
    )["abs_error"]
    mape = float(np.mean(np.abs(err / predictions["observed"])) * 100)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE_pp": float(100 * np.mean(np.abs(err))),
        "RMSE_pp": float(100 * np.sqrt(np.mean(err**2))),
        "MAPE_percent": mape,
        "SOH_relative_agreement_percent": float(100 - mape),
        "R2": float(r2_score(predictions["observed"], predictions["predicted"])),
        "E200_median": float(e200.median()),
        "E200_q90": float(e200.quantile(0.9)),
        "battery_RMSE_median": float(battery_rmse.median()),
        "battery_RMSE_q90": float(battery_rmse.quantile(0.9)),
        "battery_RMSE_max": float(battery_rmse.max()),
    }


def direct_baseline_predictions(clean: pd.DataFrame, ids: list[int], length: int, mode: str) -> pd.DataFrame:
    feat = build_feature_frame(clean, ids, length, "M1")
    cycles = np.arange(151, 201, dtype=float)
    rows = []
    for i, bid in enumerate(ids):
        y = future_target(clean[clean["battery_id"].eq(bid)])[1]
        pred = baseline_curve(feat.iloc[i], length, cycles, mode)
        rows.extend({
            "battery_id": bid,
            "cycle": int(cyc),
            "observed": float(obs),
            "predicted": float(pr),
            "model": f"{mode}_ONLY",
            "length": length,
        } for cyc, obs, pr in zip(cycles, y, pred))
    return pd.DataFrame(rows)


def loocv_trend_pca_ridge(clean: pd.DataFrame, ids: list[int], length: int, feature_set: str, baseline_mode: str) -> tuple[pd.DataFrame, dict]:
    feat = build_feature_frame(clean, ids, length, feature_set)
    x_all, _ = design_matrix(feat, feature_set)
    cycles = np.arange(151, 201, dtype=float)
    y_all = np.vstack([future_target(clean[clean["battery_id"].eq(bid)])[1] for bid in ids])
    base_all = np.vstack([baseline_curve(feat.iloc[i], length, cycles, baseline_mode) for i in range(len(ids))])
    residuals = y_all - base_all
    alphas = np.logspace(-4, 4, 13)
    rows, selected_alphas, ks = [], [], []
    for i, bid in enumerate(ids):
        train = np.arange(len(ids)) != i
        scaler = StandardScaler().fit(x_all[train])
        xtr, xte = scaler.transform(x_all[train]), scaler.transform(x_all[i:i + 1])
        pca_full = PCA().fit(residuals[train])
        k = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95) + 1)
        k = max(1, min(k, len(ids) - 2))
        pca = PCA(n_components=k).fit(residuals[train])
        scores = pca.transform(residuals[train])
        alpha = choose_alpha(xtr, scores, alphas)
        model = Ridge(alpha=alpha).fit(xtr, scores)
        score_pred = model.predict(xte)
        if score_pred.ndim == 1:
            score_pred = score_pred[:, None]
        pred_resid = pca.inverse_transform(score_pred)[0]
        pred = base_all[i] + pred_resid
        selected_alphas.append(alpha); ks.append(k)
        rows.extend({"battery_id": bid, "cycle": int(cyc), "observed": float(obs), "predicted": float(pr), "feature_set": feature_set, "length": length, "baseline": baseline_mode} for cyc, obs, pr in zip(cycles, y_all[i], pred))
    out = pd.DataFrame(rows)
    metrics = {
        "feature_set": feature_set, "length": length, "baseline": baseline_mode,
        **prediction_metrics(out),
        "alpha_median": float(np.median(selected_alphas)), "K_median": float(np.median(ks)),
    }
    return out, metrics


def direct_baseline_metrics(clean: pd.DataFrame, ids: list[int], length: int, mode: str) -> dict:
    metrics = prediction_metrics(direct_baseline_predictions(clean, ids, length, mode))
    return {"baseline": mode, "length": length, **metrics}


def one_standard_error_choice(candidates: dict[str, pd.DataFrame], complexity_order: list[str]) -> tuple[str, pd.DataFrame]:
    """按电池级MSE的一标准误规则选择最简候选。"""
    rows = []
    fold_mse = {}
    for name, pred in candidates.items():
        mse = pred.assign(sqerr=(pred["predicted"] - pred["observed"]) ** 2).groupby("battery_id")["sqerr"].mean()
        fold_mse[name] = mse
        rows.append({"candidate": name, "mean_battery_MSE": float(mse.mean())})
    table = pd.DataFrame(rows)
    best_name = table.sort_values("mean_battery_MSE").iloc[0]["candidate"]
    best_mse = fold_mse[best_name]
    best_se = float(best_mse.std(ddof=1) / np.sqrt(len(best_mse)))
    threshold = float(best_mse.mean() + best_se)
    eligible = set(table.loc[table["mean_battery_MSE"].le(threshold + 1e-18), "candidate"])
    selected = next(name for name in complexity_order if name in eligible)
    table["best_candidate"] = best_name
    table["best_SE"] = best_se
    table["one_SE_threshold"] = threshold
    table["eligible_one_SE"] = table["candidate"].isin(eligible)
    table["selected_one_SE"] = table["candidate"].eq(selected)
    return selected, table


def nested_q3_validation(clean: pd.DataFrame, train_ids: list[int], length: int = 150) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """外层LOBO评价，内层LOBO完成基线、特征组与增强模型选择。"""
    feature_order = ["M1", "M1D", "M1D2", "M2", "M2D", "M3", "M4"]
    outer_predictions, fold_rows, inner_rows = [], [], []
    for outer_bid in train_ids:
        inner_ids = [bid for bid in train_ids if bid != outer_bid]
        baseline_candidates = {
            mode: direct_baseline_predictions(clean, inner_ids, length, mode)
            for mode in ["PERSIST", "OLS", "TS", "D"]
        }
        trend_selected, _ = one_standard_error_choice(
            {f"{mode}_ONLY": baseline_candidates[mode] for mode in ["TS", "D"]},
            ["TS_ONLY", "D_ONLY"],
        )
        baseline_mode = trend_selected.replace("_ONLY", "")
        candidates = {f"{mode}_ONLY": prediction for mode, prediction in baseline_candidates.items()}
        for feature_set in feature_order:
            pred, _ = loocv_trend_pca_ridge(clean, inner_ids, length, feature_set, baseline_mode)
            candidates[feature_set] = pred
        # TS 与 OLS 同属一参数线性趋势基线；按预先规定的稳健基线职责，
        # 二者同时进入 one-SE 近优集合时优先 TS。该顺序不依据外层测试结果调整。
        complexity = ["PERSIST_ONLY", "TS_ONLY", "OLS_ONLY", "D_ONLY", *feature_order]
        selected, selection_table = one_standard_error_choice(candidates, complexity)
        selection_table.insert(0, "held_out_battery", outer_bid)
        inner_rows.append(selection_table)

        if selected.endswith("_ONLY"):
            selected_baseline = selected.replace("_ONLY", "")
            outer_pred = direct_baseline_predictions(clean, [outer_bid], length, selected_baseline)
        else:
            fitted, _ = fit_final_predict(clean, inner_ids, [outer_bid], length, selected, baseline_mode)
            observed = future_target(clean[clean["battery_id"].eq(outer_bid)])[1]
            outer_pred = fitted.rename(columns={"SOH_pred": "predicted"})
            outer_pred["observed"] = observed
            outer_pred["model"] = selected
            outer_pred["length"] = length
            outer_pred = outer_pred[["battery_id", "cycle", "observed", "predicted", "model", "length"]]
        outer_predictions.append(outer_pred)
        outer_metric = prediction_metrics(outer_pred)
        fold_rows.append({
            "held_out_battery": outer_bid,
            "inner_baseline": baseline_mode,
            "selected_model": selected,
            **outer_metric,
        })

    predictions = pd.concat(outer_predictions, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    inner_selection = pd.concat(inner_rows, ignore_index=True)
    battery_metrics = predictions.assign(
        abs_error=lambda d: (d["predicted"] - d["observed"]).abs(),
        sq_error=lambda d: (d["predicted"] - d["observed"]) ** 2,
    ).groupby("battery_id", as_index=False).agg(
        MAE=("abs_error", "mean"), RMSE=("sq_error", lambda s: float(np.sqrt(s.mean())))
    )
    summary = {
        **prediction_metrics(predictions),
        "outer_folds": len(train_ids),
        "selection_counts": folds["selected_model"].value_counts().to_dict(),
        "baseline_counts": folds["inner_baseline"].value_counts().to_dict(),
    }
    return predictions, folds, battery_metrics, summary


def fit_final_predict(clean: pd.DataFrame, train_ids: list[int], test_ids: list[int], length: int, feature_set: str, baseline_mode: str) -> tuple[pd.DataFrame, dict]:
    all_ids = train_ids + test_ids
    feat = build_feature_frame(clean, all_ids, length, feature_set)
    x_all, names = design_matrix(feat, feature_set)
    ntr = len(train_ids)
    scaler = StandardScaler().fit(x_all[:ntr])
    xtr, xte = scaler.transform(x_all[:ntr]), scaler.transform(x_all[ntr:])
    cycles = np.arange(151, 201, dtype=float)
    ytr = np.vstack([future_target(clean[clean["battery_id"].eq(bid)])[1] for bid in train_ids])
    base_tr = np.vstack([baseline_curve(feat.iloc[i], length, cycles, baseline_mode) for i in range(ntr)])
    residuals = ytr - base_tr
    pca_full = PCA().fit(residuals)
    k = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95) + 1)
    pca = PCA(n_components=max(1, k)).fit(residuals)
    scores = pca.transform(residuals)
    alpha = choose_alpha(xtr, scores, np.logspace(-4, 4, 13))
    model = Ridge(alpha=alpha).fit(xtr, scores)
    score_pred = model.predict(xte)
    if score_pred.ndim == 1:
        score_pred = score_pred[:, None]
    pred_resid = pca.inverse_transform(score_pred)
    rows = []
    for j, bid in enumerate(test_ids):
        base = baseline_curve(feat.iloc[ntr + j], length, cycles, baseline_mode)
        pred = base + pred_resid[j]
        rows.extend({"battery_id": bid, "cycle": int(cyc), "SOH_pred": float(value), "policy": feat.iloc[ntr + j]["policy"]} for cyc, value in zip(cycles, pred))
    return pd.DataFrame(rows), {"alpha": alpha, "K": k, "features": names}


def fit_direct_baseline_predict(clean: pd.DataFrame, test_ids: list[int], length: int, baseline_mode: str) -> pd.DataFrame:
    feat = build_feature_frame(clean, test_ids, length, "M1")
    cycles = np.arange(151, 201, dtype=float)
    rows = []
    for i, bid in enumerate(test_ids):
        pred = baseline_curve(feat.iloc[i], length, cycles, baseline_mode)
        rows.extend({"battery_id": bid, "cycle": int(cyc), "SOH_pred": float(value), "policy": feat.iloc[i]["policy"]} for cyc, value in zip(cycles, pred))
    return pd.DataFrame(rows)


def q3_eol(
    clean: pd.DataFrame,
    predictions: pd.DataFrame,
    test_ids: list[int],
    selected_n0: int,
    selected_model: str,
) -> pd.DataFrame:
    rows = []
    for bid in test_ids:
        obs = clean[
            clean["battery_id"].eq(bid) & clean["cycle"].between(selected_n0, 150)
        ].sort_values("cycle")
        pred = predictions[predictions["battery_id"].eq(bid)].sort_values("cycle")
        x = np.concatenate([obs["cycle"].to_numpy(float), pred["cycle"].to_numpy(float)])
        y = np.concatenate([obs["SOH_sg"].to_numpy(float), pred["SOH_pred"].to_numpy(float)])
        params = fit_eol_candidate(selected_model, x, y, selected_n0)
        rows.append({
            "battery_id": bid, "selected_n0": selected_n0, "selected_model": selected_model,
            "life_q3": params["life"], "model_a": params["a"], "model_b": params["b"], "model_c": params["c"],
        })
    return pd.DataFrame(rows)


def residual_bootstrap_life(
    clean: pd.DataFrame,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    nested_predictions: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
    selected_n0: int,
    selected_model: str,
    draws: int = 2000,
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 3)
    residual_bank = []
    for bid in train_ids:
        fold = nested_predictions[nested_predictions["battery_id"].eq(bid)].sort_values("cycle")
        residual_bank.append((fold["observed"] - fold["predicted"]).to_numpy(float))
    residual_bank = np.vstack(residual_bank)
    sampled = residual_bank[rng.integers(0, len(residual_bank), size=draws)]
    rows = []
    for bid in test_ids:
        obs = clean[
            clean["battery_id"].eq(bid) & clean["cycle"].between(selected_n0, 150)
        ].sort_values("cycle")
        point = predictions[predictions["battery_id"].eq(bid)].sort_values("cycle")["SOH_pred"].to_numpy(float)
        lives = []
        x = np.arange(selected_n0, 201, dtype=float)
        yobs = obs["SOH_sg"].to_numpy(float)
        warm_start = fit_power(x, np.concatenate([yobs, point])) if selected_model == "power" else None
        for residual in sampled:
            y = np.concatenate([yobs, point + residual])
            if selected_model == "power":
                params = fit_power_warm(x, y, warm_start)
            else:
                params = fit_eol_candidate(selected_model, x, y, selected_n0)
            if np.isfinite(params["life"]):
                lives.append(params["life"])
        q = np.quantile(lives, [0.025, 0.5, 0.975]) if lives else [np.nan, np.nan, np.nan]
        rows.append({"battery_id": bid, "life_boot_low": q[0], "life_boot_median": q[1], "life_boot_high": q[2], "bootstrap_valid": len(lives), "bootstrap_draws": draws})
    return pd.DataFrame(rows)


def q3_eol_structural_sensitivity(
    clean: pd.DataFrame,
    predictions: pd.DataFrame,
    battery_ids: list[int],
    prediction_col: str,
) -> pd.DataFrame:
    rows = []
    for battery_id in battery_ids:
        future = predictions[predictions["battery_id"].eq(battery_id)].sort_values("cycle")
        for n0 in STABLE_STARTS:
            observed = clean[
                clean["battery_id"].eq(battery_id) & clean["cycle"].between(n0, 150)
            ].sort_values("cycle")
            x = np.concatenate([observed["cycle"].to_numpy(float), future["cycle"].to_numpy(float)])
            y = np.concatenate([observed["SOH_sg"].to_numpy(float), future[prediction_col].to_numpy(float)])
            for model in ["linear", "power", "centered_quadratic"]:
                fit = fit_eol_candidate(model, x, y, n0)
                rows.append({
                    "battery_id": battery_id, "n0": n0, "model": model, "life": fit["life"],
                    "fit_success": fit["success"],
                    "b_at_lower_bound": bool(fit.get("b_at_lower_bound", False)),
                    "c_near_bound": bool(fit.get("c_near_bound", False)),
                })
    details = pd.DataFrame(rows)
    summaries = []
    for battery_id, group in details.groupby("battery_id"):
        finite = group["life"].replace([np.inf, -np.inf], np.nan).dropna()
        summaries.append({
            "battery_id": battery_id,
            "finite_scenarios": len(finite),
            "structural_life_min": float(finite.min()) if len(finite) else np.nan,
            "structural_life_median": float(finite.median()) if len(finite) else np.nan,
            "structural_life_max": float(finite.max()) if len(finite) else np.nan,
            "structural_relative_span_percent": float(100 * (finite.max() - finite.min()) / finite.median()) if len(finite) else np.nan,
        })
    return details.merge(pd.DataFrame(summaries), on="battery_id", how="left")


def q3_eol_stability_on_complete_batteries(
    clean: pd.DataFrame,
    nested_predictions: pd.DataFrame,
    train_ids: list[int],
    selected_n0: int,
    selected_model: str,
) -> pd.DataFrame:
    rows = []
    for battery_id in train_ids:
        observed_stable = clean[
            clean["battery_id"].eq(battery_id) & clean["cycle"].between(selected_n0, 150)
        ].sort_values("cycle")
        predicted_future = nested_predictions[nested_predictions["battery_id"].eq(battery_id)].sort_values("cycle")
        x_pred = np.concatenate([observed_stable["cycle"].to_numpy(float), predicted_future["cycle"].to_numpy(float)])
        y_pred = np.concatenate([observed_stable["SOH_sg"].to_numpy(float), predicted_future["predicted"].to_numpy(float)])
        predicted_fit = fit_eol_candidate(selected_model, x_pred, y_pred, selected_n0)
        observed_200 = clean[
            clean["battery_id"].eq(battery_id) & clean["cycle"].between(selected_n0, 200)
        ].sort_values("cycle")
        reference_fit = fit_eol_candidate(
            selected_model,
            observed_200["cycle"].to_numpy(float),
            observed_200["SOH_sg"].to_numpy(float),
            selected_n0,
        )
        relative_error = abs(predicted_fit["life"] - reference_fit["life"]) / reference_fit["life"]
        rows.append({
            "battery_id": battery_id, "selected_n0": selected_n0, "selected_model": selected_model,
            "life_predicted_trajectory": predicted_fit["life"], "life_observed_to_200_reference": reference_fit["life"],
            "relative_difference": relative_error, "relative_difference_percent": 100 * relative_error,
        })
    return pd.DataFrame(rows)


def fit_state_relation(clean: pd.DataFrame, train_ids: list[int], value_col: str, family: str) -> dict:
    """只用前150圈拟合 X=f(SOH, policy)；锚定模型强制通过各电池末端观测。"""
    train = clean[clean["battery_id"].isin(train_ids) & clean["cycle"].le(150)].copy()
    if family == "PERSIST":
        return {"family": family, "global": None, "policy": {}}

    if family.startswith("ANCHOR"):
        anchors = train[train["cycle"].between(141, 150)].groupby("battery_id").agg(
            SOH_anchor=("SOH_sg", "median"), X_anchor=(value_col, "median")
        )
        train = train.join(anchors, on="battery_id")
        train["dSOH"] = train["SOH_sg"] - train["SOH_anchor"]
        train["dX"] = train[value_col] - train["X_anchor"]

        def anchored_slope(frame: pd.DataFrame) -> float:
            x = frame["dSOH"].to_numpy(float)
            y = frame["dX"].to_numpy(float)
            denominator = float(x @ x)
            return float((x @ y) / denominator) if denominator > 1e-15 else 0.0

        global_fit = anchored_slope(train)
        policy_fit = {policy: anchored_slope(group) for policy, group in train.groupby("policy")}
        return {"family": family, "global": global_fit, "policy": policy_fit}

    def full_fit(frame: pd.DataFrame) -> np.ndarray:
        design = np.column_stack([np.ones(len(frame)), frame["SOH_sg"].to_numpy(float)])
        return np.linalg.lstsq(design, frame[value_col].to_numpy(float), rcond=None)[0]

    global_fit = full_fit(train)
    policy_fit = {policy: full_fit(group) for policy, group in train.groupby("policy")}
    return {"family": family, "global": global_fit, "policy": policy_fit}


def predict_state_relation(
    clean: pd.DataFrame,
    battery_id: int,
    soh: np.ndarray,
    value_col: str,
    relation: dict,
) -> np.ndarray:
    battery = clean[clean["battery_id"].eq(battery_id)].sort_values("cycle")
    policy = battery["policy"].iloc[0]
    anchor = battery[battery["cycle"].between(141, 150)]
    x_anchor = float(anchor[value_col].median())
    soh_anchor = float(anchor["SOH_sg"].median())
    family = relation["family"]
    soh = np.asarray(soh, float)
    if family == "PERSIST":
        return np.repeat(x_anchor, len(soh))
    if family.startswith("ANCHOR"):
        slope = relation["global"] if family == "ANCHOR_GLOBAL" else relation["policy"].get(policy, relation["global"])
        return x_anchor + float(slope) * (soh - soh_anchor)
    coefficients = relation["global"] if family == "FULL_GLOBAL" else relation["policy"].get(policy, relation["global"])
    return float(coefficients[0]) + float(coefficients[1]) * soh


def state_relation_lobo(
    clean: pd.DataFrame,
    summary: pd.DataFrame,
    nested_soh_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """电池级LOBO分别检验真实SOH条件下的状态映射和端到端状态预测。"""
    train_ids = summary.loc[summary["prediction_test"].eq(0), "battery_id"].astype(int).tolist()
    prediction_rows = []
    for channel, (value_col, _) in STATE_CHANNELS.items():
        for family in STATE_FAMILIES:
            for battery_id in train_ids:
                relation = fit_state_relation(clean, [bid for bid in train_ids if bid != battery_id], value_col, family)
                future = clean[
                    clean["battery_id"].eq(battery_id) & clean["cycle"].between(151, 200)
                ].sort_values("cycle")
                q3_soh = nested_soh_predictions[nested_soh_predictions["battery_id"].eq(battery_id)].sort_values("cycle")
                assert future["cycle"].tolist() == q3_soh["cycle"].tolist()
                oracle = predict_state_relation(clean, battery_id, future["SOH_sg"].to_numpy(float), value_col, relation)
                end_to_end = predict_state_relation(clean, battery_id, q3_soh["predicted"].to_numpy(float), value_col, relation)
                prediction_rows.extend({
                    "channel": channel,
                    "family": family,
                    "battery_id": battery_id,
                    "cycle": int(cycle),
                    "observed": float(observed),
                    "predicted_oracle_SOH": float(pred_oracle),
                    "predicted_Q3_SOH": float(pred_q3),
                } for cycle, observed, pred_oracle, pred_q3 in zip(
                    future["cycle"], future[value_col], oracle, end_to_end
                ))
    predictions = pd.DataFrame(prediction_rows)

    metric_rows = []
    for (channel, family), group in predictions.groupby(["channel", "family"], sort=False):
        for mode, pred_col in [("oracle_SOH", "predicted_oracle_SOH"), ("Q3_SOH", "predicted_Q3_SOH")]:
            error = group[pred_col] - group["observed"]
            battery_mse = group.assign(sq_error=error**2).groupby("battery_id")["sq_error"].mean()
            metric_rows.append({
                "channel": channel,
                "family": family,
                "SOH_input": mode,
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "battery_RMSE_median": float(np.sqrt(battery_mse).median()),
                "battery_RMSE_q90": float(np.sqrt(battery_mse).quantile(0.9)),
            })
    metrics = pd.DataFrame(metric_rows)

    selection_rows = []
    # 完整交互只检验Policy是否改变状态水平/斜率，不与锚定轨迹模型争夺最终外推资格。
    complexity = {family: i for i, family in enumerate(STATE_TRAJECTORY_FAMILIES)}
    for channel in STATE_CHANNELS:
        channel_predictions = predictions[predictions["channel"].eq(channel)]
        battery_mse = {}
        for family in STATE_FAMILIES:
            group = channel_predictions[channel_predictions["family"].eq(family)]
            battery_mse[family] = group.assign(
                sq_error=(group["predicted_oracle_SOH"] - group["observed"]) ** 2
            ).groupby("battery_id")["sq_error"].mean()
        mean_mse = {family: float(values.mean()) for family, values in battery_mse.items()}
        best_family = min(STATE_TRAJECTORY_FAMILIES, key=lambda family: mean_mse[family])
        best_se = float(battery_mse[best_family].std(ddof=1) / np.sqrt(len(battery_mse[best_family])))
        threshold = mean_mse[best_family] + best_se
        eligible = [
            family for family in STATE_TRAJECTORY_FAMILIES
            if mean_mse[family] <= threshold + 1e-18
        ]
        selected = min(eligible, key=complexity.get)
        nontrivial = min(STATE_TRAJECTORY_FAMILIES[1:], key=lambda family: mean_mse[family])
        best_complete = min(STATE_DIAGNOSTIC_FAMILIES, key=lambda family: mean_mse[family])
        improvement = 100.0 * (mean_mse["PERSIST"] - mean_mse[nontrivial]) / mean_mse["PERSIST"]
        selection_rows.append({
            "channel": channel,
            "best_trajectory_family": best_family,
            "best_nontrivial_family": nontrivial,
            "best_complete_interaction_diagnostic": best_complete,
            "selected_one_SE": selected,
            "best_mean_battery_MSE": mean_mse[best_family],
            "best_SE": best_se,
            "one_SE_threshold": threshold,
            "nontrivial_improvement_percent": improvement,
            "incremental_relation_retained": selected != "PERSIST",
        })
    return predictions, metrics, pd.DataFrame(selection_rows)


def life_summary_validation(
    clean: pd.DataFrame,
    summary: pd.DataFrame,
    soh_predictions: pd.DataFrame,
    eol: pd.DataFrame,
    state_selection: pd.DataFrame,
    selected_q1_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """冻结SOH/EOL后才读取三个完整寿命均值；结果只作间接诊断，不反向校准寿命。"""
    train_ids = summary.loc[summary["prediction_test"].eq(0), "battery_id"].astype(int).tolist()
    test_ids = summary.loc[summary["prediction_test"].eq(1), "battery_id"].astype(int).tolist()
    rows = []
    for channel, (value_col, truth_col) in STATE_CHANNELS.items():
        selected_family = state_selection.loc[
            state_selection["channel"].eq(channel), "selected_one_SE"
        ].iloc[0]
        retained = bool(state_selection.loc[
            state_selection["channel"].eq(channel), "incremental_relation_retained"
        ].iloc[0])
        relation = fit_state_relation(clean, train_ids, value_col, selected_family)
        for battery_id in test_ids:
            observed = clean[clean["battery_id"].eq(battery_id) & clean["cycle"].le(150)].sort_values("cycle")
            future = soh_predictions[soh_predictions["battery_id"].eq(battery_id)].sort_values("cycle")
            life_row = eol[eol["battery_id"].eq(battery_id)].iloc[0]
            life = max(200, int(round(float(life_row["life_q3"]))))
            tail_cycles = np.arange(201, life + 1, dtype=float)
            params = {
                "a": life_row["model_a"],
                "b": life_row["model_b"],
                "c": life_row["model_c"],
                "n0": int(life_row.get("selected_n0", 31)),
            }
            tail_soh = model_predict(selected_q1_model, params, tail_cycles) if len(tail_cycles) else np.array([])
            predicted_soh = np.concatenate([future["SOH_pred"].to_numpy(float), tail_soh])
            predicted_state = predict_state_relation(clean, battery_id, predicted_soh, value_col, relation)
            life_mean = float(np.mean(np.concatenate([observed[value_col].to_numpy(float), predicted_state])))

            true_mean = float(summary.loc[summary["battery_id"].eq(battery_id), truth_col].iloc[0])
            rows.append({
                "channel": channel,
                "family": selected_family,
                "formal_relation_retained": retained,
                "battery_id": battery_id,
                "life_q3": float(life_row["life_q3"]),
                "predicted_life_mean": life_mean,
                "true_life_mean": true_mean,
                "error": life_mean - true_mean,
            })
    validation = pd.DataFrame(rows)
    metric_rows = []
    for channel, group in validation.groupby("channel", sort=False):
        error = group["error"]
        rho = spearmanr(group["predicted_life_mean"], group["true_life_mean"])
        mape = float(100 * np.mean(np.abs(error / group["true_life_mean"])))
        metric_rows.append({
            "channel": channel,
            "family": group["family"].iloc[0],
            "formal_relation_retained": bool(group["formal_relation_retained"].iloc[0]),
            "n_test_batteries": len(group),
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error**2))),
            "bias": float(np.mean(error)),
            "MAPE_percent": mape,
            "state_agreement_percent": float(100 - mape),
            "spearman_rho": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
        })
    return validation, pd.DataFrame(metric_rows)


def q3_analysis(clean: pd.DataFrame, summary: pd.DataFrame, selected_n0: int, selected_q1_model: str) -> dict:
    train_ids = summary.loc[summary["prediction_test"].eq(0), "battery_id"].astype(int).tolist()
    test_ids = summary.loc[summary["prediction_test"].eq(1), "battery_id"].astype(int).tolist()
    global POLICIES
    POLICIES = sorted(summary["policy"].unique().tolist())

    baseline_metrics = [
        direct_baseline_metrics(clean, train_ids, 150, mode)
        for mode in ["PERSIST", "OLS", "TS", "D"]
    ]
    trend_candidates = {
        f"{mode}_ONLY": direct_baseline_predictions(clean, train_ids, 150, mode)
        for mode in ["TS", "D"]
    }
    selected_trend, trend_selection = one_standard_error_choice(trend_candidates, ["TS_ONLY", "D_ONLY"])
    baseline_mode = selected_trend.replace("_ONLY", "")
    pd.DataFrame(baseline_metrics).to_csv(Q3_RESULTS / "q3_01_四种简单基线比较.csv", index=False)
    trend_selection.to_csv(Q3_RESULTS / "q3_02_TS与SG导数基线_oneSE选择.csv", index=False)

    feature_order = ["M1", "M1D", "M1D2", "M2", "M2D", "M3", "M4"]
    ablation_rows, ablation_preds = [], []
    for fs in feature_order:
        pred, metric = loocv_trend_pca_ridge(clean, train_ids, 150, fs, baseline_mode)
        ablation_rows.append(metric); ablation_preds.append(pred)
    ablation = pd.DataFrame(ablation_rows).sort_values("RMSE").reset_index(drop=True)
    # 开发阶段增强候选内部仍用1%近优规则；最终模型由一标准误规则决定。
    best_rmse = ablation["RMSE"].min()
    eligible = ablation[ablation["RMSE"] <= best_rmse * 1.01]["feature_set"].tolist()
    selected_fs = min(eligible, key=feature_order.index)
    ablation.to_csv(Q3_RESULTS / "q3_03_增强模型特征消融.csv", index=False)
    ablation_predictions = pd.concat(ablation_preds, ignore_index=True)
    ablation_predictions.to_csv(Q3_RESULTS / "q3_04_增强模型消融逐点预测.csv", index=False)

    development_candidates = {
        f"{mode}_ONLY": direct_baseline_predictions(clean, train_ids, 150, mode)
        for mode in ["PERSIST", "OLS", "TS", "D"]
    }
    development_candidates.update({
        fs: ablation_predictions[ablation_predictions["feature_set"].eq(fs)].copy()
        for fs in feature_order
    })
    development_selected, development_selection = one_standard_error_choice(
        development_candidates, ["PERSIST_ONLY", "TS_ONLY", "OLS_ONLY", "D_ONLY", *feature_order]
    )
    development_selection.to_csv(Q3_RESULTS / "q3_05_全候选开发选择_oneSE.csv", index=False)

    nested_pred, nested_folds, nested_battery, nested_summary = nested_q3_validation(clean, train_ids, 150)
    nested_pred.to_csv(Q3_RESULTS / "q3_06_外层40折逐点预测.csv", index=False)
    nested_folds.to_csv(Q3_RESULTS / "q3_07_外层40折选型与误差.csv", index=False)
    nested_battery.to_csv(Q3_RESULTS / "q3_08_外层电池级误差.csv", index=False)
    pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v for k, v in nested_summary.items()}]).to_csv(
        Q3_RESULTS / "q3_09_外层40折精度汇总.csv", index=False
    )

    enhanced_length_rows, length_preds = [], []
    for length in [50, 75, 100, 125, 150]:
        pred, metric = loocv_trend_pca_ridge(clean, train_ids, length, selected_fs, baseline_mode)
        enhanced_length_rows.append(metric); length_preds.append(pred)
    enhanced_length_metrics = pd.DataFrame(enhanced_length_rows)
    enhanced_length_metrics.to_csv(Q3_RESULTS / "q3_10_增强模型不同早期长度.csv", index=False)
    length_metrics = pd.DataFrame([
        direct_baseline_metrics(clean, train_ids, length, mode)
        for length in [50, 75, 100, 125, 150]
        for mode in ["PERSIST", "OLS", "TS", "D"]
    ])
    length_metrics.to_csv(Q3_RESULTS / "q3_11_简单基线不同早期长度.csv", index=False)
    pd.concat(length_preds, ignore_index=True).to_csv(Q3_RESULTS / "q3_12_增强模型不同长度逐点预测.csv", index=False)

    enhanced_predictions, final_info = fit_final_predict(clean, train_ids, test_ids, 150, selected_fs, baseline_mode)
    enhanced_predictions.to_csv(Q3_RESULTS / "q3_13_测试电池增强模型预测_候选.csv", index=False)
    final_predictor = development_selected
    if development_selected.endswith("_ONLY"):
        final_baseline = development_selected.replace("_ONLY", "")
        predictions = fit_direct_baseline_predict(clean, test_ids, 150, final_baseline)
        final_info = {"alpha": np.nan, "K": np.nan, "features": []}
    elif development_selected == selected_fs:
        predictions = enhanced_predictions
    else:
        predictions, final_info = fit_final_predict(clean, train_ids, test_ids, 150, development_selected, baseline_mode)
    predictions.to_csv(Q3_RESULTS / "q3_14_九块测试电池151_200正式预测.csv", index=False)
    eol = q3_eol(clean, predictions, test_ids, selected_n0, selected_q1_model)
    boot = residual_bootstrap_life(
        clean, summary, predictions, nested_pred, train_ids, test_ids,
        selected_n0, selected_q1_model, draws=2000
    )
    test_summary = summary[summary["battery_id"].isin(test_ids)][["battery_id", "policy"]].merge(eol, on="battery_id").merge(boot, on="battery_id")
    test_summary.to_csv(Q3_RESULTS / "q3_15_九块测试电池EOL与统计区间.csv", index=False)

    structural = q3_eol_structural_sensitivity(clean, predictions, test_ids, "SOH_pred")
    structural.to_csv(Q3_RESULTS / "q3_16_九块测试电池EOL结构情景.csv", index=False)
    complete_eol_stability = q3_eol_stability_on_complete_batteries(
        clean, nested_pred, train_ids, selected_n0, selected_q1_model
    )
    complete_eol_stability.to_csv(Q3_RESULTS / "q3_17_四十块电池预测轨迹EOL稳定性.csv", index=False)

    state_predictions, state_metrics, state_selection = state_relation_lobo(clean, summary, nested_pred)
    state_predictions.to_csv(Q3_RESULTS / "q3_18_状态关系LOBO逐点预测.csv", index=False)
    state_metrics.to_csv(Q3_RESULTS / "q3_19_状态关系LOBO误差.csv", index=False)
    state_selection.to_csv(Q3_RESULTS / "q3_20_状态关系oneSE选择.csv", index=False)
    life_validation, life_metrics = life_summary_validation(
        clean, summary, predictions, eol, state_selection, selected_q1_model
    )
    life_validation.to_csv(Q3_RESULTS / "q3_21_九块测试电池完整寿命均值对照.csv", index=False)
    life_metrics.to_csv(Q3_RESULTS / "q3_22_完整寿命状态吻合度汇总.csv", index=False)
    return {
        "q1_selected_n0": selected_n0,
        "baseline_mode": baseline_mode,
        "best_enhanced_feature_set": selected_fs,
        "development_selection_one_SE": development_selected,
        "final_predictor": final_predictor,
        "final_alpha": None if not np.isfinite(final_info["alpha"]) else float(final_info["alpha"]),
        "final_K": None if not np.isfinite(final_info["K"]) else int(final_info["K"]),
        "nested_validation": nested_summary,
        "length_metrics": length_metrics.to_dict("records"),
        "test_life": test_summary.to_dict("records"),
        "complete_battery_eol_stability": {
            "median_relative_difference_percent": float(complete_eol_stability["relative_difference_percent"].median()),
            "spearman": float(spearmanr(
                complete_eol_stability["life_predicted_trajectory"],
                complete_eol_stability["life_observed_to_200_reference"],
            ).statistic),
        },
        "state_relation_selection": state_selection.to_dict("records"),
        "state_relation_metrics": state_metrics.to_dict("records"),
        "life_summary_metrics": life_metrics.to_dict("records"),
        "charge_time_variable": "cycle_train.chargetime",
        "charge_time_proxy_used": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_report(summary_out: dict) -> None:
    q1 = summary_out["q1"]
    q2 = summary_out["q2"]
    q3 = summary_out["q3"]
    kw_lines = [
        f"- {row['response']}：H={row['H']:.6f}，置换p={row['p_permutation']:.6f}（{row['p_permutation_percent']:.3f}%），eta_H²={row['eta_H2']:.6f}"
        for row in q2["kruskal"]
    ]
    state_lines = [
        f"- {row['channel']}：one-SE选择 {row['selected_one_SE']}，非平凡关系MSE改善 {row['nontrivial_improvement_percent']:.3f}%"
        for row in q3["state_relation_selection"]
    ]
    life_lines = [
        f"- {row['channel']}：MAE={row['MAE']:.8g}，MAPE={row['MAPE_percent']:.3f}%，状态吻合度={row['state_agreement_percent']:.3f}%"
        for row in q3["life_summary_metrics"]
    ]
    text = "\n".join([
        "# 最新版建模规范完整重跑结果摘要",
        "",
        "> 本报告只记录本轮运行结果；三份 `问题1/2/3（latest）(2).md` 未被写入任何实证数值。",
        "",
        "## 问题1",
        "",
        f"- 最终稳定退化起点：n0={q1['selected_n0']}。",
        f"- 最终EOL结构：{q1['selected_model']}。",
        f"- 外层40折MAE={q1['nested_MAE']:.8g}（{100*q1['nested_MAE']:.5f} SOH百分点）。",
        f"- 外层40折RMSE={q1['nested_RMSE']:.8g}（{100*q1['nested_RMSE']:.5f} SOH百分点）。",
        f"- 外层选择计数：{json.dumps(q1['nested_selection_counts'], ensure_ascii=False)}。",
        f"- EOL排序长寿命候选：{q1['long_policy']}；EOL排序短寿命候选：{q1['short_policy']}。是否可称为典型策略需结合 `q1_12_典型策略一致性审计.csv`。",
        "- EOL均为基准外推，不代表真实寿命标签。",
        "",
        "## 问题2",
        "",
        *kw_lines,
        f"- 主响应 A,D→稳定退化速率：R²={q2['main_rate_regression']['R2']:.6f}，调整R²={q2['main_rate_regression']['adjusted_R2']:.6f}，精确置换p={q2['main_rate_regression']['permutation_p']:.6f}。",
        f"- 辅助EOL响应尺度：{q2['eol_response_scale']}；仅作方向一致性辅助。",
        "",
        "## 问题3",
        "",
        f"- TS/SG导数趋势中用于增强模型的趋势基线：{q3['baseline_mode']}。",
        f"- 全候选one-SE最终模型：{q3['final_predictor']}。",
        f"- 外层40折MAE={q3['nested_validation']['MAE']:.8g}（{q3['nested_validation']['MAE_pp']:.5f} SOH百分点）。",
        f"- 外层40折RMSE={q3['nested_validation']['RMSE']:.8g}（{q3['nested_validation']['RMSE_pp']:.5f} SOH百分点）。",
        f"- MAPE={q3['nested_validation']['MAPE_percent']:.5f}%，SOH相对吻合度={q3['nested_validation']['SOH_relative_agreement_percent']:.5f}%。",
        f"- 外层模型选择计数：{json.dumps(q3['nested_validation']['selection_counts'], ensure_ascii=False)}。",
        "",
        "### 状态关系LOBO",
        "",
        *state_lines,
        "",
        "### 完整寿命状态外部一致性",
        "",
        *life_lines,
        "",
        "这些完整寿命状态吻合度不是EOL圈数准确率。",
    ])
    (RESULTS / "重跑结果摘要.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="按最新版三问建模规范完整重跑")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "重跑_最新版MD",
        help="独立结果目录；不会写入三份建模规范MD",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    configure_output_dir(output_dir.resolve())

    summary = pd.read_csv(ROOT / "battery_summary.csv")
    cycle = pd.read_csv(ROOT / "cycle_train.csv")
    assert summary["battery_id"].nunique() == 49
    assert cycle.duplicated(["battery_id", "cycle"]).sum() == 0
    assert not cycle.isna().any().any()

    clean, audit = preprocess(summary, cycle, k=5, sg_window=11, capacity_floor_fraction=0.01)
    clean.to_csv(Q1_RESULTS / "q1_01_逐循环清洗数据.csv", index=False)
    audit.to_csv(Q1_RESULTS / "q1_02_异常候选与修复审计.csv", index=False)
    battery = q1_features(summary, clean)
    battery, validation, model_predictions, selected_n0, selected_model, q1_nested, q1_parameters = q1_model_analysis(clean, battery)
    battery.to_csv(Q1_RESULTS / "q1_03_四十九块电池指标与基准EOL.csv", index=False)
    validation.to_csv(Q1_RESULTS / "q1_04_十五个稳定段结构候选验证汇总.csv", index=False)
    model_predictions.to_csv(Q1_RESULTS / "q1_05_十五个候选151_200逐点预测.csv", index=False)
    q1_nested.to_csv(Q1_RESULTS / "q1_06_外层40折稳定起点与模型选择.csv", index=False)
    q1_parameters.to_csv(Q1_RESULTS / "q1_07_四十九块电池全部候选参数.csv", index=False)
    write_q1_battery_table(battery)
    pstats = policy_statistics(battery)
    pstats.to_csv(Q1_RESULTS / "q1_08_九种策略分布统计.csv", index=False)
    q1_typical_comparison(pstats).to_csv(Q1_RESULTS / "q1_09_典型长短策略对照.csv", index=False)
    battery[[
        "battery_id", "structural_life_min", "structural_life_median", "structural_life_max",
        "structural_relative_span", "finite_eol_count",
    ]].to_csv(Q1_RESULTS / "q1_10_EOL结构不确定性_四十九块电池.csv", index=False)
    preprocessing_check = preprocessing_sensitivity(
        summary, cycle, pstats, selected_n0, selected_model
    )
    preprocessing_check.to_csv(Q1_RESULTS / "q1_11_Hampel阈值_SG窗口_容量门槛敏感性.csv", index=False)
    q1_typical_consistency_audit(pstats).to_csv(
        Q1_RESULTS / "q1_12_典型策略一致性审计.csv", index=False
    )

    q2 = q2_analysis(battery, pstats)
    q3 = q3_analysis(clean, summary, selected_n0, selected_model)

    life_sens = battery.dropna(subset=["life_200_ref"])
    life_rank = spearmanr(life_sens["life_150"], life_sens["life_200_ref"]).statistic
    summary_out = {
        "preprocess": {
            "hampel_k": 5,
            "capacity_floor_fraction": 0.01,
            "sg_window": 11,
            "capacity_candidates": int(audit.loc[audit.variable.eq("capacity"), "candidate_count"].sum()),
            "capacity_repaired": int(audit.loc[audit.variable.eq("capacity"), "repaired_count"].sum()),
            "IR_candidates": int(audit.loc[audit.variable.eq("IR"), "candidate_count"].sum()),
            "IR_repaired": int(audit.loc[audit.variable.eq("IR"), "repaired_count"].sum()),
            "Tavg_candidates": int(audit.loc[audit.variable.eq("Tavg"), "candidate_count"].sum()),
            "Tavg_repaired": int(audit.loc[audit.variable.eq("Tavg"), "repaired_count"].sum()),
            "chargetime_candidates": int(audit.loc[audit.variable.eq("chargetime"), "candidate_count"].sum()),
            "chargetime_repaired": int(audit.loc[audit.variable.eq("chargetime"), "repaired_count"].sum()),
        },
        "q1": {
            "selected_n0": selected_n0,
            "selected_model": selected_model,
            "model_validation": validation.to_dict("records"),
            "nested_selection_counts": {
                f"n0={int(n0)}|{model}": int(count)
                for (n0, model), count in q1_nested.groupby(["selected_n0", "selected_model"]).size().items()
            },
            "nested_MAE": float(q1_nested["MAE"].mean()),
            "nested_RMSE": float(np.sqrt(np.mean(q1_nested["MSE"]))),
            "nested_E200_median": float(q1_nested["E200"].median()),
            "life_150_200_spearman": float(life_rank),
            "life_relative_difference_median": float(life_sens["life_rel_diff_150_200"].median()),
            "long_policy": pstats.iloc[0]["policy"],
            "short_policy": pstats.iloc[-1]["policy"],
        },
        "q2": q2,
        "q3": q3,
    }
    summary_out["q3"]["recent_window"] = Q3_RECENT_WINDOW
    (RESULTS / "summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    source_paths = [
        ROOT / "问题1（latest）(2).md", ROOT / "问题2（latest）(2).md", ROOT / "问题3（latest）(2).md",
        ROOT / "battery_summary.csv", ROOT / "cycle_train.csv", Path(__file__).resolve(),
    ]
    manifest = pd.DataFrame([{
        "file": str(path), "size_bytes": path.stat().st_size,
        "last_modified": path.stat().st_mtime, "sha256": sha256_file(path),
    } for path in source_paths])
    manifest.to_csv(RESULTS / "输入文件与代码校验清单.csv", index=False)
    write_run_report(summary_out)
    print(json.dumps({
        "output_dir": str(RESULTS),
        "q1_selected_n0": selected_n0,
        "q1_selected_model": selected_model,
        "q1_outer_RMSE": summary_out["q1"]["nested_RMSE"],
        "q3_final_predictor": q3["final_predictor"],
        "q3_outer_RMSE": q3["nested_validation"]["RMSE"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
