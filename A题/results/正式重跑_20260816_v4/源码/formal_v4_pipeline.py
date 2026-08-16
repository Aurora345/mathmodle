"""冻结 v4 方法后，从题目 CSV 完整重跑 Q1--Q4 并传播 Q3 部分池化结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    from scripts import analysis_pipeline
except ModuleNotFoundError:
    import analysis_pipeline


ROOT = Path(__file__).resolve().parents[1]


def shrink_log(value: float, peer: float, weight: float) -> float:
    if value <= 0 or peer <= 0:
        return math.nan
    return float(math.exp(weight * math.log(value) + (1.0 - weight) * math.log(peer)))


def build_formal_q3_eol_table(base: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    """将个体二次 EOL 与同策略收缩合并为 Q3 的正式点估计和统计区间。"""
    extra = pooled[
        [
            "battery_id",
            "v4_individual_EOL",
            "peer_geometric_median_EOL",
            "individual_weight",
            "v4_partially_pooled_EOL",
            "fit_A",
            "fit_B",
            "fit_C",
        ]
    ]
    formal = base.merge(extra, on="battery_id", how="inner", validate="one_to_one")
    formal["life_q3_individual"] = formal["v4_individual_EOL"]
    formal["life_q3"] = formal["v4_partially_pooled_EOL"]
    for column in ["life_boot_low", "life_boot_median", "life_boot_high"]:
        if column in formal:
            formal[column] = [
                shrink_log(value, peer, weight)
                for value, peer, weight in zip(
                    formal[column],
                    formal["peer_geometric_median_EOL"],
                    formal["individual_weight"],
                )
            ]
    formal["selected_n0"] = 31
    formal["selected_model"] = "centered_quadratic"
    formal["model_a"] = formal["fit_A"]
    formal["model_b"] = formal["fit_B"]
    formal["model_c"] = formal["fit_C"]
    formal["eol_role"] = "Q3 formal partially pooled point estimate"
    return formal


def audit_life_propagation(
    q1_battery: pd.DataFrame,
    q2_design: pd.DataFrame,
    q3_formal: pd.DataFrame,
    q4_existing: pd.DataFrame,
) -> pd.DataFrame:
    """逐项核对寿命响应是否由同一轮 Q1/Q3 正式结果传播。"""
    q1_policy = q1_battery.groupby("policy", as_index=False).agg(
        q1_life_median=("life_150", "median")
    )
    rows: list[dict] = []
    q2_life_column = "life_median" if "life_median" in q2_design.columns else "life_150"
    for link, table, value_column in [
        ("Q1→Q2", q2_design, q2_life_column),
        ("Q1→Q4", q4_existing, "EOL_base"),
    ]:
        merged = table[["policy", value_column]].merge(q1_policy, on="policy", how="inner")
        for _, row in merged.iterrows():
            expected = float(row["q1_life_median"])
            actual = float(row[value_column])
            tolerance = max(1e-8, 1e-9 * abs(expected))
            rows.append({
                "link": link,
                "key": row["policy"],
                "expected": expected,
                "actual": actual,
                "absolute_difference": abs(actual - expected),
                "passed": bool(abs(actual - expected) <= tolerance),
            })
    for _, row in q3_formal.iterrows():
        expected = shrink_log(
            float(row["life_q3_individual"]),
            float(row["peer_geometric_median_EOL"]),
            float(row["individual_weight"]),
        )
        actual = float(row["life_q3"])
        tolerance = max(1e-8, 1e-9 * abs(expected))
        rows.append({
            "link": "Q3 pooling",
            "key": int(row["battery_id"]),
            "expected": expected,
            "actual": actual,
            "absolute_difference": abs(actual - expected),
            "passed": bool(abs(actual - expected) <= tolerance),
        })
    return pd.DataFrame(rows)


def write_formal_report(formal_dir: Path, summary: dict, audit: pd.DataFrame) -> None:
    q1 = summary["q1"]
    q2 = summary["q2"]
    q3 = summary["q3"]
    q4 = summary["q4"]
    kw = {row["response"]: row for row in q2["kruskal"]}
    pool = q3["partial_pooling"]
    acceleration = q3["acceleration"]
    text = rf"""# Q1--Q4 v4正式主管线结果摘要

## Q1：统一寿命结构

- 联合选择：$n_0={q1['selected_n0']}$，`{q1['selected_model']}`。
- 外层嵌套LOBO RMSE：{q1['nested_RMSE']:.8f}。
- 40折选择计数：{json.dumps(q1['nested_selection_counts'], ensure_ascii=False)}。
- 150圈与200圈EOL更新中位相对差：{100*q1['life_relative_difference_median']:.2f}%；排序Spearman：{q1['life_150_200_spearman']:.3f}。
- 正式典型长寿命策略：`{q1['long_policy']}`；典型短寿命策略：`{q1['short_policy']}`。

## Q2：全部寿命依赖已重算

- 九策略稳定退化速率置换：$p={kw['stable_rate']['p_permutation']:.6f}$，$\eta_H^2={kw['stable_rate']['eta_H2']:.3f}$。
- 九策略基准EOL置换：$p={kw['life_150']['p_permutation']:.6f}$，$\eta_H^2={kw['life_150']['eta_H2']:.3f}$。总体检验显著不等于任一Dunn--Holm策略对必然显著。
- $A,D\rightarrow r^{{stable}}$ 精确置换：$p={q2['main_rate_regression']['permutation_p']:.6f}$；模型方向须结合Bootstrap和LOPO解释。
- EOL辅助响应仍不是独立证据，也不用于Q4连续优化目标。

## Q3：近期预测与远期稳健化

- 151--200圈正式预测器：`{q3['final_predictor']}`；外层RMSE：{q3['nested_validation']['RMSE']:.8f}。
- 部分池化权重：个体 {pool['final_individual_weight']:.2f}，同策略 {1-pool['final_individual_weight']:.2f}；40折选权计数：{json.dumps(pool['nested_selected_weight_counts'], ensure_ascii=False)}。
- EOL更新中位差：{100*pool['nested_individual_median']:.2f}%→{100*pool['nested_pooled_median']:.2f}%；平均差：{100*pool['nested_individual_mean']:.2f}%→{100*pool['nested_pooled_mean']:.2f}%。
- 加速度辅助头：`{acceleration['selected_model']}`，未来斜率RMSE={acceleration['selected_RMSE']:.8g}；不参与EOL结构门控。

## Q4：优化结论

- 最佳已有策略：`{q4['robust_optimization']['best_existing_policy']}`。
- 局部新候选是否正式推荐：{q4['robust_optimization']['recommend_new_candidate']}。
- 最终推荐：`{q4['robust_optimization']['final_recommendation']}`。

## 全链路审计与证据边界

- Q1→Q2、Q1→Q4、Q3部分池化共 {len(audit)} 项数值传播检查，全部通过：{bool(audit['passed'].all())}。
- 151--200圈SOH和未来斜率具有真实留出观测；80% EOL没有真实标签。EOL截断变化衡量内部稳定性，不等于真实寿命准确率。
"""
    (formal_dir / "Q1-Q4_v4正式重跑结果摘要.md").write_text(text, encoding="utf-8")


def run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def copy_v4_outputs(v4_dir: Path, q3_dir: Path) -> None:
    mapping = {
        "v4_05_Q3部分池化固定权重逐电池.csv": "q3_23_部分池化固定权重逐电池.csv",
        "v4_06_Q3部分池化固定权重汇总.csv": "q3_24_部分池化固定权重汇总.csv",
        "v4_07_Q3部分池化嵌套LOBO逐电池.csv": "q3_25_部分池化嵌套LOBO逐电池.csv",
        "v4_08_Q3部分池化嵌套内层选权.csv": "q3_26_部分池化嵌套内层选权.csv",
        "v4_09_加速度四基线LOBO逐电池.csv": "q3_28_加速度四基线LOBO逐电池.csv",
        "v4_10_加速度四基线oneSE汇总.csv": "q3_29_加速度四基线oneSE汇总.csv",
        "v4_11_九块测试电池未来斜率预测.csv": "q3_30_九块测试电池未来斜率预测.csv",
    }
    for source, target in mapping.items():
        shutil.copy2(v4_dir / source, q3_dir / target)


def refresh_q3_formal_outputs(formal_dir: Path, v4_dir: Path) -> dict:
    q3_dir = formal_dir / "问题3"
    base_path = q3_dir / "q3_15_九块测试电池EOL与统计区间.csv"
    individual_path = q3_dir / "q3_15a_九块测试电池个体EOL与统计区间.csv"
    if individual_path.exists():
        base = pd.read_csv(individual_path)
    else:
        shutil.copy2(base_path, individual_path)
        base = pd.read_csv(base_path)
    pooled = pd.read_csv(v4_dir / "v4_12_九块测试电池EOL与部分池化.csv")
    formal_eol = build_formal_q3_eol_table(base, pooled)
    formal_eol.to_csv(base_path, index=False)
    formal_eol.to_csv(q3_dir / "q3_27_九块测试电池正式部分池化EOL.csv", index=False)
    copy_v4_outputs(v4_dir, q3_dir)

    clean = pd.read_csv(formal_dir / "问题1" / "q1_01_逐循环清洗数据.csv")
    summary = pd.read_csv(ROOT / "battery_summary.csv")
    predictions = pd.read_csv(q3_dir / "q3_14_九块测试电池151_200正式预测.csv")
    state_selection = pd.read_csv(q3_dir / "q3_20_状态关系oneSE选择.csv")
    life_validation, life_metrics = analysis_pipeline.life_summary_validation(
        clean,
        summary,
        predictions,
        formal_eol,
        state_selection,
        "centered_quadratic",
    )
    life_validation.to_csv(q3_dir / "q3_21_九块测试电池完整寿命均值对照.csv", index=False)
    life_metrics.to_csv(q3_dir / "q3_22_完整寿命状态吻合度汇总.csv", index=False)

    v4_summary = json.loads((v4_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        "formal_eol": formal_eol.to_dict("records"),
        "life_summary_metrics": life_metrics.to_dict("records"),
        "partial_pooling": v4_summary["partial_pooling"],
        "acceleration": v4_summary["acceleration"],
        "claim_boundary": v4_summary["claim_boundary"],
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="v4正式主管线：从原始CSV完整重跑Q1--Q4")
    parser.add_argument(
        "--formal-output-dir",
        type=Path,
        default=ROOT / "results" / "正式重跑_20260816_v4",
    )
    parser.add_argument("--skip-q4", action="store_true")
    args = parser.parse_args()
    formal_dir = args.formal_output_dir if args.formal_output_dir.is_absolute() else ROOT / args.formal_output_dir
    formal_dir = formal_dir.resolve()
    v4_dir = formal_dir / "v4方法裁决与Q3稳健化"
    q4_dir = formal_dir / "问题4"

    run_checked([
        sys.executable,
        str(ROOT / "scripts" / "analysis_pipeline.py"),
        "--output-dir",
        str(formal_dir),
    ])
    run_checked([
        sys.executable,
        str(ROOT / "scripts" / "eol_v4_final_arbitration.py"),
        "--source-results",
        str(formal_dir),
        "--output-dir",
        str(v4_dir),
    ])
    q3_extension = refresh_q3_formal_outputs(formal_dir, v4_dir)

    summary_path = formal_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["q3"]["test_life"] = q3_extension["formal_eol"]
    summary["q3"]["life_summary_metrics"] = q3_extension["life_summary_metrics"]
    summary["q3"]["partial_pooling"] = q3_extension["partial_pooling"]
    summary["q3"]["acceleration"] = q3_extension["acceleration"]
    summary["claim_boundary"] = q3_extension["claim_boundary"]

    if not args.skip_q4:
        run_checked([
            sys.executable,
            str(ROOT / "scripts" / "q4_optimization.py"),
            "--source-results",
            str(formal_dir),
            "--output-dir",
            str(q4_dir),
        ])
        summary["q4"] = json.loads((q4_dir / "summary.json").read_text(encoding="utf-8"))

    audit = audit_life_propagation(
        pd.read_csv(formal_dir / "问题1" / "q1_03_四十九块电池指标与基准EOL.csv"),
        pd.read_csv(formal_dir / "问题2" / "q2_04_策略参数与SOC暴露.csv"),
        pd.read_csv(formal_dir / "问题3" / "q3_27_九块测试电池正式部分池化EOL.csv"),
        pd.read_csv(q4_dir / "q4_01_九策略经验Pareto.csv"),
    )
    if not audit["passed"].all():
        raise AssertionError("Cross-question lifetime propagation audit failed")
    audit.to_csv(formal_dir / "Q1-Q4寿命结果全链路一致性审计.csv", index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_formal_report(formal_dir, summary, audit)
    sources = [
        ROOT / "scripts" / "analysis_pipeline.py",
        ROOT / "scripts" / "eol_accuracy_optimization.py",
        ROOT / "scripts" / "eol_v4_final_arbitration.py",
        ROOT / "scripts" / "formal_v4_pipeline.py",
        ROOT / "scripts" / "q4_optimization.py",
        ROOT / "scripts" / "make_figures.R",
    ]
    manifest = pd.DataFrame(
        [{"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in sources]
    )
    manifest.to_csv(formal_dir / "正式主管线源码校验.csv", index=False)
    print(json.dumps({
        "formal_output_dir": str(formal_dir),
        "q1_selected_n0": summary["q1"]["selected_n0"],
        "q1_selected_model": summary["q1"]["selected_model"],
        "q3_partial_pool_weight": summary["q3"]["partial_pooling"]["final_individual_weight"],
        "q4_ran": not args.skip_q4,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
