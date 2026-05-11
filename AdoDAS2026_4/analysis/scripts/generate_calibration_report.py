#!/usr/bin/env python3
"""Assemble analysis/outputs into reports/a1_dataset_calibration_report.md (md §13–§14)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.scripts.utils import load_yaml

TASKS = ("D", "A", "S")


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "N/A"
    return f"{float(x):.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    out_dir = Path(cfg["paths"]["output_dir"])
    report_dir = Path(cfg["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "a1_dataset_calibration_report.md"

    lines: list[str] = []
    lines.append("# A1 数据集与校准分析报告\n")
    lines.append("（本报告由 `generate_calibration_report.py` 根据分析输出自动生成。）\n")

    lines.append("## 总览\n")
    lines.append("本次分析包含：\n")
    lines.append("1. 标签分布分析\n")
    lines.append("2. 预测概率分布分析\n")
    lines.append("3. threshold 稳定性分析\n")
    lines.append("4. 校准 bias 分析\n")
    lines.append("5. session 质量分析\n")
    lines.append("6. feature drift 分析\n")
    lines.append("7. ensemble 分析\n")

    # --- Labels (§13.2 #1)
    lines.append("\n## 关键结论\n")
    lab_csv = _read_csv(out_dir / "label_analysis" / "label_distribution.csv")
    if lab_csv is not None and len(lab_csv) >= 2:
        tr = lab_csv[lab_csv["split"] == "train"].iloc[0]
        va = lab_csv[lab_csv["split"] == "val"].iloc[0]
        lines.append("### 1. train / val 中 D/A/S 正例率差异\n")
        for c in TASKS:
            pt = float(tr[f"{c}_pos_rate"])
            pv = float(va[f"{c}_pos_rate"])
            lines.append(f"- **{c}**: train = {_fmt(pt)}, val = {_fmt(pv)}, diff = {_fmt(abs(pv - pt))}\n")
    else:
        lines.append("### 1. train / val 正例率\n")
        lines.append("- （未找到 `label_distribution.csv`）\n")

    # --- Threshold bootstrap (§13.2 #2)
    boot = _read_csv(out_dir / "threshold_analysis" / "threshold_bootstrap.csv")
    lines.append("\n### 2. threshold 稳定性（bootstrap 最优阈值）\n")
    if boot is not None and len(boot) > 0:
        sub = boot[(boot["model"] == "baseline") & (boot["calib"] == "raw")]
        if sub.empty:
            sub = boot
        for c in TASKS:
            r = sub[sub["class"] == c]
            if r.empty:
                lines.append(f"- **{c}**: 无数据\n")
                continue
            r0 = r.iloc[0]
            std = float(r0["best_thr_std"])
            p05 = float(r0["best_thr_p05"])
            p95 = float(r0["best_thr_p95"])
            lines.append(f"- **{c}**: std = {_fmt(std)}, p05~p95 = {_fmt(p05)}~{_fmt(p95)}\n")
    elif boot is not None:
        lines.append("- `threshold_bootstrap.csv` 无数据行（通常缺少 val 预测）。\n")
    else:
        lines.append("- （未找到 `threshold_bootstrap.csv`）\n")

    # --- Calibration / pred_pos (§13.2 #3)
    lines.append("\n### 3. 校准风险（full calibration 与 pred_pos）\n")
    shrink = _read_csv(out_dir / "calibration_analysis" / "shrink_calibration_summary.csv")
    prob = _read_csv(out_dir / "prediction_analysis" / "prob_summary.csv")
    if shrink is not None and len(shrink) > 0:
        full = shrink[np.isclose(shrink["shrink"].astype(float), 1.0)]
        light = shrink[np.isclose(shrink["shrink"].astype(float), 0.4)]
        lines.append("- shrink=1.0（full）与 shrink=0.4（示例轻量）时各类 `pred_pos_0.5` 与 `target_pos_rate` 对比见 `shrink_calibration_summary.csv`。\n")
        if not full.empty and not light.empty:
            for model in ("baseline", "dynamic"):
                f_m = full[full["model"] == model]
                l_m = light[light["model"] == model]
                if f_m.empty or l_m.empty:
                    continue
                lines.append(f"- **{model}**: full 平均 |pred_pos−target| ≈ ")
                lines.append(_fmt(float(f_m["pos_rate_gap"].mean())) + "；shrink=0.4 时 ≈ ")
                lines.append(_fmt(float(l_m["pos_rate_gap"].mean())) + "\n")
        lines.append("- 若 full 校准下 pred_pos 相对 target 明显偏大/偏小，则存在“过度偏移”风险。\n")
    elif shrink is not None:
        lines.append("- `shrink_calibration_summary.csv` 存在但无数据行（通常缺少 val 预测）。\n")
    else:
        lines.append("- （未找到 shrink 汇总表）\n")
    if prob is not None:
        cal = prob[(prob["split"] == "val") & (prob["calib"] == "calibrated")]
        if not cal.empty:
            lines.append("- val **calibrated** 预测：`prob_summary.csv` 中 `pred_pos@0.5` 列可对照标签正例率。\n")

    # --- Drift (§13.2 #4)
    lines.append("\n### 4. 动态特征 / 特征组漂移\n")
    drift = _read_csv(out_dir / "feature_drift" / "feature_drift_score.csv")
    if drift is not None and len(drift):
        gmean = drift.groupby("feature_group")["test_drift_score"].mean().sort_values(ascending=False)
        top = gmean.head(5)
        lines.append("- test 相对 train 漂移（按 feature_group 平均 test_drift_score）较高的组：\n")
        for g, v in top.items():
            lines.append(f"  - {g}: {_fmt(float(v))}\n")
        hi = gmean.loc[gmean.index.isin(["body_pose", "global_motion"])]
        if not hi.empty:
            lines.append("- **body_pose / global_motion** 均值：" + ", ".join(f"{i}={_fmt(float(hi[i]))}" for i in hi.index) + "\n")
    else:
        lines.append("- （未找到 `feature_drift_score.csv`）\n")

    # --- Ensemble & strategy (§13.2 #5)
    lines.append("\n### 5. Ensemble 与推荐权重\n")
    ens_path = out_dir / "ensemble_analysis" / "recommended_ensemble.json"
    if ens_path.is_file():
        with open(ens_path) as f:
            ens = json.load(f)
        lines.append(
            f"- 推荐 baseline 权重 **{ens.get('baseline_weight', 'N/A')}**，dynamic **{ens.get('dynamic_weight', 'N/A')}**（pred_calib={ens.get('pred_calib', '?')}）。\n"
        )
        lines.append(f"- 说明：{ens.get('reason', '')}\n")
        if ens.get("note"):
            lines.append(f"- 备注：{ens['note']}\n")
        if ens.get("submission_csv"):
            lines.append(f"- 融合测试集提交示例：`{ens['submission_csv']}`\n")
    else:
        lines.append("- （未找到 `recommended_ensemble.json`）\n")

    lines.append("\n### 6. 推荐提交策略（摘要）\n")
    strat: list[str] = []

    if lab_csv is not None and len(lab_csv) >= 2:
        tr = lab_csv[lab_csv["split"] == "train"].iloc[0]
        va = lab_csv[lab_csv["split"] == "val"].iloc[0]
        if any(abs(float(va[f"{c}_pos_rate"]) - float(tr[f"{c}_pos_rate"])) > 0.05 for c in TASKS):
            strat.append("train/val 标签差异较大：倾向 **shrink calibration（0.25–0.4）**、避免盲信 full val 校准；可 **baseline 或 ensemble**（§14.1）。")

    unstable = False
    if boot is not None and len(boot):
        sub = boot[(boot["model"] == "baseline") & (boot["calib"] == "raw")]
        if sub.empty:
            sub = boot
        for c in TASKS:
            r = sub[sub["class"] == c]
            if r.empty:
                continue
            r0 = r.iloc[0]
            if float(r0["best_thr_std"]) > 0.08 or float(r0["best_thr_p95"]) - float(r0["best_thr_p05"]) > 0.20:
                unstable = True
                break
    if unstable:
        strat.append("threshold bootstrap 不稳定：**no / light calibration**，更信 raw AUROC/F1（§14.2）。")

    val_m = _read_csv(out_dir / "prediction_analysis" / "val_metric_summary.csv")
    if val_m is not None and len(val_m):
        def row(model: str, calib: str) -> pd.Series | None:
            r = val_m[(val_m["model"] == model) & (val_m["calib"] == calib)]
            return r.iloc[0] if len(r) else None

        rb, rd = row("baseline", "raw"), row("dynamic", "raw")
        cb, cd = row("baseline", "calibrated"), row("dynamic", "calibrated")
        if (
            rb is not None
            and rd is not None
            and cb is not None
            and cd is not None
            and float(rd["F1_macro_0.5"]) <= float(rb["F1_macro_0.5"]) + 0.01
            and float(cd["F1_macro_0.5"]) > float(cb["F1_macro_0.5"]) + 0.02
        ):
            strat.append("dynamic 的 raw F1 未明显高于 baseline，但 calibrated F1 明显更高：**dynamic 勿主导**，建议 **baseline+dynamic ensemble，baseline 权重 0.6–0.7**（§14.3）。")

    if drift is not None and len(drift):
        gmean = drift.groupby("feature_group")["test_drift_score"].mean()
        for g in ("body_pose", "global_motion"):
            if g in gmean.index and float(gmean[g]) > 1.5:
                strat.append(f"**{g}** drift 偏高：可考虑配置中弱化或 exclude 动态组，或加强 selective dynamics（§14.4）。")
                break

    if prob is not None:
        tr_pr = lab_csv[lab_csv["split"] == "train"].iloc[0] if lab_csv is not None and len(lab_csv) >= 2 else None
        test_rows = prob[prob["split"] == "test"]
        if tr_pr is not None and not test_rows.empty:
            for c in TASKS:
                tmean = float(tr_pr[f"{c}_pos_rate"])
                for _, r in test_rows.iterrows():
                    key = "pred_pos@0.5"
                    col = f"{key}"
                    if col not in r.index:
                        continue
                    pv = float(r[col])
                    if abs(pv - tmean) > 0.12:
                        strat.append(f"test **{r.get('name', '')}** 上 {c} 的 pred_pos@0.5 与 train 正例率差距较大，需检查校准/偏差（§14.5）。")
                        break

    if ens_path.is_file():
        with open(ens_path) as f:
            ens = json.load(f)
        strat.append(
            f"ensemble 脚本推荐权重 baseline={ens.get('baseline_weight')}（见 `recommended_ensemble.json`）。"
        )

    if not strat:
        strat.append("未触发明确规则；请结合各子目录 CSV/图做人工复核。")
    for s in strat:
        lines.append(f"- {s}\n")

    lines.append("\n## 输出索引\n")
    lines.append(f"- 根目录：`{out_dir}`\n")
    lines.append(
        "- 子目录：`label_analysis/`、`prediction_analysis/`、`threshold_analysis/`、`calibration_analysis/`、`session_quality/`、`feature_drift/`、`ensemble_analysis/`。\n"
    )

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
