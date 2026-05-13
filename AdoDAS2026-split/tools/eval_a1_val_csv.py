#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate A1 validation predictions against val.csv.

适配当前 AdoDAS2026_4 代码：
1. infer.py 输出 A1 participant-level csv:
   file_id,p_D,p_A,p_S
   其中 file_id = anon_school_anon_class_anon_pid

2. val.csv 通常是 session-level manifest:
   anon_school, anon_class, anon_pid, session, D, A, S ...
   同一个 participant 有 4 个 session，D/A/S 标签重复。
   本脚本会自动聚合成 participant-level label。

3. 指标对齐 common/runner.py:
   - raw F1@0.5
   - per-class F1[D/A/S]
   - mean F1 = mean(D/A/S F1)
   - macro AUROC
   - bias calibration: sigmoid(logit(p) + bias)，阈值仍为 0.5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


TASKS = ["D", "A", "S"]
PROB_COLS = ["p_D", "p_A", "p_S"]


LABEL_CANDIDATES = {
    "D": ["D", "d", "label_D", "y_D", "D_label", "depression", "Depression", "dep", "DEP"],
    "A": ["A", "a", "label_A", "y_A", "A_label", "anxiety", "Anxiety", "anx", "ANX"],
    "S": ["S", "s", "label_S", "y_S", "S_label", "stress", "Stress", "str", "STR"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv", required=True, help="Path to manifests/val.csv")
    parser.add_argument("--pred_csv", required=True, help="Path to prediction csv, usually val_pred.csv")
    parser.add_argument("--out_dir", required=True, help="Directory to save metric csv/json files")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bias_grid_min", type=float, default=-3.0)
    parser.add_argument("--bias_grid_max", type=float, default=3.0)
    parser.add_argument("--bias_grid_step", type=float, default=0.1)
    parser.add_argument("--thr_grid_step", type=float, default=0.005)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def safe_float_array(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").astype(float).values


def find_label_col(df: pd.DataFrame, task: str) -> str:
    for c in LABEL_CANDIDATES[task]:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find label column for {task}. "
        f"Tried {LABEL_CANDIDATES[task]}. "
        f"Current columns: {list(df.columns)}"
    )


def normalize_id_part(x) -> str:
    """
    保持和 infer.py 的 str(row[col]) 尽量一致。
    因为 val.csv 可能被 pandas 读成 int/float，这里做一点防御。
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_file_id_from_manifest(df: pd.DataFrame, session_level: bool = False) -> pd.Series:
    required = ["anon_school", "anon_class", "anon_pid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"val_csv missing columns: {missing}")

    school = df["anon_school"].map(normalize_id_part)
    cls = df["anon_class"].map(normalize_id_part)
    pid = df["anon_pid"].map(normalize_id_part)

    if session_level:
        if "session" not in df.columns:
            raise ValueError("session_level=True but val_csv has no 'session' column.")
        sess = df["session"].map(normalize_id_part)
        return school + "_" + cls + "_" + pid + "_" + sess

    return school + "_" + cls + "_" + pid


def prepare_labels(val_csv: Path, pred_file_ids: pd.Series) -> pd.DataFrame:
    """
    根据 pred_csv 的 file_id 形式，自动判断 participant-level 还是 session-level。
    当前你的 infer.py 默认 participant-level，所以通常是 participant-level。
    """
    val_df = pd.read_csv(val_csv, dtype=str)

    label_cols = {task: find_label_col(val_df, task) for task in TASKS}

    # 判断预测是否是 session-level：file_id 末尾是否包含 A01/B01/B02/B03
    # participant-level: school_class_pid
    # session-level: school_class_pid_session
    sample_ids = pred_file_ids.dropna().astype(str).head(20).tolist()
    session_like = any(
        sid.endswith("_A01") or sid.endswith("_B01") or sid.endswith("_B02") or sid.endswith("_B03")
        for sid in sample_ids
    )

    val_df = val_df.copy()
    val_df["file_id"] = build_file_id_from_manifest(val_df, session_level=session_like)

    for task in TASKS:
        val_df[task] = pd.to_numeric(val_df[label_cols[task]], errors="coerce").astype(float)

    # participant-level 时，一个人有 4 行 session，标签重复，取 first 即可
    label_df = (
        val_df[["file_id", *TASKS]]
        .dropna(subset=TASKS)
        .groupby("file_id", as_index=False)
        .first()
    )

    for task in TASKS:
        label_df[task] = label_df[task].astype(int)

    return label_df


def prepare_predictions(pred_csv: Path) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_csv, dtype=str)

    if "file_id" not in pred_df.columns:
        if all(c in pred_df.columns for c in ["anon_school", "anon_class", "anon_pid"]):
            pred_df["file_id"] = build_file_id_from_manifest(pred_df, session_level=("session" in pred_df.columns))
        else:
            raise ValueError(
                "pred_csv must contain 'file_id', or anon_school/anon_class/anon_pid columns."
            )

    # 支持概率列 p_D/p_A/p_S
    if all(c in pred_df.columns for c in PROB_COLS):
        for c in PROB_COLS:
            pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce").astype(float)

    # 兼容 logits_D/logits_A/logits_S 或 logit_D/logit_A/logit_S
    else:
        logit_candidates = [
            ["logit_D", "logit_A", "logit_S"],
            ["logits_D", "logits_A", "logits_S"],
            ["D_logit", "A_logit", "S_logit"],
        ]
        found = None
        for cand in logit_candidates:
            if all(c in pred_df.columns for c in cand):
                found = cand
                break
        if found is None:
            raise ValueError(
                f"pred_csv must contain {PROB_COLS}, "
                f"or one of logit column sets. Current columns: {list(pred_df.columns)}"
            )
        logits = pred_df[found].apply(pd.to_numeric, errors="coerce").astype(float).values
        probs = sigmoid(logits)
        for i, c in enumerate(PROB_COLS):
            pred_df[c] = probs[:, i]

    pred_df["file_id"] = pred_df["file_id"].map(normalize_id_part)

    # 如果有重复 file_id，取平均概率，防止 session-level/重复输出导致 merge 多行
    pred_df = (
        pred_df[["file_id", *PROB_COLS]]
        .dropna(subset=PROB_COLS)
        .groupby("file_id", as_index=False)
        .mean()
    )

    return pred_df


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def compute_per_class_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    name: str = "raw",
) -> Tuple[pd.DataFrame, Dict]:
    rows = []

    for i, task in enumerate(TASKS):
        y = y_true[:, i].astype(int)
        p = probs[:, i].astype(float)
        pred = (p > threshold).astype(int)  # 对齐 runner.py 里的 > 0.5

        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        auc = safe_auc(y, p)

        rows.append(
            {
                "name": name,
                "class": task,
                "threshold": threshold,
                "label_pos": float(y.mean()),
                "pred_pos": float(pred.mean()),
                "prob_mean": float(np.mean(p)),
                "prob_std": float(np.std(p)),
                "prob_p10": float(np.quantile(p, 0.10)),
                "prob_p25": float(np.quantile(p, 0.25)),
                "prob_p50": float(np.quantile(p, 0.50)),
                "prob_p75": float(np.quantile(p, 0.75)),
                "prob_p90": float(np.quantile(p, 0.90)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auroc": auc,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "name": name,
        "threshold": threshold,
        "mean_f1": float(df["f1"].mean()),
        "macro_auroc": float(np.nanmean(df["auroc"].values)),
        "D_f1": float(df.loc[df["class"] == "D", "f1"].iloc[0]),
        "A_f1": float(df.loc[df["class"] == "A", "f1"].iloc[0]),
        "S_f1": float(df.loc[df["class"] == "S", "f1"].iloc[0]),
    }
    return df, summary


def threshold_scan(y_true: np.ndarray, probs: np.ndarray, step: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    best_rows = []

    thresholds = np.arange(0.0, 1.0 + 1e-9, step)

    for i, task in enumerate(TASKS):
        y = y_true[:, i].astype(int)
        p = probs[:, i].astype(float)

        best = None
        for thr in thresholds:
            pred = (p > thr).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            precision = precision_score(y, pred, zero_division=0)
            recall = recall_score(y, pred, zero_division=0)
            pred_pos = pred.mean()

            row = {
                "class": task,
                "threshold": float(thr),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "pred_pos": float(pred_pos),
            }
            all_rows.append(row)

            if best is None or f1 > best["f1"]:
                best = row

        best_rows.append(best)

    return pd.DataFrame(all_rows), pd.DataFrame(best_rows)


def calibrate_bias(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid_min: float,
    grid_max: float,
    grid_step: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    对齐 runner.py 的 calibrate_a1_bias：
    对每个类别单独搜索 bias，使 sigmoid(logit + b) 在 0.5 阈值下 F1 最大。
    """
    logits = logit(probs)
    grid = np.arange(grid_min, grid_max + 1e-9, grid_step)

    biases = np.zeros(3, dtype=np.float64)
    rows = []

    for i, task in enumerate(TASKS):
        y = y_true[:, i].astype(int)
        best_f1 = -1.0
        best_b = 0.0
        best_pred_pos = 0.0

        for b in grid:
            p_cal = sigmoid(logits[:, i] + b)
            pred = (p_cal > 0.5).astype(int)
            f1 = f1_score(y, pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_b = float(b)
                best_pred_pos = float(pred.mean())

        biases[i] = best_b
        rows.append(
            {
                "class": task,
                "best_bias": best_b,
                "best_f1": float(best_f1),
                "pred_pos_at_best_bias": best_pred_pos,
                "label_pos": float(y.mean()),
            }
        )

    return biases, pd.DataFrame(rows)


def apply_bias(probs: np.ndarray, biases: np.ndarray, shrink: float = 1.0) -> np.ndarray:
    logits = logit(probs)
    return sigmoid(logits + shrink * biases.reshape(1, -1))


def any_positive_analysis(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """
    分析全0 / 任意阳性结构：
    后续做 any-positive gate 时很有用。
    """
    y_any = (y_true.sum(axis=1) > 0).astype(int)
    p_any_max = probs.max(axis=1)
    p_any_mean = probs.mean(axis=1)
    pred_any = ((probs > threshold).sum(axis=1) > 0).astype(int)

    rows = []
    for name, score in [("max_prob", p_any_max), ("mean_prob", p_any_mean)]:
        auc = safe_auc(y_any, score)
        rows.append(
            {
                "score": name,
                "any_label_pos": float(y_any.mean()),
                "any_pred_pos@0.5": float((score > threshold).mean()),
                "any_f1@0.5": float(f1_score(y_any, (score > threshold).astype(int), zero_division=0)),
                "any_auroc": auc,
            }
        )

    rows.append(
        {
            "score": "three_logits_any_pred",
            "any_label_pos": float(y_any.mean()),
            "any_pred_pos@0.5": float(pred_any.mean()),
            "any_f1@0.5": float(f1_score(y_any, pred_any, zero_division=0)),
            "any_auroc": float("nan"),
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    val_csv = Path(args.val_csv)
    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = prepare_predictions(pred_csv)
    label_df = prepare_labels(val_csv, pred_df["file_id"])

    merged = label_df.merge(pred_df, on="file_id", how="inner")

    if len(merged) == 0:
        raise RuntimeError(
            "No matched rows between val_csv and pred_csv. "
            "Please check file_id construction."
        )

    missing_label = set(label_df["file_id"]) - set(merged["file_id"])
    missing_pred = set(pred_df["file_id"]) - set(merged["file_id"])

    y_true = merged[TASKS].values.astype(int)
    probs = merged[PROB_COLS].values.astype(float)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    # 1. raw metrics
    raw_df, raw_summary = compute_per_class_metrics(
        y_true, probs, threshold=args.threshold, name="raw"
    )

    # 2. threshold scan
    scan_df, best_thr_df = threshold_scan(y_true, probs, step=args.thr_grid_step)

    # 3. bias calibration
    biases, bias_df = calibrate_bias(
        y_true,
        probs,
        grid_min=args.bias_grid_min,
        grid_max=args.bias_grid_max,
        grid_step=args.bias_grid_step,
    )

    shrink_rows = []
    shrink_metric_dfs = []

    for shrink in [0.0, 0.3, 0.5, 0.6, 0.7, 1.0]:
        cal_probs = apply_bias(probs, biases, shrink=shrink)
        mdf, summary = compute_per_class_metrics(
            y_true,
            cal_probs,
            threshold=args.threshold,
            name=f"bias_shrink_{shrink:.1f}",
        )
        summary["shrink"] = shrink
        summary["bias_D"] = float(biases[0] * shrink)
        summary["bias_A"] = float(biases[1] * shrink)
        summary["bias_S"] = float(biases[2] * shrink)
        shrink_rows.append(summary)
        shrink_metric_dfs.append(mdf)

    shrink_summary_df = pd.DataFrame(shrink_rows)
    all_metric_df = pd.concat([raw_df, *shrink_metric_dfs], ignore_index=True)

    # 4. probability summary
    prob_rows = []
    for i, task in enumerate(TASKS):
        p = probs[:, i]
        prob_rows.append(
            {
                "class": task,
                "label_pos": float(y_true[:, i].mean()),
                "prob_mean": float(np.mean(p)),
                "prob_std": float(np.std(p)),
                "prob_p10": float(np.quantile(p, 0.10)),
                "prob_p25": float(np.quantile(p, 0.25)),
                "prob_p50": float(np.quantile(p, 0.50)),
                "prob_p75": float(np.quantile(p, 0.75)),
                "prob_p90": float(np.quantile(p, 0.90)),
                "pred_pos@0.5": float((p > 0.5).mean()),
            }
        )
    prob_summary_df = pd.DataFrame(prob_rows)

    # 5. any-positive analysis
    any_df = any_positive_analysis(y_true, probs, threshold=args.threshold)

    # save files
    merged.to_csv(out_dir / "merged_val_predictions.csv", index=False)
    raw_df.to_csv(out_dir / "metrics_raw_per_class.csv", index=False)
    all_metric_df.to_csv(out_dir / "metrics_all_bias_shrink_per_class.csv", index=False)
    shrink_summary_df.to_csv(out_dir / "metrics_bias_shrink_summary.csv", index=False)
    scan_df.to_csv(out_dir / "threshold_scan.csv", index=False)
    best_thr_df.to_csv(out_dir / "best_thresholds.csv", index=False)
    bias_df.to_csv(out_dir / "best_biases.csv", index=False)
    prob_summary_df.to_csv(out_dir / "prob_summary.csv", index=False)
    any_df.to_csv(out_dir / "any_positive_analysis.csv", index=False)

    summary = {
        "n_label_rows": int(len(label_df)),
        "n_pred_rows": int(len(pred_df)),
        "n_matched_rows": int(len(merged)),
        "n_missing_label_file_ids": int(len(missing_label)),
        "n_missing_pred_file_ids": int(len(missing_pred)),
        "raw": raw_summary,
        "best_biases": {
            "D": float(biases[0]),
            "A": float(biases[1]),
            "S": float(biases[2]),
        },
        "best_shrink_by_val_mean_f1": shrink_summary_df.sort_values(
            "mean_f1", ascending=False
        ).iloc[0].to_dict(),
        "outputs": {
            "merged": str(out_dir / "merged_val_predictions.csv"),
            "raw_metrics": str(out_dir / "metrics_raw_per_class.csv"),
            "bias_shrink_summary": str(out_dir / "metrics_bias_shrink_summary.csv"),
            "threshold_scan": str(out_dir / "threshold_scan.csv"),
            "best_thresholds": str(out_dir / "best_thresholds.csv"),
            "best_biases": str(out_dir / "best_biases.csv"),
            "prob_summary": str(out_dir / "prob_summary.csv"),
            "any_positive_analysis": str(out_dir / "any_positive_analysis.csv"),
        },
    }

    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Matched participants/samples: {len(merged)}")
    print("-" * 80)
    print("RAW:")
    print(
        f"mean_f1={raw_summary['mean_f1']:.6f} "
        f"AUROC={raw_summary['macro_auroc']:.6f} "
        f"D/A/S={raw_summary['D_f1']:.4f}/"
        f"{raw_summary['A_f1']:.4f}/"
        f"{raw_summary['S_f1']:.4f}"
    )
    print("-" * 80)
    print("Best biases from val:")
    print(bias_df.to_string(index=False))
    print("-" * 80)
    print("Bias shrink summary:")
    print(
        shrink_summary_df[
            ["shrink", "mean_f1", "macro_auroc", "D_f1", "A_f1", "S_f1", "bias_D", "bias_A", "bias_S"]
        ].to_string(index=False)
    )
    print("-" * 80)
    print("Best thresholds:")
    print(best_thr_df.to_string(index=False))
    print("-" * 80)
    print(f"Saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()