"""Evaluate A1 participant-level predictions against a labeled manifest."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.dataset import A1_COLS
from .utils.metrics import binary_f1, macro_auroc, per_class_f1

log = logging.getLogger(__name__)

TASK_NAMES = ("D", "A", "S")


def _label_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    if {"D", "A", "S"}.issubset(df.columns):
        return "D", "A", "S"
    if set(A1_COLS).issubset(df.columns):
        return A1_COLS[0], A1_COLS[1], A1_COLS[2]
    raise KeyError("manifest must contain D/A/S or y_D/y_A/y_S columns")


def participant_table_from_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """One row per participant with file_id and binary labels (matches infer file_id)."""
    df = pd.read_csv(manifest_path)
    d_col, a_col, s_col = _label_columns(df)
    rows: list[dict[str, Any]] = []
    group_cols = ["anon_school", "anon_class", "anon_pid"]
    for (school, cls, pid), grp in df.groupby(group_cols):
        row = grp.iloc[0]
        rows.append({
            "file_id": f"{school}_{cls}_{pid}",
            "y_D": float(row[d_col]),
            "y_A": float(row[a_col]),
            "y_S": float(row[s_col]),
        })
    return pd.DataFrame(rows)


def evaluate_a1_arrays(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """probs/labels shape (N, 3) for D/A/S."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.shape != labels.shape or probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected probs/labels (N,3), got {probs.shape} vs {labels.shape}")

    pcf1 = per_class_f1(probs, labels, threshold=threshold)
    metrics: dict[str, Any] = {
        "n": int(len(labels)),
        "threshold": float(threshold),
        "mean_f1": float(binary_f1(probs, labels, threshold=threshold)),
        "macro_auroc": float(macro_auroc(probs, labels)),
        "D_f1": float(pcf1[0]),
        "A_f1": float(pcf1[1]),
        "S_f1": float(pcf1[2]),
    }
    preds = (probs >= threshold).astype(int)
    for t, name in enumerate(TASK_NAMES):
        gt = labels[:, t]
        pr = preds[:, t]
        tp = int(((pr == 1) & (gt == 1)).sum())
        metrics[f"{name}_gt_pos"] = float(gt.mean())
        metrics[f"{name}_pred_pos"] = float(pr.mean())
        metrics[f"{name}_prob_mean"] = float(probs[:, t].mean())
        metrics[f"{name}_precision"] = float(tp / max(int(pr.sum()), 1))
        metrics[f"{name}_recall"] = float(tp / max(int(gt.sum()), 1))
    return metrics


def evaluate_a1_pred_csv(
    pred_csv: str | Path,
    manifest_csv: str | Path,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    pred_path = Path(pred_csv)
    manifest_path = Path(manifest_csv)
    pred_df = pd.read_csv(pred_path)
    for col in ("file_id", "p_D", "p_A", "p_S"):
        if col not in pred_df.columns:
            raise KeyError(f"Prediction CSV missing column {col}: {pred_path}")

    labels_df = participant_table_from_manifest(manifest_path)
    merged = pred_df.merge(labels_df, on="file_id", how="inner")
    missing_pred = len(labels_df) - len(merged)
    extra_pred = len(pred_df) - len(merged)
    if missing_pred > 0:
        log.warning(
            "%d / %d manifest participants missing from predictions (%s)",
            missing_pred,
            len(labels_df),
            pred_path.name,
        )
    if extra_pred > 0:
        log.warning(
            "%d prediction file_ids not in manifest (dropped)",
            extra_pred,
        )
    if merged.empty:
        raise RuntimeError(
            f"No overlapping file_id between {pred_path} and {manifest_path}"
        )

    probs = merged[["p_D", "p_A", "p_S"]].values.astype(np.float64)
    labels = merged[["y_D", "y_A", "y_S"]].values.astype(np.float64)
    metrics = evaluate_a1_arrays(probs, labels, threshold=threshold)
    metrics["pred_csv"] = str(pred_path.resolve())
    metrics["manifest_csv"] = str(manifest_path.resolve())
    metrics["n_merged"] = int(len(merged))
    metrics["n_manifest"] = int(len(labels_df))
    metrics["n_pred"] = int(len(pred_df))
    return metrics


def write_a1_metrics_json(metrics: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return out


def format_a1_metrics_log(metrics: dict[str, Any], *, title: str = "A1 val metrics") -> str:
    lines = [
        f"{title}: mean_f1={metrics['mean_f1']:.4f} macro_auroc={metrics['macro_auroc']:.4f} "
        f"(n={metrics.get('n_merged', metrics.get('n', '?'))})",
        f"  per-class F1: D={metrics['D_f1']:.4f} A={metrics['A_f1']:.4f} S={metrics['S_f1']:.4f}",
    ]
    for name in TASK_NAMES:
        lines.append(
            f"  {name}: gt_pos={metrics[f'{name}_gt_pos']:.3f} "
            f"pred_pos={metrics[f'{name}_pred_pos']:.3f} "
            f"P={metrics[f'{name}_precision']:.3f} R={metrics[f'{name}_recall']:.3f}"
        )
    return "\n".join(lines)


def log_a1_metrics(metrics: dict[str, Any], *, title: str = "A1 val metrics") -> None:
    log.info(format_a1_metrics_log(metrics, title=title))
