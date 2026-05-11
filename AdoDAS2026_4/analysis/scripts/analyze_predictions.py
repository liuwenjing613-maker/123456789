#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.scripts.utils import (
    binary_macro_f1,
    load_yaml,
    macro_auprc,
    macro_auroc,
    per_class_metrics_at_threshold,
    pred_pos_rate,
    read_label_file,
    read_prediction_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

TASKS = ("D", "A", "S")


def prob_stats(probs: np.ndarray) -> dict:
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    row = {
        "prob_mean": float(np.mean(probs)),
        "prob_std": float(np.std(probs)),
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
    }
    for q in qs:
        row[f"prob_p{q:02d}"] = float(np.percentile(probs, q))
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        row[f"pred_pos@{thr}"] = float(np.mean(probs >= thr))
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    out_dir = Path(cfg["paths"]["output_dir"]) / "prediction_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("plot", {}).get("dpi", 200))
    lab_cfg = cfg["labels"]
    id_col = lab_cfg.get("id_col", "person_id")

    val_labels = read_label_file(
        lab_cfg["val_label_path"], id_col, tuple(lab_cfg.get("class_cols", ["D", "A", "S"]))
    )
    val_label_map = val_labels.set_index(id_col)

    prob_rows = []
    metric_rows = []

    pred_specs = cfg.get("predictions") or []

    hist_by_class: dict[str, list[tuple[str, np.ndarray]]] = {t: [] for t in TASKS}

    for spec in pred_specs:
        name = spec["name"]
        path = Path(spec["path"])
        if not path.is_file():
            log.error("Missing prediction file: %s (%s)", path, name)
            continue
        pred = read_prediction_file(path, id_col)
        for j, t in enumerate(TASKS):
            col = f"p_{t}"
            probs = pred[col].values.astype(np.float64)
            base = {
                "name": name,
                "model": spec.get("model", ""),
                "split": spec.get("split", ""),
                "calib": spec.get("calib", ""),
                "class": t,
            }
            st = prob_stats(probs)
            prob_rows.append({**base, **st})
            hist_by_class[t].append((name, probs))

        if spec.get("split") == "val":
            merged = pred.set_index(id_col).join(val_label_map, how="inner")
            if len(merged) == 0:
                log.warning("No overlap val predictions with labels for %s", name)
                continue
            P = merged[["p_D", "p_A", "p_S"]].values.astype(np.float64)
            Y = merged[["D", "A", "S"]].values.astype(int)
            m = per_class_metrics_at_threshold(P, Y, 0.5)
            metric_rows.append(
                {
                    "name": name,
                    "model": spec.get("model", ""),
                    "calib": spec.get("calib", ""),
                    "label_pos_rate_D": float(Y[:, 0].mean()),
                    "label_pos_rate_A": float(Y[:, 1].mean()),
                    "label_pos_rate_S": float(Y[:, 2].mean()),
                    "F1_macro_0.5": binary_macro_f1(P, Y, 0.5),
                    "AUROC_macro": macro_auroc(P, Y),
                    "AUPRC_macro": macro_auprc(P, Y),
                    "pred_pos_D_0.5": float(pred_pos_rate(P, 0.5)[0]),
                    "pred_pos_A_0.5": float(pred_pos_rate(P, 0.5)[1]),
                    "pred_pos_S_0.5": float(pred_pos_rate(P, 0.5)[2]),
                    "f1_D": m["f1"][0],
                    "f1_A": m["f1"][1],
                    "f1_S": m["f1"][2],
                    "precision_D": m["precision"][0],
                    "recall_D": m["recall"][0],
                }
            )

    pd.DataFrame(prob_rows).to_csv(out_dir / "prob_summary.csv", index=False)
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(out_dir / "val_metric_summary.csv", index=False)

    # Histograms per class
    for t in TASKS:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, arr in hist_by_class[t]:
            ax.hist(arr, bins=40, alpha=0.45, density=True, label=name[:40])
        ax.set_title(f"Probability histogram class={t}")
        ax.set_xlabel("p")
        ax.legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"prob_hist_all_{t}.png", dpi=dpi)
        plt.close(fig)

    # Boxplot: merge val raw files by model
    val_raw = [spec for spec in pred_specs if spec.get("split") == "val" and spec.get("calib") == "raw"]
    for t in TASKS:
        series = []
        labels = []
        for spec in val_raw:
            path = Path(spec["path"])
            if not path.is_file():
                continue
            df = read_prediction_file(path, id_col)
            series.append(df[f"p_{t}"].values)
            labels.append(spec.get("model", spec["name"]))
        if len(series) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(series, tick_labels=labels)
        ax.set_title(f"Val raw prob boxplot class={t}")
        fig.tight_layout()
        fig.savefig(out_dir / f"prob_boxplot_by_model_{t}.png", dpi=dpi)
        plt.close(fig)

    log.info("Prediction analysis -> %s", out_dir)


if __name__ == "__main__":
    main()
