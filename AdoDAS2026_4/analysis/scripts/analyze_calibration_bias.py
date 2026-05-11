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
from sklearn.metrics import f1_score, roc_auc_score

from analysis.scripts.utils import (
    binary_macro_f1,
    load_calibration_biases,
    load_yaml,
    logit,
    macro_auroc,
    pred_pos_rate,
    read_label_file,
    read_prediction_file,
    save_json,
    sigmoid,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    out_dir = Path(cfg["paths"]["output_dir"]) / "calibration_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("plot", {}).get("dpi", 200))
    cal_cfg = cfg.get("calibration", {})
    shrinks = [float(x) for x in cal_cfg.get("shrink_values", [0.0, 0.25, 0.4, 0.6, 0.8, 1.0])]
    alpha = float(cal_cfg.get("alpha_pos_rate_penalty", 0.7))
    wt = float(cal_cfg.get("target_pos_rate_train_weight", 0.7))
    wv = float(cal_cfg.get("target_pos_rate_val_weight", 0.3))

    lab_cfg = cfg["labels"]
    id_col = lab_cfg.get("id_col", "person_id")
    train_df = read_label_file(
        lab_cfg["train_label_path"], id_col, tuple(lab_cfg.get("class_cols", ["D", "A", "S"]))
    )
    val_df = read_label_file(
        lab_cfg["val_label_path"], id_col, tuple(lab_cfg.get("class_cols", ["D", "A", "S"]))
    )
    train_rate = train_df[["D", "A", "S"]].mean().values
    val_rate = val_df[["D", "A", "S"]].mean().values
    if np.max(np.abs(val_rate - train_rate)) > 0.05:
        wt, wv = 0.8, 0.2
    target_pos = wt * train_rate + wv * val_rate

    exports = cfg.get("exports") or {}
    pred_by_name = {x["name"]: Path(x["path"]) for x in (cfg.get("predictions") or [])}

    summary_rows = []
    recommendations: dict = {}

    for model_name in ("baseline", "dynamic"):
        cal_json = (exports.get(model_name) or {}).get("calibration_json")
        raw_name = f"{model_name}_val_raw"
        if raw_name not in pred_by_name:
            log.warning("Missing %s in predictions", raw_name)
            continue
        bias_full = load_calibration_biases(cal_json or "")
        if bias_full is None:
            log.warning("No biases for %s", model_name)
            bias_full = np.zeros(3)

        pred = read_prediction_file(pred_by_name[raw_name], id_col)
        val_lab = val_df.set_index(id_col)
        merged = pred.set_index(id_col).join(val_lab, how="inner")
        P0 = merged[["p_D", "p_A", "p_S"]].values.astype(np.float64)
        Y = merged[["D", "A", "S"]].values.astype(int)

        best_score = -1e9
        best_shrink = 0.4
        for s in shrinks:
            b = bias_full * s
            Z = logit(P0) + b.reshape(1, -1)
            P = sigmoid(Z)
            f1 = binary_macro_f1(P, Y, 0.5)
            ppos = pred_pos_rate(P, 0.5)
            gap = float(np.mean(np.abs(ppos - target_pos)))
            score = f1 - alpha * gap
            for j, c in enumerate(["D", "A", "S"]):
                summary_rows.append(
                    {
                        "model": model_name,
                        "shrink": s,
                        "class": c,
                        "F1_0.5": float(
                            f1_score(Y[:, j], (P[:, j] >= 0.5).astype(int), zero_division=0.0)
                        ),
                        "pred_pos_0.5": float(ppos[j]),
                        "target_pos_rate": float(target_pos[j]),
                        "pos_rate_gap": float(abs(ppos[j] - target_pos[j])),
                        "AUROC_dim": float(
                            roc_auc_score(Y[:, j], P[:, j])
                            if len(np.unique(Y[:, j])) > 1
                            else float("nan")
                        ),
                        "prob_mean": float(np.mean(P[:, j])),
                        "score": float(f1 - alpha * abs(ppos[j] - target_pos[j])),
                    }
                )
            if score > best_score:
                best_score = score
                best_shrink = s

        recommendations[model_name] = {
            "recommended_shrink": float(best_shrink),
            "recommended_bias": (bias_full * best_shrink).tolist(),
            "score": float(best_score),
        }

        for j, c in enumerate(["D", "A", "S"]):
            f1s, gaps = [], []
            for s in shrinks:
                b = bias_full * s
                P = sigmoid(logit(P0) + b.reshape(1, -1))
                f1s.append(
                    f1_score(Y[:, j], (P[:, j] >= 0.5).astype(int), zero_division=0.0)
                )
                gaps.append(abs(pred_pos_rate(P, 0.5)[j] - target_pos[j]))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(shrinks, f1s, marker="o")
            ax.set_xlabel("shrink")
            ax.set_ylabel("F1@0.5")
            ax.set_title(f"Shrink vs F1 {model_name} class={c}")
            fig.tight_layout()
            fig.savefig(out_dir / f"shrink_f1_curve_{model_name}_{c}.png", dpi=dpi)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(shrinks, gaps, marker="o")
            ax.set_xlabel("shrink")
            ax.set_ylabel("|pred_pos - target|")
            ax.set_title(f"Shrink vs pos gap {model_name} class={c}")
            fig.tight_layout()
            fig.savefig(out_dir / f"shrink_pred_pos_curve_{model_name}_{c}.png", dpi=dpi)
            plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(out_dir / "shrink_calibration_summary.csv", index=False)
    save_json(out_dir / "recommended_bias.json", recommendations)
    log.info("Calibration bias analysis -> %s", out_dir)


if __name__ == "__main__":
    main()
