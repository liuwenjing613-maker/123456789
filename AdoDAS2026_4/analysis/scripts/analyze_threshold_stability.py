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

from analysis.scripts.utils import load_yaml, read_label_file, read_prediction_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def binary_metrics_vs_thresholds(probs: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized binary metrics for each threshold (probs, y same length)."""
    y = y.astype(np.int64)
    pred = probs[:, None] >= thresholds[None, :]
    tp = (pred & (y[:, None] == 1)).sum(axis=0).astype(np.float64)
    fp = (pred & (y[:, None] == 0)).sum(axis=0).astype(np.float64)
    fn = ((~pred) & (y[:, None] == 1)).sum(axis=0).astype(np.float64)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    pred_pos = pred.mean(axis=0).astype(np.float64)
    return {"f1": f1, "precision": prec, "recall": rec, "pred_pos_rate": pred_pos}


def scan_threshold_1d(probs: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    m = binary_metrics_vs_thresholds(probs, y, thresholds)
    return pd.DataFrame(
        {
            "threshold": thresholds.astype(np.float64),
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "pred_pos_rate": m["pred_pos_rate"],
        }
    )


def bootstrap_best_threshold(
    probs: np.ndarray, y: np.ndarray, thresholds: np.ndarray, rng: np.random.Generator, B: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per bootstrap sample: best threshold by F1 (same rule as before, fully vectorized)."""
    n = len(probs)
    best_thrs = np.empty(B, dtype=np.float64)
    best_f1s = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        pv, yv = probs[idx], y[idx]
        f1s = binary_metrics_vs_thresholds(pv, yv, thresholds)["f1"]
        j = int(np.argmax(f1s))
        best_thrs[b] = float(thresholds[j])
        best_f1s[b] = float(f1s[j])
    return best_thrs, best_f1s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    out_dir = Path(cfg["paths"]["output_dir"]) / "threshold_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("plot", {}).get("dpi", 200))
    th_cfg = cfg.get("threshold", {})
    tmin = float(th_cfg.get("min_thr", 0.05))
    tmax = float(th_cfg.get("max_thr", 0.95))
    nt = int(th_cfg.get("num_thr", 181))
    B = int(th_cfg.get("bootstrap_n", 1000))
    seed = int(th_cfg.get("bootstrap_seed", 42))
    thresholds = np.linspace(tmin, tmax, nt)

    lab_cfg = cfg["labels"]
    id_col = lab_cfg.get("id_col", "person_id")
    val_labels = read_label_file(
        lab_cfg["val_label_path"], id_col, tuple(lab_cfg.get("class_cols", ["D", "A", "S"]))
    )
    val_label_map = val_labels.set_index(id_col)

    scan_parts = []
    boot_parts = []
    classes = ["D", "A", "S"]

    val_specs = [
        s
        for s in (cfg.get("predictions") or [])
        if s.get("split") == "val" and Path(s["path"]).is_file()
    ]
    log.info(
        "Threshold stability: %d val prediction file(s), bootstrap_n=%d, num_thr=%d",
        len(val_specs),
        B,
        nt,
    )

    rng = np.random.default_rng(seed)

    for spec in val_specs:
        model = spec.get("model", "")
        calib = spec.get("calib", "")
        pred = read_prediction_file(Path(spec["path"]), id_col)
        merged = pred.set_index(id_col).join(val_label_map, how="inner")
        if len(merged) < 10:
            log.warning("Too few merged rows for %s", spec["name"])
            continue
        for j, c in enumerate(classes):
            probs = merged[f"p_{c}"].values.astype(np.float64)
            y = merged[c].values.astype(int)
            df_scan = scan_threshold_1d(probs, y, thresholds)
            df_scan.insert(0, "class", c)
            df_scan.insert(0, "calib", calib)
            df_scan.insert(0, "model", model)
            df_scan.insert(0, "name", spec["name"])
            scan_parts.append(df_scan)

            # curves
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df_scan["threshold"], df_scan["f1"], label="F1")
            ax.set_xlabel("threshold")
            ax.set_title(f"F1 vs threshold {model} {calib} class={c}")
            fig.tight_layout()
            fig.savefig(out_dir / f"f1_threshold_curve_{model}_{calib}_{c}.png", dpi=dpi)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df_scan["threshold"], df_scan["pred_pos_rate"], label="pred_pos")
            ax.set_xlabel("threshold")
            ax.set_title(f"pred_pos vs threshold {model} {calib} class={c}")
            fig.tight_layout()
            fig.savefig(out_dir / f"pred_pos_threshold_curve_{model}_{calib}_{c}.png", dpi=dpi)
            plt.close(fig)

            # bootstrap best threshold (vectorized inner threshold loop)
            best_thrs, best_f1s = bootstrap_best_threshold(probs, y, thresholds, rng, B)
            boot_parts.append(
                {
                    "name": spec["name"],
                    "model": model,
                    "calib": calib,
                    "class": c,
                    "best_thr_mean": float(best_thrs.mean()),
                    "best_thr_std": float(best_thrs.std()),
                    "best_thr_p05": float(np.percentile(best_thrs, 5)),
                    "best_thr_p25": float(np.percentile(best_thrs, 25)),
                    "best_thr_p50": float(np.percentile(best_thrs, 50)),
                    "best_thr_p75": float(np.percentile(best_thrs, 75)),
                    "best_thr_p95": float(np.percentile(best_thrs, 95)),
                    "best_f1_mean": float(best_f1s.mean()),
                    "best_f1_std": float(best_f1s.std()),
                }
            )

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(best_thrs, bins=30, density=True)
            ax.set_title(f"Bootstrap best threshold {model} {calib} class={c}")
            fig.tight_layout()
            fig.savefig(out_dir / f"bootstrap_threshold_hist_{model}_{calib}_{c}.png", dpi=dpi)
            plt.close(fig)

    scan_path = out_dir / "threshold_scan.csv"
    if scan_parts:
        pd.concat(scan_parts, ignore_index=True).to_csv(scan_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "name",
                "model",
                "calib",
                "class",
                "threshold",
                "f1",
                "precision",
                "recall",
                "pred_pos_rate",
            ]
        ).to_csv(scan_path, index=False)

    boot_path = out_dir / "threshold_bootstrap.csv"
    if boot_parts:
        pd.DataFrame(boot_parts).to_csv(boot_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "name",
                "model",
                "calib",
                "class",
                "best_thr_mean",
                "best_thr_std",
                "best_thr_p05",
                "best_thr_p25",
                "best_thr_p50",
                "best_thr_p75",
                "best_thr_p95",
                "best_f1_mean",
                "best_f1_std",
            ]
        ).to_csv(boot_path, index=False)

    log.info("Threshold analysis -> %s", out_dir)


if __name__ == "__main__":
    main()
