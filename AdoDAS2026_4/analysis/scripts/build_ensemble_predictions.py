#!/usr/bin/env python3
"""Ensemble baseline + dynamic probabilities: p = w * p_base + (1-w) * p_dyn."""
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from analysis.scripts.utils import (
    binary_macro_f1,
    ensure_dir,
    load_yaml,
    macro_auroc,
    pred_pos_rate,
    read_label_file,
    read_prediction_file,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

TASKS = ("D", "A", "S")


def find_pred_path(cfg: dict, model: str, split: str, calib: str) -> Path | None:
    for spec in cfg.get("predictions") or []:
        if spec.get("model") == model and spec.get("split") == split and spec.get("calib") == calib:
            return Path(spec["path"])
    return None


def prob_quantiles(probs: np.ndarray) -> dict[str, float]:
    return {
        "prob_mean": float(np.mean(probs)),
        "prob_p50": float(np.percentile(probs, 50)),
        "prob_p90": float(np.percentile(probs, 90)),
    }


def ensemble_probs(pb: np.ndarray, pdyn: np.ndarray, w: float) -> np.ndarray:
    return w * pb + (1.0 - w) * pdyn


def per_class_f1_prec_rec(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> tuple[list[float], list[float], list[float]]:
    pred = (p >= thr).astype(int)
    f1s, precs, recs = [], [], []
    for j in range(3):
        f1s.append(float(f1_score(y[:, j], pred[:, j], zero_division=0.0)))
        precs.append(float(precision_score(y[:, j], pred[:, j], zero_division=0.0)))
        recs.append(float(recall_score(y[:, j], pred[:, j], zero_division=0.0)))
    return f1s, precs, recs


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> list[float]:
    out = []
    for j in range(3):
        col = y[:, j]
        if len(np.unique(col)) < 2:
            out.append(float("nan"))
        else:
            out.append(float(roc_auc_score(col, p[:, j])))
    return out


def recommend_weight(
    val_df: pd.DataFrame,
    val_rates: np.ndarray,
    weights: list[float],
) -> tuple[float, dict]:
    """Prefer baseline-heavy w in {0.6, 0.7}; penalize pred_pos far from val prevalence."""
    if val_df.empty:
        return 0.7, {"reason": "no val metrics; default w=0.7"}

    wcol = "weight_baseline" if "weight_baseline" in val_df.columns else "weight"

    def macro_f1_row(w: float) -> float:
        r = val_df[val_df[wcol] == w]
        if r.empty:
            return -1.0
        return float(r["macro_F1_0.5"].values[0])

    global_best_w = max(weights, key=lambda w: macro_f1_row(w))
    global_best_f1 = macro_f1_row(global_best_w)

    preferred = [w for w in (0.7, 0.6, 0.65, 0.8, 0.55, 0.5) if w in weights]

    def gap_penalty(w: float) -> float:
        r = val_df[val_df[wcol] == w]
        if r.empty:
            return 1.0
        gaps = []
        for j, c in enumerate(TASKS):
            pr = float(r[f"pred_pos_{c}_0.5"].values[0])
            gaps.append(abs(pr - float(val_rates[j])))
        return float(np.mean(gaps))

    scored = []
    for w in preferred:
        if macro_f1_row(w) < 0:
            continue
        scored.append((w, macro_f1_row(w), gap_penalty(w)))

    if not scored:
        chosen = global_best_w
        reason = f"no preferred grid point; fallback to global best F1 w={chosen}"
    else:
        scored.sort(key=lambda t: (-(t[1] - 0.35 * t[2]), -t[0]))
        chosen = scored[0][0]
        reason = (
            f"prefer baseline-heavy w among {{0.5–0.8}} using macro_F1 - 0.35*mean|pred_pos-val_rate|; "
            f"picked w={chosen}"
        )

    meta = {
        "reason": reason,
        "global_best_weight": float(global_best_w),
        "global_best_macro_F1_0.5": float(global_best_f1),
        "chosen_macro_F1_0.5": float(macro_f1_row(chosen)),
        "preferred_candidates": preferred,
    }
    if global_best_w != chosen and global_best_f1 - macro_f1_row(chosen) > 0.02:
        meta["note"] = (
            f"Global best F1 was at w={global_best_w} (macro F1={global_best_f1:.4f}); "
            f"chosen w={chosen} per stability / pred_pos rules (md §10.5)."
        )
    return float(chosen), meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    out_root = Path(cfg["paths"]["output_dir"]) / "ensemble_analysis"
    ensure_dir(out_root)
    dpi = int(cfg.get("plot", {}).get("dpi", 200))
    ens = cfg.get("ensemble") or {}
    baseline_name = ens.get("baseline_model_name", "baseline")
    dynamic_name = ens.get("dynamic_model_name", "dynamic")
    calib = str(ens.get("pred_calib", "raw"))
    weights = [float(x) for x in ens.get("weights", [round(i / 10, 1) for i in range(11)])]

    lab_cfg = cfg["labels"]
    id_col = lab_cfg.get("id_col", "person_id")

    path_b_val = find_pred_path(cfg, baseline_name, "val", calib)
    path_d_val = find_pred_path(cfg, dynamic_name, "val", calib)
    path_b_test = find_pred_path(cfg, baseline_name, "test", calib)
    path_d_test = find_pred_path(cfg, dynamic_name, "test", calib)

    val_rates = np.zeros(3)
    val_df_lab = read_label_file(
        lab_cfg["val_label_path"], id_col, tuple(lab_cfg.get("class_cols", ["D", "A", "S"]))
    )
    val_rates = val_df_lab[["D", "A", "S"]].mean().values.astype(np.float64)

    val_rows = []
    curves: dict[str, list[tuple[float, float]]] = {c: [] for c in TASKS}

    if path_b_val and path_d_val and path_b_val.is_file() and path_d_val.is_file():
        pred_b = read_prediction_file(path_b_val, id_col)
        pred_d = read_prediction_file(path_d_val, id_col)
        merged = pred_b.merge(pred_d, on=id_col, suffixes=("_b", "_d"), how="inner")
        if len(merged) < 10:
            log.warning("Too few merged val rows for ensemble (%s)", len(merged))
        else:
            lab = val_df_lab.set_index(id_col)
            m2 = merged.set_index(id_col).join(lab, how="inner")
            if len(m2) < 10:
                log.warning("Too few val rows after label join (%s)", len(m2))
            else:
                Pb = m2[[f"p_{c}_b" for c in TASKS]].values.astype(np.float64)
                Pd = m2[[f"p_{c}_d" for c in TASKS]].values.astype(np.float64)
                Y = m2[["D", "A", "S"]].values.astype(int)

                for w in weights:
                    P = ensemble_probs(Pb, Pd, w)
                    macro_f1 = binary_macro_f1(P, Y, 0.5)
                    macro_auc = macro_auroc(P, Y)
                    f1s, precs, recs = per_class_f1_prec_rec(Y, P, 0.5)
                    aucs = per_class_auroc(Y, P)
                    ppos = pred_pos_rate(P, 0.5)
                    row: dict = {
                        "weight_baseline": w,
                        "weight_dynamic": 1.0 - w,
                        "macro_F1_0.5": macro_f1,
                        "macro_AUROC": macro_auc,
                        "macro_precision_0.5": float(np.mean(precs)),
                        "macro_recall_0.5": float(np.mean(recs)),
                    }
                    for j, c in enumerate(TASKS):
                        row[f"F1_{c}_0.5"] = f1s[j]
                        row[f"AUROC_{c}"] = aucs[j]
                        row[f"precision_{c}_0.5"] = precs[j]
                        row[f"recall_{c}_0.5"] = recs[j]
                        row[f"pred_pos_{c}_0.5"] = float(ppos[j])
                        curves[c].append((w, f1s[j]))
                    val_rows.append(row)

    val_out = pd.DataFrame(val_rows)
    if not val_out.empty:
        val_out.to_csv(out_root / "ensemble_val_summary.csv", index=False)

    for c in TASKS:
        pts = curves[c]
        if not pts:
            continue
        xs = [t[0] for t in pts]
        ys = [t[1] for t in pts]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("baseline weight w")
        ax.set_ylabel(f"F1@0.5 ({c})")
        ax.set_title(f"Ensemble F1 vs weight (class {c})")
        ax.set_xlim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(out_root / f"ensemble_weight_curve_{c}.png", dpi=dpi)
        plt.close(fig)

    chosen_w, meta = recommend_weight(val_out, val_rates, weights)
    test_rows = []
    submission_path: str | None = None
    if path_b_test and path_d_test and path_b_test.is_file() and path_d_test.is_file():
        tb = read_prediction_file(path_b_test, id_col)
        td = read_prediction_file(path_d_test, id_col)
        merged_t = tb.merge(td, on=id_col, suffixes=("_b", "_d"), how="inner")
        if len(merged_t) == 0:
            log.warning("No overlapping test ids for ensemble")
        else:
            Pb_t = merged_t[[f"p_{c}_b" for c in TASKS]].values.astype(np.float64)
            Pd_t = merged_t[[f"p_{c}_d" for c in TASKS]].values.astype(np.float64)
            ids = merged_t[id_col].values
            for w in weights:
                P = ensemble_probs(Pb_t, Pd_t, w)
                ppos = pred_pos_rate(P, 0.5)
                row = {
                    "weight_baseline": w,
                    "weight_dynamic": 1.0 - w,
                    "pred_pos_D_0.5": float(ppos[0]),
                    "pred_pos_A_0.5": float(ppos[1]),
                    "pred_pos_S_0.5": float(ppos[2]),
                }
                for j, c in enumerate(TASKS):
                    q = prob_quantiles(P[:, j])
                    row[f"prob_mean_{c}"] = q["prob_mean"]
                    row[f"prob_p50_{c}"] = q["prob_p50"]
                    row[f"prob_p90_{c}"] = q["prob_p90"]
                test_rows.append(row)

            P_star = ensemble_probs(Pb_t, Pd_t, chosen_w)
            sub = pd.DataFrame(
                {
                    id_col: ids,
                    "p_D": P_star[:, 0],
                    "p_A": P_star[:, 1],
                    "p_S": P_star[:, 2],
                }
            )
            wtag = int(round(chosen_w * 100))
            sub_path = out_root / f"test_submission_ensemble_w{wtag:03d}.csv"
            sub.to_csv(sub_path, index=False)
            submission_path = str(sub_path)
    else:
        log.warning("Missing test prediction files for ensemble; skip test summary / submission")

    meta.update(
        {
            "baseline_weight": round(float(chosen_w), 4),
            "dynamic_weight": round(float(1.0 - chosen_w), 4),
            "pred_calib": calib,
            "submission_csv": submission_path,
        }
    )
    save_json(out_root / "recommended_ensemble.json", meta)

    if test_rows:
        pd.DataFrame(test_rows).to_csv(out_root / "ensemble_test_summary.csv", index=False)

    log.info("Ensemble analysis -> %s", out_root)


if __name__ == "__main__":
    main()
