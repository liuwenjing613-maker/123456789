#!/usr/bin/env python3
"""Hard per-class ensemble of three A1 models + per-class bias search on val."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

LABELS = ["D", "A", "S"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_npz(path: Path | str) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {
        "ids": data["ids"].astype(str),
        "logits": data["logits"].astype(np.float32),
    }
    if "labels" in data.files:
        out["labels"] = data["labels"].astype(np.float32)
    return out


def align_by_ids(ref: dict, other: dict) -> dict:
    """Reorder `other` so ids match `ref` ids order."""
    ref_ids = list(ref["ids"])
    index = {fid: i for i, fid in enumerate(other["ids"])}

    missing = [fid for fid in ref_ids if fid not in index]
    if missing:
        raise ValueError(f"Missing ids in other logits: {missing[:5]}, total={len(missing)}")

    order = [index[fid] for fid in ref_ids]
    aligned = {
        "ids": other["ids"][order],
        "logits": other["logits"][order],
    }
    if "labels" in other:
        aligned["labels"] = other["labels"][order]
    return aligned


def build_hard_ensemble_logits(baseline: dict, full: dict, lite: dict) -> np.ndarray:
    """D: full dynaug, A: baseline, S: dynaug-lite."""
    final_logits = np.zeros_like(baseline["logits"], dtype=np.float32)
    final_logits[:, 0] = full["logits"][:, 0]
    final_logits[:, 1] = baseline["logits"][:, 1]
    final_logits[:, 2] = lite["logits"][:, 2]
    return final_logits


def search_bias_per_class(
    logits: np.ndarray,
    labels: np.ndarray,
    step: float,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    biases_list: list[float] = []
    per_class: list[dict] = []

    grid = np.arange(lo, hi + 1e-9, step)

    for k, name in enumerate(LABELS):
        y = labels[:, k].astype(int)
        best_bias = 0.0
        best_f1 = -1.0
        best_pred_pos = 0.0
        best_prob_mean = 0.0

        for b in grid:
            prob = sigmoid(logits[:, k] + b)
            pred = (prob >= 0.5).astype(int)
            f1 = float(f1_score(y, pred, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_bias = float(b)
                best_pred_pos = float(pred.mean())
                best_prob_mean = float(prob.mean())

        biases_list.append(best_bias)
        per_class.append(
            {
                "label": name,
                "bias": best_bias,
                "f1": best_f1,
                "pred_pos": best_pred_pos,
                "prob_mean": best_prob_mean,
            }
        )

    biases = np.asarray(biases_list, dtype=np.float32)
    probs = sigmoid(logits + biases[None, :])
    preds = (probs >= 0.5).astype(int)

    f1s = [f1_score(labels[:, k].astype(int), preds[:, k], zero_division=0) for k in range(3)]

    aucs: list[float | None] = []
    for k in range(3):
        try:
            aucs.append(float(roc_auc_score(labels[:, k].astype(int), probs[:, k])))
        except ValueError:
            aucs.append(None)

    report = {
        "biases": {LABELS[i]: float(biases[i]) for i in range(3)},
        "per_class": per_class,
        "mean_f1": float(np.mean(f1s)),
        "f1_D": float(f1s[0]),
        "f1_A": float(f1s[1]),
        "f1_S": float(f1s[2]),
        "mean_auc": None if aucs[0] is None else float(np.nanmean([a for a in aucs if a is not None])),
        "auc_D": aucs[0],
        "auc_A": aucs[1],
        "auc_S": aucs[2],
    }

    return biases, probs, preds, report


def save_probs_csv(path: Path, ids: np.ndarray, probs: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "file_id": ids,
            "p_D": probs[:, 0],
            "p_A": probs[:, 1],
            "p_S": probs[:, 2],
        }
    )
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A1 per-class hard ensemble + bias calibration on val.")
    p.add_argument("--baseline-val", required=True)
    p.add_argument("--full-val", required=True)
    p.add_argument("--lite-val", required=True)
    p.add_argument("--baseline-test", default=None)
    p.add_argument("--full-test", default=None)
    p.add_argument("--lite-test", default=None)
    p.add_argument(
        "--val-only",
        action="store_true",
        help="Only val: search biases and write val report/calibration/probs; no test npz or submission CSV.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bias-step", type=float, default=0.05)
    p.add_argument("--bias-lo", type=float, default=-3.0)
    p.add_argument("--bias-hi", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.val_only:
        missing = [
            n
            for n, v in (
                ("--baseline-test", args.baseline_test),
                ("--full-test", args.full_test),
                ("--lite-test", args.lite_test),
            )
            if v is None
        ]
        if missing:
            raise SystemExit(
                "Missing test npz arguments: "
                + ", ".join(missing)
                + ". Pass all three test paths, or use --val-only to evaluate on val only."
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_val = load_npz(args.baseline_val)
    full_val = align_by_ids(baseline_val, load_npz(args.full_val))
    lite_val = align_by_ids(baseline_val, load_npz(args.lite_val))

    if "labels" not in baseline_val:
        raise ValueError("baseline val npz must contain labels")

    labels = baseline_val["labels"]
    if "labels" in full_val and not np.allclose(labels, full_val["labels"]):
        raise ValueError("full val labels do not match baseline labels")
    if "labels" in lite_val and not np.allclose(labels, lite_val["labels"]):
        raise ValueError("lite val labels do not match baseline labels")

    val_logits = build_hard_ensemble_logits(baseline_val, full_val, lite_val)

    biases, val_probs, _val_preds, report = search_bias_per_class(
        val_logits,
        labels,
        step=args.bias_step,
        lo=args.bias_lo,
        hi=args.bias_hi,
    )

    save_probs_csv(out_dir / "val_ensemble_probs.csv", baseline_val["ids"], val_probs)

    with open(out_dir / "ensemble_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    calibration = {
        "rule": {"D": "full_dynaug", "A": "baseline", "S": "dynaug_lite"},
        "biases": {LABELS[i]: float(biases[i]) for i in range(3)},
        "bias_step": args.bias_step,
        "bias_lo": args.bias_lo,
        "bias_hi": args.bias_hi,
    }
    with open(out_dir / "ensemble_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)

    print("===== VAL ENSEMBLE REPORT =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"biases (D,A,S): {biases.tolist()}")
    print(f"mean_f1: {report['mean_f1']:.4f}")

    if args.val_only:
        print(f"Saved val-only outputs to: {out_dir}")
        return

    baseline_test = load_npz(args.baseline_test)
    full_test = align_by_ids(baseline_test, load_npz(args.full_test))
    lite_test = align_by_ids(baseline_test, load_npz(args.lite_test))

    test_logits = build_hard_ensemble_logits(baseline_test, full_test, lite_test)
    test_probs = sigmoid(test_logits + biases[None, :])
    test_probs = np.clip(test_probs, 0.0, 1.0)

    save_probs_csv(out_dir / "test_ensemble_submission.csv", baseline_test["ids"], test_probs)

    print(f"Saved outputs to: {out_dir}")
    print(f"test probs min/max: {test_probs.min():.6f} / {test_probs.max():.6f}")


if __name__ == "__main__":
    main()
