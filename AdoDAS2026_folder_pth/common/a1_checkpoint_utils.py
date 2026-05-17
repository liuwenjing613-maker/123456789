"""A1 checkpoint selection helpers for AdoDAS2026_folder_pth."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .utils.metrics import binary_f1


EPS = 1e-8


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checkpoint_bias_path(checkpoint_path: Path) -> Path:
    """Sidecar lives under checkpoints/calibration/<stem>.bias.json."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.parent / "calibration" / f"{checkpoint_path.stem}.bias.json"


def apply_a1_logit_bias(probs: np.ndarray, biases: np.ndarray, shrink: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probs, 1e-6, 1.0 - 1e-6))
    shifted = logits + float(shrink) * np.asarray(biases, dtype=np.float64).reshape(1, -1)
    return 1.0 / (1.0 + np.exp(-shifted))


def mean_f1_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    return float(binary_f1(probs, labels, threshold=threshold))


def shrink_f1_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    biases: np.ndarray,
    shrink: float,
    threshold: float = 0.5,
) -> float:
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    cal_probs = apply_a1_logit_bias(raw_probs, biases, shrink)
    return mean_f1_from_probs(cal_probs, labels, threshold=threshold)


def resolve_safe_submit_mode(raw_f1: float, shrink03: float, shrink05: float) -> tuple[str, float]:
    """Tie order: raw > shrink0.3 > shrink0.5."""
    best = max(raw_f1, shrink03, shrink05)
    if abs(best - raw_f1) <= EPS:
        return "raw", 0.0
    if abs(best - shrink03) <= EPS:
        return "shrink0.3", 0.3
    return "shrink0.5", 0.5


def _is_better_max(score: float, best_score: float, tie_score: float, best_tie: float) -> bool:
    if score > best_score + EPS:
        return True
    if abs(score - best_score) <= EPS and tie_score > best_tie + EPS:
        return True
    return False


def update_best_raw_f1_state(state: dict, epoch: int, raw_f1: float, auroc: float) -> bool:
    changed = _is_better_max(raw_f1, state["score"], auroc, state["tie_auc"])
    if changed:
        state["score"] = float(raw_f1)
        state["tie_auc"] = float(auroc)
        state["epoch"] = int(epoch)
    return changed


def update_best_auc_state(state: dict, epoch: int, auroc: float, raw_f1: float) -> bool:
    changed = _is_better_max(auroc, state["score"], raw_f1, state["tie_raw_f1"])
    if changed:
        state["score"] = float(auroc)
        state["tie_raw_f1"] = float(raw_f1)
        state["epoch"] = int(epoch)
    return changed


def update_best_safe_submit_state(
    state: dict,
    epoch: int,
    safe_submit_f1: float,
    macro_auroc: float,
    raw_f1: float,
) -> bool:
    """Primary checkpoint: max safe_submit_f1; tie AUROC; tie raw_f1."""
    if safe_submit_f1 > state["score"] + EPS:
        changed = True
    elif abs(safe_submit_f1 - state["score"]) <= EPS and macro_auroc > state["tie_auc"] + EPS:
        changed = True
    elif (
        abs(safe_submit_f1 - state["score"]) <= EPS
        and abs(macro_auroc - state["tie_auc"]) <= EPS
        and raw_f1 > state["tie_raw_f1"] + EPS
    ):
        changed = True
    else:
        changed = False
    if changed:
        state["score"] = float(safe_submit_f1)
        state["tie_auc"] = float(macro_auroc)
        state["tie_raw_f1"] = float(raw_f1)
        state["epoch"] = int(epoch)
    return changed


def save_a1_bias_sidecar(
    checkpoint_path: Path,
    *,
    run_name: str,
    epoch: int,
    biases,
    checkpoint_type: str,
    raw_mean_f1: float,
    macro_auroc: float,
    shrink0_3_f1: float,
    shrink0_5_f1: float,
    safe_submit_f1: float,
    safe_submit_mode: str,
    safe_submit_shrink: float,
    full_calibrated_mean_f1: float | None = None,
    full_biases: dict | None = None,
    val_metrics_raw: dict | None = None,
    val_metrics_calibrated: dict | None = None,
    pred_pos_raw: dict | None = None,
    pred_pos_calibrated: dict | None = None,
    source: str = "official_val_for_single_run",
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    bias_path = checkpoint_bias_path(checkpoint_path)

    if isinstance(biases, dict):
        bias_vector = [float(biases["D"]), float(biases["A"]), float(biases["S"])]
    else:
        bias_vector = [float(biases[0]), float(biases[1]), float(biases[2])]

    fb = full_biases or {"D": bias_vector[0], "A": bias_vector[1], "S": bias_vector[2]}

    payload = {
        "version": 1,
        "task": "a1",
        "bias_type": "logit_additive",
        "bias_order": ["D", "A", "S"],
        "bias_vector": bias_vector,
        "biases": {"D": bias_vector[0], "A": bias_vector[1], "S": bias_vector[2]},
        "full_biases": fb,
        "threshold": 0.5,
        "threshold_rule": ">=0.5",
        "checkpoint_type": checkpoint_type,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_stem": checkpoint_path.stem,
        "checkpoint_relpath": str(Path("checkpoints") / checkpoint_path.name),
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        "run_name": run_name,
        "epoch": int(epoch),
        "source": source,
        "selection_metric": "safe_submit_f1",
        "raw_f1": float(raw_mean_f1),
        "raw_mean_f1": float(raw_mean_f1),
        "macro_auroc": float(macro_auroc),
        "shrink0.3_f1": float(shrink0_3_f1),
        "shrink0.5_f1": float(shrink0_5_f1),
        "safe_submit_f1": float(safe_submit_f1),
        "safe_submit_mode": str(safe_submit_mode),
        "safe_submit_shrink": float(safe_submit_shrink),
        "full_calibrated_mean_f1": full_calibrated_mean_f1,
        "raw_metrics": val_metrics_raw or {"mean_f1": raw_mean_f1, "macro_auroc": macro_auroc},
        "full_calibrated_metrics": val_metrics_calibrated or {},
        "pred_pos_raw": pred_pos_raw or {},
        "pred_pos_calibrated": pred_pos_calibrated or {},
        "created_for": "A1 logit bias: sigmoid(logit(p) + shrink * bias)",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    bias_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bias_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return bias_path


def write_checkpoint_meta(
    ckpt_dir: Path,
    *,
    run_name: str,
    selections: dict[str, dict[str, Any]],
    early_stop_metric: str,
    checkpoint_selection_mode: str = "safe_submit",
) -> Path:
    meta = {
        "run_name": run_name,
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "early_stop_metric": early_stop_metric,
        "checkpoints": selections,
        "recommended_default_submission": "best_safe_submit.pt + --a1_bias_mode none",
        "secondary_submission": "best_safe_submit.pt + --a1_bias_mode auto --a1_use_sidecar_shrink",
        "optional_submission": "best_raw_f1.pt + --a1_bias_mode none",
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = ckpt_dir / "checkpoint_meta.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out


_SELECTION_SUMMARY_FIELDS = (
    "checkpoint_name",
    "epoch",
    "raw_mean_f1",
    "macro_auroc",
    "safe_submit_f1",
    "safe_submit_mode",
    "safe_submit_shrink",
    "full_calibrated_mean_f1",
    "reference_f1_raw",
    "recommended_default_submission",
)


def _selection_summary_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    ordered = [k for k in _SELECTION_SUMMARY_FIELDS if k in all_keys]
    ordered.extend(sorted(all_keys - set(ordered)))
    return ordered


def write_selection_summary(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = run_dir / "selection_summary.json"
    csv_path = run_dir / "selection_summary.csv"
    payload = {
        "rows": rows,
        "recommended_default_submission": "best_safe_submit.pt + --a1_bias_mode none",
        "secondary_submission": "best_safe_submit.pt + --a1_bias_mode auto --a1_use_sidecar_shrink",
        "optional_submission": "best_raw_f1.pt + --a1_bias_mode none",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if rows:
        fieldnames = _selection_summary_fieldnames(rows)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return json_path, csv_path


def build_a1_epoch_record(
    epoch: int,
    val_metrics: dict[str, Any],
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    cal_biases: np.ndarray,
) -> dict[str, Any]:
    raw_f1 = float(val_metrics["mean_f1"])
    auroc = float(val_metrics["auroc"])
    shrink03 = shrink_f1_from_logits(logits_np, labels_np, cal_biases, shrink=0.3)
    shrink05 = shrink_f1_from_logits(logits_np, labels_np, cal_biases, shrink=0.5)
    safe_submit_f1 = float(max(raw_f1, shrink03, shrink05))
    mode, shrink = resolve_safe_submit_mode(raw_f1, shrink03, shrink05)
    return {
        "epoch": int(epoch),
        "raw_mean_f1": raw_f1,
        "macro_auroc": auroc,
        "shrink0.3_f1": float(shrink03),
        "shrink0.5_f1": float(shrink05),
        "safe_submit_f1": safe_submit_f1,
        "safe_submit_mode": mode,
        "safe_submit_shrink": shrink,
        "full_calibrated_mean_f1": float(val_metrics.get("mean_f1_calibrated", raw_f1)),
        "full_biases": cal_biases.tolist(),
        "val_metrics_raw": val_metrics.get("val_metrics_raw", {}),
        "val_metrics_calibrated": val_metrics.get("val_metrics_calibrated", {}),
        "pred_pos_raw": val_metrics.get("pred_pos_raw", {}),
        "pred_pos_calibrated": val_metrics.get("pred_pos_calibrated", {}),
    }
