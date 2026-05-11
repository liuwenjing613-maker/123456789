"""Shared helpers for A1 calibration / dataset analysis (matplotlib only)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

log = logging.getLogger(__name__)

EPS = 1e-6


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read_label_file(
    path: str | Path,
    id_col: str = "person_id",
    class_cols: tuple[str, ...] = ("D", "A", "S"),
) -> pd.DataFrame:
    """Build person-level labels. Supports manifest-style CSV with anon_* and y_D/y_A/y_S."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    df = pd.read_csv(path)

    if {"anon_school", "anon_class", "anon_pid"}.issubset(df.columns):
        df = df.copy()
        df[id_col] = (
            df["anon_school"].astype(str)
            + "_"
            + df["anon_class"].astype(str)
            + "_"
            + df["anon_pid"].astype(str)
        )
        ymap = {"D": "y_D", "A": "y_A", "S": "y_S"}
        for c in class_cols:
            yc = ymap.get(c, c)
            if yc not in df.columns:
                raise KeyError(f"Expected column {yc!r} in {path}")
            df[c] = pd.to_numeric(df[yc], errors="coerce").fillna(0).astype(int).clip(0, 1)
        g = df.groupby(id_col, as_index=False)[list(class_cols)].first()
        return g

    missing = set(class_cols) - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing label columns {missing}")
    if id_col not in df.columns:
        raise KeyError(f"{path}: missing id column {id_col!r}")
    out = df[[id_col] + list(class_cols)].copy()
    for c in class_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return out.drop_duplicates(subset=[id_col])


def read_prediction_file(
    path: str | Path,
    id_col: str = "person_id",
    prob_cols: tuple[str, ...] = ("p_D", "p_A", "p_S"),
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path)

    if "file_id" in df.columns and id_col not in df.columns:
        df = df.rename(columns={"file_id": id_col})
    elif {"anon_school", "anon_class", "anon_pid"}.issubset(df.columns) and id_col not in df.columns:
        df = df.copy()
        df[id_col] = (
            df["anon_school"].astype(str)
            + "_"
            + df["anon_class"].astype(str)
            + "_"
            + df["anon_pid"].astype(str)
        )

    cols = list(prob_cols)
    alt = {f"p_{x}": f"p_{x}" for x in ["D", "A", "S"]}
    if all(c in df.columns for c in ["D", "A", "S"]) and not all(c in df.columns for c in cols):
        df = df.rename(columns={"D": "p_D", "A": "p_A", "S": "p_S"})
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing probability columns {missing}")
    out = df[[id_col] + cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_bias_probs(probs: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """probs (N,3), biases (3,) on logits."""
    z = logit(probs)
    return sigmoid(z + biases.reshape(1, -1))


def calibrate_a1_bias_grid(
    logits: np.ndarray, labels: np.ndarray, grid_min=-3.0, grid_max=3.0, grid_step=0.1
) -> tuple[np.ndarray, list[float]]:
    """Per-task bias maximizing sklearn F1 (same as training runner)."""
    grid = np.arange(grid_min, grid_max + grid_step, grid_step)
    biases = np.zeros(3, dtype=np.float64)
    best_f1s: list[float] = []
    for t in range(3):
        best_f1 = -1.0
        best_b = 0.0
        for b in grid:
            pr = sigmoid(logits[:, t] + b)
            pred = (pr > 0.5).astype(int)
            f1 = f1_score(labels[:, t], pred, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_b = b
        biases[t] = best_b
        best_f1s.append(float(best_f1))
    return biases, best_f1s


def binary_macro_f1(probs: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> float:
    pred = (probs >= thr).astype(int)
    fs = [
        f1_score(labels[:, i], pred[:, i], zero_division=0.0) for i in range(probs.shape[1])
    ]
    return float(np.mean(fs))


def macro_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(probs.shape[1]):
        y = labels[:, c]
        if len(np.unique(y)) < 2:
            scores.append(0.0)
        else:
            scores.append(float(roc_auc_score(y, probs[:, c])))
    return float(np.mean(scores))


def macro_auprc(probs: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(probs.shape[1]):
        y = labels[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            scores.append(0.0)
        else:
            scores.append(float(average_precision_score(y, probs[:, c])))
    return float(np.mean(scores))


def per_class_metrics_at_threshold(
    probs: np.ndarray, labels: np.ndarray, thr: float = 0.5
) -> dict[str, list[float]]:
    pred = (probs >= thr).astype(int)
    out: dict[str, list[float]] = {"f1": [], "precision": [], "recall": []}
    for i in range(3):
        out["f1"].append(float(f1_score(labels[:, i], pred[:, i], zero_division=0.0)))
        out["precision"].append(float(precision_score(labels[:, i], pred[:, i], zero_division=0.0)))
        out["recall"].append(float(recall_score(labels[:, i], pred[:, i], zero_division=0.0)))
    return out


def pred_pos_rate(probs: np.ndarray, thr: float) -> np.ndarray:
    return (probs >= thr).mean(axis=0)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_calibration_biases(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    b = data.get("biases")
    if not b or len(b) != 3:
        return None
    return np.asarray(b, dtype=np.float64)
