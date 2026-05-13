#!/usr/bin/env python3
"""Offline tool: build checkpoint-matched A1 bias sidecar from raw val predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.runner import calibrate_a1_bias, save_a1_bias_sidecar


def _file_id_from_manifest_row(row: pd.Series) -> str:
    return f"{row['anon_school']}_{row['anon_class']}_{row['anon_pid']}"


def _file_id_from_pred_row(row: pd.Series) -> str:
    if "file_id" in row.index and pd.notna(row["file_id"]) and str(row["file_id"]).strip():
        return str(row["file_id"]).strip()
    return f"{row['anon_school']}_{row['anon_class']}_{row['anon_pid']}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--pred_csv", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--selection_reason", type=str, required=True)
    p.add_argument("--grid_min", type=float, default=-3.0)
    p.add_argument("--grid_max", type=float, default=3.0)
    p.add_argument("--grid_step", type=float, default=0.1)
    p.add_argument("--epoch", type=int, default=None, help="Epoch to record in sidecar (default: from checkpoint if present)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    val_path = Path(args.val_csv).expanduser().resolve()
    pred_path = Path(args.pred_csv).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()

    if not val_path.exists():
        raise FileNotFoundError(val_path)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    val_df = pd.read_csv(val_path)
    pred_df = pd.read_csv(pred_path)

    required_val = {"anon_school", "anon_class", "anon_pid", "y_D", "y_A", "y_S"}
    if not required_val.issubset(val_df.columns):
        raise ValueError(f"val_csv must contain columns {sorted(required_val)}")
    val_df = val_df.copy()
    val_df["_fid"] = val_df.apply(_file_id_from_manifest_row, axis=1)
    val_df = val_df.drop_duplicates(subset=["_fid"], keep="first")

    pred_cols = set(pred_df.columns)
    if "file_id" not in pred_cols:
        need = {"anon_school", "anon_class", "anon_pid", "p_D", "p_A", "p_S"}
        if not need.issubset(pred_cols):
            raise ValueError(
                "pred_csv must have file_id,p_D,p_A,p_S or anon_school,anon_class,anon_pid,p_D,p_A,p_S"
            )
    need_p = {"p_D", "p_A", "p_S"}
    if not need_p.issubset(pred_df.columns):
        raise ValueError("pred_csv must contain p_D, p_A, p_S")
    pred_df = pred_df.copy()
    pred_df["_fid"] = pred_df.apply(_file_id_from_pred_row, axis=1)
    pred_df = pred_df.drop_duplicates(subset=["_fid"], keep="first")

    merged = val_df.merge(pred_df, on="_fid", how="inner", suffixes=("", "_pred"))
    if merged.empty:
        raise ValueError("No overlapping rows between val_csv and pred_csv (check file_id / anon_* keys).")

    labels = merged[["y_D", "y_A", "y_S"]].values.astype(np.float64)
    probs = merged[["p_D", "p_A", "p_S"]].values.astype(np.float64)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(probs / (1.0 - probs))

    biases, _ = calibrate_a1_bias(
        logits, labels, grid_min=args.grid_min, grid_max=args.grid_max, grid_step=args.grid_step
    )
    bias_dict = {"D": float(biases[0]), "A": float(biases[1]), "S": float(biases[2])}

    run_name = ckpt_path.parent.parent.name
    epoch = args.epoch
    if epoch is None:
        try:
            try:
                sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                sd = torch.load(ckpt_path, map_location="cpu")
            epoch = int(sd.get("epoch", 0)) if isinstance(sd, dict) else 0
        except Exception:
            epoch = 0

    out_path = save_a1_bias_sidecar(
        ckpt_path,
        run_name=run_name,
        epoch=epoch,
        biases=bias_dict,
        selection_reason=str(args.selection_reason),
        val_metrics_raw={},
        val_metrics_calibrated={},
        pred_pos_raw={},
        pred_pos_calibrated={},
        source="retrofitted_from_val_pred",
    )
    print(json.dumps({"wrote": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
