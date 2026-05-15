#!/usr/bin/env python3
"""Build participant-level K-fold manifests from official train.csv only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def label_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    if {"D", "A", "S"}.issubset(df.columns):
        return "D", "A", "S"
    if {"y_D", "y_A", "y_S"}.issubset(df.columns):
        return "y_D", "y_A", "y_S"
    raise KeyError("train manifest must contain D/A/S or y_D/y_A/y_S columns")


def participant_table(train_df: pd.DataFrame) -> pd.DataFrame:
    d_col, a_col, s_col = label_columns(train_df)
    gcols = ["anon_school", "anon_class", "anon_pid"]
    rows = []
    for key, grp in train_df.groupby(gcols):
        row = grp.iloc[0]
        d, a, s = int(row[d_col]), int(row[a_col]), int(row[s_col])
        combo = f"{d}{a}{s}"
        rows.append({
            "anon_school": key[0],
            "anon_class": key[1],
            "anon_pid": key[2],
            "label_combo": combo,
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", default="/home/adodas/dataset/manifests")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--kfold", type=int, choices=[3, 5], required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_path = Path(args.manifest_dir) / "train.csv"
    train_df = pd.read_csv(train_path)
    d_col, a_col, s_col = label_columns(train_df)
    part_df = participant_table(train_df)
    y = part_df["label_combo"].values

    try:
        splitter = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(part_df, y))
        print(f"StratifiedKFold on label_combo (n={len(part_df)} participants)")
    except ValueError as e:
        print(f"WARNING: StratifiedKFold failed ({e}); fallback to KFold")
        splitter = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(part_df))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        tr_parts = set(map(tuple, part_df.iloc[tr_idx][["anon_school", "anon_class", "anon_pid"]].values))
        va_parts = set(map(tuple, part_df.iloc[va_idx][["anon_school", "anon_class", "anon_pid"]].values))
        if tr_parts & va_parts:
            raise RuntimeError(f"fold {fold_idx}: participant leakage between train and val")

        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        def _mask(part_set):
            keys = list(zip(train_df["anon_school"], train_df["anon_class"], train_df["anon_pid"]))
            return [k in part_set for k in keys]

        tr_df = train_df[_mask(tr_parts)]
        va_df = train_df[_mask(va_parts)]
        tr_df.to_csv(fold_dir / "train.csv", index=False)
        va_df.to_csv(fold_dir / "val.csv", index=False)

        summary_rows.append({
            "fold": fold_idx,
            "n_train_participants": len(tr_parts),
            "n_val_participants": len(va_parts),
            "n_train_rows": len(tr_df),
            "n_val_rows": len(va_df),
            "D_train_pos_rate": tr_df[d_col].mean(),
            "A_train_pos_rate": tr_df[a_col].mean(),
            "S_train_pos_rate": tr_df[s_col].mean(),
            "D_val_pos_rate": va_df[d_col].mean(),
            "A_val_pos_rate": va_df[a_col].mean(),
            "S_val_pos_rate": va_df[s_col].mean(),
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "fold_summary.csv", index=False)
    print(f"Wrote {args.kfold} folds to {out_dir}")


if __name__ == "__main__":
    main()
