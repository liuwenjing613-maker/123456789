#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create internal train/val splits for AdoDAS A1 (GroupKFold by group mode).

See ``lwj/A1_val_test_consistency_best_plan.md`` Part B.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def build_group(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "school":
        return df["anon_school"].astype(str)
    if mode == "school_class":
        return df["anon_school"].astype(str) + "_" + df["anon_class"].astype(str)
    if mode == "participant":
        return (
            df["anon_school"].astype(str)
            + "_"
            + df["anon_class"].astype(str)
            + "_"
            + df["anon_pid"].astype(str)
        )
    raise ValueError(f"Unknown group mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--group-mode",
        type=str,
        default="school_class",
        choices=["school", "school_class", "participant"],
    )
    parser.add_argument("--train-name", type=str, default="train.csv")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = manifest_dir / args.train_name
    df = pd.read_csv(train_path)

    groups = build_group(df, args.group_mode)

    print("rows:", len(df))
    print("unique groups:", groups.nunique())
    print("group mode:", args.group_mode)

    gkf = GroupKFold(n_splits=args.n_splits)
    dummy_y = np.zeros(len(df), dtype=np.int64)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, dummy_y, groups=groups)):
        split_dir = out_dir / f"split_{fold}_{args.group_mode}"
        split_dir.mkdir(parents=True, exist_ok=True)

        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()

        tr.to_csv(split_dir / "train.csv", index=False)
        va.to_csv(split_dir / "val.csv", index=False)

        print(f"\n===== split {fold} =====")
        print("train rows:", len(tr), "val rows:", len(va))
        key = ["anon_school", "anon_class", "anon_pid"]
        if all(c in tr.columns for c in key):
            print("train participants:", tr[key].drop_duplicates().shape[0])
            print("val participants:", va[key].drop_duplicates().shape[0])

        for col in ["D", "A", "S"]:
            if col in tr.columns and col in va.columns:
                print(
                    col,
                    "train pos:",
                    round(float(tr[col].mean()), 4),
                    "val pos:",
                    round(float(va[col].mean()), 4),
                )

    print("\nDone. Splits saved to:", out_dir)


if __name__ == "__main__":
    main()
