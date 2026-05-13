#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create internal train/val splits for AdoDAS A1.

Usage:
python tools/make_internal_splits.py \
  --manifest-dir /home/adodas/dataset/manifests \
  --out-dir /home/adodas/dataset/manifests_internal \
  --n-splits 3 \
  --group-mode school_class \
  --require-feature-root /home/adodas/dataset
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def filter_rows_with_train_feature_dir(df: pd.DataFrame, feature_root: Path) -> pd.DataFrame:
    """Keep only sessions whose participant has extracted features under ``train/`` on disk.

    Official ``train.csv`` can list participants who have no ``{root}/train/.../{pid}/`` tree
    (no extraction or different layout). Internal CV with ``val_sequence_path_split: train``
    will otherwise hit val batches where every row fails to load ``sequence.npz``.
    """
    keys = df[["anon_school", "anon_class", "anon_pid"]].drop_duplicates()
    ok: set[tuple[str, str, str]] = set()
    for r in keys.itertuples(index=False):
        pid_dir = (
            feature_root
            / "train"
            / str(r.anon_school)
            / str(r.anon_class)
            / str(r.anon_pid)
        )
        if pid_dir.is_dir():
            ok.add((str(r.anon_school), str(r.anon_class), str(r.anon_pid)))

    def _in_ok(row: pd.Series) -> bool:
        return (
            str(row["anon_school"]),
            str(row["anon_class"]),
            str(row["anon_pid"]),
        ) in ok

    mask = df.apply(_in_ok, axis=1)
    out = df.loc[mask].reset_index(drop=True)
    return out


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


def main():
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
    parser.add_argument(
        "--require-feature-root",
        type=str,
        default=None,
        help=(
            "If set (e.g. /home/adodas/dataset), drop manifest rows whose participant folder "
            "``{root}/train/{school}/{class}/{pid}/`` does not exist. Recommended for internal "
            "CV when using val_sequence_path_split=train."
        ),
    )
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = manifest_dir / args.train_name
    df = pd.read_csv(train_path)

    if args.require_feature_root:
        root = Path(args.require_feature_root).expanduser()
        n_before = len(df)
        p_before = (
            df[["anon_school", "anon_class", "anon_pid"]].drop_duplicates().shape[0]
        )
        df = filter_rows_with_train_feature_dir(df, root)
        n_after = len(df)
        p_after = (
            df[["anon_school", "anon_class", "anon_pid"]].drop_duplicates().shape[0]
        )
        print(
            "require-feature-root:",
            root,
            f"=> rows {n_before} -> {n_after}; participants {p_before} -> {p_after}",
        )

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
        print(
            "train participants:",
            tr[["anon_school", "anon_class", "anon_pid"]].drop_duplicates().shape[0],
        )
        print(
            "val participants:",
            va[["anon_school", "anon_class", "anon_pid"]].drop_duplicates().shape[0],
        )

        for col in ["D", "A", "S"]:
            if col in tr.columns and col in va.columns:
                print(
                    col,
                    "train pos:", round(float(tr[col].mean()), 4),
                    "val pos:", round(float(va[col].mean()), 4),
                )

    print("\nDone. Splits saved to:", out_dir)


if __name__ == "__main__":
    main()
