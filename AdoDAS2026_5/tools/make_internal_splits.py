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


def _session_has_core_features(
    root: Path,
    row: pd.Series,
    path_split: str,
    audio_ssl_model_tag: str,
    video_ssl_model_tag: str,
) -> bool:
    base = (
        root
        / path_split
        / str(row["anon_school"])
        / str(row["anon_class"])
        / str(row["anon_pid"])
    )
    session = str(row["session"])
    core_paths = [
        base / "audio" / "mel_mfcc" / session / "sequence.npz",
        base / "audio" / "ssl_embed" / audio_ssl_model_tag / session / "sequence.npz",
        base / "video" / "vision_ssl_embed" / video_ssl_model_tag / session / "sequence.npz",
    ]
    return all(p.exists() for p in core_paths)


def filter_participants_with_core_features(
    df: pd.DataFrame,
    feature_root: Path,
    path_split: str,
    audio_ssl_model_tag: str,
    video_ssl_model_tag: str,
) -> pd.DataFrame:
    key = ["anon_school", "anon_class", "anon_pid"]
    keep_keys: set[tuple[str, str, str]] = set()
    dropped: list[tuple[str, str, str]] = []

    for values, group in df.groupby(key, sort=False):
        ok = any(
            _session_has_core_features(
                feature_root,
                row,
                path_split,
                audio_ssl_model_tag,
                video_ssl_model_tag,
            )
            for _, row in group.iterrows()
        )
        values_str = tuple(str(x) for x in values)
        if ok:
            keep_keys.add(values_str)
        else:
            dropped.append(values_str)

    mask = df[key].astype(str).apply(tuple, axis=1).isin(keep_keys)
    out = df.loc[mask].copy()
    print(
        "feature filter:",
        "participants before=", df[key].drop_duplicates().shape[0],
        "after=", out[key].drop_duplicates().shape[0],
        "dropped=", len(dropped),
    )
    if dropped:
        print("dropped first 20:", dropped[:20])
    return out


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
    parser.add_argument("--feature-root", type=str, default=None)
    parser.add_argument("--sequence-path-split", type=str, default="train")
    parser.add_argument("--audio-ssl-model-tag", type=str, default="wav2vec2-chinese-xlsr")
    parser.add_argument("--video-ssl-model-tag", type=str, default="dinov2-large")
    parser.add_argument(
        "--filter-missing-core",
        action="store_true",
        help="Drop participants with no loadable core session before GroupKFold.",
    )
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = manifest_dir / args.train_name
    df = pd.read_csv(train_path)
    if args.filter_missing_core:
        if args.feature_root is None:
            raise ValueError("--filter-missing-core requires --feature-root")
        df = filter_participants_with_core_features(
            df,
            Path(args.feature_root),
            args.sequence_path_split,
            args.audio_ssl_model_tag,
            args.video_ssl_model_tag,
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
    print(
        "\nTip (internal CV): val.csv participants load features from "
        "`{feature_root}/train/...` on disk. `train.py` auto-sets "
        "`val_sequence_path_split: train` when manifest_dir contains "
        "`manifests_internal` or a `split_N_*` folder name; you can still set it "
        "explicitly in YAML."
    )


if __name__ == "__main__":
    main()
