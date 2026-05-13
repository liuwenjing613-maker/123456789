#!/usr/bin/env python3
"""Print triple vs anon_pid uniqueness stats (plan Part A §2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", type=str, default="/home/adodas/dataset/manifests")
    args = p.parse_args()
    manifest_dir = Path(args.manifest_dir)

    for name in ["train.csv", "val.csv", "test_hidden.csv", "test.csv"]:
        path = manifest_dir / name
        if not path.exists():
            continue

        print("\n===== checking", path, "=====")
        m = pd.read_csv(path)
        need_cols = ["anon_school", "anon_class", "anon_pid"]
        missing = [c for c in need_cols if c not in m.columns]
        if missing:
            print("missing cols:", missing)
            continue

        g = m[need_cols].drop_duplicates()
        print("rows:", len(m))
        print("unique school_class_pid:", len(g))
        print("unique anon_pid only:", m["anon_pid"].nunique())

        dup = (
            g.groupby("anon_pid")
            .agg(
                n_school=("anon_school", "nunique"),
                n_class=("anon_class", "nunique"),
                n_rows=("anon_pid", "size"),
            )
            .query("n_rows > 1")
        )

        print("duplicate anon_pid count:", len(dup))
        if len(dup) > 0:
            print(dup.head(20))


if __name__ == "__main__":
    main()
