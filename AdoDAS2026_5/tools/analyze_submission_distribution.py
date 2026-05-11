#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize A1 submission CSV probability columns (val vs test drift checks)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(path: Path) -> None:
    df = pd.read_csv(path)
    print("\n===== ", path, " =====")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head())

    for c in ["p_D", "p_A", "p_S"]:
        if c not in df.columns:
            continue
        s = df[c]
        print(f"\n[{c}]")
        print("mean:", round(float(s.mean()), 6))
        print("std :", round(float(s.std()), 6))
        print("min :", round(float(s.min()), 6))
        print("max :", round(float(s.max()), 6))
        print("pred_pos@0.5:", round(float((s >= 0.5).mean()), 6))
        qs = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        print(qs.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    args = parser.parse_args()

    for p in args.csvs:
        summarize(Path(p))


if __name__ == "__main__":
    main()
