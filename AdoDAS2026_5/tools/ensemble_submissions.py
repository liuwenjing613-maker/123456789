#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weighted average of A1 submission CSVs (same file_id order)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_item(item: str) -> tuple[float, Path]:
    w, p = item.split(":", 1)
    return float(w), Path(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="format: weight:path.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    items = [parse_item(x) for x in args.inputs]
    total_w = sum(w for w, _ in items)
    if total_w <= 0:
        raise ValueError("sum of weights must be positive")

    base = None
    probs = None

    for w, path in items:
        df = pd.read_csv(path)
        if base is None:
            base = df[["file_id"]].copy()
            probs = df[["p_D", "p_A", "p_S"]].astype(float) * (w / total_w)
        else:
            if not base["file_id"].equals(df["file_id"]):
                df = df.set_index("file_id").loc[base["file_id"]].reset_index()
            probs += df[["p_D", "p_A", "p_S"]].astype(float) * (w / total_w)

    out = pd.concat([base, probs], axis=1)
    out.to_csv(args.output, index=False)
    print("saved:", args.output)
    print(out.head())
    for c in ["p_D", "p_A", "p_S"]:
        print(c, "mean=", out[c].mean(), "pred_pos@0.5=", (out[c] >= 0.5).mean())


if __name__ == "__main__":
    main()
