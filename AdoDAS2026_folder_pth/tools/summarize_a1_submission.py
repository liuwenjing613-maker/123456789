#!/usr/bin/env python3
"""Summarize A1 submission CSV probability distribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_submission_df(df: pd.DataFrame) -> dict:
    n = len(df)
    out: dict = {"n": n}
    for task, col in [("D", "p_D"), ("A", "p_A"), ("S", "p_S")]:
        if col not in df.columns:
            continue
        p = pd.to_numeric(df[col], errors="coerce").astype(float).values
        out[task] = {
            "prob_mean": float(np.mean(p)),
            "prob_std": float(np.std(p)),
            "p10": float(np.percentile(p, 10)),
            "p50": float(np.percentile(p, 50)),
            "p90": float(np.percentile(p, 90)),
            "pred_pos@0.5": float((p >= 0.5).mean()),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    stats = summarize_submission_df(df)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".pred_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
