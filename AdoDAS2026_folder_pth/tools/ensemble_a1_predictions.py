#!/usr/bin/env python3
"""Average A1 fold probabilities (not labels) and optional OOF bias shrink."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.a1_checkpoint_utils import apply_a1_logit_bias


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_csvs", nargs="+", required=True, help="Fold prediction csvs with file_id,p_D,p_A,p_S")
    p.add_argument("--output", required=True)
    p.add_argument("--oof_bias_json", default=None)
    p.add_argument("--shrink", type=float, default=0.0)
    args = p.parse_args()

    dfs = [pd.read_csv(x) for x in args.pred_csvs]
    base = dfs[0][["file_id"]].copy()
    for col in ("p_D", "p_A", "p_S"):
        stacked = np.stack([df.set_index("file_id")[col].reindex(base["file_id"]).values for df in dfs], axis=0)
        base[col] = stacked.mean(axis=0)

    if args.oof_bias_json and args.shrink > 0:
        with open(args.oof_bias_json, encoding="utf-8") as f:
            bias_data = json.load(f)
        b = bias_data.get("biases", {})
        biases = np.array([b["D"], b["A"], b["S"]], dtype=np.float64)
        probs = base[["p_D", "p_A", "p_S"]].values.astype(np.float64)
        cal = apply_a1_logit_bias(probs, biases, args.shrink)
        base[["p_D", "p_A", "p_S"]] = cal

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
