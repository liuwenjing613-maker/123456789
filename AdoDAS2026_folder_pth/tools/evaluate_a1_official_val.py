#!/usr/bin/env python3
"""Compute mean F1 and macro AUROC for A1 predictions on official val.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.eval_a1 import (
    evaluate_a1_pred_csv,
    format_a1_metrics_log,
    log_a1_metrics,
    write_a1_metrics_json,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_csv", required=True)
    p.add_argument("--manifest", default="/home/adodas/dataset/manifests/val.csv")
    p.add_argument("--output_json", default=None, help="Default: <pred_csv>.metrics.json")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    pred_path = Path(args.pred_csv)
    out_json = (
        Path(args.output_json)
        if args.output_json
        else pred_path.with_suffix(pred_path.suffix + ".metrics.json")
    )
    metrics = evaluate_a1_pred_csv(
        pred_path,
        args.manifest,
        threshold=args.threshold,
    )
    write_a1_metrics_json(metrics, out_json)
    print(format_a1_metrics_log(metrics))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
