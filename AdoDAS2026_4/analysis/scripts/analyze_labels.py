#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.scripts.utils import ensure_dir, load_yaml, read_label_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def summarize_split(df: pd.DataFrame, split: str) -> dict:
    n = len(df)
    D = df["D"].values
    A = df["A"].values
    S = df["S"].values
    pos = D + A + S
    row = {
        "split": split,
        "num_samples": n,
        "D_pos_count": int(D.sum()),
        "D_pos_rate": float(D.mean()) if n else 0.0,
        "A_pos_count": int(A.sum()),
        "A_pos_rate": float(A.mean()) if n else 0.0,
        "S_pos_count": int(S.sum()),
        "S_pos_rate": float(S.mean()) if n else 0.0,
        "all_zero_rate": float(np.mean((D + A + S) == 0)),
        "DA_pos_rate": float(np.mean((D == 1) & (A == 1))),
        "DS_pos_rate": float(np.mean((D == 1) & (S == 1))),
        "AS_pos_rate": float(np.mean((A == 1) & (S == 1))),
        "DAS_all_pos_rate": float(np.mean((D == 1) & (A == 1) & (S == 1))),
        "num_positive_0_rate": float(np.mean(pos == 0)),
        "num_positive_1_rate": float(np.mean(pos == 1)),
        "num_positive_2_rate": float(np.mean(pos == 2)),
        "num_positive_3_rate": float(np.mean(pos == 3)),
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    lab_cfg = cfg["labels"]
    out_root = Path(cfg["paths"]["output_dir"]) / "label_analysis"
    ensure_dir(out_root)
    dpi = int(cfg.get("plot", {}).get("dpi", 200))

    train = read_label_file(
        lab_cfg["train_label_path"],
        lab_cfg.get("id_col", "person_id"),
        tuple(lab_cfg.get("class_cols", ["D", "A", "S"])),
    )
    val = read_label_file(
        lab_cfg["val_label_path"],
        lab_cfg.get("id_col", "person_id"),
        tuple(lab_cfg.get("class_cols", ["D", "A", "S"])),
    )

    rows = [summarize_split(train, "train"), summarize_split(val, "val")]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_root / "label_distribution.csv", index=False)
    with open(out_root / "label_distribution.json", "w") as f:
        json.dump(rows, f, indent=2)

    # bar: pos rates
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, [rows[0]["D_pos_rate"], rows[0]["A_pos_rate"], rows[0]["S_pos_rate"]], w, label="train")
    ax.bar(x + w / 2, [rows[1]["D_pos_rate"], rows[1]["A_pos_rate"], rows[1]["S_pos_rate"]], w, label="val")
    ax.set_xticks(x)
    ax.set_xticklabels(["D", "A", "S"])
    ax.set_ylabel("positive rate")
    ax.legend()
    ax.set_title("Label positive rate: train vs val")
    fig.tight_layout()
    fig.savefig(out_root / "label_pos_rate_bar.png", dpi=dpi)
    plt.close(fig)

    # comorbidity
    keys = ["DA_pos_rate", "DS_pos_rate", "AS_pos_rate", "DAS_all_pos_rate"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    ax.bar(x - w / 2, [rows[0][k] for k in keys], w, label="train")
    ax.bar(x + w / 2, [rows[1][k] for k in keys], w, label="val")
    ax.set_xticks(x)
    ax.set_xticklabels(["D+A", "D+S", "A+S", "D+A+S"])
    ax.set_ylabel("rate")
    ax.legend()
    ax.set_title("Comorbidity rates")
    fig.tight_layout()
    fig.savefig(out_root / "comorbidity_bar.png", dpi=dpi)
    plt.close(fig)

    log.info("Label analysis -> %s", out_root)


if __name__ == "__main__":
    main()
