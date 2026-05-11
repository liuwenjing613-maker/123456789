#!/usr/bin/env python3
"""Export logits via tools/export_a1_logits.py and write raw/calibrated prediction CSVs."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from analysis.scripts.utils import load_calibration_biases, load_yaml, sigmoid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_export(repo: Path, checkpoint: Path, split: str, out_npz: Path, batch_size: int | None) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo / "tools" / "export_a1_logits.py"),
        "--checkpoint",
        str(checkpoint),
        "--split",
        split,
        "--out",
        str(out_npz),
    ]
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))


def write_probs_csv(ids: np.ndarray, probs: np.ndarray, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"person_id": ids, "p_D": probs[:, 0], "p_A": probs[:, 1], "p_S": probs[:, 2]}
    )
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skip-export", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    repo = Path(cfg["paths"]["repo_root"])
    exp = cfg.get("exports") or {}
    inter = Path(exp.get("intermediate_dir", repo / "analysis/outputs/intermediate_logits"))
    inter.mkdir(parents=True, exist_ok=True)

    base = exp.get("baseline") or {}
    dyn = exp.get("dynamic") or {}

    plan = [
        ("baseline", "val", base.get("checkpoint"), inter / "baseline_val_logits.npz"),
        ("baseline", "test_hidden", base.get("checkpoint"), inter / "baseline_test_hidden_logits.npz"),
        ("dynamic", "val", dyn.get("checkpoint"), inter / "dynamic_val_logits.npz"),
        ("dynamic", "test_hidden", dyn.get("checkpoint"), inter / "dynamic_test_hidden_logits.npz"),
    ]

    if exp.get("enabled") and not args.skip_export:
        for _name, split, ckpt, npz_out in plan:
            if not ckpt:
                raise ValueError("exports.baseline.checkpoint / dynamic.checkpoint required when export enabled")
            ckpt_p = Path(ckpt)
            if not ckpt_p.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
            run_export(repo, ckpt_p, split, npz_out, args.batch_size)

    b_base = load_calibration_biases(base.get("calibration_json", ""))
    b_dyn = load_calibration_biases(dyn.get("calibration_json", ""))

    def process_pair(npz_path: Path, raw_csv: Path, cal_csv: Path, bias: np.ndarray | None) -> None:
        if not npz_path.is_file():
            log.warning("Skip (missing npz): %s", npz_path)
            return
        z = np.load(npz_path)
        ids = z["ids"]
        logits = z["logits"].astype(np.float64)
        write_probs_csv(ids, sigmoid(logits), raw_csv)
        if bias is not None:
            write_probs_csv(ids, sigmoid(logits + bias.reshape(1, -1)), cal_csv)

    process_pair(
        inter / "baseline_val_logits.npz",
        inter / "baseline_val_raw.csv",
        inter / "baseline_val_calibrated.csv",
        b_base,
    )
    process_pair(
        inter / "dynamic_val_logits.npz",
        inter / "dynamic_val_raw.csv",
        inter / "dynamic_val_calibrated.csv",
        b_dyn,
    )
    process_pair(
        inter / "baseline_test_hidden_logits.npz",
        inter / "baseline_test_hidden_raw.csv",
        inter / "baseline_test_hidden_calibrated.csv",
        b_base,
    )
    process_pair(
        inter / "dynamic_test_hidden_logits.npz",
        inter / "dynamic_test_hidden_raw.csv",
        inter / "dynamic_test_hidden_calibrated.csv",
        b_dyn,
    )


if __name__ == "__main__":
    main()
