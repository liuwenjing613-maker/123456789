#!/usr/bin/env python3
"""Run full A1 dataset / calibration analysis pipeline (md §11)."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.scripts.utils import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRIPTS = ROOT / "analysis" / "scripts"


def run_step(name: str, script: str, config: Path, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), "--config", str(config)]
    if extra:
        cmd.extend(extra)
    log.info("=== %s ===", name)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="analysis/configs/analysis_a1.yaml")
    p.add_argument("--prepare-predictions", action="store_true", help="Run prepare_prediction_csvs first")
    p.add_argument("--skip-labels", action="store_true")
    p.add_argument("--skip-predictions", action="store_true")
    p.add_argument("--skip-threshold", action="store_true")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--skip-session", action="store_true")
    p.add_argument("--skip-drift", action="store_true")
    p.add_argument("--skip-ensemble", action="store_true")
    p.add_argument("--skip-report", action="store_true")
    p.add_argument("--prepare-extra", nargs="*", default=[], help="Extra args for prepare_prediction_csvs")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    load_yaml(cfg_path)

    if args.prepare_predictions:
        run_step(
            "prepare_prediction_csvs",
            "prepare_prediction_csvs.py",
            cfg_path,
            list(args.prepare_extra) if args.prepare_extra else None,
        )

    if not args.skip_labels:
        run_step("analyze_labels", "analyze_labels.py", cfg_path)
    if not args.skip_predictions:
        run_step("analyze_predictions", "analyze_predictions.py", cfg_path)
    if not args.skip_threshold:
        run_step("analyze_threshold_stability", "analyze_threshold_stability.py", cfg_path)
    if not args.skip_calibration:
        run_step("analyze_calibration_bias", "analyze_calibration_bias.py", cfg_path)
    if not args.skip_session:
        run_step("analyze_session_quality", "analyze_session_quality.py", cfg_path)
    if not args.skip_drift:
        run_step("analyze_feature_drift", "analyze_feature_drift.py", cfg_path)
    if not args.skip_ensemble:
        run_step("build_ensemble_predictions", "build_ensemble_predictions.py", cfg_path)
    if not args.skip_report:
        run_step("generate_calibration_report", "generate_calibration_report.py", cfg_path)

    log.info("All requested steps finished.")


if __name__ == "__main__":
    main()
