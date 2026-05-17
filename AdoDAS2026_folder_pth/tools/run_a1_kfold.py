#!/usr/bin/env python3
"""K-fold driver: kfold=0 single train; kfold=3/5 OOF + ensemble."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def kfold0(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--task", "a1",
        "--config", args.config,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ]
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    _run(cmd)


def write_fold_config(
    base_cfg: dict,
    fold_manifest: Path,
    fold_out: Path,
    fold_idx: int,
    work_dir: Path,
) -> Path:
    cfg = dict(base_cfg)
    cfg["manifest_dir"] = str(fold_manifest)
    cfg["output_dir"] = str(fold_out)
    cfg["train_sequence_path_split"] = "train"
    cfg["val_sequence_path_split"] = "train"
    ckpt_sel = cfg.get("checkpoint_selection", {}) or {}
    if not isinstance(ckpt_sel, dict):
        ckpt_sel = {}
    cfg["checkpoint_selection"] = ckpt_sel
    cfg_dir = work_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out = cfg_dir / f"fold_{fold_idx}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out


def find_latest_run(output_dir: Path) -> Path:
    runs = sorted((output_dir / "runs").glob("a1__*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No runs under {output_dir}/runs")
    return runs[-1]


def kfold_n(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    manifest_root = work / "manifests" / f"kfold_{args.kfold}"
    _run([
        sys.executable, str(ROOT / "tools" / "make_a1_kfold_manifests.py"),
        "--manifest_dir", str(Path(args.manifest_dir)),
        "--out_dir", str(manifest_root),
        "--kfold", str(args.kfold),
        "--seed", str(args.seed),
    ])

    with open(args.config, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    fs = base_cfg.pop("feature_selection", {}) or {}
    if isinstance(fs, dict):
        base_cfg.update(fs)

    oof_parts: list[pd.DataFrame] = []
    ckpt_name = args.checkpoint_name

    for fold_idx in range(args.kfold):
        fold_manifest = manifest_root / f"fold_{fold_idx}"
        fold_out = work / f"fold_{fold_idx}" / "outputs" / "a1"
        fold_cfg = write_fold_config(base_cfg, fold_manifest, fold_out, fold_idx, work)
        _run([
            sys.executable, str(ROOT / "train.py"),
            "--task", "a1",
            "--config", str(fold_cfg),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
        ])
        run_dir = find_latest_run(fold_out)
        ckpt = run_dir / "checkpoints" / ckpt_name
        fold_val_pred = work / "oof" / f"fold_{fold_idx}_val_pred.csv"
        fold_val_pred.parent.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable, str(ROOT / "infer.py"),
            "--task", "a1",
            "--checkpoint", str(ckpt),
            "--config", str(fold_cfg),
            "--manifest", str(fold_manifest / "val.csv"),
            "--split", "val",
            "--output", str(fold_val_pred),
            "--a1_bias_mode", "none",
        ])
        part = pd.read_csv(fold_val_pred)
        part["fold"] = fold_idx
        part["checkpoint_name"] = ckpt_name
        oof_parts.append(part)

    oof_dir = work / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.concat(oof_parts, ignore_index=True)
    oof_df.to_csv(oof_dir / "oof_predictions.csv", index=False)

    official_manifest = Path(args.manifest_dir)
    val_preds = []
    test_preds = []
    for fold_idx in range(args.kfold):
        fold_cfg = oof_dir.parent / "configs" / f"fold_{fold_idx}.yaml"
        fold_out = work / f"fold_{fold_idx}" / "outputs" / "a1"
        run_dir = find_latest_run(fold_out)
        ckpt = run_dir / "checkpoints" / ckpt_name
        vp = work / "ensemble" / f"fold_{fold_idx}_official_val.csv"
        tp = work / "ensemble" / f"fold_{fold_idx}_test.csv"
        vp.parent.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable, str(ROOT / "infer.py"),
            "--task", "a1", "--checkpoint", str(ckpt), "--config", str(fold_cfg),
            "--manifest", str(official_manifest / "val.csv"), "--split", "val",
            "--output", str(vp), "--a1_bias_mode", "none",
        ])
        _run([
            sys.executable, str(ROOT / "infer.py"),
            "--task", "a1", "--checkpoint", str(ckpt), "--config", str(fold_cfg),
            "--manifest", str(official_manifest / "test_hidden.csv"),
            "--split", "test_hidden",
            "--output", str(tp), "--a1_bias_mode", "none",
        ])
        val_preds.append(vp)
        test_preds.append(tp)

    ens_dir = work / "ensemble"
    _run([
        sys.executable, str(ROOT / "tools" / "ensemble_a1_predictions.py"),
        "--pred_csvs", *map(str, val_preds),
        "--output", str(ens_dir / "official_val_ensemble_raw.csv"),
    ])
    _run([
        sys.executable, str(ROOT / "tools" / "ensemble_a1_predictions.py"),
        "--pred_csvs", *map(str, test_preds),
        "--output", str(ens_dir / "test_ensemble_raw.csv"),
    ])

    report = work / "final_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"# K-fold {args.kfold} report\n\n")
        f.write(f"- checkpoint for OOF/ensemble: `{ckpt_name}`\n")
        f.write(f"- oof: `{oof_dir / 'oof_predictions.csv'}`\n")
        f.write(f"- val ensemble: `{ens_dir / 'official_val_ensemble_raw.csv'}`\n")
        f.write(f"- test ensemble: `{ens_dir / 'test_ensemble_raw.csv'}`\n")
        f.write("\nRecommended submissions:\n")
        f.write("1. test_ensemble_raw.csv\n")
        f.write("2. (optional) shrink0.3 after fitting OOF bias offline\n")
    print(report)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "tasks/a1/default.yaml"))
    p.add_argument("--kfold", type=int, default=0, choices=[0, 3, 5])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work_dir", default=str(ROOT / "outputs_folder_pth/a1/kfold0_baseline"))
    p.add_argument("--manifest_dir", default="/home/adodas/dataset/manifests")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--checkpoint_name", default="best_safe_submit.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.kfold == 0:
        kfold0(args)
    else:
        kfold_n(args)
