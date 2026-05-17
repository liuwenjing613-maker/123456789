#!/usr/bin/env python3
"""K-fold driver: kfold=0 single train; kfold=3/5 OOF + ensemble with resume."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.eval_a1 import (
    evaluate_a1_pred_csv,
    format_a1_metrics_log,
    write_a1_metrics_json,
)

PHASES = ("all", "train_oof", "merge_oof", "ensemble")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fold_out_dir(work: Path, fold_idx: int) -> Path:
    return work / f"fold_{fold_idx}" / "outputs" / "a1"


def fold_oof_path(work: Path, fold_idx: int) -> Path:
    return work / "oof" / f"fold_{fold_idx}_val_pred.csv"


def ensemble_val_path(work: Path, fold_idx: int) -> Path:
    return work / "ensemble" / f"fold_{fold_idx}_official_val.csv"


def ensemble_test_path(work: Path, fold_idx: int) -> Path:
    return work / "ensemble" / f"fold_{fold_idx}_test.csv"


def find_latest_run(output_dir: Path) -> Path:
    runs = sorted((output_dir / "runs").glob("a1__*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No runs under {output_dir}/runs")
    return runs[-1]


def fold_ckpt_path(work: Path, fold_idx: int, ckpt_name: str) -> Path | None:
    fold_out = fold_out_dir(work, fold_idx)
    if not (fold_out / "runs").exists():
        return None
    try:
        run_dir = find_latest_run(fold_out)
    except FileNotFoundError:
        return None
    ckpt = run_dir / "checkpoints" / ckpt_name
    return ckpt if ckpt.is_file() else None


def is_fold_train_oof_done(work: Path, fold_idx: int, ckpt_name: str) -> bool:
    return fold_ckpt_path(work, fold_idx, ckpt_name) is not None and fold_oof_path(
        work, fold_idx
    ).is_file()


def is_ensemble_fold_done(work: Path, fold_idx: int) -> bool:
    return ensemble_val_path(work, fold_idx).is_file() and ensemble_test_path(
        work, fold_idx
    ).is_file()


def load_oof_part(work: Path, fold_idx: int, ckpt_name: str) -> pd.DataFrame:
    path = fold_oof_path(work, fold_idx)
    if not path.is_file():
        raise FileNotFoundError(f"Missing OOF predictions: {path}")
    part = pd.read_csv(path)
    part["fold"] = fold_idx
    part["checkpoint_name"] = ckpt_name
    return part


def load_base_cfg(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    fs = base_cfg.pop("feature_selection", {}) or {}
    if isinstance(fs, dict):
        base_cfg.update(fs)
    return base_cfg


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


def progress_path(work: Path) -> Path:
    return work / "kfold_progress.json"


def load_progress(work: Path) -> dict[str, Any]:
    path = progress_path(work)
    if not path.is_file():
        return {"folds": {}, "updated_at": None}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_progress(work: Path, progress: dict[str, Any]) -> None:
    progress["updated_at"] = _utc_now()
    path = progress_path(work)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def mark_progress(
    work: Path,
    progress: dict[str, Any],
    fold_idx: int,
    step: str,
    *,
    write: bool,
) -> None:
    key = str(fold_idx)
    progress.setdefault("folds", {}).setdefault(key, {})[step] = _utc_now()
    if write:
        save_progress(work, progress)


def ensure_manifests(args: argparse.Namespace, work: Path, manifest_root: Path) -> None:
    summary = manifest_root / "fold_summary.csv"
    if summary.is_file() and not args.remake_manifests:
        print(f"SKIP make_a1_kfold_manifests (exists: {summary})", flush=True)
        return
    _run([
        sys.executable,
        str(ROOT / "tools" / "make_a1_kfold_manifests.py"),
        "--manifest_dir",
        str(Path(args.manifest_dir)),
        "--out_dir",
        str(manifest_root),
        "--kfold",
        str(args.kfold),
        "--seed",
        str(args.seed),
    ])


def phase_train_oof(
    args: argparse.Namespace,
    work: Path,
    manifest_root: Path,
    base_cfg: dict[str, Any],
) -> list[pd.DataFrame]:
    ckpt_name = args.checkpoint_name
    oof_parts: list[pd.DataFrame] = []
    progress = load_progress(work) if args.write_progress else {"folds": {}}

    for fold_idx in range(args.kfold):
        if fold_idx < args.start_fold:
            if is_fold_train_oof_done(work, fold_idx, ckpt_name):
                print(f"SKIP fold_{fold_idx} train_oof (before start_fold, already done)", flush=True)
                oof_parts.append(load_oof_part(work, fold_idx, ckpt_name))
            else:
                raise RuntimeError(
                    f"fold_{fold_idx} is incomplete but start_fold={args.start_fold}. "
                    f"Lower --start_fold or complete fold_{fold_idx} first."
                )
            continue

        if args.skip_completed and is_fold_train_oof_done(work, fold_idx, ckpt_name):
            print(f"SKIP fold_{fold_idx} train_oof (already done)", flush=True)
            oof_parts.append(load_oof_part(work, fold_idx, ckpt_name))
            continue

        fold_manifest = manifest_root / f"fold_{fold_idx}"
        fold_out = fold_out_dir(work, fold_idx)
        fold_cfg = write_fold_config(base_cfg, fold_manifest, fold_out, fold_idx, work)
        _run([
            sys.executable,
            str(ROOT / "train.py"),
            "--task",
            "a1",
            "--config",
            str(fold_cfg),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ])
        run_dir = find_latest_run(fold_out)
        ckpt = run_dir / "checkpoints" / ckpt_name
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found after training: {ckpt}")

        fold_val_pred = fold_oof_path(work, fold_idx)
        fold_val_pred.parent.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable,
            str(ROOT / "infer.py"),
            "--task",
            "a1",
            "--checkpoint",
            str(ckpt),
            "--config",
            str(fold_cfg),
            "--manifest",
            str(fold_manifest / "val.csv"),
            "--split",
            "val",
            "--output",
            str(fold_val_pred),
            "--a1_bias_mode",
            "none",
        ])
        part = load_oof_part(work, fold_idx, ckpt_name)
        oof_parts.append(part)
        mark_progress(work, progress, fold_idx, "train_oof", write=args.write_progress)
        print(f"fold_{fold_idx} train_oof done -> {fold_val_pred}", flush=True)

    return oof_parts


def _metrics_json_path(pred_csv: Path) -> Path:
    return pred_csv.with_suffix(pred_csv.suffix + ".metrics.json")


def eval_official_val_predictions(
    pred_csv: Path,
    manifest_csv: Path,
    *,
    title: str,
    refresh: bool = False,
) -> dict[str, Any]:
    """Write <pred>.csv.metrics.json and print F1/AUROC summary."""
    pred_csv = Path(pred_csv)
    manifest_csv = Path(manifest_csv)
    out_json = _metrics_json_path(pred_csv)
    if (
        not refresh
        and out_json.is_file()
        and out_json.stat().st_mtime >= pred_csv.stat().st_mtime
    ):
        with open(out_json, encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"SKIP eval (up-to-date): {out_json.name}", flush=True)
    else:
        metrics = evaluate_a1_pred_csv(pred_csv, manifest_csv)
        metrics["eval_title"] = title
        write_a1_metrics_json(metrics, out_json)
        print(format_a1_metrics_log(metrics, title=title), flush=True)
        print(f"Wrote {out_json}", flush=True)
    return metrics


def phase_merge_oof(
    args: argparse.Namespace,
    work: Path,
    oof_parts: list[pd.DataFrame] | None = None,
) -> Path:
    ckpt_name = args.checkpoint_name
    if oof_parts is None:
        oof_parts = []
        missing: list[int] = []
        for fold_idx in range(args.kfold):
            if fold_oof_path(work, fold_idx).is_file():
                oof_parts.append(load_oof_part(work, fold_idx, ckpt_name))
            else:
                missing.append(fold_idx)
        if missing:
            raise FileNotFoundError(
                f"Missing OOF files for folds {missing}. Run --phase train_oof first."
            )

    if len(oof_parts) != args.kfold:
        raise RuntimeError(
            f"Expected {args.kfold} OOF parts, got {len(oof_parts)}. "
            "Check fold_*_val_pred.csv under oof/."
        )

    oof_dir = work / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    out = oof_dir / "oof_predictions.csv"
    pd.concat(oof_parts, ignore_index=True).to_csv(out, index=False)
    print(f"Wrote {out}", flush=True)

    if args.write_progress:
        progress = load_progress(work)
        progress["merge_oof"] = _utc_now()
        save_progress(work, progress)
    return out


def phase_ensemble(args: argparse.Namespace, work: Path) -> Path:
    ckpt_name = args.checkpoint_name
    official_manifest = Path(args.manifest_dir)
    val_preds: list[Path] = []
    test_preds: list[Path] = []
    progress = load_progress(work) if args.write_progress else {"folds": {}}

    for fold_idx in range(args.kfold):
        fold_cfg = work / "configs" / f"fold_{fold_idx}.yaml"
        if not fold_cfg.is_file():
            raise FileNotFoundError(
                f"Missing fold config: {fold_cfg}. Run train_oof phase first."
            )
        ckpt = fold_ckpt_path(work, fold_idx, ckpt_name)
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint for fold_{fold_idx}: {ckpt_name}. Run train_oof first."
            )

        vp = ensemble_val_path(work, fold_idx)
        tp = ensemble_test_path(work, fold_idx)
        vp.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_completed and is_ensemble_fold_done(work, fold_idx):
            print(f"SKIP fold_{fold_idx} ensemble infer (already done)", flush=True)
        else:
            # Official val/test use dataset/val and dataset/test_hidden; fold yaml sets
            # val_sequence_path_split=train for internal K-fold val only.
            _run([
                sys.executable,
                str(ROOT / "infer.py"),
                "--task",
                "a1",
                "--checkpoint",
                str(ckpt),
                "--config",
                str(fold_cfg),
                "--manifest",
                str(official_manifest / "val.csv"),
                "--split",
                "val",
                "--path_split",
                "val",
                "--output",
                str(vp),
                "--a1_bias_mode",
                "none",
            ])
            _run([
                sys.executable,
                str(ROOT / "infer.py"),
                "--task",
                "a1",
                "--checkpoint",
                str(ckpt),
                "--config",
                str(fold_cfg),
                "--manifest",
                str(official_manifest / "test_hidden.csv"),
                "--split",
                "test_hidden",
                "--path_split",
                "test_hidden",
                "--output",
                str(tp),
                "--a1_bias_mode",
                "none",
            ])
            mark_progress(work, progress, fold_idx, "ensemble", write=args.write_progress)

        if args.eval_official_val:
            eval_official_val_predictions(
                vp,
                official_manifest / "val.csv",
                title=f"official val fold_{fold_idx}",
            )

        val_preds.append(vp)
        test_preds.append(tp)

    ens_dir = work / "ensemble"
    val_out = ens_dir / "official_val_ensemble_raw.csv"
    test_out = ens_dir / "test_ensemble_raw.csv"
    _run([
        sys.executable,
        str(ROOT / "tools" / "ensemble_a1_predictions.py"),
        "--pred_csvs",
        *map(str, val_preds),
        "--output",
        str(val_out),
    ])
    _run([
        sys.executable,
        str(ROOT / "tools" / "ensemble_a1_predictions.py"),
        "--pred_csvs",
        *map(str, test_preds),
        "--output",
        str(test_out),
    ])

    val_metrics_all: dict[str, Any] = {}
    if args.eval_official_val:
        ens_metrics = eval_official_val_predictions(
            val_out,
            official_manifest / "val.csv",
            title="official val ensemble",
        )
        val_metrics_all["ensemble"] = ens_metrics
        for fold_idx in range(args.kfold):
            vp = ensemble_val_path(work, fold_idx)
            mj = _metrics_json_path(vp)
            if mj.is_file():
                with open(mj, encoding="utf-8") as f:
                    val_metrics_all[f"fold_{fold_idx}"] = json.load(f)
        summary_path = ens_dir / "official_val_metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(val_metrics_all, f, indent=2, ensure_ascii=False)
        print(f"Wrote {summary_path}", flush=True)

    report = work / "final_report.md"
    oof_dir = work / "oof"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"# K-fold {args.kfold} report\n\n")
        f.write(f"- checkpoint for OOF/ensemble: `{ckpt_name}`\n")
        f.write(f"- oof: `{oof_dir / 'oof_predictions.csv'}`\n")
        f.write(f"- val ensemble: `{val_out}`\n")
        f.write(f"- test ensemble: `{test_out}`\n")
        if val_metrics_all:
            f.write("\n## Official val metrics (threshold=0.5)\n\n")
            f.write("| model | mean_f1 | macro_auroc | D_f1 | A_f1 | S_f1 |\n")
            f.write("|-------|---------|-------------|------|------|------|\n")
            for key in sorted(val_metrics_all.keys(), key=lambda k: (k != "ensemble", k)):
                m = val_metrics_all[key]
                f.write(
                    f"| {key} | {m['mean_f1']:.4f} | {m['macro_auroc']:.4f} | "
                    f"{m['D_f1']:.4f} | {m['A_f1']:.4f} | {m['S_f1']:.4f} |\n"
                )
            f.write(f"\nDetails: `{ens_dir / 'official_val_metrics_summary.json'}`\n")
        f.write("\nRecommended submissions:\n")
        f.write("1. test_ensemble_raw.csv\n")
        f.write("2. (optional) shrink0.3 after fitting OOF bias offline\n")
    print(report, flush=True)

    if args.write_progress:
        progress = load_progress(work)
        progress["ensemble"] = _utc_now()
        save_progress(work, progress)
    return test_out


def kfold0(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--task",
        "a1",
        "--config",
        args.config,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
    ]
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    _run(cmd)


def kfold_n(args: argparse.Namespace) -> None:
    if args.start_fold < 0 or args.start_fold >= args.kfold:
        raise ValueError(f"--start_fold must be in [0, {args.kfold - 1}], got {args.start_fold}")

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    manifest_root = work / "manifests" / f"kfold_{args.kfold}"
    phase = args.phase

    if phase in ("all", "train_oof"):
        ensure_manifests(args, work, manifest_root)

    base_cfg = load_base_cfg(args.config)
    oof_parts: list[pd.DataFrame] | None = None

    if phase in ("all", "train_oof"):
        oof_parts = phase_train_oof(args, work, manifest_root, base_cfg)

    if phase in ("all", "merge_oof"):
        phase_merge_oof(args, work, oof_parts)

    if phase in ("all", "ensemble"):
        phase_ensemble(args, work)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K-fold driver with resume: skip completed folds, run by phase.",
    )
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
    p.add_argument(
        "--start_fold",
        type=int,
        default=0,
        help="Only train folds with index >= start_fold (earlier folds still loaded for OOF if done).",
    )
    p.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="all | train_oof | merge_oof | ensemble",
    )
    p.add_argument(
        "--skip_completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip fold steps when checkpoint/OOF/ensemble outputs already exist.",
    )
    p.add_argument(
        "--remake_manifests",
        action="store_true",
        help="Regenerate kfold manifests even if fold_summary.csv exists.",
    )
    p.add_argument(
        "--write_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write work_dir/kfold_progress.json after each completed step.",
    )
    p.add_argument(
        "--eval_official_val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After official val predictions, compute mean F1 and macro AUROC vs val.csv.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.kfold == 0:
        if args.phase != "all" or args.start_fold != 0:
            print("WARNING: --phase/--start_fold ignored for kfold=0", flush=True)
        kfold0(args)
    else:
        kfold_n(args)
