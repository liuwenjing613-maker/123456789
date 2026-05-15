#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from common.data.dataset import FeatureConfig
from common.data.grouped_dataset import (
    GroupedParticipantDataset,
    grouped_collate_fn,
    maybe_default_internal_val_sequence_path_split,
    path_split_for_yaml,
)
from common.models.grouped_model import CORALHead, GroupedModel
from common.models.heads import A1Head, A2OrdinalHead
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone
from common.a1_checkpoint_utils import checkpoint_bias_path, sha256_file
from common.runner import (
    _normalize_decode_method,
    generate_submission_grouped,
    setup_logging,
)
from common.utils.ckpt import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["a1", "a2"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test_hidden")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--a1_bias_mode",
        choices=["none", "auto", "path"],
        default="none",
        help="A1 bias: none=raw; auto=checkpoint sidecar; path=--a1_bias_path",
    )
    parser.add_argument("--a1_bias_path", type=str, default=None)
    parser.add_argument(
        "--a1_bias_shrink",
        type=float,
        default=0.0,
        help="Multiply logit bias (e.g. 0.3/0.5); ignored when --a1_use_sidecar_shrink",
    )
    parser.add_argument(
        "--a1_use_sidecar_shrink",
        action="store_true",
        help="Use safe_submit_shrink from bias sidecar (auto/path modes)",
    )
    parser.add_argument(
        "--allow_legacy_a1_bias",
        action="store_true",
        help="Allow fallback to calibration/a1_bias_grouped.json (default off)",
    )
    parser.add_argument(
        "--dump_pred_stats",
        action="store_true",
        help="Write .pred_stats.json next to output csv",
    )
    return parser.parse_args()


def load_config(config_path: str | None, checkpoint_path: Path) -> dict:
    if config_path is None:
        candidate = checkpoint_path.parent.parent / "config_used.yaml"
        config_path = str(candidate)
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    feature_selection = cfg.pop("feature_selection", {}) or {}
    if isinstance(feature_selection, dict):
        cfg.update(feature_selection)
    return cfg


def resolve_a1_bias(
    checkpoint_path: Path,
    *,
    mode: str,
    bias_path: str | None = None,
    shrink: float = 0.0,
    use_sidecar_shrink: bool = False,
    allow_legacy_a1_bias: bool = False,
) -> tuple[list[float] | None, dict | None, Path | None]:
    checkpoint_path = Path(checkpoint_path)
    if mode == "none":
        return None, None, None
    if mode == "path":
        if not bias_path:
            raise ValueError("--a1_bias_mode path requires --a1_bias_path")
        candidate = Path(bias_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"[A1-BIAS] Bias path not found: {candidate}")
    elif mode == "auto":
        candidate = checkpoint_bias_path(checkpoint_path)
        if not candidate.exists():
            legacy_same_dir = checkpoint_path.with_suffix(".bias.json")
            if legacy_same_dir.exists():
                candidate = legacy_same_dir
        if not candidate.exists() and allow_legacy_a1_bias:
            legacy = checkpoint_path.parent.parent / "calibration" / "a1_bias_grouped.json"
            if legacy.exists():
                candidate = legacy
        if not candidate.exists():
            raise FileNotFoundError(
                f"[A1-BIAS] No matched bias for {checkpoint_path}\n"
                f"Expected: {checkpoint_bias_path(checkpoint_path)}\n"
                f"Use --a1_bias_mode none for raw inference."
            )
    else:
        raise ValueError(f"Unknown a1_bias_mode: {mode}")

    with open(candidate, encoding="utf-8") as f:
        payload = json.load(f)
    if "bias_vector" in payload:
        vec = payload["bias_vector"]
    elif "biases" in payload:
        b = payload["biases"]
        vec = [b["D"], b["A"], b["S"]] if isinstance(b, dict) else b
    else:
        raise ValueError(f"Invalid bias file: {candidate}")

    effective_shrink = float(shrink)
    if use_sidecar_shrink:
        if "safe_submit_shrink" not in payload:
            raise ValueError(
                f"[A1-BIAS] --a1_use_sidecar_shrink set but safe_submit_shrink missing in {candidate}"
            )
        effective_shrink = float(payload["safe_submit_shrink"])
    vec = [float(x) * effective_shrink for x in vec]

    if candidate.name.endswith(".bias.json") and candidate.name != "a1_bias_grouped.json":
        expected_name = payload.get("checkpoint_name")
        if expected_name and expected_name != checkpoint_path.name:
            raise ValueError(
                f"[A1-BIAS] checkpoint_name mismatch: bias={expected_name} "
                f"checkpoint={checkpoint_path.name}"
            )
        expected_sha = payload.get("checkpoint_sha256")
        if expected_sha:
            actual_sha = sha256_file(checkpoint_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"[A1-BIAS] checkpoint_sha256 mismatch\n"
                    f"bias:   {expected_sha}\nactual: {actual_sha}"
                )
    return vec, payload, candidate


def load_calibration(run_dir: Path, task: str) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    if task == "a1":
        return None, None, _normalize_decode_method("argmax")
    path = run_dir / "calibration" / "a2_threshold_offsets_grouped.json"
    if not path.exists():
        return None, None, _normalize_decode_method("expectation")
    with open(path) as f:
        data = json.load(f)
    selected_method = _normalize_decode_method(data.get("selected_decode_method", "expectation"))
    strategies = data.get("strategies", {})
    selected_strategy = data.get("selected_strategy", "")
    offsets = None
    if selected_strategy in strategies and "offsets" in strategies[selected_strategy]:
        offsets = torch.tensor(strategies[selected_strategy]["offsets"], dtype=torch.float32)
    return None, offsets, selected_method


def dump_pred_stats(csv_path: Path, sub: pd.DataFrame) -> None:
    from tools.summarize_a1_submission import summarize_submission_df

    stats = summarize_submission_df(sub)
    out = csv_path.with_suffix(csv_path.suffix + ".pred_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote pred stats: {out}")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(args.config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = checkpoint_path.parent.parent
    setup_logging(run_dir / "logs", f"infer_{args.task}")
    note = maybe_default_internal_val_sequence_path_split(cfg)
    if note:
        logging.getLogger().info(note)

    manifest_dir = Path(cfg.get("manifest_dir", "/home/adodas/dataset/manifests"))
    manifest_path = Path(args.manifest) if args.manifest else manifest_dir / f"{args.split}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg.get("feature_root", defaults.feature_root),
        audio_features=cfg.get("audio_features", defaults.audio_features),
        video_features=cfg.get("video_features", defaults.video_features),
        audio_ssl_model_tag=cfg.get("audio_ssl_model_tag", defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg.get("video_ssl_model_tag", defaults.video_ssl_model_tag),
        mask_policy=cfg.get("mask_policy", defaults.mask_policy),
        core_audio=cfg.get("core_audio", defaults.core_audio),
        core_video=cfg.get("core_video", defaults.core_video),
    )

    ds = GroupedParticipantDataset(
        manifest_path,
        feat_cfg,
        split=args.split,
        path_split=path_split_for_yaml(cfg, args.split),
    )
    preload = bool(cfg.get("preload", True))
    num_workers = int(cfg.get("num_workers", 8))
    if preload:
        ds.preload(desc=f"Preload {args.split}")
        num_workers = 0

    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=grouped_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    dims = ds.feature_dims
    bb_cfg = BackboneConfig(
        audio_group_dims={n: dims[n] for n in feat_cfg.audio_sequence_features if n in dims},
        audio_pooled_group_dims={n: dims[n] for n in feat_cfg.audio_pooled_features if n in dims},
        video_group_dims={n: dims[n] for n in feat_cfg.video_features if n in dims},
        d_adapter=cfg.get("d_adapter", 64),
        d_model=cfg.get("d_model", 256),
        tcn_layers=cfg.get("tcn_layers", 6),
        tcn_kernel_size=cfg.get("tcn_kernel_size", 3),
        asp_alpha=cfg.get("asp_alpha", 0.5),
        asp_beta=cfg.get("asp_beta", 0.5),
        dropout=cfg.get("dropout", 0.2),
        d_shared=cfg.get("d_shared", 256),
    )
    grouped_model = GroupedModel(
        backbone=MTCNBackbone(bb_cfg),
        d_shared=bb_cfg.d_shared,
        aggregator_method=cfg.get("aggregator", "mlp"),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    if args.task == "a1":
        task_head = A1Head(bb_cfg.d_shared).to(device)
    else:
        task_head = (
            CORALHead(bb_cfg.d_shared).to(device)
            if bool(cfg.get("use_coral", False))
            else A2OrdinalHead(bb_cfg.d_shared).to(device)
        )

    state = load_checkpoint(checkpoint_path, grouped_model, optimizer=None)
    task_head.load_state_dict(state["head_state_dict"])
    grouped_model.eval()
    task_head.eval()

    if args.task == "a1":
        a1_vec, _, _ = resolve_a1_bias(
            checkpoint_path,
            mode=args.a1_bias_mode,
            bias_path=args.a1_bias_path,
            shrink=args.a1_bias_shrink,
            use_sidecar_shrink=args.a1_use_sidecar_shrink,
            allow_legacy_a1_bias=args.allow_legacy_a1_bias,
        )
        a1_biases = None if a1_vec is None else np.asarray(a1_vec, dtype=np.float32)
        _, a2_offsets, selected_decode_method = None, None, _normalize_decode_method("argmax")
    else:
        a1_biases = None
        _, a2_offsets, selected_decode_method = load_calibration(run_dir, args.task)

    pids, sessions, preds = generate_submission_grouped(
        grouped_model=grouped_model,
        task_head=task_head,
        loader=loader,
        device=device,
        task=args.task,
        use_amp=bool(cfg.get("amp", True)),
        desc=f"Infer {args.split}",
        submission_level=cfg.get("submission_level", "participant"),
        a1_biases=a1_biases,
        decode_method=selected_decode_method,
        a2_threshold_offsets=None if a2_offsets is None else a2_offsets.to(device),
    )

    manifest_df = pd.read_csv(manifest_path)
    file_ids: list[str] = []
    filtered_preds: list = []
    submission_level = cfg.get("submission_level", "participant")
    if submission_level == "participant":
        pid_to_info = {}
        for _, row in manifest_df.iterrows():
            pid = str(row["anon_pid"])
            pid_to_info.setdefault(pid, (str(row["anon_school"]), str(row["anon_class"])))
        for pid, pred in zip(pids, preds):
            info = pid_to_info.get(str(pid))
            if info is None:
                continue
            school, cls = info
            file_ids.append(f"{school}_{cls}_{pid}")
            filtered_preds.append(pred)
    else:
        pid_to_info = {
            (str(row["anon_pid"]), str(row["session"])): (
                str(row["anon_school"]),
                str(row["anon_class"]),
            )
            for _, row in manifest_df.iterrows()
        }
        for pid, sess, pred in zip(pids, sessions, preds):
            key = (str(pid), str(sess))
            info = pid_to_info.get(key)
            if info is None:
                continue
            school, cls = info
            file_ids.append(f"{school}_{cls}_{key[0]}_{key[1]}")
            filtered_preds.append(pred)

    if args.task == "a1":
        sub = pd.DataFrame(
            {
                "file_id": file_ids,
                "p_D": [float(p[0]) for p in filtered_preds],
                "p_A": [float(p[1]) for p in filtered_preds],
                "p_S": [float(p[2]) for p in filtered_preds],
            }
        )
    else:
        sub = pd.DataFrame({"file_id": file_ids})
        for idx, col in enumerate([f"d{i:02d}" for i in range(1, 22)]):
            sub[col] = [int(p[idx]) for p in filtered_preds]

    output_path = (
        Path(args.output)
        if args.output
        else run_dir / "submissions" / f"submission_{args.task}_{args.split}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(output_path)

    if args.task == "a1" and args.dump_pred_stats:
        dump_pred_stats(output_path, sub)


if __name__ == "__main__":
    main()
