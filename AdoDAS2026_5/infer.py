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
from common.runner import (
    _normalize_decode_method,
    a1_build_participant_submission_rows,
    generate_submission_grouped,
    setup_logging,
    sha256_file,
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
        default="auto",
        help="A1 bias mode. none: no bias. auto: load checkpoint sidecar .bias.json. path: use --a1_bias_path.",
    )
    parser.add_argument(
        "--a1_bias_path",
        type=str,
        default=None,
        help="Explicit A1 bias json path. Used when --a1_bias_mode path.",
    )
    parser.add_argument(
        "--a1_bias_shrink",
        type=float,
        default=1.0,
        help="Multiply loaded logit bias by this factor. Use 1.0 for exact matched bias, 0.3/0.5 for shrink.",
    )
    parser.add_argument(
        "--allow_legacy_run_bias",
        action="store_true",
        help="Allow fallback to calibration/a1_bias_grouped.json. Default False to avoid checkpoint-bias mismatch.",
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
    if not isinstance(feature_selection, dict):
        raise TypeError("feature_selection must be a mapping in the config YAML")
    cfg.update(feature_selection)
    return cfg


def resolve_a1_bias(
    checkpoint_path: Path,
    *,
    mode: str,
    bias_path: str | None = None,
    shrink: float = 1.0,
    allow_legacy_run_bias: bool = False,
) -> tuple[list[float] | None, dict | None, Path | None]:
    checkpoint_path = Path(checkpoint_path)

    if mode == "none":
        print("[A1-BIAS] mode=none, no bias will be applied.")
        return None, None, None

    if mode == "path":
        if not bias_path:
            raise ValueError("--a1_bias_mode path requires --a1_bias_path")
        candidate = Path(bias_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"[A1-BIAS] Bias path not found: {candidate}")
    elif mode == "auto":
        candidate = checkpoint_path.with_suffix(".bias.json")

        if not candidate.exists() and allow_legacy_run_bias:
            run_dir = checkpoint_path.parent.parent
            legacy = run_dir / "calibration" / "a1_bias_grouped.json"
            if legacy.exists():
                candidate = legacy

        if not candidate.exists():
            raise FileNotFoundError(
                f"[A1-BIAS] Cannot find matched bias for checkpoint: {checkpoint_path}\n"
                f"Expected sidecar: {checkpoint_path.with_suffix('.bias.json')}\n"
                f"Use --a1_bias_mode none for raw inference, or create sidecar bias first."
            )
    else:
        raise ValueError(f"Unknown a1_bias_mode: {mode}")

    with open(candidate, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "bias_vector" in payload:
        vec = payload["bias_vector"]
    elif "biases" in payload:
        b = payload["biases"]
        if isinstance(b, dict):
            vec = [b["D"], b["A"], b["S"]]
        else:
            vec = b
    else:
        raise ValueError(f"Invalid bias file: {candidate}. Missing bias_vector/biases.")

    vec = [float(x) * float(shrink) for x in vec]

    if candidate.name.endswith(".bias.json") and candidate.name != "a1_bias_grouped.json":
        expected_name = payload.get("checkpoint_name")
        if expected_name and expected_name != checkpoint_path.name:
            raise ValueError(
                f"[A1-BIAS] Bias checkpoint_name mismatch: "
                f"bias says {expected_name}, current checkpoint is {checkpoint_path.name}"
            )

        expected_sha = payload.get("checkpoint_sha256")
        if expected_sha:
            actual_sha = sha256_file(checkpoint_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"[A1-BIAS] checkpoint_sha256 mismatch for {checkpoint_path}\n"
                    f"bias file: {expected_sha}\n"
                    f"actual:    {actual_sha}"
                )

    print("[A1-BIAS] mode=", mode)
    print("[A1-BIAS] checkpoint=", checkpoint_path)
    print("[A1-BIAS] bias_file=", candidate)
    print("[A1-BIAS] shrink=", shrink)
    print("[A1-BIAS] applied_bias_vector=", vec)
    print("[A1-BIAS] bias_epoch=", payload.get("epoch"))
    print("[A1-BIAS] selection_reason=", payload.get("selection_reason"))

    return vec, payload, candidate


def load_calibration(run_dir: Path, task: str) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    calibration_dir = run_dir / "calibration"
    if task == "a1":
        a1_decode_placeholder = _normalize_decode_method("argmax")
        return None, None, a1_decode_placeholder

    path = calibration_dir / "a2_threshold_offsets_grouped.json"
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


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(args.config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = checkpoint_path.parent.parent
    setup_logging(run_dir / "logs", f"infer_{args.task}")
    note_internal = maybe_default_internal_val_sequence_path_split(cfg)
    if note_internal:
        logging.getLogger().info(note_internal)

    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))
    manifest_path = Path(args.manifest) if args.manifest else manifest_dir / f"{args.split}.csv"
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if args.output:
        out_candidate = Path(args.output).expanduser().resolve()
        if out_candidate == manifest_path:
            raise ValueError(
                "Refusing to write --output to the same file as the input manifest. "
                "That would replace anon_school/session rows with file_id,p_D,p_A,p_S and break all later infer. "
                "Use e.g. /home/adodas/dataset/results/<run_name>_a1_test_hidden.csv"
            )

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
        pre_tcn_processing=cfg.get("pre_tcn_processing", defaults.pre_tcn_processing),
    )

    ds = GroupedParticipantDataset(
        manifest_path,
        feat_cfg,
        split=args.split,
        path_split=path_split_for_yaml(cfg, args.split),
    )
    ds.log_pre_tcn_diagnostics()
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
        if bool(cfg.get("use_coral", False)):
            task_head = CORALHead(bb_cfg.d_shared).to(device)
        else:
            task_head = A2OrdinalHead(bb_cfg.d_shared).to(device)

    state = load_checkpoint(checkpoint_path, grouped_model, optimizer=None)
    task_head.load_state_dict(state["head_state_dict"])
    grouped_model.eval()
    task_head.eval()

    a1_bias_payload: dict | None = None
    a1_resolved_bias_path: Path | None = None
    if args.task == "a1":
        a1_vec, a1_bias_payload, a1_resolved_bias_path = resolve_a1_bias(
            checkpoint_path,
            mode=args.a1_bias_mode,
            bias_path=args.a1_bias_path,
            shrink=args.a1_bias_shrink,
            allow_legacy_run_bias=args.allow_legacy_run_bias,
        )
        a1_biases_np = None if a1_vec is None else np.asarray(a1_vec, dtype=np.float32)
        a2_offsets, selected_decode_method = None, _normalize_decode_method("argmax")
    else:
        a1_biases_np = None
        a1_resolved_bias_path = None
        _, a2_offsets, selected_decode_method = load_calibration(run_dir, args.task)
    use_amp = bool(cfg.get("amp", True))
    submission_level = cfg.get("submission_level", "participant")

    pids, sessions, preds = generate_submission_grouped(
        grouped_model=grouped_model,
        task_head=task_head,
        loader=loader,
        device=device,
        task=args.task,
        use_amp=use_amp,
        desc=f"Infer {args.split}",
        submission_level=submission_level,
        a1_biases=a1_biases_np,
        decode_method=selected_decode_method,
        a2_threshold_offsets=None if a2_offsets is None else a2_offsets.to(device),
    )

    manifest_df = pd.read_csv(manifest_path)
    file_ids: list[str] = []
    filtered_preds: list = []
    if submission_level == "participant":
        file_ids, preds_arr = a1_build_participant_submission_rows(pids, preds, manifest_df)
        filtered_preds = [preds_arr[i] for i in range(len(preds_arr))]
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
        # Parse file_ids to extract anon_school, anon_class, anon_pid
        anon_schools = []
        anon_classes = []
        anon_pids = []
        for fid in file_ids:
            parts = fid.split('_')
            if len(parts) >= 5:  # SCH_001_CLS_0015_P000224
                anon_schools.append(parts[1])  # 001
                anon_classes.append(parts[3])  # 0015
                anon_pids.append(parts[4][1:])  # 000224 (remove P)
            else:
                # Fallback
                anon_schools.append("")
                anon_classes.append("")
                anon_pids.append("")
        sub = pd.DataFrame(
            {
                "anon_school": anon_schools,
                "anon_class": anon_classes,
                "anon_pid": anon_pids,
                "p_D": [float(pred[0]) for pred in filtered_preds],
                "p_A": [float(pred[1]) for pred in filtered_preds],
                "p_S": [float(pred[2]) for pred in filtered_preds],
            }
        )
    else:
        sub = pd.DataFrame({"file_id": file_ids})
        for idx, col in enumerate([f"d{i:02d}" for i in range(1, 22)]):
            sub[col] = [int(pred[idx]) for pred in filtered_preds]

    output_path = Path(args.output) if args.output else run_dir / "submissions" / f"submission_{args.task}_{args.split}.csv"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)

    meta_path = output_path.parent / "submission_meta.json"
    submission_meta: dict = {
        "task": args.task,
        "checkpoint": str(checkpoint_path.resolve()),
        "output_csv": output_path.name,
    }
    if args.task == "a1":
        submission_meta.update(
            {
                "a1_bias_mode": args.a1_bias_mode,
                "a1_bias_file": str(a1_resolved_bias_path.resolve()) if a1_resolved_bias_path else None,
                "a1_bias_shrink": float(args.a1_bias_shrink),
                "a1_bias_vector": None if a1_biases_np is None else [float(x) for x in a1_biases_np.tolist()],
                "bias_epoch": (a1_bias_payload or {}).get("epoch"),
                "selection_reason": (a1_bias_payload or {}).get("selection_reason"),
            }
        )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(submission_meta, f, indent=2, ensure_ascii=False)

    print(output_path)


if __name__ == "__main__":
    main()
