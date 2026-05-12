#!/usr/bin/env python3
"""Export raw A1 logits (no checkpoint bias calibration) for val or test_hidden."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data.dataset import FeatureConfig
from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from common.models.grouped_model import GroupedModel
from common.models.heads import A1Head
from common.models.mtcn_backbone import BackboneConfig, MTCNBackbone
from common.runner import _to_device, setup_logging
from common.utils.ckpt import load_checkpoint


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export A1 logits to .npz for ensemble.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoints/best.pt")
    p.add_argument("--split", default="val", choices=["val", "test_hidden"])
    p.add_argument("--manifest", default=None, help="Override manifest CSV (default: manifest_dir/<split>.csv)")
    p.add_argument("--config", default=None, help="Override config (default: run_dir/config_used.yaml)")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    p.add_argument("--out", required=True, help="Output .npz path")
    return p.parse_args()


@torch.no_grad()
def collect_a1_logits_participant(
    grouped_model: GroupedModel,
    task_head: A1Head,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    has_labels: bool,
    desc: str,
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    grouped_model.eval()
    task_head.eval()
    all_logits: list[np.ndarray] = []
    all_ids: list[str] = []
    all_labels: list[np.ndarray] = []

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        flat_batch = _to_device(batch["flat_batch"], device)
        session_valid = batch["session_valid"].to(device)
        b = batch["n_participants"]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            out = grouped_model(flat_batch, b, session_valid)
            logits = task_head(out["participant_repr"]).float().cpu().numpy()
        all_logits.append(logits)

        for school, cls, pid in zip(batch["anon_schools"], batch["anon_classes"], batch["anon_pids"]):
            all_ids.append(f"{school}_{cls}_{pid}")

        if has_labels:
            all_labels.append(batch["participant_y_a1"].numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    if has_labels:
        labels_arr = np.concatenate(all_labels, axis=0)
    else:
        labels_arr = None
    return all_ids, logits_arr, labels_arr


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(args.config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = checkpoint_path.parent.parent
    setup_logging(run_dir / "logs", "export_a1_logits")

    submission_level = cfg.get("submission_level", "participant")
    if submission_level != "participant":
        raise ValueError(
            "export_a1_logits.py only supports participant-level A1 logits; "
            f"got submission_level={submission_level!r}"
        )

    manifest_dir = Path(cfg.get("manifest_dir", "/media/k3nwong/Data1/test/outputs/data"))
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
        pre_tcn_processing=cfg.get("pre_tcn_processing", defaults.pre_tcn_processing),
    )

    ds = GroupedParticipantDataset(manifest_path, feat_cfg, split=args.split)
    ds.log_pre_tcn_diagnostics()
    preload = bool(cfg.get("preload", True))
    num_workers = int(cfg.get("num_workers", 8))
    if preload:
        ds.preload(desc=f"Preload {args.split}")
        num_workers = 0

    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 64))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
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
    task_head = A1Head(bb_cfg.d_shared).to(device)

    state = load_checkpoint(checkpoint_path, grouped_model, optimizer=None)
    task_head.load_state_dict(state["head_state_dict"])
    grouped_model.eval()
    task_head.eval()

    use_amp = bool(cfg.get("amp", True))
    has_labels = args.split == "val"

    ids, logits, labels = collect_a1_logits_participant(
        grouped_model,
        task_head,
        loader,
        device,
        use_amp,
        has_labels,
        desc=f"Export A1 logits {args.split}",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = np.asarray(["D", "A", "S"], dtype=str)

    if labels is not None:
        np.savez_compressed(
            out_path,
            ids=np.asarray(ids, dtype=str),
            logits=logits.astype(np.float32),
            labels=labels.astype(np.float32),
            columns=columns,
        )
    else:
        np.savez_compressed(
            out_path,
            ids=np.asarray(ids, dtype=str),
            logits=logits.astype(np.float32),
            columns=columns,
        )

    print(f"Saved logits to: {out_path}")
    print(f"logits shape: {logits.shape}")
    print(f"n ids: {len(ids)}")
    if labels is not None:
        print(f"labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
