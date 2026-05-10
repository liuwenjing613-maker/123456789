#!/usr/bin/env python3
"""
仅使用 train split 的 manifest 扫描所有 session 行，计算各特征组 mean/std 并写入 npz。

禁止用 val/test 参与统计（数据泄露）。请只传入 train.csv。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# 保证可从仓库根目录运行: python scripts/compute_feature_stats.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.data.dataset import FeatureConfig, align_to_grid  # noqa: E402
from common.data.feature_io import load_egemaps_pooled, load_sequence  # noqa: E402
from common.data.safenorm_stats import (  # noqa: E402
    FeatureStatsAccumulator,
    save_stats_npz,
)


def _load_yaml_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    feature_selection = cfg.pop("feature_selection", {}) or {}
    if not isinstance(feature_selection, dict):
        raise TypeError("feature_selection must be a mapping")
    cfg.update(feature_selection)
    return cfg


def _load_raw_groups(row: pd.Series, root: Path, split: str, cfg: FeatureConfig, modality: str):
    from common.data.feature_io import SequenceData

    feat_list = cfg.audio_sequence_features if modality == "audio" else cfg.video_features
    groups: dict[str, SequenceData] = {}
    for feat_name in feat_list:
        tag = None
        if feat_name == "ssl_embed":
            tag = cfg.audio_ssl_model_tag
        elif feat_name == "vision_ssl_embed":
            tag = cfg.video_ssl_model_tag
        try:
            seq = load_sequence(
                root,
                split,
                str(row["anon_school"]),
                str(row["anon_class"]),
                str(row["anon_pid"]),
                modality,
                feat_name,
                str(row["session"]),
                model_tag=tag,
            )
            groups[feat_name] = seq
        except FileNotFoundError:
            pass
    return groups


def main() -> None:
    p = argparse.ArgumentParser(description="Compute train-only SafeNorm feature stats (mean/std).")
    p.add_argument("--manifest", type=Path, required=True, help="Path to train.csv (train split only)")
    p.add_argument("--config", type=Path, required=True, help="Training YAML (same as train.py --config)")
    p.add_argument("--output", type=Path, required=True, help="Output .npz path")
    p.add_argument("--feature-root", type=Path, default=None, help="Override feature_root from config")
    p.add_argument("--split", type=str, default="train", help="Split name passed to load_sequence (default train)")
    args = p.parse_args()

    cfg_dict = _load_yaml_config(args.config)
    if args.feature_root is not None:
        cfg_dict["feature_root"] = str(args.feature_root)

    defaults = FeatureConfig()
    feat_cfg = FeatureConfig(
        feature_root=cfg_dict.get("feature_root", defaults.feature_root),
        audio_features=cfg_dict.get("audio_features", defaults.audio_features),
        video_features=cfg_dict.get("video_features", defaults.video_features),
        audio_ssl_model_tag=cfg_dict.get("audio_ssl_model_tag", defaults.audio_ssl_model_tag),
        video_ssl_model_tag=cfg_dict.get("video_ssl_model_tag", defaults.video_ssl_model_tag),
        mask_policy=cfg_dict.get("mask_policy", defaults.mask_policy),
        core_audio=cfg_dict.get("core_audio", defaults.core_audio),
        core_video=cfg_dict.get("core_video", defaults.core_video),
    )

    manifest = pd.read_csv(args.manifest)
    required = {"anon_school", "anon_class", "anon_pid", "session"}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {missing}")

    root = Path(feat_cfg.feature_root)
    acc = FeatureStatsAccumulator()
    n_ok = 0
    n_skip = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Accumulate stats"):
        audio_raw = _load_raw_groups(row, root, args.split, feat_cfg, "audio")
        video_raw = _load_raw_groups(row, root, args.split, feat_cfg, "video")
        all_groups = {}
        for k, v in audio_raw.items():
            all_groups[f"audio/{k}"] = v
        for k, v in video_raw.items():
            all_groups[f"video/{k}"] = v
        if not all_groups:
            n_skip += 1
            continue
        try:
            aligned_feats, aligned_masks, _grid, _T = align_to_grid(
                all_groups, feat_cfg.grid_step_ms, feat_cfg.tolerance_ms
            )
        except Exception:
            n_skip += 1
            continue

        for key, feat in aligned_feats.items():
            _modality, name = key.split("/", 1)
            acc.update_sequence(name, feat.astype(np.float64), aligned_masks[key])

        if "egemaps" in feat_cfg.audio_pooled_features:
            eg = load_egemaps_pooled(
                root,
                args.split,
                str(row["anon_school"]),
                str(row["anon_class"]),
                str(row["anon_pid"]),
                str(row["session"]),
            )
            if eg is not None:
                acc.update_pooled_vector("egemaps", np.asarray(eg, dtype=np.float64), present=True)

        n_ok += 1

    sn = cfg_dict.get("pre_tcn_safenorm") or {}
    eps = float(sn.get("eps", 1e-6))
    stats = acc.finalize(eps=eps)
    save_stats_npz(
        args.output,
        stats,
        norm_type=str(sn.get("norm_type", "mean_std")),
        eps=eps,
    )
    print(f"Wrote {args.output} ({len(stats)} groups), sessions_ok={n_ok}, sessions_skip={n_skip}")


if __name__ == "__main__":
    main()
