#!/usr/bin/env python3
"""Audit manifest participants for missing sequence features on disk."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data.dataset import FeatureConfig, SESSIONS
from common.data.grouped_dataset import GroupedParticipantDataset, path_split_for_yaml


def main() -> None:
    p = argparse.ArgumentParser(description="List participants with no loadable sessions")
    p.add_argument("--manifest", required=True, help="train.csv or val.csv path")
    p.add_argument("--feature_root", default="/home/adodas/dataset")
    p.add_argument("--path_split", default=None, help="Disk subfolder under feature_root (default: infer from --config)")
    p.add_argument("--config", default=None, help="Optional yaml for feature_selection + path_split")
    p.add_argument("--split", default="val", choices=["train", "val", "test_hidden"])
    p.add_argument("--output", default=None, help="Write dropped participants CSV here")
    args = p.parse_args()

    cfg_dict: dict = {"feature_root": args.feature_root}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        fs = loaded.pop("feature_selection", {}) or {}
        if isinstance(fs, dict):
            loaded.update(fs)
        cfg_dict.update(loaded)

    feat_cfg = FeatureConfig(
        feature_root=cfg_dict.get("feature_root", args.feature_root),
        audio_features=cfg_dict.get("audio_features", FeatureConfig().audio_features),
        video_features=cfg_dict.get("video_features", FeatureConfig().video_features),
        audio_ssl_model_tag=cfg_dict.get("audio_ssl_model_tag", FeatureConfig().audio_ssl_model_tag),
        video_ssl_model_tag=cfg_dict.get("video_ssl_model_tag", FeatureConfig().video_ssl_model_tag),
        mask_policy=cfg_dict.get("mask_policy", FeatureConfig().mask_policy),
        core_audio=cfg_dict.get("core_audio", FeatureConfig().core_audio),
        core_video=cfg_dict.get("core_video", FeatureConfig().core_video),
    )

    path_split = args.path_split or path_split_for_yaml(cfg_dict, args.split) or args.split

    manifest = pd.read_csv(args.manifest)
    group_cols = ["anon_school", "anon_class", "anon_pid"]
    rows_out = []

    probe_ds = GroupedParticipantDataset(
        args.manifest,
        feat_cfg,
        split=args.split,
        path_split=path_split,
    )

    # Build lookup of kept participants from filtered dataset
    kept_keys = {
        (p["anon_school"], p["anon_class"], p["anon_pid"])
        for p in probe_ds.participants
    }

    for (school, cls, pid), group in manifest.groupby(group_cols):
        key = (str(school), str(cls), str(pid))
        if key in kept_keys:
            continue
        sess_rows = {str(r["session"]): r for _, r in group.iterrows()}
        missing_sessions = [s for s in SESSIONS if s not in sess_rows]
        rows_out.append({
            "anon_school": school,
            "anon_class": cls,
            "anon_pid": pid,
            "n_manifest_sessions": len(sess_rows),
            "missing_session_names": ",".join(missing_sessions),
        })

    total = manifest.groupby(group_cols).ngroups
    dropped = len(rows_out)
    kept = total - dropped
    print(f"manifest={args.manifest}")
    print(f"path_split={path_split}  split={args.split}")
    print(f"participants total={total} kept={kept} dropped={dropped}")

    if rows_out:
        df = pd.DataFrame(rows_out)
        out = Path(args.output) if args.output else Path(args.manifest).with_suffix(".no_features.csv")
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows to {out}")
        print(df.head(10).to_string(index=False))
    else:
        print("All participants have at least one loadable session.")


if __name__ == "__main__":
    main()
