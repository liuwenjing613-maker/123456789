#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import yaml

project_root = Path("/home/adodas/AdoDAS2026-split")

base_config = project_root / "configs_pretcn" / "a1_baseline_official.yaml"
out_dir = project_root / "configs_internal"
out_dir.mkdir(parents=True, exist_ok=True)

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

for split in range(3):
    new_cfg = dict(cfg)

    new_cfg["manifest_dir"] = f"/home/adodas/dataset/manifests_internal/split_{split}_school_class"
    new_cfg["output_dir"] = f"/home/adodas/AdoDAS2026-split/outputs_internal/split{split}/a1_baseline_low_session"
    new_cfg["val_sequence_path_split"] = "train"

    new_cfg["dropout"] = 0.3
    new_cfg["weight_decay"] = 0.02
    new_cfg["label_smoothing"] = 0.03
    new_cfg["feature_noise_std"] = 0.01
    new_cfg["session_drop_prob"] = 0.10

    new_cfg["session_loss_weight"] = 0.05
    new_cfg["session_type_loss_weight"] = 0.05

    path = out_dir / f"a1_baseline_low_session_split{split}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)

    print("wrote:", path)
