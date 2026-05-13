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
    new_cfg["output_dir"] = f"/home/adodas/AdoDAS2026-split/outputs_internal/split{split}/a1_baseline_official"

    # internal split 的 val.csv 对应的序列文件通常仍在 train 路径下
    new_cfg["val_sequence_path_split"] = "train"

    path = out_dir / f"a1_baseline_official_split{split}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)

    print("wrote:", path)
