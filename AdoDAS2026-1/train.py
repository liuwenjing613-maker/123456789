#!/usr/bin/env python3
"""
训练入口。指定 GPU：export CUDA_VISIBLE_DEVICES=1,6 或 python train.py --gpu 1,6 --task a1 ...
（PyTorch 日志里的 cuda:0 / OOM「GPU 0」是逻辑编号 = 可见列表中的第一张。）
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from common.gpu_argv import apply_remove_gpu_arg  # noqa: E402

apply_remove_gpu_arg()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from common.runner import main


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--task" and len(sys.argv) > 2:
        task = sys.argv[2]
        config_path = f"tasks/{task}/default.yaml"
        if Path(config_path).exists():
            sys.argv.insert(1, "--config")
            sys.argv.insert(2, config_path)

    main()
