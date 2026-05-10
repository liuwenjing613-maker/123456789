#!/usr/bin/env python3
"""
训练入口。指定 GPU 的推荐方式（任选其一，需在 import torch 前生效）：

  export CUDA_VISIBLE_DEVICES=1,6
  python train.py --task a1 --config ...

或（由本仓库在 import torch 前解析）：

  python train.py --gpu 1,6 --task a1 --config ...

说明：设置后 PyTorch 仍显示 cuda:0，那是「可见列表里的第 1 张卡」；
OOM 报错里的 GPU 0 也是这个逻辑编号，不一定是机器上的物理 0 号卡。
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
    # Parse task argument to load correct config
    if len(sys.argv) > 1 and sys.argv[1] == "--task" and len(sys.argv) > 2:
        task = sys.argv[2]
        config_path = f"tasks/{task}/default.yaml"
        if Path(config_path).exists():
            sys.argv.insert(1, "--config")
            sys.argv.insert(2, config_path)

    main()
