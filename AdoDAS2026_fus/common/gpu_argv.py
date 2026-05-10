"""
Parse --gpu before `import torch` so CUDA only sees the intended physical GPU(s).

PyTorch always names the first visible device `cuda:0`; OOM text "GPU 0" means that
logical index, not necessarily physical PCI index 0.
"""
from __future__ import annotations

import os
import sys


def apply_remove_gpu_arg(argv: list[str] | None = None) -> bool:
    """
    If argv contains `--gpu ID` or `--gpu=ID`, set os.environ['CUDA_VISIBLE_DEVICES'] and
    remove those tokens from argv (so argparse does not fail).

    Returns True if CUDA_VISIBLE_DEVICES was set from argv (possibly overriding shell).
    """
    if argv is None:
        argv = sys.argv
    original = list(argv)
    out: list[str] = []
    i = 0
    changed = False
    while i < len(original):
        a = original[i]
        if a == "--gpu" and i + 1 < len(original):
            val = original[i + 1].strip()
            if val:
                os.environ["CUDA_VISIBLE_DEVICES"] = val
                changed = True
            i += 2
            continue
        if a.startswith("--gpu=") and len(a) > len("--gpu="):
            val = a.split("=", 1)[1].strip()
            if val:
                os.environ["CUDA_VISIBLE_DEVICES"] = val
                changed = True
            i += 1
            continue
        out.append(a)
        i += 1
    if len(out) != len(original):
        argv[:] = out
    return changed
