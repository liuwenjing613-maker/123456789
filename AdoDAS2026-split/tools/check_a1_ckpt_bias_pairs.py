#!/usr/bin/env python3
"""Verify each checkpoints/*.pt has a matching sidecar *.bias.json for A1 runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.runner import checkpoint_bias_path, sha256_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"[NO] checkpoints directory missing: {ckpt_dir}")
        sys.exit(1)

    any_missing = False
    for pt_path in sorted(ckpt_dir.glob("*.pt")):
        bias_path = checkpoint_bias_path(pt_path)
        if not bias_path.exists():
            print(f"[NO] {pt_path.name:24s} -> missing sidecar bias")
            any_missing = True
            continue
        with open(bias_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        name_ok = data.get("checkpoint_name") in (None, "", pt_path.name)
        sha_ok = True
        exp_sha = data.get("checkpoint_sha256")
        if exp_sha:
            try:
                sha_ok = sha256_file(pt_path) == exp_sha
            except OSError:
                sha_ok = False
        tag = "[OK]" if name_ok and sha_ok else "[NO]"
        if not name_ok or not sha_ok:
            any_missing = True
        vec = data.get("bias_vector", [0.0, 0.0, 0.0])
        ep = data.get("epoch", "?")
        reason = data.get("selection_reason", "")
        extra = ""
        if not name_ok:
            extra += f" name_mismatch(want {pt_path.name})"
        if not sha_ok:
            extra += " sha256_mismatch"
        print(
            f"{tag} {pt_path.name:24s} -> {bias_path.name:30s} epoch={ep} "
            f"bias=[{vec[0]:+.2f},{vec[1]:+.2f},{vec[2]:+.2f}] {reason}{extra}"
        )

    sys.exit(1 if any_missing else 0)


if __name__ == "__main__":
    main()
