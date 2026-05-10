"""
Pre-TCN dynamics augmentation (legacy single-window and selective_v2 per-group multi-window).

Mask-aware: invalid frames are zeroed; rolling/delta use the same mask rules as the original pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def dynamics_aug_num_components(
    use_delta: bool,
    use_abs_delta: bool,
    use_rolling_mean: bool,
    use_rolling_std: bool,
) -> int:
    """Match legacy `dynamics_augment_sequence` concat count: D_aug = D * n."""
    n = 1
    if use_delta or use_abs_delta:
        if use_delta:
            n += 1
        if use_abs_delta:
            n += 1
    if use_rolling_mean:
        n += 1
    if use_rolling_std:
        n += 1
    return n


def masked_rolling_mean(
    x: np.ndarray,
    mask: np.ndarray,
    window: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    t_len, d_dim = x.shape
    out = np.zeros_like(x, dtype=np.float32)
    half = window // 2

    for t in range(t_len):
        left = max(0, t - half)
        right = min(t_len, t + half + 1)
        local_x = x[left:right]
        local_m = mask[left:right]
        if np.count_nonzero(local_m) > 0:
            out[t] = local_x[local_m].mean(axis=0)
        else:
            out[t] = 0.0

    out[~mask] = 0.0
    return out


def masked_rolling_std(
    x: np.ndarray,
    mask: np.ndarray,
    window: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    t_len, d_dim = x.shape
    out = np.zeros_like(x, dtype=np.float32)
    half = window // 2

    for t in range(t_len):
        left = max(0, t - half)
        right = min(t_len, t + half + 1)
        local_x = x[left:right]
        local_m = mask[left:right]
        if np.count_nonzero(local_m) > 1:
            out[t] = local_x[local_m].std(axis=0)
        else:
            out[t] = 0.0

    out[~mask] = 0.0
    return out


def dynamics_augment_sequence(
    x: np.ndarray,
    mask: np.ndarray,
    window: int,
    use_delta: bool,
    use_abs_delta: bool,
    use_rolling_mean: bool,
    use_rolling_std: bool,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    x[~mask] = 0.0

    parts: list[np.ndarray] = [x]

    if use_delta or use_abs_delta:
        delta = np.zeros_like(x, dtype=np.float32)
        delta[1:] = x[1:] - x[:-1]
        valid_delta = np.zeros_like(mask, dtype=bool)
        valid_delta[1:] = mask[1:] & mask[:-1]
        delta[~valid_delta] = 0.0
        if use_delta:
            parts.append(delta)
        if use_abs_delta:
            parts.append(np.abs(delta))

    if use_rolling_mean:
        parts.append(masked_rolling_mean(x, mask, window))
    if use_rolling_std:
        parts.append(masked_rolling_std(x, mask, window))

    x_aug = np.concatenate(parts, axis=-1).astype(np.float32)
    x_aug[~mask] = 0.0
    x_aug = np.nan_to_num(x_aug, nan=0.0, posinf=0.0, neginf=0.0)
    x_aug = np.clip(x_aug, -10.0, 10.0)
    return x_aug


def apply_dynamics_aug_to_groups(
    groups: dict[str, torch.Tensor],
    mask: np.ndarray,
    apply_names: frozenset[str],
    window: int,
    use_delta: bool,
    use_abs_delta: bool,
    use_rolling_mean: bool,
    use_rolling_std: bool,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    m = np.asarray(mask, dtype=bool)
    for name, t in groups.items():
        if name in apply_names:
            x = np.asarray(t.detach().cpu().numpy(), dtype=np.float32, order="C")
            x_aug = dynamics_augment_sequence(
                x,
                m,
                window=window,
                use_delta=use_delta,
                use_abs_delta=use_abs_delta,
                use_rolling_mean=use_rolling_mean,
                use_rolling_std=use_rolling_std,
            )
            out[name] = torch.from_numpy(x_aug)
        else:
            out[name] = t.float().clone()
    return out


def selective_v2_num_components(cfg: dict[str, Any]) -> int:
    """Number of concat blocks (raw + ...), each width D, for D_aug = D * n."""
    c = _coerce_group_cfg(cfg)
    n = 1
    if c["use_delta"] or c["use_abs_delta"]:
        if c["use_delta"]:
            n += 1
        if c["use_abs_delta"]:
            n += 1
    for _w in c["windows"]:
        if c["use_rolling_mean"]:
            n += 1
        if c["use_rolling_std"]:
            n += 1
    return n


def _coerce_group_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    w = cfg.get("windows", [])
    if w is None:
        w = []
    if not isinstance(w, (list, tuple)):
        w = [w]
    windows = [int(x) for x in w if int(x) > 0]
    return {
        "use_delta": bool(cfg.get("use_delta", False)),
        "use_abs_delta": bool(cfg.get("use_abs_delta", False)),
        "use_rolling_mean": bool(cfg.get("use_rolling_mean", False)),
        "use_rolling_std": bool(cfg.get("use_rolling_std", False)),
        "windows": windows,
    }


def dynamics_augment_sequence_selective(
    x: np.ndarray,
    mask: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    c = _coerce_group_cfg(cfg)
    x = np.asarray(x, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    x[~mask] = 0.0

    parts: list[np.ndarray] = [x]

    if c["use_delta"] or c["use_abs_delta"]:
        delta = np.zeros_like(x, dtype=np.float32)
        delta[1:] = x[1:] - x[:-1]
        valid_delta = np.zeros_like(mask, dtype=bool)
        valid_delta[1:] = mask[1:] & mask[:-1]
        delta[~valid_delta] = 0.0
        if c["use_delta"]:
            parts.append(delta)
        if c["use_abs_delta"]:
            parts.append(np.abs(delta))

    for w in c["windows"]:
        if c["use_rolling_mean"]:
            parts.append(masked_rolling_mean(x, mask, w))
        if c["use_rolling_std"]:
            parts.append(masked_rolling_std(x, mask, w))

    x_aug = np.concatenate(parts, axis=-1).astype(np.float32)
    x_aug[~mask] = 0.0
    x_aug = np.nan_to_num(x_aug, nan=0.0, posinf=0.0, neginf=0.0)
    x_aug = np.clip(x_aug, -10.0, 10.0)
    return x_aug


def apply_dynamics_aug_to_groups_selective(
    groups: dict[str, torch.Tensor],
    mask: np.ndarray,
    group_cfgs: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    m = np.asarray(mask, dtype=bool)
    for name, t in groups.items():
        if name in group_cfgs:
            x = np.asarray(t.detach().cpu().numpy(), dtype=np.float32, order="C")
            x_aug = dynamics_augment_sequence_selective(x, m, group_cfgs[name])
            out[name] = torch.from_numpy(x_aug)
        else:
            out[name] = t.float().clone()
    return out
