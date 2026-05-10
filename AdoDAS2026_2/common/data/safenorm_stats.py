"""
TCN 前的 SafeNorm：仅用 train split 统计的 mean/std，在 Dataset 阶段做归一化 + clip。

与模型侧「低维跳过 LayerNorm」配合：数据侧统一尺度，模型侧避免 per-frame LN 破坏物理量。
统计与应用都必须使用各特征组自身的 aligned mask，避免无效帧污染 mean/std。
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# npz 内数组名前缀，与组名拼接（组名中不含双下划线即可）
_MEAN_PREFIX = "mean__"
_STD_PREFIX = "std__"
_META_KEY = "safenorm_meta.json"


class FeatureStatsAccumulator:
    """按特征组累计 sum / sqsum / count，仅统计 mask 为 True 的时间步。"""

    def __init__(self) -> None:
        self._sum: dict[str, np.ndarray | None] = {}
        self._sqsum: dict[str, np.ndarray | None] = {}
        self._count: dict[str, float] = defaultdict(float)

    def update_sequence(
        self, group_name: str, x: np.ndarray, mask: np.ndarray
    ) -> None:
        """
        x: (T, D) 对齐后的序列特征
        mask: (T,) bool，与数据集 aligned_masks 一致
        """
        valid = mask.astype(bool)
        if valid.sum() == 0:
            return
        x = x.astype(np.float64, copy=False)
        x_valid = x[valid]
        d = x_valid.shape[1]
        cur_sum = x_valid.sum(axis=0)
        cur_sq = (x_valid ** 2).sum(axis=0)
        n = float(x_valid.shape[0])

        if self._sum.get(group_name) is None:
            self._sum[group_name] = cur_sum
            self._sqsum[group_name] = cur_sq
        else:
            self._sum[group_name] = self._sum[group_name] + cur_sum  # type: ignore[operator]
            self._sqsum[group_name] = self._sqsum[group_name] + cur_sq  # type: ignore[operator]
        self._count[group_name] += n

    def update_pooled_vector(
        self, group_name: str, x: np.ndarray, present: bool
    ) -> None:
        """pooled 特征 egemaps 等：(D,) 整段会话一个向量；present=False 则跳过。"""
        if not present:
            return
        x = x.astype(np.float64, copy=False).ravel()
        d = x.size
        cur_sum = x
        cur_sq = x ** 2
        n = 1.0

        if self._sum.get(group_name) is None:
            self._sum[group_name] = cur_sum.copy()
            self._sqsum[group_name] = cur_sq.copy()
        else:
            self._sum[group_name] = self._sum[group_name] + cur_sum  # type: ignore[operator]
            self._sqsum[group_name] = self._sqsum[group_name] + cur_sq  # type: ignore[operator]
        self._count[group_name] += n

    def finalize(self, eps: float = 1e-6) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        for group_name, s in self._sum.items():
            if s is None or self._count[group_name] < 1.0:
                continue
            cnt = max(self._count[group_name], 1.0)
            mean = (s / cnt).astype(np.float32)
            sqmean = (self._sqsum[group_name] / cnt).astype(np.float64)  # type: ignore[operator]
            var = np.maximum(sqmean - (mean.astype(np.float64) ** 2), eps)
            std = np.sqrt(var).astype(np.float32)
            out[group_name] = {"mean": mean, "std": std}
        return out


def save_stats_npz(
    path: str | Path,
    stats: dict[str, dict[str, np.ndarray]],
    norm_type: str = "mean_std",
    eps: float = 1e-6,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, d in stats.items():
        arrays[_MEAN_PREFIX + name] = d["mean"]
        arrays[_STD_PREFIX + name] = d["std"]
    meta = {"norm_type": norm_type, "eps": eps, "groups": sorted(stats.keys())}
    np.savez_compressed(path, **arrays)
    with open(path.with_suffix(path.suffix + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_stats_npz(path: str | Path) -> dict[str, Any]:
    """返回 {"stats": {name: {mean, std}}, "meta": {...}}；文件不存在则 stats 为空 dict。"""
    path = Path(path)
    if not path.is_file():
        log.warning("SafeNorm stats file not found: %s", path)
        return {"stats": {}, "meta": {}}
    raw = np.load(path, allow_pickle=False)
    stats: dict[str, dict[str, np.ndarray]] = {}
    mean_keys = [k for k in raw.files if k.startswith(_MEAN_PREFIX)]
    for mk in mean_keys:
        name = mk[len(_MEAN_PREFIX) :]
        sk = _STD_PREFIX + name
        if sk not in raw.files:
            continue
        stats[name] = {
            "mean": raw[mk].astype(np.float32),
            "std": raw[sk].astype(np.float32),
        }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = json.load(f)
    log.info("Loaded SafeNorm stats for %d groups from %s", len(stats), path)
    return {"stats": stats, "meta": meta}


def apply_mean_std_clip(
    x: np.ndarray,
    group_name: str,
    stats: dict[str, dict[str, np.ndarray]],
    eps: float = 1e-6,
    clip_value: float | None = 5.0,
) -> np.ndarray:
    """
    x: (T, D) 或 (D,) pooled
    若该组不在 stats 中，返回原数组（float32），并 debug 一次。
    """
    if group_name not in stats:
        log.debug("SafeNorm: no stats for group %r, pass-through", group_name)
        return x.astype(np.float32, copy=False)
    mean = stats[group_name]["mean"]
    std = stats[group_name]["std"]
    y = (x.astype(np.float32) - mean) / (std.astype(np.float32) + eps)
    if clip_value is not None:
        y = np.clip(y, -clip_value, clip_value)
    return y.astype(np.float32)
