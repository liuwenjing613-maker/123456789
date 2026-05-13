from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataset import (
    SESSIONS, SESSION_TO_IDX, ITEM_COLS, A1_COLS,
    FeatureConfig, align_to_grid,
)
from .feature_io import SequenceData, load_egemaps_pooled, load_sequence
from .pre_tcn_processing import (
    apply_dynamics_aug_to_groups,
    apply_dynamics_aug_to_groups_selective,
    dynamics_aug_num_components,
    selective_v2_num_components,
)

log = logging.getLogger(__name__)


def path_split_for_yaml(cfg: Mapping[str, Any] | None, logical_split: str) -> str | None:
    """Which subfolder under ``feature_root`` holds ``sequence.npz`` for this logical split.

    Internal CV manifests often use ``val.csv`` for a fold's validation *labels*, while extracted
    features for those participants still live under the global ``train/`` tree. In that case set
    ``val_sequence_path_split: train`` (or ``sequence_path_split: {val: train}``).
    """
    if not cfg:
        return None
    flat_keys = {
        "train": "train_sequence_path_split",
        "val": "val_sequence_path_split",
        "test_hidden": "test_hidden_sequence_path_split",
    }
    fk = flat_keys.get(logical_split)
    if fk:
        v = cfg.get(fk)
        if v is not None and str(v).strip():
            return str(v).strip()
    nested = cfg.get("sequence_path_split")
    if isinstance(nested, dict):
        v = nested.get(logical_split)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def maybe_default_internal_val_sequence_path_split(cfg: dict[str, Any]) -> str | None:
    """If ``manifest_dir`` is an internal CV layout, default val loader to ``train/`` on disk.

    GroupKFold ``val.csv`` rows are still official-train participants; extracted
    ``sequence.npz`` almost always lives under ``{feature_root}/train/...``, not
    ``val/``. Call after loading YAML; mutates ``cfg`` only when unset.
    """
    vs = cfg.get("val_sequence_path_split")
    if vs is not None and str(vs).strip():
        return None
    nested = cfg.get("sequence_path_split")
    if isinstance(nested, dict):
        v = nested.get("val")
        if v is not None and str(v).strip():
            return None
    md = Path(str(cfg.get("manifest_dir", ""))).expanduser()
    md_str = str(md)
    last = md.name
    looks_internal = "manifests_internal" in md_str or (
        last.startswith("split_") and re.match(r"split_\d+_", last) is not None
    )
    if not looks_internal:
        return None
    cfg["val_sequence_path_split"] = "train"
    return (
        "manifest_dir looks like internal CV; set val_sequence_path_split=train "
        "(load val.csv participants from feature_root/train/, not val/)."
    )


def _filter_selective_map(
    raw: Mapping[str, Any],
    exclude: frozenset[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        kn = str(k)
        if kn in exclude:
            log.warning("[PreTCN] selective_v2: skip %r in %s (exclude)", kn, label)
            continue
        if not isinstance(v, dict):
            raise TypeError(
                f"dynamics_aug.mode=selective_v2: {label}[{kn!r}] must be a mapping, got {type(v)}"
            )
        out[kn] = dict(v)
    return out


def _parse_pre_tcn_dynamics_cfg(
    cfg: FeatureConfig,
) -> tuple[
    bool,
    bool,
    str,
    int,
    bool,
    bool,
    bool,
    bool,
    frozenset[str],
    frozenset[str],
    int,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Parse DynamicsAug: legacy list-style apply_* or selective_v2 per-group dicts."""
    raw = getattr(cfg, "pre_tcn_processing", None)
    pre: Mapping[str, Any] = raw if isinstance(raw, dict) else {}

    pre_enabled = bool(pre.get("enabled", False))
    dyn_block = pre.get("dynamics_aug") or {}
    if not isinstance(dyn_block, dict):
        dyn_block = {}

    dyn_enabled = bool(dyn_block.get("enabled", False))

    exclude = frozenset(str(x) for x in (dyn_block.get("exclude") or (
        "ssl_embed",
        "vision_ssl_embed",
        "mel_mfcc",
        "egemaps",
    )))

    mode_raw = dyn_block.get("mode", "legacy")
    mode_str = str(mode_raw).strip().lower() if mode_raw is not None else "legacy"
    if mode_str in ("", "uniform", "legacy"):
        mode_str = "legacy"
    elif mode_str != "selective_v2":
        raise ValueError(f"Unknown dynamics_aug.mode: {mode_raw!r}")

    window = int(dyn_block.get("window", 5))
    use_delta = bool(dyn_block.get("use_delta", True))
    use_abs_delta = bool(dyn_block.get("use_abs_delta", True))
    use_rm = bool(dyn_block.get("use_rolling_mean", True))
    use_rs = bool(dyn_block.get("use_rolling_std", True))

    selective_audio: dict[str, dict[str, Any]] = {}
    selective_video: dict[str, dict[str, Any]] = {}

    if mode_str == "selective_v2":
        aa = dyn_block.get("apply_audio")
        av = dyn_block.get("apply_video")
        if not isinstance(aa, dict):
            raise TypeError(
                "dynamics_aug.mode=selective_v2 requires apply_audio to be a mapping (dict)"
            )
        if not isinstance(av, dict):
            raise TypeError(
                "dynamics_aug.mode=selective_v2 requires apply_video to be a mapping (dict)"
            )
        selective_audio = _filter_selective_map(aa, exclude, "apply_audio")
        selective_video = _filter_selective_map(av, exclude, "apply_video")
        apply_audio = frozenset(selective_audio.keys())
        apply_video = frozenset(selective_video.keys())
        n_comp = 1
        return (
            pre_enabled,
            dyn_enabled,
            mode_str,
            window,
            use_delta,
            use_abs_delta,
            use_rm,
            use_rs,
            apply_audio,
            apply_video,
            n_comp,
            selective_audio,
            selective_video,
        )

    def _filt(names: list[str]) -> frozenset[str]:
        return frozenset(n for n in names if n and n not in exclude)

    apply_audio = _filt(
        list(dyn_block.get("apply_audio") or ["vad"]),
    )
    apply_video = _filt(
        list(
            dyn_block.get("apply_video")
            or [
                "vad_agg",
                "headpose_geom",
                "face_behavior",
                "qc_stats",
                "body_pose",
                "global_motion",
            ]
        ),
    )

    n_comp = dynamics_aug_num_components(use_delta, use_abs_delta, use_rm, use_rs)
    return (
        pre_enabled,
        dyn_enabled,
        mode_str,
        window,
        use_delta,
        use_abs_delta,
        use_rm,
        use_rs,
        apply_audio,
        apply_video,
        n_comp,
        selective_audio,
        selective_video,
    )


class GroupedParticipantDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        cfg: FeatureConfig,
        split: str,
        session_drop_prob: float = 0.0,
        path_split: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.path_split = path_split.strip() if isinstance(path_split, str) and path_split.strip() else split
        self.root = Path(cfg.feature_root)
        self.session_drop_prob = float(session_drop_prob)

        if self.path_split != self.split:
            log.info(
                "GroupedParticipantDataset logical split=%r -> loading sequences from %r/ "
                "(manifest %s)",
                self.split,
                self.path_split,
                Path(manifest_path).name,
            )

        manifest = pd.read_csv(manifest_path)

        group_cols = ["anon_school", "anon_class", "anon_pid"]
        grouped = manifest.groupby(group_cols)

        self.participants: list[dict[str, Any]] = []
        for (school, cls, pid), group in grouped:
            sess_rows = {}
            for _, row in group.iterrows():
                sess = str(row["session"])
                sess_rows[sess] = row

            any_row = group.iloc[0]
            y_a1 = np.array([float(any_row.get(c, -1)) for c in A1_COLS], dtype=np.float32)
            y_a2 = np.array([float(any_row.get(c, -1)) for c in ITEM_COLS], dtype=np.float32)

            self.participants.append({
                "anon_school": str(school),
                "anon_class": str(cls),
                "anon_pid": str(pid),
                "sess_rows": sess_rows,
                "y_a1": y_a1,
                "y_a2": y_a2,
            })

        (
            self._pre_tcn_processing_enabled,
            self._dynamics_aug_enabled,
            self._dynamics_aug_mode,
            self._dynamics_aug_window,
            self._dynamics_aug_use_delta,
            self._dynamics_aug_use_abs_delta,
            self._dynamics_aug_use_rolling_mean,
            self._dynamics_aug_use_rolling_std,
            self._dynamics_aug_apply_audio,
            self._dynamics_aug_apply_video,
            self._dynamics_aug_n_components,
            self._dynamics_aug_selective_audio_cfgs,
            self._dynamics_aug_selective_video_cfgs,
        ) = _parse_pre_tcn_dynamics_cfg(cfg)
        self._dynamics_aug_active = (
            self._pre_tcn_processing_enabled and self._dynamics_aug_enabled
        )

        self._feature_dims: dict[str, int] | None = None
        self._cache: list[dict | None] | None = None

    @property
    def feature_dims(self) -> dict[str, int]:
        if self._feature_dims is None:
            self._feature_dims = self._probe_dims()
        return self._feature_dims

    def _probe_dims(self) -> dict[str, int]:
        info = self.participants[0]
        sess_rows = info["sess_rows"]
        any_sess = list(sess_rows.keys())[0]
        row = sess_rows[any_sess]
        dims: dict[str, int] = {}
        for name, seq in self._load_raw_groups(row, "audio").items():
            dims[name] = int(seq.features.shape[1])
        for name, seq in self._load_raw_groups(row, "video").items():
            dims[name] = int(seq.features.shape[1])
        if "egemaps" in self.cfg.audio_pooled_features:
            eg = load_egemaps_pooled(
                self.root, self.path_split,
                str(row["anon_school"]), str(row["anon_class"]),
                str(row["anon_pid"]), str(row["session"]),
            )
            if eg is not None:
                dims["egemaps"] = len(eg)

        # 与 _load_single_session 中 DynamicsAug 一致：被增强的序列维从 D -> D * n_components
        if self._dynamics_aug_active:
            if self._dynamics_aug_mode == "selective_v2":
                for name, gcfg in self._dynamics_aug_selective_audio_cfgs.items():
                    if name == "egemaps" or name not in dims:
                        continue
                    dims[name] = int(dims[name] * selective_v2_num_components(gcfg))
                for name, gcfg in self._dynamics_aug_selective_video_cfgs.items():
                    if name == "egemaps" or name not in dims:
                        continue
                    dims[name] = int(dims[name] * selective_v2_num_components(gcfg))
            else:
                mul = self._dynamics_aug_n_components
                for name in list(dims.keys()):
                    if name == "egemaps":
                        continue
                    if name in self._dynamics_aug_apply_audio or name in self._dynamics_aug_apply_video:
                        dims[name] = int(dims[name] * mul)
        return dims

    def _raw_sequence_dims_one_row(self) -> dict[str, int]:
        """首条 manifest 行的原始特征宽（不做 DynamicsAug），供日志 raw -> aug 对照。"""
        info = self.participants[0]
        sess_rows = info["sess_rows"]
        any_sess = list(sess_rows.keys())[0]
        row = sess_rows[any_sess]
        raw: dict[str, int] = {}
        for name, seq in self._load_raw_groups(row, "audio").items():
            raw[name] = int(seq.features.shape[1])
        for name, seq in self._load_raw_groups(row, "video").items():
            raw[name] = int(seq.features.shape[1])
        if "egemaps" in self.cfg.audio_pooled_features:
            eg = load_egemaps_pooled(
                self.root, self.path_split,
                str(row["anon_school"]), str(row["anon_class"]),
                str(row["anon_pid"]), str(row["session"]),
            )
            if eg is not None:
                raw["egemaps"] = len(eg)
        return raw

    def log_pre_tcn_diagnostics(self) -> None:
        """训练启动时打印 DynamicsAug 是否生效及各组维度（文档 §12）。"""
        if not self._pre_tcn_processing_enabled:
            log.info("[PreTCN] pre_tcn_processing.enabled=False")
            return
        if not self._dynamics_aug_enabled:
            log.info("[PreTCN] pre_tcn_processing.enabled=True, dynamics_aug.enabled=False")
            return

        log.info("[PreTCN] DynamicsAug enabled: True")
        log.info("[PreTCN] mode: %s", self._dynamics_aug_mode)
        if self._dynamics_aug_mode == "selective_v2":
            for k in sorted(self._dynamics_aug_selective_audio_cfgs):
                log.info("[PreTCN]   audio[%s]: %s", k, self._dynamics_aug_selective_audio_cfgs[k])
            for k in sorted(self._dynamics_aug_selective_video_cfgs):
                log.info("[PreTCN]   video[%s]: %s", k, self._dynamics_aug_selective_video_cfgs[k])
        else:
            log.info("[PreTCN] window: %s", self._dynamics_aug_window)
            log.info(
                "[PreTCN] apply_audio: %s",
                sorted(self._dynamics_aug_apply_audio),
            )
            log.info(
                "[PreTCN] apply_video: %s",
                sorted(self._dynamics_aug_apply_video),
            )
            log.info(
                "[PreTCN] use_delta=%s use_abs_delta=%s use_rolling_mean=%s use_rolling_std=%s",
                self._dynamics_aug_use_delta,
                self._dynamics_aug_use_abs_delta,
                self._dynamics_aug_use_rolling_mean,
                self._dynamics_aug_use_rolling_std,
            )

        raw = self._raw_sequence_dims_one_row()
        aug = self.feature_dims
        if self._dynamics_aug_mode == "selective_v2":
            targets = sorted(
                set(self._dynamics_aug_selective_audio_cfgs) | set(self._dynamics_aug_selective_video_cfgs)
            )
        else:
            targets = sorted(self._dynamics_aug_apply_audio | self._dynamics_aug_apply_video)
        for name in targets:
            if name not in aug:
                continue
            if name in raw:
                log.info("[PreTCN] %s: %s -> %s", name, raw[name], aug[name])
            else:
                log.info(
                    "[PreTCN] %s: (raw missing on probe row) -> %s",
                    name,
                    aug[name],
                )

    def _load_raw_groups(self, row, modality: str) -> dict[str, SequenceData]:
        cfg = self.cfg
        feat_list = cfg.audio_sequence_features if modality == "audio" else cfg.video_features
        groups: dict[str, SequenceData] = {}
        for feat_name in feat_list:
            tag: str | None = None
            if feat_name == "ssl_embed":
                tag = cfg.audio_ssl_model_tag
            elif feat_name == "vision_ssl_embed":
                tag = cfg.video_ssl_model_tag
            try:
                seq = load_sequence(
                    self.root, self.path_split,
                    str(row["anon_school"]), str(row["anon_class"]),
                    str(row["anon_pid"]),
                    modality, feat_name, str(row["session"]),
                    model_tag=tag,
                )
                groups[feat_name] = seq
            except FileNotFoundError:
                pass
        return groups

    def _compute_modality_mask(
        self, mask_parts, mask_names, core_names, policy, T
    ) -> np.ndarray:
        if not mask_parts:
            return np.zeros(T, dtype=bool)
        if policy == "or":
            return np.any(np.stack(mask_parts), axis=0)
        if policy == "and_core":
            core_masks = [m for m, n in zip(mask_parts, mask_names) if n in core_names]
            if core_masks:
                return np.all(np.stack(core_masks), axis=0)
            return np.any(np.stack(mask_parts), axis=0)
        if policy == "require_k":
            k = max(1, len(core_names))
            stacked = np.stack(mask_parts)
            return np.sum(stacked, axis=0) >= k
        raise ValueError(f"Unknown mask_policy: {policy!r}")

    def _load_single_session(self, row) -> dict[str, Any] | None:
        """Load features for a single session. Returns None on failure."""
        cfg = self.cfg
        try:
            audio_raw = self._load_raw_groups(row, "audio")
            video_raw = self._load_raw_groups(row, "video")

            all_groups = {}
            for k, v in audio_raw.items():
                all_groups[f"audio/{k}"] = v
            for k, v in video_raw.items():
                all_groups[f"video/{k}"] = v

            if not all_groups:
                return None

            aligned_feats, aligned_masks, grid_ms, T = align_to_grid(
                all_groups, cfg.grid_step_ms, cfg.tolerance_ms
            )

            audio_groups: dict[str, torch.Tensor] = {}
            video_groups: dict[str, torch.Tensor] = {}
            audio_mask_parts, audio_mask_names = [], []
            video_mask_parts, video_mask_names = [], []

            for key, feat in aligned_feats.items():
                modality, name = key.split("/", 1)
                mask = aligned_masks[key]
                t = torch.from_numpy(feat.astype(np.float32))
                if modality == "audio":
                    audio_groups[name] = t
                    audio_mask_parts.append(mask)
                    audio_mask_names.append(name)
                else:
                    video_groups[name] = t
                    video_mask_parts.append(mask)
                    video_mask_names.append(name)

            mask_audio = self._compute_modality_mask(
                audio_mask_parts, audio_mask_names, cfg.core_audio, cfg.mask_policy, T
            )
            mask_video = self._compute_modality_mask(
                video_mask_parts, video_mask_names, cfg.core_video, cfg.mask_policy, T
            )

            vad_signal = np.zeros(T, dtype=np.float32)
            if "audio/vad" in aligned_feats:
                v = aligned_feats["audio/vad"]
                vad_signal = v[:, 0].astype(np.float32) * aligned_masks["audio/vad"].astype(np.float32)
            elif "video/vad_agg" in aligned_feats:
                v = aligned_feats["video/vad_agg"]
                vad_signal = v[:, 0].astype(np.float32) * aligned_masks["video/vad_agg"].astype(np.float32)

            qc_quality = np.zeros(T, dtype=np.float32)
            if "video/qc_stats" in aligned_feats:
                v = aligned_feats["video/qc_stats"]
                qc_quality = v[:, 0].astype(np.float32) * aligned_masks["video/qc_stats"].astype(np.float32)

            # ----- Pre-TCN DynamicsAug（TCN 前动态特征；不改 TCN / ASP / Head）-----
            if self._dynamics_aug_active:
                if self._dynamics_aug_mode == "selective_v2":
                    audio_groups = apply_dynamics_aug_to_groups_selective(
                        audio_groups,
                        mask_audio,
                        self._dynamics_aug_selective_audio_cfgs,
                    )
                    video_groups = apply_dynamics_aug_to_groups_selective(
                        video_groups,
                        mask_video,
                        self._dynamics_aug_selective_video_cfgs,
                    )
                else:
                    audio_groups = apply_dynamics_aug_to_groups(
                        audio_groups,
                        mask_audio,
                        self._dynamics_aug_apply_audio,
                        window=self._dynamics_aug_window,
                        use_delta=self._dynamics_aug_use_delta,
                        use_abs_delta=self._dynamics_aug_use_abs_delta,
                        use_rolling_mean=self._dynamics_aug_use_rolling_mean,
                        use_rolling_std=self._dynamics_aug_use_rolling_std,
                    )
                    video_groups = apply_dynamics_aug_to_groups(
                        video_groups,
                        mask_video,
                        self._dynamics_aug_apply_video,
                        window=self._dynamics_aug_window,
                        use_delta=self._dynamics_aug_use_delta,
                        use_abs_delta=self._dynamics_aug_use_abs_delta,
                        use_rolling_mean=self._dynamics_aug_use_rolling_mean,
                        use_rolling_std=self._dynamics_aug_use_rolling_std,
                    )
            # ----- Pre-TCN DynamicsAug 结束 -----

            dims = self.feature_dims
            audio_pooled_groups: dict[str, torch.Tensor] = {}
            pooled_presence: dict[str, bool] = {}
            if "egemaps" in cfg.audio_pooled_features:
                egemaps = load_egemaps_pooled(
                    self.root, self.path_split,
                    str(row["anon_school"]), str(row["anon_class"]),
                    str(row["anon_pid"]), str(row["session"]),
                )
                audio_pooled_groups["egemaps"] = (
                    torch.from_numpy(egemaps) if egemaps is not None
                    else torch.zeros(dims.get("egemaps", 88))
                )
                pooled_presence["egemaps"] = egemaps is not None

            for name in cfg.audio_features:
                if name not in audio_groups and name not in cfg.audio_pooled_features and name in dims:
                    audio_groups[name] = torch.zeros(T, dims[name])
            for name in cfg.video_features:
                if name not in video_groups and name in dims:
                    video_groups[name] = torch.zeros(T, dims[name])

            session_idx = SESSION_TO_IDX.get(str(row["session"]), 0)

            return {
                "audio_groups": audio_groups,
                "audio_pooled_groups": audio_pooled_groups,
                "video_groups": video_groups,
                "mask_audio": torch.from_numpy(mask_audio),
                "mask_video": torch.from_numpy(mask_video),
                "vad_signal": torch.from_numpy(vad_signal),
                "qc_quality": torch.from_numpy(qc_quality),
                "audio_pooled_present": pooled_presence,
                "session_idx": session_idx,
                "seq_len": T,
                "session": str(row["session"]),
            }
        except Exception as e:
            log.debug(f"Failed to load session {row.get('session', '?')} for {row.get('anon_pid', '?')}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.participants)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache is not None and self._cache[idx] is not None:
            sample = self._cache[idx]
        else:
            sample = self._load_participant(idx)

        if self.split == "train" and self.session_drop_prob > 0.0:
            return self._apply_session_dropout(sample)
        return sample

    def _load_participant(self, idx: int) -> dict[str, Any]:
        info = self.participants[idx]
        sessions_data = []
        session_valid = []

        for sess_name in SESSIONS:
            if sess_name in info["sess_rows"]:
                data = self._load_single_session(info["sess_rows"][sess_name])
                if data is not None:
                    sessions_data.append(data)
                    session_valid.append(True)
                else:
                    sessions_data.append(None)
                    session_valid.append(False)
            else:
                sessions_data.append(None)
                session_valid.append(False)

        return {
            "sessions": sessions_data,
            "session_valid": np.array(session_valid, dtype=bool),
            "y_a1": torch.from_numpy(info["y_a1"]),
            "y_a2": torch.from_numpy(info["y_a2"]),
            "anon_pid": info["anon_pid"],
            "anon_school": info["anon_school"],
            "anon_class": info["anon_class"],
            "session_names": SESSIONS,
        }

    def _apply_session_dropout(self, sample: dict[str, Any]) -> dict[str, Any]:
        valid_indices = [
            idx for idx, is_valid in enumerate(sample["session_valid"].tolist())
            if is_valid and sample["sessions"][idx] is not None
        ]
        if len(valid_indices) <= 1 or np.random.random() >= self.session_drop_prob:
            return sample

        drop_idx = int(np.random.choice(valid_indices))
        sessions = list(sample["sessions"])
        sessions[drop_idx] = None
        session_valid = np.array(sample["session_valid"], copy=True)
        session_valid[drop_idx] = False

        return {
            **sample,
            "sessions": sessions,
            "session_valid": session_valid,
        }

    def preload(self, desc: str | None = None) -> float:
        n = len(self)
        if desc is None:
            desc = f"Preload {self.split}"
        self._cache = [None] * n
        errors = 0
        for i in tqdm(range(n), desc=desc, dynamic_ncols=True):
            try:
                self._cache[i] = self._load_participant(i)
            except Exception as exc:
                errors += 1
                if errors <= 3:
                    log.warning(f"Preload: participant {i} failed: {exc}")
        if errors > 0:
            log.warning(f"Preload: {errors}/{n} participants failed")
        gb = self._estimate_cache_bytes() / 1024**3
        log.info(f"Preloaded {n - errors}/{n} participants ({gb:.1f} GB in RAM)")
        return gb

    def _estimate_cache_bytes(self) -> int:
        total = 0
        if self._cache is None:
            return 0
        for sample in self._cache:
            if sample is None:
                continue
            for sess in sample.get("sessions", []):
                if sess is None:
                    continue
                for v in sess.values():
                    if isinstance(v, torch.Tensor):
                        total += v.nelement() * v.element_size()
                    elif isinstance(v, dict):
                        for vv in v.values():
                            if isinstance(vv, torch.Tensor):
                                total += vv.nelement() * vv.element_size()
        return total

    @property
    def is_preloaded(self) -> bool:
        return self._cache is not None


def grouped_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    B = len(batch)
    all_sessions = []  
    session_types = [] 
    session_valid_list = []
    flat_pids = []
    flat_sess_names = []

    for b_idx, sample in enumerate(batch):
        session_valid_list.append(sample["session_valid"])
        for s_idx, sess_data in enumerate(sample["sessions"]):
            if sess_data is not None:
                all_sessions.append(sess_data)
                session_types.append(s_idx)
                flat_pids.append(sample["anon_pid"])
                flat_sess_names.append(SESSIONS[s_idx])
            else:
                dims = batch[0]["sessions"]
                ref = None
                for s in sample["sessions"]:
                    if s is not None:
                        ref = s
                        break
                if ref is None:
                    for other in batch:
                        for s in other["sessions"]:
                            if s is not None:
                                ref = s
                                break
                        if ref is not None:
                            break

                if ref is not None:
                    dummy = _make_dummy_session(ref)
                    all_sessions.append(dummy)
                    session_types.append(s_idx)
                    flat_pids.append(sample["anon_pid"])
                    flat_sess_names.append(SESSIONS[s_idx])

    if not all_sessions:
        hints: list[str] = []
        for sample in batch[: min(4, B)]:
            n_ok = sum(1 for s in sample["sessions"] if s is not None)
            keys = list(sample.get("session_names", SESSIONS))
            missing = [keys[i] for i, s in enumerate(sample["sessions"]) if s is None]
            hints.append(
                f"school={sample.get('anon_school')!r} class={sample.get('anon_class')!r} "
                f"pid={sample.get('anon_pid')!r} loaded={n_ok}/{len(sample['sessions'])} "
                f"session_valid={sample['session_valid'].tolist()} missing_sessions={missing[:6]}"
            )
        raise RuntimeError(
            "No valid sessions in batch: every participant had no loadable session "
            "(all entries in sample['sessions'] are None). "
            "Common causes: (1) manifest `session` values do not match "
            f"{SESSIONS}; (2) `feature_root`/`split` path wrong for this manifest "
            f"(expect sequence.npz under {{root}}/{{split}}/...); (3) core features missing on disk so "
            "`_load_single_session` returns None; (4) internal-CV manifests: val.csv labels may refer to a "
            "fold's validation set while sequences still live under the global `train/` tree — set "
            "`val_sequence_path_split: train` (or `sequence_path_split: {{val: train}}`) in the YAML. "
            "First samples: "
            + " | ".join(hints)
        )

    n_flat = len(all_sessions)
    T_max = max(s["seq_len"] for s in all_sessions)

    audio_names = list(all_sessions[0]["audio_groups"].keys())
    pooled_audio_names = list(all_sessions[0]["audio_pooled_groups"].keys())
    video_names = list(all_sessions[0]["video_groups"].keys())

    def _pad_groups(names, key):
        result = {}
        for n in names:
            D = all_sessions[0][key][n].shape[-1]
            t = torch.zeros(n_flat, T_max, D)
            for i, s in enumerate(all_sessions):
                L = s["seq_len"]
                t[i, :L] = s[key][n]
            result[n] = t
        return result

    def _pad_1d(key, dtype=torch.float32):
        t = torch.zeros(n_flat, T_max, dtype=dtype)
        for i, s in enumerate(all_sessions):
            L = s["seq_len"]
            t[i, :L] = s[key]
        return t

    pad_mask = torch.ones(n_flat, T_max, dtype=torch.bool)
    for i, s in enumerate(all_sessions):
        pad_mask[i, :s["seq_len"]] = False

    flat_batch = {
        "audio_groups": _pad_groups(audio_names, "audio_groups"),
        "audio_pooled_groups": {
            name: torch.stack([s["audio_pooled_groups"][name] for s in all_sessions])
            for name in pooled_audio_names
        },
        "video_groups": _pad_groups(video_names, "video_groups"),
        "mask_audio": _pad_1d("mask_audio", torch.bool),
        "mask_video": _pad_1d("mask_video", torch.bool),
        "pad_mask": pad_mask,
        "vad_signal": _pad_1d("vad_signal"),
        "qc_quality": _pad_1d("qc_quality"),
        "audio_pooled_present": {
            name: torch.tensor(
                [s["audio_pooled_present"].get(name, False) for s in all_sessions],
                dtype=torch.bool,
            )
            for name in pooled_audio_names
        },
        "session_idx": torch.tensor([s["session_idx"] for s in all_sessions], dtype=torch.long),
        "seq_len": torch.tensor([s["seq_len"] for s in all_sessions], dtype=torch.long),
        "anon_pid": flat_pids,
        "session": flat_sess_names,
    }

    return {
        "flat_batch": flat_batch,
        "participant_y_a1": torch.stack([b["y_a1"] for b in batch]),
        "participant_y_a2": torch.stack([b["y_a2"] for b in batch]),
        "session_valid": torch.from_numpy(np.stack(session_valid_list)),
        "session_types": torch.tensor(session_types, dtype=torch.long),
        "n_participants": B,
        "anon_pids": [b["anon_pid"] for b in batch],
        "anon_schools": [b["anon_school"] for b in batch],
        "anon_classes": [b["anon_class"] for b in batch],
        "flat_sessions": flat_sess_names,
        "flat_pids": flat_pids,
    }


def _make_dummy_session(ref: dict[str, Any]) -> dict[str, Any]:
    """Create a zero-filled dummy session matching reference dims."""
    T = 1  # minimal length
    audio_groups = {k: torch.zeros(T, v.shape[-1]) for k, v in ref["audio_groups"].items()}
    video_groups = {k: torch.zeros(T, v.shape[-1]) for k, v in ref["video_groups"].items()}
    return {
        "audio_groups": audio_groups,
        "audio_pooled_groups": {
            k: torch.zeros_like(v) for k, v in ref["audio_pooled_groups"].items()
        },
        "video_groups": video_groups,
        "mask_audio": torch.zeros(T, dtype=torch.bool),
        "mask_video": torch.zeros(T, dtype=torch.bool),
        "vad_signal": torch.zeros(T),
        "qc_quality": torch.zeros(T),
        "audio_pooled_present": {
            k: False for k in ref["audio_pooled_groups"].keys()
        },
        "session_idx": 0,
        "seq_len": T,
        "session": "A01",
    }
