#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data.feature_io import load_sequence
from analysis.scripts.utils import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _rolling_std_mean(tmean: np.ndarray, window: int = 5) -> float | None:
    """Mean rolling std of 1D series (numpy only; avoids pandas in hot path)."""
    tmean = np.asarray(tmean, dtype=np.float64)
    if len(tmean) < window:
        return None
    k = np.ones(window, dtype=np.float64) / window
    ma = np.convolve(tmean, k, mode="valid")
    c2 = np.convolve(tmean * tmean, k, mode="valid")
    var = np.clip(c2 - ma * ma, 0.0, None)
    rs = np.sqrt(var)
    return float(np.mean(rs)) if rs.size else None


def _one_person_feature_row(
    root: Path,
    split: str,
    session: str,
    group: str,
    school: str,
    cls: str,
    pid: str,
    ssl_a: str | None,
    ssl_v: str | None,
) -> dict | str:
    """Return stats dict for one person, \"miss\" if load failed, None if skipped (egemaps)."""
    if group == "egemaps":
        return "skip"
    tag = ssl_a if group == "ssl_embed" else (ssl_v if group == "vision_ssl_embed" else None)
    try:
        seq = load_sequence(
            root,
            split,
            school,
            cls,
            pid,
            "audio" if group in ("mel_mfcc", "vad", "egemaps", "ssl_embed") else "video",
            group,
            session,
            tag,
        )
    except Exception:
        return "miss"
    x = seq.features.astype(np.float64)
    m = seq.valid_mask.astype(np.float64)
    if x.size == 0:
        return "miss"
    w = x * m[:, None]
    mean_abs = float(np.mean(np.abs(w)))
    std_v = float(np.std(x[m > 0]) if (m > 0).any() else 0.0)
    if x.shape[0] > 1:
        d = np.diff(x, axis=0)
        fd = float(np.mean(np.abs(d)))
        tmean = x.mean(axis=1)
        rs_m = _rolling_std_mean(tmean, 5)
    else:
        fd = 0.0
        rs_m = None
    return {"mean_abs": mean_abs, "std": std_v, "fd": fd, "rs": rs_m}


def pooled_feature_stats(
    root: Path,
    split: str,
    session: str,
    group: str,
    sample_persons: list[tuple[str, str, str]],
    ssl_a: str | None,
    ssl_v: str | None,
    max_workers: int,
) -> dict:
    mean_abs_list, std_list, fd_list, miss = [], [], [], 0
    roll_std_list = []
    if not sample_persons:
        return {
            "mean_abs": float("nan"),
            "std": float("nan"),
            "frame_diff_abs_mean": float("nan"),
            "rolling_std_mean": float("nan"),
            "missing_rate": 1.0,
            "zero_rate": float("nan"),
            "valid_ratio": float("nan"),
            "dim_mean_std": float("nan"),
        }

    workers = max(1, min(max_workers, len(sample_persons)))
    if workers == 1:
        for school, cls, pid in sample_persons:
            r = _one_person_feature_row(root, split, session, group, school, cls, pid, ssl_a, ssl_v)
            if r == "skip":
                continue
            if r == "miss":
                miss += 1
                continue
            mean_abs_list.append(r["mean_abs"])
            std_list.append(r["std"])
            fd_list.append(r["fd"])
            if r["rs"] is not None:
                roll_std_list.append(r["rs"])
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _one_person_feature_row, root, split, session, group, school, cls, pid, ssl_a, ssl_v
                ): (school, cls, pid)
                for school, cls, pid in sample_persons
            }
            for fut in as_completed(futs):
                r = fut.result()
                if r == "skip":
                    continue
                if r == "miss":
                    miss += 1
                    continue
                mean_abs_list.append(r["mean_abs"])
                std_list.append(r["std"])
                fd_list.append(r["fd"])
                if r["rs"] is not None:
                    roll_std_list.append(r["rs"])
    if not mean_abs_list:
        return {
            "mean_abs": float("nan"),
            "std": float("nan"),
            "frame_diff_abs_mean": float("nan"),
            "rolling_std_mean": float("nan"),
            "missing_rate": 1.0,
            "zero_rate": float("nan"),
            "valid_ratio": float("nan"),
            "dim_mean_std": float("nan"),
        }
    return {
        "mean_abs": float(np.mean(mean_abs_list)),
        "std": float(np.mean(std_list)),
        "frame_diff_abs_mean": float(np.mean(fd_list)) if fd_list else float("nan"),
        "rolling_std_mean": float(np.mean(roll_std_list)) if roll_std_list else float("nan"),
        "missing_rate": miss / max(len(sample_persons), 1),
        "zero_rate": float("nan"),
        "valid_ratio": float("nan"),
        "dim_mean_std": float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    feat_root = Path(paths["feature_root"])
    man_dir = Path(paths["manifest_dir"])
    out_dir = Path(paths["output_dir"]) / "feature_drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("plot", {}).get("dpi", 200))
    fcfg = cfg.get("features", {})
    groups = [g for g in fcfg.get("analyze_groups", []) if g != "egemaps"]
    sessions = fcfg.get("session_ids", ["A01", "B01", "B02", "B03"])
    max_p = int(fcfg.get("max_participants_per_split", 400))
    drift_cap_raw = fcfg.get("feature_drift_max_participants")
    if drift_cap_raw is None:
        drift_cap = min(max_p, 120)
    else:
        drift_cap = int(drift_cap_raw)
    if drift_cap <= 0:
        drift_cap = min(max_p, 120)
    io_workers = int(fcfg.get("feature_io_workers", 8))
    ssl_a = fcfg.get("audio_ssl_model_tag")
    ssl_v = fcfg.get("video_ssl_model_tag")
    splits = fcfg.get("splits", ["train", "val", "test_hidden"])
    log.info(
        "Feature drift: splits=%s sessions=%s groups=%d persons_per_split=%d (min of max_participants=%d and drift_cap=%d) workers=%d",
        splits,
        sessions,
        len(groups),
        min(max_p, drift_cap) if drift_cap > 0 else max_p,
        max_p,
        drift_cap,
        io_workers,
    )

    def sample_persons(split: str) -> list[tuple[str, str, str]]:
        mp = man_dir / f"{split}.csv"
        if not mp.is_file():
            return []
        df = pd.read_csv(mp)
        df["pk"] = (
            df["anon_school"].astype(str)
            + "_"
            + df["anon_class"].astype(str)
            + "_"
            + df["anon_pid"].astype(str)
        )
        n_take = min(max_p, drift_cap) if drift_cap > 0 else max_p
        keys = df["pk"].unique()[:n_take]
        sset = set(keys)
        out = []
        seen = set()
        for _, r in df.iterrows():
            pk = r["pk"]
            if pk not in sset or pk in seen:
                continue
            seen.add(pk)
            out.append((str(r["anon_school"]), str(r["anon_class"]), str(r["anon_pid"])))
            if len(out) >= n_take:
                break
        return out

    stats_rows = []
    done = 0
    total_units = max(1, len(splits) * len(sessions) * len(groups))
    for split in splits:
        persons = sample_persons(split)
        if not persons:
            continue
        for session in sessions:
            for g in groups:
                st = pooled_feature_stats(
                    feat_root, split, session, g, persons, ssl_a, ssl_v, io_workers
                )
                stats_rows.append(
                    {
                        "split": split,
                        "session": session,
                        "feature_group": g,
                        **st,
                    }
                )
                done += 1
                if done % max(1, total_units // 10) == 0 or done == total_units:
                    log.info("Feature drift progress %d/%d (split=%s session=%s group=%s)", done, total_units, split, session, g)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "feature_stats.csv", index=False)

    # drift vs train (mean_abs)
    drift_rows = []
    train_ref = stats_df[stats_df["split"] == "train"]
    for session in sessions:
        for g in groups:
            tr = train_ref[(train_ref["session"] == session) & (train_ref["feature_group"] == g)]
            if tr.empty:
                continue
            m0 = float(tr["mean_abs"].values[0])
            s0 = float(tr["std"].values[0]) + 1e-6

            def mean_abs(split: str) -> float | None:
                sub = stats_df[
                    (stats_df["split"] == split)
                    & (stats_df["session"] == session)
                    & (stats_df["feature_group"] == g)
                ]
                if sub.empty:
                    return None
                return float(sub["mean_abs"].values[0])

            m_val = mean_abs("val")
            m_test = mean_abs("test_hidden")
            val_drift = abs(m_val - m0) / s0 if m_val is not None else float("nan")
            test_drift = abs(m_test - m0) / s0 if m_test is not None else float("nan")
            drift_rows.append(
                {
                    "session": session,
                    "feature_group": g,
                    "val_drift_score": val_drift,
                    "test_drift_score": test_drift,
                    "test_minus_val_drift": (
                        test_drift - val_drift
                        if test_drift == test_drift and val_drift == val_drift
                        else float("nan")
                    ),
                }
            )

    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(out_dir / "feature_drift_score.csv", index=False)

    if not drift_df.empty:
        piv = drift_df.pivot_table(
            index="feature_group", columns="session", values="test_drift_score", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_title("Feature drift (test vs train, mean_abs)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "feature_drift_heatmap.png", dpi=dpi)
        plt.close(fig)

    log.info("Feature drift -> %s", out_dir)


if __name__ == "__main__":
    main()
