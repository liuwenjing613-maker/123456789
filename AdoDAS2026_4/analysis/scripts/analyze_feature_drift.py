#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
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


def pooled_feature_stats(
    root: Path,
    split: str,
    session: str,
    group: str,
    sample_persons: list[tuple[str, str, str]],
    ssl_a: str | None,
    ssl_v: str | None,
) -> dict:
    mean_abs_list, std_list, fd_list, miss = [], [], [], 0
    roll_std_list = []
    for school, cls, pid in sample_persons:
        tag = ssl_a if group == "ssl_embed" else (ssl_v if group == "vision_ssl_embed" else None)
        try:
            if group == "egemaps":
                continue
            seq = load_sequence(
                root, split, school, cls, pid, "audio" if group in ("mel_mfcc", "vad", "egemaps", "ssl_embed") else "video",
                group,
                session,
                tag,
            )
        except Exception:
            miss += 1
            continue
        x = seq.features.astype(np.float64)
        m = seq.valid_mask.astype(np.float64)
        if x.size == 0:
            miss += 1
            continue
        w = x * m[:, None]
        mean_abs_list.append(float(np.mean(np.abs(w))))
        std_list.append(float(np.std(x[m > 0]) if (m > 0).any() else 0.0))
        if x.shape[0] > 1:
            d = np.diff(x, axis=0)
            fd_list.append(float(np.mean(np.abs(d))))
            # rolling std window 5 on mean across dims
            tmean = x.mean(axis=1)
            if len(tmean) >= 5:
                rs = pd.Series(tmean).rolling(5).std().dropna().values
                if len(rs):
                    roll_std_list.append(float(np.mean(rs)))
        else:
            fd_list.append(0.0)
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
    max_p = int(fcfg.get("max_participants_per_split", 200))
    ssl_a = fcfg.get("audio_ssl_model_tag")
    ssl_v = fcfg.get("video_ssl_model_tag")
    splits = fcfg.get("splits", ["train", "val", "test_hidden"])

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
        keys = df["pk"].unique()[:max_p]
        sset = set(keys)
        out = []
        seen = set()
        for _, r in df.iterrows():
            pk = r["pk"]
            if pk not in sset or pk in seen:
                continue
            seen.add(pk)
            out.append((str(r["anon_school"]), str(r["anon_class"]), str(r["anon_pid"])))
            if len(out) >= max_p:
                break
        return out

    stats_rows = []
    for split in splits:
        persons = sample_persons(split)
        if not persons:
            continue
        for session in sessions:
            for g in groups:
                st = pooled_feature_stats(feat_root, split, session, g, persons, ssl_a, ssl_v)
                stats_rows.append(
                    {
                        "split": split,
                        "session": session,
                        "feature_group": g,
                        **st,
                    }
                )

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
