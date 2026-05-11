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


def iter_manifest_rows(manifest_path: Path, max_persons: int | None) -> list[dict]:
    df = pd.read_csv(manifest_path)
    df = df.dropna(subset=["anon_pid"])
    df["person_key"] = (
        df["anon_school"].astype(str)
        + "_"
        + df["anon_class"].astype(str)
        + "_"
        + df["anon_pid"].astype(str)
    )
    keys = df["person_key"].unique().tolist()
    if max_persons is not None and max_persons > 0:
        keys = keys[:max_persons]
    keyset = set(keys)
    sub = df[df["person_key"].isin(keyset)]
    rows: list[dict] = []
    for r in sub.itertuples(index=False):
        rows.append(
            {
                "split": None,
                "person_key": r.person_key,
                "school": str(r.anon_school),
                "cls": str(r.anon_class),
                "pid": str(r.anon_pid),
                "session": str(r.session),
            }
        )
    return rows


def session_stats_for_row(
    root: Path, split: str, school: str, cls: str, pid: str, session: str, ssl_a: str, ssl_v: str
) -> dict | None:
    out = {"T": np.nan, "valid_ratio": np.nan, "speech_ratio": np.nan, "qc_mean": np.nan, "missing": 1.0}
    try:
        vad = load_sequence(root, split, school, cls, pid, "audio", "vad", session, None)
        T = len(vad.timestamps_ms)
        vm = vad.valid_mask.astype(np.float64)
        feats = vad.features.astype(np.float64)
        out["T"] = float(T)
        out["valid_ratio"] = float(vm.mean()) if T else 0.0
        out["speech_ratio"] = float(np.mean(feats[:, 0] * vm) / max(vm.mean(), 1e-6)) if T else 0.0
    except Exception:
        return out
    try:
        qc = load_sequence(root, split, school, cls, pid, "video", "qc_stats", session, None)
        qcm = qc.features.astype(np.float64)
        qv = qc.valid_mask.astype(np.float64)
        out["qc_mean"] = float(np.mean(qcm[:, 0] * qv) / max(qv.mean(), 1e-6))
    except Exception:
        out["qc_mean"] = float("nan")
    out["missing"] = 0.0
    return out


def _session_task(
    args: tuple[Path, str, dict, str, str],
) -> dict | None:
    feat_root, split, br, ssl_a, ssl_v = args
    st = session_stats_for_row(
        feat_root, split, br["school"], br["cls"], br["pid"], br["session"], ssl_a, ssl_v
    )
    if st is None:
        return None
    out = dict(br)
    out["split"] = split
    return {**out, **st}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    feat_root = Path(paths["feature_root"])
    man_dir = Path(paths["manifest_dir"])
    out_dir = Path(paths["output_dir"]) / "session_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("plot", {}).get("dpi", 200))
    fcfg = cfg.get("features", {})
    max_p = int(fcfg.get("max_participants_per_split", 400))
    sq_raw = fcfg.get("session_quality_max_participants")
    if sq_raw is None:
        sq_max = min(max_p, 200)
    else:
        sq_max = max(1, int(sq_raw))
    io_workers = int(fcfg.get("feature_io_workers", 8))
    ssl_a = fcfg.get("audio_ssl_model_tag", "wav2vec2-chinese-xlsr")
    ssl_v = fcfg.get("video_ssl_model_tag", "dinov2-large")

    tasks: list[tuple[Path, str, dict, str, str]] = []
    for split in fcfg.get("splits", ["train", "val", "test_hidden"]):
        mp = man_dir / f"{split}.csv"
        if not mp.is_file():
            log.warning("Manifest missing: %s", mp)
            continue
        base_rows = iter_manifest_rows(mp, sq_max)
        log.info(
            "Session quality split=%s: %d manifest rows (unique persons cap=%d) workers=%d",
            split,
            len(base_rows),
            sq_max,
            io_workers,
        )
        for br in base_rows:
            tasks.append((feat_root, split, dict(br), ssl_a, ssl_v))

    summaries: list[dict] = []
    if not tasks:
        pass
    elif io_workers <= 1:
        for t in tasks:
            row = _session_task(t)
            if row is not None:
                summaries.append(row)
    else:
        n = len(tasks)
        done = 0
        step = max(1, n // 20)
        with ThreadPoolExecutor(max_workers=min(io_workers, max(1, n))) as ex:
            futs = {ex.submit(_session_task, t): t for t in tasks}
            for fut in as_completed(futs):
                row = fut.result()
                if row is not None:
                    summaries.append(row)
                done += 1
                if done % step == 0 or done == n:
                    log.info("Session quality progress %d/%d", done, n)

    if not summaries:
        log.error("No session quality rows computed")
        return

    df = pd.DataFrame(summaries)
    grp = df.groupby(["split", "session"]).agg(
        num_samples=("person_key", "count"),
        avg_T=("T", "mean"),
        median_T=("T", "median"),
        p10_T=("T", lambda s: float(np.percentile(s.dropna(), 10)) if len(s.dropna()) else np.nan),
        p90_T=("T", lambda s: float(np.percentile(s.dropna(), 90)) if len(s.dropna()) else np.nan),
        valid_ratio_mean=("valid_ratio", "mean"),
        valid_ratio_std=("valid_ratio", "std"),
        speech_ratio_mean=("speech_ratio", "mean"),
        speech_ratio_std=("speech_ratio", "std"),
        qc_mean=("qc_mean", "mean"),
        qc_std=("qc_mean", "std"),
        missing_rate=("missing", "mean"),
    ).reset_index()
    grp.to_csv(out_dir / "session_quality_summary.csv", index=False)

    # simple bar charts by session
    for col, fname, title in [
        ("avg_T", "session_length_bar.png", "Mean sequence length T"),
        ("valid_ratio_mean", "session_valid_ratio_bar.png", "Mean valid ratio"),
        ("speech_ratio_mean", "session_speech_ratio_bar.png", "Mean speech proxy (vad ch0)"),
        ("qc_mean", "session_qc_bar.png", "Mean QC (dim0)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        piv = grp.pivot(index="session", columns="split", values=col)
        piv.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.legend(title="split")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=dpi)
        plt.close(fig)

    log.info("Session quality -> %s", out_dir)


if __name__ == "__main__":
    main()
