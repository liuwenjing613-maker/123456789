#!/usr/bin/env python3
"""Fill p_D, p_A, p_S in a baseline CSV from test_hidden-style predictions.

test_hidden.csv: columns file_id, p_D, p_A, p_S
  file_id example: SCH_001_CLS_0015_P000224

baseline CSV: columns anon_school, anon_class, anon_pid, p_D, p_A, p_S

Rows are matched on (anon_school, anon_class, anon_pid). Order of baseline rows is kept.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def file_id_to_key(file_id: str) -> tuple[str, str, str]:
    parts = file_id.strip().split("_")
    if len(parts) < 5 or parts[0] != "SCH" or parts[2] != "CLS":
        raise ValueError(f"Unexpected file_id format: {file_id!r}")
    anon_school = f"{parts[0]}_{parts[1]}"
    anon_class = f"{parts[2]}_{parts[3]}"
    anon_pid = "_".join(parts[4:])
    return anon_school, anon_class, anon_pid


def load_hidden(path: Path) -> dict[tuple[str, str, str], tuple[str, str, str]]:
    out: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("file_id"):
                continue
            key = file_id_to_key(row["file_id"])
            out[key] = (row["p_D"], row["p_A"], row["p_S"])
    return out


def merge(
    hidden_path: Path,
    baseline_path: Path,
    output_path: Path,
) -> tuple[int, int, int]:
    hidden = load_hidden(hidden_path)
    rows: list[dict[str, str]] = []
    missing: list[tuple[str, str, str]] = []

    with baseline_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise SystemExit("baseline has no header")
        for row in reader:
            key = (row["anon_school"], row["anon_class"], row["anon_pid"])
            if key in hidden:
                p_d, p_a, p_s = hidden[key]
                row["p_D"], row["p_A"], row["p_S"] = p_d, p_a, p_s
            else:
                missing.append(key)
            rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(hidden), len(rows), len(missing)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hidden",
        type=Path,
        default=Path(__file__).resolve().parent / "kfold3_fold0_best_safe_submit_shrink05.csv",
        help="Source CSV with file_id and probabilities",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).resolve().parent / "result_kfold3_fold0_best_safe_submit_shrink05.csv",
        help="Baseline CSV to update (anon_school, anon_class, anon_pid, ...)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite --baseline)",
    )
    args = p.parse_args()
    out = args.output or args.baseline

    n_keys, n_rows, n_missing = merge(args.hidden, args.baseline, out)
    print(f"hidden keys: {n_keys}")
    print(f"baseline data rows: {n_rows}")
    print(f"rows without matching hidden key: {n_missing}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()