#!/usr/bin/env python3
"""
将提交 CSV 中的 anon_school / anon_class / anon_pid 从纯数字格式
转为与官方示例一致的命名：SCH_{ddd}、CLS_{dddd}、P{dddddd}。

表头保持不变：anon_school,anon_class,anon_pid,p_D,p_A,p_S

用法：
  python convert_submission_id_format.py -i in.csv -o out.csv
  python convert_submission_id_format.py in.csv              # 覆盖写回 in.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def normalize_cell(raw: str, prefix: str, width: int) -> str:
    """若已是带前缀格式则先剥掉再按位数补零重编码。"""
    s = raw.strip()
    if prefix == "SCH_" and s.startswith("SCH_"):
        s = s[4:]
    elif prefix == "CLS_" and s.startswith("CLS_"):
        s = s[4:]
    return f"{prefix}{int(s):0{width}d}"


def convert_row(row: list[str]) -> list[str]:
    if len(row) < 3:
        return row
    out = list(row)
    out[0] = normalize_cell(out[0], "SCH_", 3)
    out[1] = normalize_cell(out[1], "CLS_", 4)
    # anon_pid: P + 6 位数字（与 dataset/results/result.csv 一致）
    pid = out[2].strip()
    if pid.startswith("P") and pid[1:].isdigit():
        pid = pid[1:]
    out[2] = f"P{int(pid):06d}"
    return out


def run(in_path: Path, out_path: Path) -> int:
    if not in_path.is_file():
        print(f"错误：找不到输入文件 {in_path}", file=sys.stderr)
        return 1

    rows: list[list[str]] = []
    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("错误：空文件", file=sys.stderr)
            return 1
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            rows.append(convert_row(row))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"已写入 {out_path}，共 {len(rows)} 行数据（不含表头）。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="提交 CSV 匿名 ID 列格式转换")
    ap.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="输入 CSV（默认可通过 -i 指定）",
    )
    ap.add_argument("-i", "--input-file", type=Path, help="输入 CSV")
    ap.add_argument("-o", "--output", type=Path, help="输出 CSV（省略则覆盖输入）")
    args = ap.parse_args()

    in_path = args.input_file or args.input
    if in_path is None:
        ap.print_help()
        return 2

    out_path = args.output if args.output is not None else in_path
    return run(in_path, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
