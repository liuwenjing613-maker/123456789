import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.manifest_dir) / "train.csv")
    groups = df["anon_school"].astype(str) + "_" + df["anon_class"].astype(str)
    
    out_dir = Path(args.out_dir)
    gkf = GroupKFold(n_splits=args.n_splits)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, np.zeros(len(df)), groups=groups)):
        split_dir = out_dir / f"split_{fold}_school_class"
        split_dir.mkdir(parents=True, exist_ok=True)
        df.iloc[tr_idx].to_csv(split_dir / "train.csv", index=False)
        df.iloc[va_idx].to_csv(split_dir / "val.csv", index=False)
        print(f"✅ Split {fold} 创建成功: 训练集 {len(tr_idx)} 行, 验证集 {len(va_idx)} 行.")

if __name__ == "__main__":
    main()
