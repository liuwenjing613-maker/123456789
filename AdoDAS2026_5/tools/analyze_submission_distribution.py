import argparse
import pandas as pd

def summarize(path):
    try:
        df = pd.read_csv(path)
        print('\n===== 分析文件:', path.split('/')[-1], ' =====')
        for c in ['p_D', 'p_A', 'p_S']:
            if c not in df.columns: continue
            s = df[c].astype(float)
            print(f'[{c}] 预测阳性比例 (>=0.5): {round((s >= 0.5).mean()*100, 2)}%  |  平均概率: {round(s.mean(), 4)}')
    except Exception as e:
        print("读取失败:", e)

parser = argparse.ArgumentParser()
parser.add_argument('csvs', nargs='+')
args = parser.parse_args()
for p in args.csvs: summarize(p)
