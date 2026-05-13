#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

manifest_dir = Path('/home/adodas/dataset/manifests')

for name in ['train.csv', 'val.csv', 'test_hidden.csv', 'test.csv']:
    path = manifest_dir / name
    if not path.exists():
        continue

    print('\n===== checking', path, '=====')
    m = pd.read_csv(path)

    need_cols = ['anon_school', 'anon_class', 'anon_pid']
    missing = [c for c in need_cols if c not in m.columns]
    if missing:
        print('missing cols:', missing)
        continue

    g = m[need_cols].drop_duplicates()

    print('rows:', len(m))
    print('unique school_class_pid:', len(g))
    print('unique anon_pid only:', m['anon_pid'].nunique())

    dup = (
        g.groupby('anon_pid')
         .agg(
             n_school=('anon_school', 'nunique'),
             n_class=('anon_class', 'nunique'),
             n_rows=('anon_pid', 'size'),
         )
         .query('n_rows > 1')
    )

    print('duplicate anon_pid count:', len(dup))
    if len(dup) > 0:
        print(dup.head(20))
