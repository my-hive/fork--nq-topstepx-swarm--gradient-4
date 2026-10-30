import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from strategy import signals
from sweep3 import build, SIZES, i0, i1, SIDE, B
df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
t, c = df.ct_sec.values, df.close.values
a = np.asarray(signals(df))
b = build(t, c, i0, i1, SIDE, B, SIZES, 8*3600+35*60, 14*3600+30*60, 40.0, 80.0, 0.0)
d = np.where(a != b)[0]
print("len", len(a), "diffs", len(d))
if len(d):
    print(df.iloc[d[:10]][["session","ct_sec","close"]])
    print("strat", a[d[:10]], "mine", b[d[:10]])
