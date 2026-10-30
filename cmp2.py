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
print("DIFFS", len(d))
for k in d[:6]:
    print("row", k, "ct", t[k], "px", c[k], "strat", a[k], "mine", b[k])
# first session entry region
s0 = np.where(df.session.values == df.session.values[0])[0]
w = s0[(t[s0] >= 30900) & (t[s0] < 30920)]
for k in w:
    print(k, t[k], c[k], a[k], b[k])
