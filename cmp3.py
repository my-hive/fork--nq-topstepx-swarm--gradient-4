import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
t, c = df.ct_sec.values, df.close.values
s0 = np.where(df.session.values == df.session.values[0])[0]
tt, cc = t[s0], c[s0]
w = np.where((tt >= 8*3600+30*60) & (tt < 8*3600+35*60))[0]
onv = cc[w[0]] - cc[0]
mv = cc[w[-1]] - cc[w[0]]
print("w", w[0], w[-1], "onv", onv, "mv", mv)
# also via strategy's groupby approach
for name, idx in df.groupby("session", sort=False).indices.items():
    tt2, cc2 = t[idx], c[idx]
    m = (tt2 >= 8*3600+30*60) & (tt2 < 8*3600+35*60)
    where = np.where(m)[0]
    onv2 = cc2[where[0]] - cc2[0]
    mv2 = cc2[where[-1]] - cc2[where[0]]
    print("gb", onv2, mv2)
    break
