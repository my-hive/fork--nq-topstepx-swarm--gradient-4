import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import sweep3
df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, c = df.ct_sec.values, df.close.values
NS = sess.max() + 1
ON_LO, ON_HI = sweep3.ON_LO, sweep3.ON_HI
print("ON bounds", ON_LO, ON_HI)
B2 = np.full(NS, 4, dtype=np.int64)
for s in range(NS):
    idx = np.where(sess == s)[0]
    tt, cc = t[idx], c[idx]
    w = np.where((tt >= 8*3600+30*60) & (tt < 8*3600+35*60))[0]
    if len(w) == 0: continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0: continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    B2[s] = (0 if hot else 2) + (1 if agree else 0)
print("match", np.array_equal(B2, sweep3.B), "diffs", (B2 != sweep3.B).sum())
print("B2[0:5]", B2[:5], "sweep3.B[0:5]", sweep3.B[:5])
