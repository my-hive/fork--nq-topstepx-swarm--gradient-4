import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, c = df.ct_sec.values, df.close.values
NS = sess.max() + 1
for s in range(2):
    idx = np.where(sess == s)[0]
    tt, cc = t[idx], c[idx]
    w = np.where((tt >= 8*3600+30*60) & (tt < 8*3600+35*60))[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    print("s", s, "sess0", df.session.values[idx[0]], "wlen", len(w), "onv", onv, "mv", mv,
          "sign eq", np.sign(mv) == np.sign(onv), "onv!=0", onv != 0)
    print("  true agree =", (np.sign(mv) == np.sign(onv)) if onv != 0 else True)
