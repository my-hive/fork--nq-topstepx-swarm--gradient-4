import numpy as np, pandas as pd
DF = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
t, c = DF.ct_sec.values, DF.close.values
n_debug = 0
for _, idx in DF.groupby("session", sort=False).indices.items():
    tt, cc = t[idx], c[idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    if not m.any(): continue
    w = np.where(m)[0]
    i930 = np.where(tt >= 9*3600)[0]
    i1100 = np.where(tt >= 11*3600)[0]
    print("sess rows:", len(idx), "9:30 idx:", i930[0] if len(i930) else None,
          "11:00 idx:", i1100[0] if len(i1100) else None,
          "tt range:", tt[0], tt[-1])
    print("c930-835:", cc[i930[0]] - cc[w[-1]], "c1100-835:", cc[i1100[0]] - cc[w[-1]])
    n_debug += 1
    if n_debug >= 3: break
