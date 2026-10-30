import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from strategy import signals
from mg5_orth import build, i0, i1, SIDE, ONV, AGRE, IMB, t, c, sess
df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
a = np.asarray(signals(df))
b = build(t, c, i0, i1, SIDE, ONV, AGRE, IMB, on_lo=50.0, on_hi=225.0, sz_ha=8, sz_hd=7, sz_cd=6,
          sz_ca_mom=0, shrink=2, imb_th=0.05, flip_dsz=1, flip_cutoff=0,
          stop=40.0, tgt=80.0, f_stop=40.0, f_tgt=60.0, m_stop=40.0, m_tgt=80.0)
d = np.where(a != b)[0]
print("DIFFS", len(d))
if len(d):
    ses = df.session.values
    import collections
    cnt = collections.Counter(ses[d])
    for k, v in cnt.most_common(8):
        print(k, v)
    k = d[0]
    print("first diff row", k, "ct", t[k], "strat", a[k], "mine", b[k])
