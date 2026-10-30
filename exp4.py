"""After leg-1 stop-out (40pts against fade), what does continuation in breakout dir do?"""
import numpy as np, pandas as pd
from exp2 import gate, START

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
EXIT = 14 * 3600 + 30 * 60
rows = []
for _, (idx, side, sz, mv, onv) in gate(df).items():
    if sz == 0:
        continue
    t, c = df.ct_sec.values[idx], df.close.values[idx]
    pos, entry, stopped = side, None, None
    for k in range(len(idx)):
        if t[k] < START or t[k] >= EXIT:
            continue
        if entry is None and t[k] >= START:
            entry, k0 = c[k], k
        elif entry is not None:
            pnl = (c[k] - entry) * pos
            if pnl <= -40:
                stopped = (k, c[k], -side)  # breakout dir = -side
                break
            if pnl >= 80:
                break
    if stopped is None:
        continue
    k, px, bdir = stopped
    # continuation: MFE/MAE of bdir position from stop point, within 60 min and to EXIT
    hh = EXIT
    mfe = mae = 0.0
    eod = 0.0
    for j in range(k + 1, len(idx)):
        if t[j] >= hh:
            eod = (c[j-1] - px) * bdir
            break
        pnl = (c[j] - px) * bdir
        mfe, mae = max(mfe, pnl), min(mae, pnl)
    else:
        eod = (c[-1] - px) * bdir
    rows.append(dict(session=str(df.session.values[idx[0]])[:10], mv=mv, onv=abs(onv),
                     mfe=mfe, mae=mae, eod=eod, half=int(idx[0] > len(df) // 2)))
r = pd.DataFrame(rows)
print("n stop-outs:", len(r))
print(r[["mfe", "mae", "eod"]].describe().loc[["mean", "25%", "50%", "75%"]])
print("\nby half:")
print(r.groupby("half").agg(n=("eod", "size"), eod=("eod", "mean"), win50=("eod", lambda x: (x > 50).mean()),
                            lose50=("eod", lambda x: (x < -50).mean())))
print("\nby onv bucket:")
print(r.groupby(pd.cut(r.onv, [0, 50, 175, 10000])).agg(n=("eod", "size"), eod=("eod", "mean")))
print("\nby mv size:")
print(r.groupby(pd.cut(abs(r.mv), [0, 20, 40, 80, 1000])).agg(n=("eod", "size"), eod=("eod", "mean")))
