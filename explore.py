"""Explore per-day outcomes of current gate scheme: what do traded days do after 8:35?"""
import numpy as np
import pandas as pd

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 175.0

rows = []
for s, g in df.groupby("session", sort=False):
    t, c, h, l = g.ct_sec.values, g.close.values, g.high.values, g.low.values
    m = (t >= 8 * 3600 + 30 * 60) & (t < START)
    if not m.any():
        continue
    w = np.where(m)[0]
    onv = c[w[0]] - c[0]
    mv = c[w[-1]] - c[w[0]]
    if mv == 0:
        continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    bucket = ("hot" if hot else "cold") + ("+agree" if agree else "+dis")
    side = -1 if mv >= 0 else 1
    # simulate fade 8:35 entry, track MFE/MAE in pts, time to stop/target
    entry = c[w[-1] + 1] if w[-1] + 1 < len(c) else None
    pos = side
    mfe = mae = 0.0
    t_exit = None
    stop = tgt = False
    for k in range(w[-1] + 1, len(c)):
        if t[k] >= EXIT:
            t_exit = t[k] - START
            break
        pnl = (c[k] - entry) * pos
        mfe, mae = max(mfe, pnl), min(mae, pnl)
        if pnl <= -40:
            stop = True; t_exit = t[k] - START; break
        if pnl >= 80:
            tgt = True; t_exit = t[k] - START; break
    else:
        t_exit = (t[-1] - START) if len(c) > w[-1] + 1 else 0
    eod = None
    if entry is not None and t_exit is not None:
        eod = (c[min(w[-1] + 1, len(c)-1)] - entry) * 0  # placeholder
    rows.append(dict(session=str(s)[:10], bucket=bucket, mv=mv, onv=onv, onv_abs=abs(onv),
                     side=side, mfe=mfe, mae=mae, stop=stop, tgt=tgt,
                     t_exit=(t_exit or 0) / 60.0, eod_pnl=(c[-1] - entry) * pos if entry else 0))

r = pd.DataFrame(rows)
print(r.groupby("bucket").agg(n=("session", "count"), win=("tgt", "mean"), loss=("stop", "mean"),
                               mfe=("mfe", "median"), mae=("mae", "median"),
                               eod=("eod_pnl", "mean"), tmin=("t_exit", "median")))
# where does time go: distribution of exit times
print("\nexit-time minutes by bucket:")
print(r.groupby("bucket").t_exit.describe()[["25%", "50%", "75%"]])
# losing days: big mv? big onv?
tr = r[r.bucket != "cold+agree"]
print("\ntraded day count:", len(tr), " skipped cold+agree:", (r.bucket == "cold+agree").sum())
print("\nMAE beyond -40 (how deep do losers go):")
print((tr.groupby(pd.cut(tr.onv_abs, [0, 50, 100, 175, 250, 400, 10000]))
        .agg(n=("session", "count"), eod=("eod_pnl", "mean"), win=("tgt", "mean"), loss=("stop", "mean"))))
