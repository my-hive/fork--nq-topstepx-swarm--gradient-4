"""exp12: diagnose current base 0.5382 — day PnL by gate bucket, split by train half; extreme-|onv| and weekday effects."""
import numpy as np, pandas as pd
from harness import DF

START, EXIT = 8*3600+35*60, 14*3600+30*60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0
IMB_TH, IMB_SH = 0.05, 3

t, c, bv, sv = DF.ct_sec.values, DF.close.values, DF.buy_vol.values, DF.sell_vol.values
rows = []
for _, idx in DF.groupby("session", sort=False).indices.items():
    tt, cc = t[idx], c[idx]
    m = (tt >= 8*3600+30*60) & (tt < START)
    if not m.any(): continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0: continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    BV, SV = bv[idx][w].sum(), sv[idx][w].sum()
    ia = (BV - SV) / max(BV + SV, 1) * np.sign(mv)
    bucket = ("hot_" if hot else "cold_") + ("agree" if agree else "dis") + (("_imb" if ia >= IMB_TH else "") if hot or not agree else "")
    sz = (9 if agree else 7) if hot else (6 if not agree else 0)
    if sz > 0 and ia >= IMB_TH: sz = max(1, sz - 3)
    # simulate the trade manually
    side = -1 if mv >= 0 else 1
    pos, entry, done = 0, 0.0, False
    pnl = 0.0
    for k in range(len(idx)):
        if tt[k] < START or tt[k] >= EXIT: 
            if pos != 0: pnl += pos * (cc[k] - entry); pos = 0
        elif pos == 0 and not done: pos, entry, done = side * sz, cc[k], True
        elif pos != 0:
            p = (cc[k] - entry) * np.sign(pos)
            if p <= -STOP_PTS or p >= TGT_PTS:
                pnl += pos * (cc[k] - entry); pos = 0
    rows.append(dict(bucket=bucket, onv=onv, mv=mv, ia=ia, sz=sz, pnl=pnl * 2, date=str(_)[:10]))
df = pd.DataFrame(rows)
df["half"] = np.where(df.index < len(df) / 2, "h1", "h2")
print(df.groupby(["bucket", "half"]).agg(n=("pnl", "size"), tot=("pnl", "sum"), avg=("pnl", "mean")).round(1))
print("\nby |onv| decile x half:")
df["absonv"] = df.onv.abs()
print(df.groupby([pd.cut(df.absonv, [-1, 25, 50, 100, 175, 250, 400, 10000]), "half"]).agg(n=("pnl", "size"), avg=("pnl", "mean"), tot=("pnl", "sum")).round(1))
print("\nweekday x half:")
df["wd"] = pd.to_datetime(df.date).dt.dayofweek
print(df.groupby(["wd", "half"]).agg(n=("pnl", "size"), avg=("pnl", "mean")).round(1))
print("\nmv size decile x half:")
df["absmv"] = df.mv.abs()
print(df.groupby([pd.cut(df.absmv, [-1, 10, 20, 35, 60, 10000]), "half"]).agg(n=("pnl", "size"), avg=("pnl", "mean"), tot=("pnl", "sum")).round(1))
print("\ndir x half:")
print(df.groupby([np.where(df.mv > 0, "up", "down"), "half"]).agg(n=("pnl", "size"), avg=("pnl", "mean")).round(1))
