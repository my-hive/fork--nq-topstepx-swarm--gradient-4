"""gu4_dow: Thursday anatomy. Per-DOW pnl/half/bucket; Thursday-specific surgery options."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
from strategy import signals

START0 = 8 * 3600 + 35 * 60
ON_LO, ON_HI = 50.0, 225.0

target = np.asarray(signals(DF), dtype=np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)
n = len(day_pnl); mid = n // 2

t, c = DF.ct_sec.values, DF.close.values
bv, sv = DF.buy_vol.values, DF.sell_vol.values
dates = pd.to_datetime(DF.session.values)
rows = []
for s_i, (_, idx) in enumerate(DF.groupby("session", sort=False).indices.items()):
    tt, cc = t[idx], c[idx]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < START0)
    if not m.any(): continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
    imb = (bv[idx][w].sum() - sv[idx][w].sum()) / max(bv[idx][w].sum() + sv[idx][w].sum(), 1)
    bucket = ("hot_" if not (ON_LO <= abs(onv) <= ON_HI) else "cold_") + \
             ("agree" if (np.sign(mv) == np.sign(onv) if onv != 0 else True) else "dis")
    rows.append(dict(s=s_i, dow=dates[idx[0]].dayofweek, bucket=bucket, onv=onv, mv=mv,
                     imb_along=imb * np.sign(mv) if mv else 0,
                     pnl=day_pnl[s_i], dmin=day_min[s_i],
                     half="h1" if s_i < mid else "h2"))
d = pd.DataFrame(rows)
names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
print("per-DOW pnl by half:")
print(d.pivot_table(index="dow", columns="half", values="pnl", aggfunc=["count", "sum", "mean"]).round(0))
print("\nThu only, per bucket x half:")
th = d[d.dow == 3]
print(th.groupby(["bucket", "half"]).agg(n=("pnl", "size"), tot=("pnl", "sum"), avg=("pnl", "mean"),
      dll=("dmin", lambda x: int((x <= -990).sum()))).round(0))
print("\nThu worst days:")
print(th.nsmallest(10, "pnl")[["s", "bucket", "onv", "mv", "imb_along", "pnl", "dmin", "half"]].to_string(index=False))
print("\nall-DOW dll day counts:")
print(d.assign(dll=(d.dmin <= -990)).groupby("dow").dll.sum())
