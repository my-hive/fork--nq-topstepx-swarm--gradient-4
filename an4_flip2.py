"""Feature table for flip days on current base: winners vs DLL losers."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals, ON_LO, ON_HI, IMB_TH
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, SESS.max()+1)
sd = pd.to_datetime(df.session.values)
rows = []
for _, idx in df.groupby("session", sort=False).indices.items():
    p = target[idx]
    nz = p[p != 0]
    if len(nz) < 2 or np.sign(nz[0]) == np.sign(nz[-1]): continue  # flips only
    tt, cc = df.ct_sec.values[idx], df.close.values[idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    where = np.where(m)[0]
    onv = cc[where[0]] - cc[0]; mv = cc[where[-1]] - cc[where[0]]
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    imb = (df.buy_vol.values[idx][where].sum() - df.sell_vol.values[idx][where].sum()) / max(df.buy_vol.values[idx][where].sum() + df.sell_vol.values[idx][where].sum(), 1)
    s = df.session.values[idx[0]]
    si = list(df.session.unique()).index(s)
    rows.append(dict(date=str(sd[idx[0]])[:10], dow=sd[idx[0]].dayofweek, pnl=day_pnl[si], dmin=day_min[si],
                     onv=round(onv,1), mv=round(mv,1), hot=hot, agree=agree, imb_along=round(imb*np.sign(mv),3),
                     big_agree=agree and abs(onv)>200, ntr=trades[si]))
r = pd.DataFrame(rows)
r["win"] = r.pnl > 0
print(r.sort_values("pnl").to_string())
print("\nwinners vs losers by class:")
print(r.groupby(["win","big_agree","hot","agree"]).size())
print("\nimbalong stats:"); print(r.groupby("win").imb_along.describe().round(3))
print("\nabs(mv) stats:"); print(r.groupby("win").mv.apply(lambda x: x.abs().describe().round(1)))
print("\nabs(onv) stats:"); print(r.groupby("win").onv.apply(lambda x: x.abs().describe().round(1)))
