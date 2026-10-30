"""gu-4: per-regime day pnl analysis on 13582df base — where does profit live, what can scale?"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
import strategy as S

target = np.nan_to_num(np.asarray(S.signals(DF), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)
dates = pd.to_datetime(DF.session.unique())

# recompute regime per session (mirror strategy logic)
rows = []
for _, idx in DF.groupby("session", sort=False).indices.items():
    tt, cc = DF.ct_sec.values[idx], DF.close.values[idx]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60)
    if not m.any():
        continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    hot = not (50.0 <= abs(onv) <= 200.0)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    imb = (DF.buy_vol.values[idx][w].sum() - DF.sell_vol.values[idx][w].sum()) / max(
        DF.buy_vol.values[idx][w].sum() + DF.sell_vol.values[idx][w].sum(), 1)
    reg = ("bigON" if abs(onv) > 200 else "smallON" if hot else "cold") + ("-agr" if agree else "-dis")
    rows.append((DF.session.values[idx[0]], reg, imb * np.sign(mv)))

regs = pd.DataFrame(rows, columns=["sess", "reg", "imb"]).set_index("sess")
pnl = pd.Series(day_pnl, index=dates)
tr = pd.Series(trades, index=dates)
res = pnl.to_frame("pnl").join(tr.to_frame("ct"))
res = res.join(regs, on=res.index)
res["dow"] = res.index.dayofweek

print(res.groupby("reg").agg(n=("pnl", "size"), sum=("pnl", "sum"), mean=("pnl", "mean"),
                             win=("pnl", lambda x: (x > 0).mean()), ct=("ct", "sum")).round(0))
print("\nby reg+dow (top/bottom):")
g = res.groupby([res.reg, res.dow]).agg(n=("pnl", "size"), sum=("pnl", "sum"), mean=("pnl", "mean")).round(0)
print(g[g.n >= 5].sort_values("mean").to_string())

print("\nskipped days pnl:", res[res.reg.isna()].pnl.sum(), res.reg.isna().sum())
# big winning days (consistency risk)
print("\ntop-10 day pnl:", res.pnl.nlargest(10).round(0).tolist())
