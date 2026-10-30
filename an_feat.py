"""Feature table for all traded sessions under BASE CD8: bucket, outcome, features.
Find what separates loss days from win days (esp. Sep/Oct -600s)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
from exp_flip import signals

target = signals(DF, BST=30.0, BTG=95.0)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)
dates = pd.to_datetime(DF.session.unique())
t, c, bv, sv = DF.ct_sec.values, DF.close.values, DF.buy_vol.values, DF.sell_vol.values
hi, lo = DF.high.values, DF.low.values
srt = np.where(np.r_[True, SESS[1:] != SESS[:-1]])[0]
end = np.r_[srt[1:] - 1, len(SESS) - 1]
START = 8 * 3600 + 35 * 60

rows = []
for s in range(len(dates)):
    i0, i1 = srt[s], end[s]
    tt, cc = t[i0:i1 + 1], c[i0:i1 + 1]
    w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < START))[0]
    if len(w) == 0:
        continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    hot = not (50 <= abs(onv) <= 200)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    bsum, ssum = bv[i0 + w].sum(), sv[i0 + w].sum()
    imb = (bsum - ssum) / max(bsum + ssum, 1) * np.sign(mv)
    pvol = bsum + ssum
    ext = hi[i0 + w].max() if mv > 0 else lo[i0 + w].min()
    retr = abs(ext - cc[w[-1]])
    onrng = hi[i0:i0 + w[0]].max() - lo[i0:i0 + w[0]].min()
    md = dates[s].month * 100 + dates[s].day
    bucket = ("hot_agree" if hot and agree else "hot_dis" if hot else "cold_dis" if not agree else "cold_agree")
    rows.append(dict(date=str(dates[s].date()), dow=dates[s].dayofweek, bucket=bucket,
                     pnl=day_pnl[s], abs_onv=abs(onv), abs_mv=abs(mv), imb=imb, pvol=pvol,
                     retr=retr, onrng=onrng, holiday=int(md >= 1210 or md <= 102)))

r = pd.DataFrame(rows)
r.to_csv("day_features.csv", index=False)
traded = r[(r.pnl != 0) & (r.holiday == 0)]
print("traded days:", len(traded))
print("\nby bucket:")
print(traded.groupby("bucket").pnl.agg(["count", "mean", "sum", "min", "max"]).round(0))
loss = traded[traded.pnl < 0]
win = traded[traded.pnl > 500]
print(f"\nloss days n={len(loss)}:")
print(loss[["date", "dow", "bucket", "pnl", "abs_onv", "abs_mv", "imb", "pvol", "retr", "onrng"]].round(2).to_string())
print(f"\nbig-win (>500) days n={len(win)}:")
print(win[["date", "dow", "bucket", "pnl", "abs_onv", "abs_mv", "imb", "pvol", "retr", "onrng"]].round(2).to_string())
print("\nfeature medians loss vs win:")
print(pd.DataFrame({"loss": loss[["abs_mv", "imb", "pvol", "retr", "onrng"]].median(),
                    "win": win[["abs_mv", "imb", "pvol", "retr", "onrng"]].median()}).round(2))
