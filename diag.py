"""Per-bucket diagnostics + new gate candidates (volume imbalance, push size, DOW)."""
import numpy as np
import pandas as pd
import sweep
from sweep import evaluate, show, bounds, t, c, NS, START

df = sweep.df
buy, sell, vol = df.buy_vol.values, df.sell_vol.values, df.volume.values
dates = df.session.values

# per-day features
rows = []
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc = t[i0:i1], c[i0:i1]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
    if not m.any():
        continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    bv, sv = buy[i0:i1][w].sum(), sell[i0:i1][w].sum()
    imb = (bv - sv) / max(bv + sv, 1)
    rows.append(dict(s=s, date=dates[i0], onv=onv, mv=mv, imb=imb, push=abs(mv),
                     bucket=sweep.buckets[s], side=sweep.sides[s]))
F = pd.DataFrame(rows)
F["half"] = np.where(F.s < NS // 2, "h1", "h2")
print("bucket pops (0=hot+dis,1=hot+agree,2=cold+dis,3=cold+agree):")
print(F.groupby(["bucket", "half"]).size())
print("\npush |mv| stats by bucket:")
print(F.groupby("bucket").push.describe()[["count", "mean", "50%", "max"]])
print("\nimbalance stats by bucket:")
print(F.groupby("bucket").imb.describe()[["count", "mean", "50%"]])

# fade outcome per bucket (approx: sign of (entry-to-exit) pnl using simple 40/80 walk is complex;
# use next-day proxy: move from 8:35 close to 14:30 close reversed sign correlation)
out = []
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc = t[i0:i1], c[i0:i1]
    m835 = tt >= START
    m1430 = tt < 14 * 3600 + 30 * 60
    if not (m835.any() and m1430.any()):
        continue
    e = cc[np.where(m835)[0][0]]
    x = cc[np.where(m143.any() if False else m1430)[0][-1]]
    out.append(dict(s=s, bucket=sweep.buckets[s], ret_fade=(e - x)))  # fade pnl in pts (before stop/target)
O = pd.DataFrame(out).merge(F[["s", "half"]], on="s")
print("\nfade pts 8:35->14:30 by bucket/half (mean, sum):")
print(O.groupby(["bucket", "half"]).ret_fade.agg(["count", "mean", "sum"]).round(1))
