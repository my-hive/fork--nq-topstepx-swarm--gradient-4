"""Cold-agree day fade PnL (paper): what if we faded cold-agree days at small size?
Cold-agree = |onv| in [50,200] (dead zone) and push agrees with onv."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60

dates = pd.to_datetime(DF.session.unique())
NS = len(dates)
t, c = DF.ct_sec.values, DF.close.values
srt = np.where(np.r_[True, SESS[1:] != SESS[:-1]])[0]
end = np.r_[srt[1:] - 1, len(SESS) - 1]

rows = []
for s in range(NS):
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
    if hot or not agree:
        continue  # cold-agree only
    md = dates[s].month * 100 + dates[s].day
    if md >= 1210 or md <= 102:
        continue
    # paper fade at 8:35, stop 40 tgt 80 (like moderate agree, stand aside after stop)
    side = -1 if mv >= 0 else 1
    pos, entry, state = 0, 0.0, 0
    pnl = 0.0
    for k in range(len(tt)):
        if tt[k] < START or tt[k] >= EXIT:
            pos = 0
        elif state == 0:
            pos, entry, state = side, cc[k], 1
        elif pos != 0 and state == 1:
            p = (cc[k] - entry) * side
            if p <= -40:
                pnl, state, pos = -40, 4, 0
            elif p >= 80:
                pnl, state, pos = 80, 4, 0
    if state == 1:  # still open at exit: mark to market
        pnl = (cc[-1] - entry) * side
    rows.append(dict(date=str(dates[s])[:10], pnl=pnl, half=int(s > NS / 2),
                     onv=abs(onv), mv=mv))

r = pd.DataFrame(rows)
print(f"n cold-agree days: {len(r)}")
print(r.pnl.describe().round(1))
print("\nby half:"); print(r.groupby("half").pnl.agg(["count", "mean", "sum"]).round(1))
print("\nwin rate:", (r.pnl > 0).mean().round(3), " stop-out rate:", (r.pnl <= -40).mean().round(3))
r.to_csv("cold_agree_days.csv", index=False)
