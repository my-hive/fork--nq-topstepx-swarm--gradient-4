"""gu-4: holdout-risk estimate for 13582df: pass rate by calendar quarter + rolling 30-session pass count."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
import strategy as S

target = np.nan_to_num(np.asarray(S.signals(DF), dtype=float)).astype(np.int64)
day_pnl, day_min, _ = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                   DF.low.values, DF.close.values, target, SESS.max() + 1)
dates = pd.to_datetime(DF.session.unique())
n = len(day_pnl)

# every possible 30-session window pass
res = []
for st in range(n - 29):
    bal, peak, best, days = 50000.0, 50000.0, 0.0, 0
    out = (False, 30)
    for s in range(st, st + 30):
        thr = min(peak - 2000, 50000)
        if bal + day_min[s] <= thr:
            out = (False, days); break
        bal += day_pnl[s]; days += 1
        best, peak = max(best, day_pnl[s]), max(peak, bal)
        profit = bal - 50000
        if profit >= 3000 and days >= 2 and best < 0.5 * profit:
            out = (True, days); break
    res.append((dates[st], out[0], out[1]))

r = pd.DataFrame(res, columns=["start", "pass", "days"])
r["q"] = r.start.dt.to_period("Q")
print(r.groupby("q").agg(n=("pass", "size"), rate=("pass", "mean"), days=("days", "mean")).round(3))
print("\nfailing window starts:", r[~r["pass"]].start.dt.strftime("%Y-%m-%d").tolist())
print("overall rate:", r["pass"].mean().round(4), "days:", r[r["pass"]].days.mean().round(2))

# per-quarter day pnl
q = pd.Series(day_pnl, index=dates).groupby(dates.to_period("Q")).agg(["mean", "sum", "count"])
print("\nquarter pnl:\n", q.round(0))
