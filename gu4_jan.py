"""gu-4: diagnose Jan 2026 bust cluster (10 failing window starts, invisible to official halves)."""
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
pnl = pd.Series(day_pnl, index=dates)
mind = pd.Series(day_min, index=dates)

# reproduce a failing attempt from 2026-01-12
st = list(dates).index(pd.Timestamp("2026-01-12"))
bal, peak, best, days = 50000.0, 50000.0, 0.0, 0
for s in range(st, st + 30):
    thr = min(peak - 2000, 50000)
    if bal + mind.iloc[s] <= thr:
        print(f"BUST at {dates[s].date()} day {s-st+1}: bal={bal:.0f} day_min={mind.iloc[s]:.0f} thr={thr:.0f}")
        break
    bal += pnl.iloc[s]; days += 1
    best, peak = max(best, pnl.iloc[s]), max(peak, bal)
    if days <= 14:
        print(f"{dates[s].date()} d{days}: pnl={pnl.iloc[s]:+.0f} min={mind.iloc[s]:+.0f} bal={bal:.0f}")

print("\nJan-Feb 2026 worst days:")
jj = pnl["2026-01-01":"2026-02-28"]
print(jj.sort_values().head(12).round(0).to_string())
print("\nJan-Feb 2026 stats: mean", jj.mean().round(0), "win", (jj > 0).mean().round(2), "dll hits",
      (mind["2026-01-01":"2026-02-28"] <= -1000).sum())
