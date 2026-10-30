"""gu-4: decompose pass delay on 13582df: target-hit day vs consistency-satisfied day."""
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

def first_target_day(dp, st):
    bal, best, d = 50000.0, 0.0, 0
    for s in range(st, st + 30):
        # check MLL
        thr = min(max(50000.0, bal) - 2000, 50000.0)
        if bal + day_min[s] <= thr:
            return None, None
        bal += dp[s]; d += 1
        best = max(best, dp[s])
        profit = bal - 50000.0
        if profit >= 3000:
            ok = best < 0.5 * profit
            return d, (d if ok else None)

n = len(day_pnl)
d1s, dds, cons_delayed = [], [], 0
for half in (range(0, n // 2 - 29), range(n // 2, n - 29)):
    for st in half:
        d1, dp_ = first_target_day(day_pnl, st)
        d1s.append(d1)
        # actual pass day from sim.attempts
        dds.append(dp_)
d1s = np.array([x for x in d1s if x is not None])
dds = np.array([x for x in dds if x is not None])
print("attempts with target hit:", len(d1s), "passed:", len(dds))
print("mean first-target day:", d1s.mean().round(2), "mean pass day:", dds.mean().round(2))
print("consistency-delayed attempts:", (dds > d1s).sum(), "mean extra days:", (dds - d1s)[dds > d1s].mean().round(2) if (dds > d1s).any() else 0)
