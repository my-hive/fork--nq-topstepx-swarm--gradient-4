"""Failure-mode analysis of current strategy: why do attempts fail?"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
from strategy import signals
from harness import DF, SESS
import sim

target = np.nan_to_num(np.asarray(signals(DF), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)
n, mid, W = len(day_pnl), len(day_pnl) // 2, 30
print("day_pnl: mean %.0f  p25 %.0f  p50 %.0f  p75 %.0f  min %.0f  win %.3f" % (
    day_pnl.mean(), *np.percentile(day_pnl, [25, 50, 75]), day_pnl.min(), (day_pnl > 0).mean()))
print("day_min: min %.0f  frac < -900: %.3f" % (day_min.min(), (day_min < -900).mean()))

for name, lo, hi in (("h1", 0, mid), ("h2", mid, n)):
    modes = {}
    for st in range(lo, hi - W + 1):
        bal, peak_eod, best, days = 50000.0, 50000.0, 0.0, 0
        mode = "timeout"
        for s in range(st, st + W):
            thr = min(peak_eod - 2000.0, 50000.0)
            if bal + day_min[s] <= thr:
                mode = "MLL_bust"; break
            bal += day_pnl[s]; days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - 50000.0
            if profit >= 3000.0:
                if days >= 2 and best < 0.5 * profit:
                    mode = "PASS"; break
                elif best >= 0.5 * profit:
                    mode = "consistency"; break
        modes[mode] = modes.get(mode, 0) + 1
    print(name, modes, "rate", modes.get("PASS", 0) / sum(modes.values()))

# day_pnl autocorrelation: do wins cluster?
dp = day_pnl - day_pnl.mean()
print("\nlag1 corr %.3f  lag2 %.3f" % (np.corrcoef(dp[:-1], dp[1:])[0, 1],
                                        np.corrcoef(dp[:-2], dp[2:])[0, 1]))
# histogram of day_pnl
print(pd.Series(day_pnl).describe())
print((day_pnl > 1400).sum(), "days > 1400;", (day_pnl < -600).sum(), "days < -600")
