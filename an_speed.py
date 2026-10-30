"""Analyze BASE CD8 day PnL structure: DLL hits, loss days, win days, per-half."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
from exp_re import signals

target = signals(DF)  # BASE CD8
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)
n = len(day_pnl)
mid = n // 2
sc, rates, passed = sim.score(day_pnl, day_min)
print("score", sc, rates, "days", np.mean(passed))

dates = pd.to_datetime(DF.session.unique())
pnl = pd.Series(day_pnl, index=dates)
mind = pd.Series(day_min, index=dates)
dll = mind <= -1000 + 1e-9
print(f"\nDLL-hit days: {dll.sum()} ({dll.mean():.2%})")
print(dll[dll].index.strftime("%Y-%m-%d").tolist())
print("\nloss days:", (pnl < 0).sum(), "win days:", (pnl > 0).sum(), "flat:", (pnl == 0).sum())
print("day pnl quantiles:\n", pnl.describe([.05, .25, .5, .75, .95]).round(0))

# where do attempts slow down: distribution of passed days
res = sim.attempts(day_pnl, day_min, 0, mid) + sim.attempts(day_pnl, day_min, mid, n)
pd_days = [d for p, d in res if p]
failed = [st for (p, d), st in zip(res, range(n - 30 + 1)) if not p]
print(f"\npassed days: min={min(pd_days)} p25={np.percentile(pd_days,25):.0f} med={np.median(pd_days):.0f} p75={np.percentile(pd_days,75):.0f} max={max(pd_days)}")
print("failed attempt start idx:", failed)

# DLL days by half
print("\nDLL days by half:", int(dll[:mid].sum()), int(dll[mid:].sum()))
# biggest loss days and biggest win days
print("\nworst 8 days:\n", pnl.nsmallest(8).round(0))
print("\nbest 8 days:\n", pnl.nlargest(8).round(0))
