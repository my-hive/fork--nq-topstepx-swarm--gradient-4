"""Slow-attempt anatomy under big30/95 variant: which windows take >18 days, what's in them."""
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
n = len(day_pnl)
dates = pd.to_datetime(DF.session.unique())
pnl = pd.Series(day_pnl, index=dates)
mind = pd.Series(day_min, index=dates)

res = sim.attempts(day_pnl, day_min, 0, n // 2) + sim.attempts(day_pnl, day_min, n // 2, n)
slow = [(st, d) for st, (p, d) in enumerate(res) if p and d >= 18]
print("slow attempts (>=18 days):", len(slow))
for st, d in slow:
    win = pnl.iloc[st:st + d]
    print(f"\nstart {dates[st].date()} days={d} pnl_sum={win.sum():.0f} "
          f"best={win.max():.0f} nwin={(win > 0).sum()} nflat={(win == 0).sum()} nloss={(win < 0).sum()}")
    print(win[win != 0].round(0).to_string())
