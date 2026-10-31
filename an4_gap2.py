"""Identify windows with smallest gap to passing one day earlier; show their day composition."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, SESS.max()+1)
n = len(day_pnl); mid = n//2
dates = pd.to_datetime(df.session.unique())
rows = []
for lo, hi in [(0, mid), (mid, n)]:
    for st in range(lo, hi - 30 + 1):
        bal, peak, best, days = 50000.0, 50000.0, 0.0, 0
        hist = []
        passed = False
        for s in range(st, st + 30):
            thr = min(peak - 2000, 50000)
            if bal + day_min[s] <= thr: break
            bal += day_pnl[s]; days += 1
            best, peak = max(best, day_pnl[s]), max(peak, bal)
            profit = bal - 50000
            ok = profit >= 3000 and days >= 2 and best < 0.5*profit
            hist.append((s, profit, best, ok))
            if ok: passed = True; break
        if not passed or len(hist) < 2: continue
        s_prev, profit_prev, best_prev, _ = hist[-2]
        need = max(3000 - profit_prev, 0)
        if best_prev >= 0.5*(profit_prev + need): need = max(need, 2*best_prev - profit_prev + 1)
        rows.append(dict(start=st, date=str(dates[st])[:10], pass_days=len(hist), gap=need,
                         prev_day=str(dates[s_prev])[:10], prev_pnl=day_pnl[s_prev]))
r = pd.DataFrame(rows).sort_values("gap")
print(r.head(12).to_string())
