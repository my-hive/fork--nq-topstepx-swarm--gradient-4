"""Per passed attempt: $ short of passing one day earlier (and 2 days earlier)."""
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
gaps1, gaps2 = [], []
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
            hist.append((profit, best, days, ok))
            if ok: passed = True; break
        if not passed or len(hist) < 2: continue
        # shortfall at day d-1 (pass one day earlier): need profit>=3000, days>=2, consistency
        for d in (len(hist)-2, len(hist)-3):
            if d < 1: continue
            profit, best, ndays, _ = hist[d]
            need = max(3000 - profit, 0)
            # consistency at that point: best < 0.5*(profit+need)... approximate: need also 2*best-profit if best large
            if best >= 0.5 * (profit + need):
                need = max(need, 2*best - profit + 1)
            (gaps1 if d == len(hist)-2 else gaps2).append(need)
g1 = np.array(gaps1); g2 = np.array(gaps2)
print("windows:", len(g1))
for q in [0, 10, 25, 50, 100, 200, 400, 800]:
    print(f"  gap1 <= {q}$: {(g1 <= q).sum()} windows")
print("gap1 sorted (first 25):", np.sort(g1)[:25].astype(int).tolist())
print("gap2 sorted (first 15):", np.sort(g2)[:15].astype(int).tolist() if len(g2) else "n/a")
print("mean days saved if all gap1 closed:", (len(g1))/159)
