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
res = []
for half, (lo, hi) in enumerate([(0, mid), (mid, n)]):
    for st in range(lo, hi - 30 + 1):
        bal, peak, best, days = 50000.0, 50000.0, 0.0, 0
        passed = False
        for s in range(st, st + 30):
            thr = min(peak - 2000, 50000)
            if bal + day_min[s] <= thr: break
            bal += day_pnl[s]; days += 1
            best, peak = max(best, day_pnl[s]), max(peak, bal)
            profit = bal - 50000
            if profit >= 3000 and days >= 2 and best < 0.5*profit:
                passed = True; break
        res.append(dict(half=half, start=st, days=days if passed else 99, passed=passed))
r = pd.DataFrame(res)
p = r[r.passed]
print("passed:", len(p), "mean days:", p.days.mean())
print(p.days.value_counts().sort_index().to_string())
print("\nh1 mean", p[p.half==0].days.mean().round(3), "h2 mean", p[p.half==1].days.mean().round(3))
print("\nslowest 10:")
for _, row in p.nlargest(10, "days").iterrows():
    st = int(row.start)
    seg = day_pnl[st:st+row.days]
    print(f"start {dates[st].date()} half{row.half} days {row.days} cum {seg.sum():.0f} pnls {list(np.round(seg))}")
