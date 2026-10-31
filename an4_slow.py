import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, sess.max()+1)
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
        res.append(dict(start=st, half=half, days=days if passed else 99, passed=passed,
                        start_date=str(dates[st])[:10]))
r = pd.DataFrame(res)
print("h1 days mean", r[r.half==0].days.mean().round(3), "h2", r[r.half==1].days.mean().round(3))
print("slowest 12 windows:")
print(r.nlargest(12, "days").to_string())
# composition of slow windows: count zero/neg/dll days in first 12 sessions
def comp(row):
    seg = day_pnl[row.start:row.start+12]
    return dict(start=row.start_date, days=row.days, zero=(seg==0).sum(), neg=(seg<0).sum(),
                dll=(day_min[row.start:row.start+12] <= -1000).sum(), pnl=seg.sum().round(0))
print("\nslow window composition (first 12 sessions):")
for _, row in r.nlargest(8, "days").iterrows():
    print(comp(row))
print("\nfast window composition:")
for _, row in r.nsmallest(5, "days").iterrows():
    print(comp(row))
