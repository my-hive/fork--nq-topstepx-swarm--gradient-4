import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, sess.max()+1)
n = len(day_pnl)
base_sc, base_rates, base_passed = sim.score(day_pnl, day_min)
base_days = np.mean(base_passed)
print("base days", round(base_days,6), "rates", base_rates)

def avg_days(dp):
    _,_,p = sim.score(dp, day_min)
    return np.mean(p) if p else np.nan

rows = []
for i in range(n):
    if day_pnl[i] == 0: continue
    dp = day_pnl.copy(); dp[i] += 200.0
    rows.append((i, day_pnl[i], avg_days(dp) - base_days))
r = pd.DataFrame(rows, columns=["i","pnl","ddays_per_200"])
r["date"] = pd.to_datetime(df.session.unique())[r.i.values]
r = r.sort_values("ddays_per_200")
print("most negative ddays (boosting these days helps most):")
print(r.head(15).to_string())
print("\nmost positive ddays (these days' losses hurt):")
print(r.tail(15).to_string())
print("\nsum all ddays:", r.ddays_per_200.sum().round(4))
r.to_csv("sens4.csv", index=False)
