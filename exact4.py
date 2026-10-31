import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, sess.max()+1)
sc, rates, passed = sim.score(day_pnl, day_min)
print(f"score {sc:.9f} rates {rates} avg_days {np.mean(passed):.6f} pnl {day_pnl.sum():.0f} worst {day_pnl.min():.0f}")
big = day_pnl[day_pnl > 1400]
print("days>1400:", len(big), big[:20])
