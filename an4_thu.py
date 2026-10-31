import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, SESS.max()+1)
d = pd.DataFrame({"date": pd.to_datetime(df.session.unique()), "pnl": day_pnl, "ntr": trades})
thu = d[d.date.dt.dayofweek == 3]
print(thu[thu.ntr > 0].to_string())
