import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.expanduser("."))
from strategy import signals
sys.path.insert(0, "eval")
import sim

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, sess.max()+1)
sessions = df.session.unique()
d = pd.DataFrame({"session": sessions, "pnl": day_pnl, "dmin": day_min, "ntr": trades})
d["date"] = pd.to_datetime(d.session)
d = d.sort_values("date").reset_index(drop=True)
print("n sess", len(d), "total", d.pnl.sum())
print("worst 15 days:")
print(d.nsmallest(15, "pnl")[["session","pnl","dmin","ntr"]].to_string())
print("best 10 days:")
print(d.nlargest(10, "pnl")[["session","pnl","ntr"]].to_string())
print("DLL-hit days (dmin<=-1000):", (d.dmin<=-1000).sum())
print("neg days:", (d.pnl<0).sum(), "sum neg:", d.pnl[d.pnl<0].sum())
print("pos days:", (d.pnl>0).sum(), "sum pos:", d.pnl[d.pnl>0].sum())
print("zero days:", (d.pnl==0).sum())
d.to_csv("day_pnl4.csv", index=False)
