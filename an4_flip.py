import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, sess.max()+1)
sessions = df.session.unique()
dp = pd.DataFrame({"session": sessions, "pnl": day_pnl, "dmin": day_min, "ntr": trades})
dp["date"] = pd.to_datetime(dp.session)
# detect flips: sign change of nonzero positions within session
flip_days = set()
for _, idx in df.groupby("session", sort=False).indices.items():
    p = target[idx]
    nz = p[p != 0]
    if len(nz) >= 2 and (np.sign(nz[0]) != np.sign(nz[-1])):
        flip_days.add(df.session.values[idx[0]])
dp["flipped"] = dp.session.isin(flip_days)
t = dp[dp.pnl != 0]
print("flip days:", t.flipped.sum(), "pnl", t[t.flipped].pnl.sum(), "mean", t[t.flipped].pnl.mean().round(0))
print("  wins", (t[t.flipped].pnl>0).sum(), "losses", (t[t.flipped].pnl<0).sum(), "DLLhits", (t[t.flipped].dmin<=-1000).sum())
print("no-flip days:", (~t.flipped).sum(), "pnl", t[~t.flipped].pnl.sum(), "mean", t[~t.flipped].pnl.mean().round(0))
print("  wins", (t[~t.flipped].pnl>0).sum(), "losses", (t[~t.flipped].pnl<0).sum(), "DLLhits", (t[~t.flipped].dmin<=-1000).sum())
print("\nflip day pnl dist:"); print(t[t.flipped].pnl.describe().round(0))
print("\nflip losses detail:"); print(t[t.flipped & (t.pnl<0)][["date","pnl","dmin","ntr"]].to_string())
print("\nflip wins detail:"); print(t[t.flipped & (t.pnl>0)][["date","pnl","ntr"]].to_string())
