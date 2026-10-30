"""cold_dis Monday analysis + skip test."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from strategy import signals
from eval.sim import run_days, score, attempts

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
sess_ids = pd.DatetimeIndex(df.session.unique())
tg = np.asarray(signals(df), dtype=np.int64)
dp, dm, tr = run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
D = dict(t=df.ct_sec.values, c=df.close.values)
dows = pd.to_datetime(df.session.values).dayofweek
groups = list(df.groupby("session", sort=False).indices.items())
rows = []
for si, (_, idx) in enumerate(groups):
    tt, cc = D['t'][idx], D['c'][idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    w = np.where(m)[0]
    if len(w)==0: continue
    onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
    if mv == 0: continue
    hot = not (50 <= abs(onv) <= 200)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    if not hot and not agree:
        rows.append((si, str(sess_ids[si].date()), dows[idx[0]], onv, mv, dp[si]))
r = pd.DataFrame(rows, columns=["si","date","dow","onv","mv","pnl"])
print(r.groupby("dow").pnl.agg(["sum","count","mean"]).round(0))
print("\nall cold_dis Monday days:")
print(r[r.dow==0].to_string(index=False))
