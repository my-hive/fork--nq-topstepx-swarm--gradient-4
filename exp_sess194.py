"""Identify classes of the MLL-bust days: 74,76,84,108,109,194 and their pnl/min."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from strategy import signals
from eval.sim import run_days

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess_ids = pd.DatetimeIndex(df.session.unique())
sess = pd.factorize(df.session, sort=True)[0]
D = dict(t=df.ct_sec.values, c=df.close.values, hi=df.high.values, lo=df.low.values,
         bv=df.buy_vol.values, sv=df.sell_vol.values)
dows = pd.to_datetime(df.session.values).dayofweek
tg = np.asarray(signals(df), dtype=np.int64)
dp, dm, tr = run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)

for si in [69, 74, 76, 84, 87, 92, 107, 108, 109, 174, 194]:
    idx = None
    for _, ii in df.groupby("session", sort=False).indices.items():
        pass
# map session idx -> row block
starts = {}
for _, ii in df.groupby("session", sort=False).indices.items():
    pass
groups = list(df.groupby("session", sort=False).indices.items())
for si in [69, 74, 76, 84, 87, 92, 107, 108, 109, 174, 194]:
    _, idx = groups[si]
    tt, cc = D['t'][idx], D['c'][idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
    hot = not (50 <= abs(onv) <= 200)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    big = abs(onv) > 200
    vol = D['bv'][idx][w].sum() + D['sv'][idx][w].sum()
    imb = (D['bv'][idx][w].sum() - D['sv'][idx][w].sum()) / max(vol, 1)
    ext = D['hi'][idx][w].max() if mv > 0 else D['lo'][idx][w].min()
    retr = abs(ext - cc[w[-1]])
    cls = ("big_agree" if agree else "big_dis") if big else (("hot_agree" if agree else "hot_dis") if hot else "cold_dis")
    print(f"si={si} {sess_ids[si].date()} dow={dows[idx[0]]} {cls:10s} onv={onv:+.0f} mv={mv:+.0f} vol={vol:.0f} imb={imb:+.2f} retr={retr:.1f} pnl={dp[si]:.0f} dmin={dm[si]:.0f}")
