"""Which days bust the h1-failing windows under 98e5e788 (RETR_MIN 8)?"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from strategy import signals
from eval.sim import run_days, attempts, WINDOW, START, TARGET, MLL

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess_ids = pd.DatetimeIndex(df.session.unique())
sess = pd.factorize(df.session, sort=True)[0]
tg = np.asarray(signals(df), dtype=np.int64)
dp, dm, tr = run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
n = len(dp); mid = n//2

fails = []
for st in range(0, mid - WINDOW + 1):
    bal, peak_eod, best, days = START, START, 0.0, 0
    ok = False
    for s in range(st, st + WINDOW):
        thr = min(peak_eod - MLL, START)
        if bal + dm[s] <= thr: fails.append((st, s)); break
        bal += dp[s]; days += 1
        best = max(best, dp[s]); peak_eod = max(peak_eod, bal)
        if bal - START >= TARGET and days >= 2 and best < 0.5*(bal-START): ok = True; break
    else:
        fails.append((st, -1))
print("h1 fails:", len(fails))
bust_days = sorted(set(s for _, s in fails if s >= 0))
print("bust days:", [(s, str(sess_ids[s].date()), round(dp[s]), round(dm[s])) for s in bust_days])
# what classes traded on those days
D = dict(t=df.ct_sec.values, c=df.close.values, hi=df.high.values, lo=df.low.values,
         bv=df.buy_vol.values, sv=df.sell_vol.values)
dows = pd.to_datetime(df.session.values).dayofweek
groups = list(df.groupby("session", sort=False).indices.items())
for s in bust_days:
    _, idx = groups[s]
    tt, cc = D['t'][idx], D['c'][idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
    hot = not (50 <= abs(onv) <= 200)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    big = abs(onv) > 200
    ext = D['hi'][idx][w].max() if mv > 0 else D['lo'][idx][w].min()
    retr = abs(ext - cc[w[-1]])
    cls = ("big_agree" if agree else "big_dis") if big else (("hot_agree" if agree else "hot_dis") if hot else "cold_dis")
    print(f"  si={s} {sess_ids[s].date()} dow={dows[idx[0]]} {cls} onv={onv:+.0f} mv={mv:+.0f} retr={retr:.1f} pnl={dp[s]:.0f} dmin={dm[s]:.0f}")
