"""Final failure anatomy on 7107298e: which sessions kill the failing windows."""
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

def fail_windows(lo, hi):
    out = []
    for st in range(lo, hi - WINDOW + 1):
        bal, peak_eod, best, days = START, START, 0.0, 0
        ok = False
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - MLL, START)
            if bal + dm[s] <= thr: out.append((st, s, "MLL@%d" % s)); break
            bal += dp[s]; days += 1
            best = max(best, dp[s]); peak_eod = max(peak_eod, bal)
            if bal - START >= TARGET and days >= 2 and best < 0.5*(bal-START): ok = True; break
        else:
            out.append((st, -1, "slow"))
    return out

for name, lo, hi in [("h1", 0, mid), ("h2", mid, n)]:
    fw = fail_windows(lo, hi)
    print(f"{name}: {len(fw)} fails")
    for st, s, why in fw:
        # days contributing within failing window up to bust
        seg = dp[st:st+WINDOW]
        print(f"  start={st} {why}: window pnl sum={seg.sum():.0f} worst={seg.min():.0f} ndays<-700={(seg<-700).sum()}")
print("\nsession idx of MLL bust days:", sorted(set(s for _,s,w in fail_windows(0,n) if s>=0)))
print("dates:", [str(sess_ids[s].date()) for _,s,w in fail_windows(0,n) if s>=0][:20])
