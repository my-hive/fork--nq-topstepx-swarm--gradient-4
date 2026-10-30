# map failing 30d windows on s_24ff1d60 (24ff1d60 = 1.0062): failure mode per window
import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim

m = importlib.import_module('s_24ff1d60')
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
tg = np.asarray(m.signals(df)).astype(np.int64)
dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max() + 1)
print("n_sessions", len(dp))
WINDOW = 30; START, TARGET, MLL = 50000.0, 3000.0, 2000.0

def att(st):
    bal, peak_eod, best, days, mode = START, START, 0.0, 0, 'slow'
    for s in range(st, st + WINDOW):
        thr = min(peak_eod - MLL, START)
        if bal + dm[s] <= thr:
            return (False, s - st, f'bust@+{s-st}', dp[s])
        bal += dp[s]; days += 1
        best, peak_eod = max(best, dp[s]), max(peak_eod, bal)
        profit = bal - START
        if profit >= TARGET and days >= 2:
            return (True, days, 'ok', best)
        if profit >= TARGET and days >= 2 and best >= 0.5 * profit:
            mode = 'consistency'
    return (False, days, mode, best)

fails = []
for st in range(0, len(dp) - WINDOW + 1):
    ok, d, mode, aux = att(st)
    if not ok:
        fails.append((st, d, mode, round(aux, 0)))
print("failing windows:", len(fails), "of", len(dp) - WINDOW + 1)
for f in fails:
    print(" st=%d d=%d %s aux=%s sessions[%d..%d]=%s..%s" % (f[0], f[1], f[2], f[3], f[0], f[0] + 29, pd.to_datetime(df.session.unique()[f[0]]).date(), pd.to_datetime(df.session.unique()[f[0] + 29]).date()))
# day pnl around the failing region
un = pd.to_datetime(df.session.unique())
bad = sorted(set(s for st, d, mode, aux in fails for s in range(st, st + 30) if mode.startswith('bust') and st <= s <= st + d))
print("bust days:", [(int(x), str(un[x].date()), round(dp[x], 0)) for x in bad])
print("worst days:", [(int(i), str(un[i].date()), round(dp[i], 0)) for i in np.argsort(dp)[:15]])
