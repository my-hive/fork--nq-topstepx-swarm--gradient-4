# allrate compare across leaderboard bases + shrink=3 variant on top of CD8
import sys, os, subprocess, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
WINDOW, START, TARGET, MLL = 30, 50000.0, 3000.0, 2000.0

def allrate(dp, dm):
    fails = []
    for st in range(0, len(dp) - WINDOW + 1):
        bal, peak_eod, best, days, ok = START, START, 0.0, 0, False
        for s in range(st, st + WINDOW):
            if bal + dm[s] <= min(peak_eod - MLL, START): break
            bal += dp[s]; days += 1
            best, peak_eod = max(best, dp[s]), max(peak_eod, bal)
            p = bal - START
            if p >= TARGET and days >= 2 and best < 0.5 * p: ok = True; break
        if not ok: fails.append(st)
    return fails

for sha, name in [("4afc9a7", "CD7"), ("6b84e7dc", "gu4-ft68"), ("f426721a", "gu1-ft70"), ("ff15e00", "HEAD-mine")]:
    src = subprocess.run(["git", "show", f"{sha}:strategy.py"], capture_output=True, text=True).stdout
    open('_sw.py', 'w').write(src)
    if '_sw' in sys.modules: del sys.modules['_sw']
    m = importlib.import_module('_sw')
    tg = np.asarray(m.signals(df)).astype(np.int64)
    dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max() + 1)
    sc = sim.score(dp, dm)[0]
    f = allrate(dp, dm)
    print(f"{name} ({sha[:7]}): base={sc:.4f} fails={len(f)} {f} dll={(dp<=-990).sum()} pnl={dp.sum():.0f}", flush=True)
