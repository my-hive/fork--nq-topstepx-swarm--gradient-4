# grid on MY HEAD base (ff15e00, fails=0, dll=5): CD x hot-dis; maximize base keeping fails==0
import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim
BASE = open('strategy.py').read()

def build(cd, hd):
    s = BASE.replace("sz = 10 if agree else 9  # hot-dis plateau 8-10; 10+ cliff on tail cuts", f"sz = 10 if agree else {hd}")
    s = s.replace("sz = 6 if not agree else 0", f"sz = {cd} if not agree else 0")
    open('_sw.py', 'w').write(s)
    if '_sw' in sys.modules: del sys.modules['_sw']
    return importlib.import_module('_sw')

df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
WINDOW, START, TARGET, MLL = 30, 50000.0, 3000.0, 2000.0

def allrate(dp, dm):
    n = 0
    for st in range(0, len(dp) - WINDOW + 1):
        bal, peak_eod, best, days, ok = START, START, 0.0, 0, False
        for s in range(st, st + WINDOW):
            if bal + dm[s] <= min(peak_eod - MLL, START): break
            bal += dp[s]; days += 1
            best, peak_eod = max(best, dp[s]), max(peak_eod, bal)
            p = bal - START
            if p >= TARGET and days >= 2 and best < 0.5 * p: ok = True; break
        if not ok: n += 1
    return n

for cd in (5, 6, 7, 8):
    for hd in (8, 9, 10):
        m = build(cd, hd)
        tg = np.asarray(m.signals(df)).astype(np.int64)
        dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max() + 1)
        sc = sim.score(dp, dm)[0]
        print(f"CD={cd} HD={hd}: base={sc:.4f} fails={allrate(dp, dm)} dll={(dp<=-990).sum()} pnl={dp.sum():.0f}", flush=True)
