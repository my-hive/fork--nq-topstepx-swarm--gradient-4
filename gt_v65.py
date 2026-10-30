# grid: imb-shrink depth x flip-stop on 24ff1d60 base; target: save window cluster st=93..104 w/o losing base
import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim
BASE = open('_base24.py').read()

def build(shrink=2, fstop=45.0, ftgt=68.0, fsz_add=2):
    s = BASE.replace("sz = max(1, sz - 2)", f"sz = max(1, sz - {shrink})")
    s = s.replace("F_STOP_PTS, F_TGT_PTS = 45.0, 68.0", f"F_STOP_PTS, F_TGT_PTS = {fstop}, {ftgt}")
    s = s.replace("-side * (sz + 2)", f"-side * (sz + {fsz_add})")
    open('_sw.py', 'w').write(s)
    m = importlib.import_module('_sw'); importlib.reload(m)
    return m

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

for shrink, fstop, fsz in [(3, 45, 2), (4, 45, 2), (2, 30, 2), (2, 35, 2), (3, 35, 2), (2, 45, 1), (3, 45, 1), (2, 40, 3)]:
    m = build(shrink, fstop, 68.0, fsz)
    tg = np.asarray(m.signals(df)).astype(np.int64)
    dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max() + 1)
    sc = sim.score(dp, dm)[0]
    fails = allrate(dp, dm)
    gap = [f for f in fails if 78 <= f <= 106]
    print(f"shrink={shrink} fstop={fstop} fsz_add={fsz}: base={sc:.4f} fails={len(fails)} gap={gap} dll={(dp<=-990).sum()} pnl={dp.sum():.0f}", flush=True)
