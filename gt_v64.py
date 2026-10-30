# variants on 24ff1d60 (1.0062) targeting DLL-day loss cap; objective: all-window pass rate + base score
import sys, os, re, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim

BASE_SRC = open('_base24.py').read()

def build(v):
    s = BASE_SRC
    if v == 'v0': pass
    elif v == 'flipoff':  # ablation: no flip leg
        s = s.replace("""                    else:                           # counters drift / big agree: real momentum
                        pos, entry, state = -side * (sz + 2), cc[k], 3""",
"""                    else:
                        pos, state = 0, 4""")
    elif v == 'fs40':
        s = s.replace("F_STOP_PTS, F_TGT_PTS = 45.0, 68.0", "F_STOP_PTS, F_TGT_PTS = 40.0, 68.0")
    elif v == 'fsz1':  # flip size sz+1
        s = s.replace("-side * (sz + 2)", "-side * (sz + 1)")
    elif v == 'v1':  # skip flip when loss-streak throttled
        s = s.replace("""        if sum(hist[-LOSS_K:]) < LOSS_TH:
            sz = int(round(sz * 0.5))  # drawdown streak: halve size to survive trailing MLL""",
"""        throttled = sum(hist[-LOSS_K:]) < LOSS_TH
        if throttled:
            sz = int(round(sz * 0.5))  # drawdown streak: halve size to survive trailing MLL""")
        s = s.replace("""                    else:                           # counters drift / big agree: real momentum
                        pos, entry, state = -side * (sz + 2), cc[k], 3""",
"""                    elif not throttled:
                        pos, entry, state = -side * (sz + 2), cc[k], 3
                    else:
                        pos, state = 0, 4""")
    elif v == 'v2':  # budget-capped flip stop: total fade+flip stop pts*ct <= 470 (~$940)
        s = s.replace("""                    else:                           # counters drift / big agree: real momentum
                        pos, entry, state = -side * (sz + 2), cc[k], 3""",
"""                    else:
                        fl = sz + 2
                        cap = 470 - sz * (30.0 if abs(onv) > 200 else 40.0)
                        fstop = max(20.0, min(45.0, cap / fl))
                        pos, entry, state, fs = -side * fl, cc[k], 3, fstop""")
        s = s.replace("F_STOP_PTS, F_TGT_PTS = 45.0, 68.0", "F_TGT_PTS = 68.0")
        s = s.replace("if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:", "if pnl <= -fs or pnl >= F_TGT_PTS:")
        s = s.replace("pos, entry, state = side * sz, cc[k], 1", "pos, entry, state, fs = side * sz, cc[k], 1, 0.0")
    open('_sw.py', 'w').write(s)
    assert v in ('v0', 'flipoff') or ('throttled' in s) == (v == 'v1') or v in ('fs40', 'fsz1', 'v2'), f"replace failed for {v}"
    assert v not in ('v0', 'flipoff', 'fs40', 'fsz1') or s != BASE_SRC or v == 'v0', f"no change for {v}"
    m = importlib.import_module('_sw'); importlib.reload(m)
    return m

df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
WINDOW, START, TARGET, MLL = 30, 50000.0, 3000.0, 2000.0

def allrate(dp, dm):
    n = len(dp); out = []
    for st in range(0, n - WINDOW + 1):
        bal, peak_eod, best, days, ok = START, START, 0.0, 0, False
        for s in range(st, st + WINDOW):
            if bal + dm[s] <= min(peak_eod - MLL, START): break
            bal += dp[s]; days += 1
            best, peak_eod = max(best, dp[s]), max(peak_eod, bal)
            p = bal - START
            if p >= TARGET and days >= 2 and best < 0.5 * p: ok = True; break
        out.append(ok)
    return np.mean(out), int(np.sum(out))

for v in ('v0', 'flipoff', 'fs40', 'fsz1', 'v1', 'v2'):
    m = build(v)
    tg = np.asarray(m.signals(df)).astype(np.int64)
    dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max() + 1)
    sc = sim.score(dp, dm)[0]
    r, npass = allrate(dp, dm)
    dll = int((dp <= -990).sum())
    print(f"{v}: base={sc:.4f} allrate={r:.4f} ({npass}/186) dll_days={dll} total_pnl={dp.sum():.0f}", flush=True)
