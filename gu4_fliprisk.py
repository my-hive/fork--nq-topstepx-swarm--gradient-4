"""gu-4: reduce Jan-Feb DLL cluster: flip-leg risk trims. Score = official + full-window rate (starts 0..n-30)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
import strategy as S

def day_pnls(sig):
    target = np.nan_to_num(np.asarray(sig, dtype=float)).astype(np.int64)
    return sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                        DF.low.values, DF.close.values, target, SESS.max() + 1)[:2]

def full_rate(dp, dm):
    n = len(dp)
    res = []
    for st in range(n - 29):
        bal, peak, best, days = 50000.0, 50000.0, 0.0, 0
        out = (False, 30)
        for s in range(st, st + 30):
            thr = min(peak - 2000, 50000)
            if bal + dm[s] <= thr:
                out = (False, days); break
            bal += dp[s]; days += 1
            best, peak = max(best, dp[s]), max(peak, bal)
            if bal - 50000 >= 3000 and days >= 2 and best < 0.5 * (bal - 50000):
                out = (True, days); break
        res.append(out)
    p = [x[0] for x in res]
    return np.mean(p), np.mean([d for x, d in res if x]) if any(p) else 30

def run(sig, name):
    dp, dm = day_pnls(sig)
    sc, rates, passed = sim.score(dp, dm)
    fr, fd = full_rate(dp, dm)
    print(f"{name:44s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} FULLrate={fr:.3f} FULLdays={fd:.1f} "
          f"dll={(dm <= -1000).sum()} pnl={dp.sum():.0f}")
    return sc

src = open("strategy.py").read()

def variant(subs):
    s = src
    for a, b in subs:
        assert a in s, f"NOT FOUND: {a[:60]}"
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "v", "exec"), ns)
    return ns["signals"](DF)

run(S.signals(DF), "BASE 13582df")

FLIPCHK = "                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:"

# C: flip stop 35
run(variant([("F_STOP_PTS, F_TGT_PTS = 45.0, 68.0", "F_STOP_PTS, F_TGT_PTS = 35.0, 68.0")]), "flip stop 35")

# B: flip size = sz
run(variant([("-side * (sz + 2), cc[k], 3", "-side * sz, cc[k], 3")]), "flip sz (no +2)")

# D: skip flip if fade loss >= $600 (sz*stop*2)
run(variant([("""                    else:                           # counters drift / big agree: real momentum
                        pos, entry, state = -side * (sz + 2), cc[k], 3""",
              """                    elif sz * stop * 2.0 < 600:              # small fade loss only: flip
                        pos, entry, state = -side * (sz + 2), cc[k], 3
                    else:
                        pos, state = 0, 4""")]), "skip flip if fade loss>=$600")

# A: adaptive flip stop capping day risk at $950
sA = src.replace("-side * (sz + 2), cc[k], 3",
                 "-side * (sz + 2), cc[k], 3")
sA = sA.replace("        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)",
                "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)\n        fstop_c = max(10.0, (950.0 - sz * stop * 2.0) / (2.0 * (sz + 2)))")
sA = sA.replace(FLIPCHK, "                if pnl <= -min(F_STOP_PTS, fstop_c) or pnl >= F_TGT_PTS:")
ns = {}; exec(compile(sA, "v", "exec"), ns)
run(ns["signals"](DF), "flip stop adaptive $950 day cap")
