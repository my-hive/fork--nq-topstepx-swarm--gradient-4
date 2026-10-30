"""gu-4: DLL-risk-capped size-up: bigON-agr/smallON-agr sz 11 with flip-leg risk cap <= $950."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS

def day_pnls(sig):
    target = np.nan_to_num(np.asarray(sig, dtype=float)).astype(np.int64)
    return sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                        DF.low.values, DF.close.values, target, SESS.max() + 1)[:2]

def run(sig, name, shift=0):
    dp, dm = day_pnls(sig)
    if shift:
        dp, dm = dp[shift:], dm[shift:]
    sc, rates, passed = sim.score(dp, dm)
    print(f"{name:44s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={dp.sum():.0f} worst={dp.min():.0f}")
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

SZ = """        if hot:
            sz = 10 if agree else 7
        else:
            sz = 8 if not agree else 0"""
FLIP = """                        pos, entry, state = -side * (sz + 2), cc[k], 3"""

# A: bigON-agr 11, flip risk cap
A = variant([
    (SZ, SZ.replace("sz = 10 if agree else 7", "sz = 11 if (agree and abs(onv) > 200) else 10 if agree else 7")),
    (FLIP, """                        fsz = min(sz + 2, 11)
                        pos, entry, state = -side * fsz, cc[k], 3"""),
    ("            elif pos != 0 and state == 3:\n                pnl = (cc[k] - entry) * np.sign(pos)",
     "            elif pos != 0 and state == 3:\n                pnl = (cc[k] - entry) * np.sign(pos)\n                fstop = min(F_STOP_PTS, 475.0 / abs(pos))")])
A = A  # need F_STOP replace inside state3 check
sA = src
sA = sA.replace("sz = 10 if agree else 7", "sz = 11 if (agree and abs(onv) > 200) else 10 if agree else 7")
sA = sA.replace(FLIP, """                        fsz = min(sz + 2, 11)
                        pos, entry, state = -side * fsz, cc[k], 3""")
sA = sA.replace("                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:",
                "                if pnl <= -min(F_STOP_PTS, 475.0 / abs(pos)) or pnl >= F_TGT_PTS:")
ns = {}; exec(compile(sA, "v", "exec"), ns)
sig = ns["signals"](DF)
run(sig, "bigON-agr 11 + flip cap 475$/leg")
for sh in (10, 25, 40):
    run(sig, f"  shift{sh}", shift=sh)

# B: bigON-agr 11, flip sz cap 11, stop unchanged 45 -> risk 990
sB = src.replace("sz = 10 if agree else 7", "sz = 11 if (agree and abs(onv) > 200) else 10 if agree else 7")
sB = sB.replace(FLIP, """                        pos, entry, state = -side * min(sz + 2, 11), cc[k], 3""")
ns = {}; exec(compile(sB, "v", "exec"), ns)
run(ns["signals"](DF), "bigON-agr 11 + flip sz cap 11")

# C: smallON-agr 11 (fade risk 11*40*2=880 ok; flip cap as A)
sC = src.replace("sz = 10 if agree else 7", "sz = 11 if agree else 7")
sC = sC.replace(FLIP, """                        fsz = min(sz + 2, 11)
                        pos, entry, state = -side * fsz, cc[k], 3""")
sC = sC.replace("                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:",
                "                if pnl <= -min(F_STOP_PTS, 475.0 / abs(pos)) or pnl >= F_TGT_PTS:")
ns = {}; exec(compile(sC, "v", "exec"), ns)
run(ns["signals"](DF), "all-agr 11 + flip cap")
