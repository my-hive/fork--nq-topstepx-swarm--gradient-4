"""gu-4: revive dead-zone cold+agree days at small size (53/214 flat days slow every attempt)."""
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
    print(f"{name:34s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={dp.sum():.0f} worst={dp.min():.0f}")
    return sc

src = open("strategy.py").read()

def variant(subs):
    s = src
    for a, b in subs:
        assert a in s, f"NOT FOUND: {a[:60]}"
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "variant", "exec"), ns)
    return ns["signals"](DF)

SZ = """        if hot:
            sz = 10 if agree else 7
        else:
            sz = 8 if not agree else 0"""

for csz in [2, 3, 4, 5, 6]:
    B = SZ.replace("sz = 8 if not agree else 0", f"sz = 8 if not agree else {csz}")
    sig = variant([(SZ, B)])
    run(sig, f"cold-agr sz {csz}")
    if csz == 3:
        for sh in (10, 25, 40):
            run(sig, f"cold-agr sz {csz} shift{sh}", shift=sh)

# tighter brackets for cold-agr small size: stop 30 tgt 60
for csz in [3, 4]:
    B = SZ.replace("sz = 8 if not agree else 0", f"sz = 8 if not agree else {csz}")
    s2 = src.replace(SZ, B)
    s2 = s2.replace(
        "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)",
        f"        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else ((30.0, 60.0) if agree else (40.0, 80.0))")
    ns = {}
    exec(compile(s2, "v", "exec"), ns)
    run(ns["signals"](DF), f"cold-agr sz {csz} 30/60")
