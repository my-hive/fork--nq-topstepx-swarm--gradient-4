"""gu-4: push-size conditioning on 13582df: does bigger 8:30-8:35 push = bigger fade edge?"""
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

for mvth in [40, 50, 60]:
    B = SZ + f"""
        if sz > 0 and abs(mv) >= {mvth}:
            sz += 1"""
    sig = variant([(SZ, B)])
    run(sig, f"|mv|>={mvth} sz+1")
    if mvth == 50:
        for sh in (10, 25, 40):
            run(sig, f"|mv|>={mvth} sz+1 shift{sh}", shift=sh)

for mvth in [30, 40]:
    B = SZ + f"""
        if sz > 0 and abs(mv) < {mvth}:
            sz = max(1, sz - 2)"""
    run(variant([(SZ, B)]), f"|mv|<{mvth} sz-2")
