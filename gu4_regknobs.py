"""gu-4: regime-targeted speed knobs on 13582df: bigON-agr sz 11, cold-dis Monday skip/halve."""
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

run(variant([(SZ, SZ.replace("sz = 10 if agree else 7", "sz = 11 if agree else 7"))]), "bigON+smallON-agr 11")

# bigON-agr only: size 11 when abs(onv)>200 and agree
B = SZ.replace("sz = 10 if agree else 7",
               "sz = 11 if (agree and abs(onv) > 200) else 10 if agree else 7")
sig = variant([(SZ, B)])
run(sig, "bigON-agr 11")
for sh in (10, 25, 40):
    run(sig, f"bigON-agr 11 shift{sh}", shift=sh)

B2 = SZ.replace("sz = 10 if agree else 7",
                "sz = 12 if (agree and abs(onv) > 200) else 10 if agree else 7")
run(variant([(SZ, B2)]), "bigON-agr 12")

# cold-dis Monday skip / halve
DOW = """        dow = dows[idx[0]]"""
run(variant([(DOW, DOW + """
        if not hot and not agree and dow == 0:
            sz = 0""")]), "cold-dis Mon skip")
run(variant([(DOW, DOW + """
        if not hot and not agree and dow == 0:
            sz = int(round(sz * 0.5))""")]), "cold-dis Mon halve")
