"""gu-4: base vs small-ON84 shift robustness comparison."""
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

src = open("strategy.py").read()
def variant():
    s = src.replace(
        "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 80.0)",
        "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)")
    ns = {}
    exec(compile(s, "variant", "exec"), ns)
    return ns["signals"](DF)

for name, sig in [("base", S.signals(DF)), ("small84", variant())]:
    dp, dm = day_pnls(sig)
    for sh in [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]:
        d2, m2 = (dp[sh:], dm[sh:]) if sh else (dp, dm)
        sc, rates, passed = sim.score(d2, m2)
        print(f"{name} shift{sh:3d}: {sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={np.mean(passed) if passed else float('nan'):.1f}")
