"""gu-4: micro-timing START shifts +-1s and fill-mechanics bracket recenter."""
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

def run(sig, name):
    dp, dm = day_pnls(sig)
    sc, rates, passed = sim.score(dp, dm)
    print(f"{name:40s} score={sc:.7f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.3f} pnl={dp.sum():.0f}")
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

for off in [-2, -1, 1, 2]:
    run(variant([("START, EXIT = 8 * 3600 + 35 * 60,", f"START, EXIT = 8 * 3600 + 35 * 60 + {off},")]), f"START +{off}s")

# push measure window 8:30-8:35: try 8:30-8:34 / 8:31-8:35
for a, b in [("m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)", "m = (tt >= 8 * 3600 + 30 * 60) & (tt < START - 60)")]:
    run(variant([(a, b)]), "push window to 8:34")
for a, b in [("m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)", "m = (tt >= 8 * 3600 + 31 * 60) & (tt < START)")]:
    run(variant([(a, b)]), "push window from 8:31")
