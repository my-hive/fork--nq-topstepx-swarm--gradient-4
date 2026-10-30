"""gu-4: holiday-skip window edges on bda6c57 base (unclaimed knob)."""
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
    print(f"{name:30s} score={sc:.7f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
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

H = "        if md >= 1210 or md <= 102:  # start 1210-1212 plateau (h1 1.0); 1215 leaves Dec8-9 windows short"
for lo, hi in [(1205, 102), (1208, 102), (1212, 102), (1210, 105), (1210, 108), (1210, 100), (1210, 95)]:
    run(variant([(H, f"        if md >= {lo} or md <= {hi}:")]), f"skip {lo}-{hi}")
