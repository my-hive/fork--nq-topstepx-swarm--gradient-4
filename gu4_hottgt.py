"""gu-4: HOT-regime fade target sweep (winner: hot tgt 84, cold 80)."""
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

def variant(ht):
    s = src.replace(
        "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 80.0)",
        f"        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, {ht}.0)")
    ns = {}
    exec(compile(s, "variant", "exec"), ns)
    return ns["signals"](DF)

for ht in [80, 82, 84, 86, 88]:
    sig = variant(ht)
    run(sig, f"hot tgt {ht}")
    if ht in (84, 86):
        for sh in (10, 25, 40):
            run(sig, f"hot tgt {ht} shift{sh}", shift=sh)
