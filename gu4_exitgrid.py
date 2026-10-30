"""gu-4: fine EXIT grid around 14:30 on 13582df."""
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

def run(sig, name, shift=0):
    dp, dm = day_pnls(sig)
    if shift:
        dp, dm = dp[shift:], dm[shift:]
    sc, rates, passed = sim.score(dp, dm)
    print(f"{name:34s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={dp.sum():.0f} worst={dp.min():.0f}")
    return sc

src = open("strategy.py").read()

def variant(e):
    s = src.replace("14 * 3600 + 30 * 60", e)
    ns = {}
    exec(compile(s, "v", "exec"), ns)
    return ns["signals"](DF)

for hh, mm in [(13, 50), (14, 10), (14, 20), (14, 40), (14, 50)]:
    sig = variant(f"{hh} * 3600 + {mm} * 60")
    run(sig, f"EXIT {hh}:{mm:02d}")
    if mm in (20, 40):
        for sh in (10, 25, 40):
            run(sig, f"EXIT {hh}:{mm:02d} shift{sh}", shift=sh)
